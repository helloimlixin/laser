#!/bin/bash
# Sweep: Quantized coefficient AR prior over existing stage-1 checkpoints.
#
# Takes trained stage-1 checkpoints, re-extracts tokens with quantized
# coefficients (coeff_bins > 0), and trains stage-2 AR models.
#
# Sweep axes:
#   - coeff_bins: 256 vs 512 vs 1024
#   - coeff_quantization: uniform vs mu_law
#   - AR model size: small (d=256) vs large (d=512)
#
# Usage:
#   ./scripts/sweep_quantized_coeffs.sh
#   CASE_FILTER=q512_uniform_large ./scripts/sweep_quantized_coeffs.sh
#   DRY_RUN=1 ./scripts/sweep_quantized_coeffs.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -- Cluster ----------------------------------------------------------------
PARTITION="${PARTITION:-auto}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-96000}"

# -- Data -------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
DATASET="${DATASET:-celeba}"

# -- Stage 1 checkpoint (reuse existing trained model) ----------------------
# Default: use the fast_r checkpoint from v3 sweep.
STAGE1_CKPT="${STAGE1_CKPT:-auto}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-/scratch/$USER/runs/src_pq_sweep_v3_10ep/src_pqv3_10ep_fast_r}"

# -- Training ---------------------------------------------------------------
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
STAGE2_LR="${STAGE2_LR:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# -- Output / logging ------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/src_quant_coeff_sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_PREFIX="${RUN_PREFIX:-src_qcoeff}"
JOB_PREFIX="${JOB_PREFIX:-srcqc}"

CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

# -- Container (Amarel) ----------------------------------------------------
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_src_py311}"

# -- Pre-flight -------------------------------------------------------------
echo "=== Login node GPU check ==="
nvidia-smi 2>/dev/null || echo "(no GPUs on login node -- compute nodes will have them)"
echo ""

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

# -- Partition auto-selection -----------------------------------------------
auto_select_partition() {
  if [[ -n "${PARTITION:-}" && "$PARTITION" != "auto" ]]; then
    return 0
  fi
  local best="" best_idle=-1
  while read -r part avail tl nodes _rest; do
    [[ -z "${part:-}" || "$part" == "PARTITION" ]] && continue
    part="${part%\*}"
    case "$part" in gpu|gpu-redhat|cgpu) ;; *) continue ;; esac
    local idle
    idle=$(echo "$nodes" | grep -oP '\d+(?=/\d+/\d+$)' 2>/dev/null || echo "0")
    idle="${idle:-0}"
    if (( idle > best_idle )); then
      best="$part"; best_idle=$idle
    fi
  done < <(sinfo -s 2>/dev/null)
  PARTITION="${best:-gpu-redhat}"
  echo "[Sweep] auto-selected PARTITION=$PARTITION (idle=$best_idle)"
}
auto_select_partition

mkdir -p "$OUT_ROOT"

# -- Cases: name | coeff_bins | coeff_quantization | d_model | n_heads | n_layers | d_ff
#
# Earlier successful quantized run used: bins=512, uniform, d=512, h=8, l=6, ff=2048
# This sweep explores bin counts, quantization schemes, and model sizes.
#
cases=(
  # -- Baseline matches earlier success: 512 bins, uniform, large model --
  "q512_uniform_large|512|uniform|512|8|6|2048"
  # -- 512 bins, uniform, small model (faster, compare capacity) --
  "q512_uniform_small|512|uniform|256|4|6|512"
  # -- 256 bins (fewer bins = easier classification, coarser coeffs) --
  "q256_uniform_large|256|uniform|512|8|6|2048"
  "q256_uniform_small|256|uniform|256|4|6|512"
  # -- 1024 bins (finer bins = better recon, harder classification) --
  "q1024_uniform_large|1024|uniform|512|8|6|2048"
  # -- mu-law quantization (better resolution near zero) --
  "q512_mulaw_large|512|mu_law|512|8|6|2048"
  "q256_mulaw_large|256|mu_law|512|8|6|2048"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name coeff_bins coeff_quantization d_model n_heads n_layers d_ff <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  run_dir="$OUT_ROOT/$run_name"

  echo "[Sweep] $case_name  bins=$coeff_bins  quant=$coeff_quantization  d=$d_model"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry run -- skipped)"
    submitted=$((submitted + 1))
    continue
  fi

  mkdir -p "$run_dir"

  # -- Compute coeff_mu for mu-law quantization ----------------------------
  COEFF_MU=0
  if [[ "$coeff_quantization" == "mu_law" ]]; then
    COEFF_MU=255
  fi

  # -- Build the runner script ---------------------------------------------
  RUNNER="$run_dir/run_${case_name}.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/bin/bash
set -euo pipefail

echo "=== GPU inventory ==="
nvidia-smi
echo ""

# Unset SLURM variables that confuse Lightning's DDP auto-detection.
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODELIST 2>/dev/null || true

export PYTHONUSERBASE="$PYDEPS"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"

pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib 2>/dev/null || true

cd "$PROJECT_DIR"

STAGE2_DIR="$run_dir/stage2"
TOKEN_CACHE="$run_dir/token_cache_q${coeff_bins}_${coeff_quantization}.pt"
mkdir -p "\$STAGE2_DIR"

# Find stage-1 checkpoint
STAGE1_CKPT="$STAGE1_CKPT"
if [[ "\$STAGE1_CKPT" == "auto" ]]; then
  STAGE1_CKPT="\$(find "$STAGE1_RUN_DIR/stage1" -name '*.ckpt' -path '*/checkpoints/*' | sort | tail -1)"
fi
if [[ -z "\$STAGE1_CKPT" || ! -f "\$STAGE1_CKPT" ]]; then
  echo "ERROR: No stage-1 checkpoint found. Set STAGE1_CKPT or check STAGE1_RUN_DIR." >&2
  echo "  STAGE1_RUN_DIR=$STAGE1_RUN_DIR" >&2
  exit 1
fi
echo "Using stage-1 checkpoint: \$STAGE1_CKPT"

echo ""
echo "========================================"
echo "TOKEN EXTRACTION (quantized: bins=$coeff_bins, $coeff_quantization)"
echo "========================================"

python extract_token_cache.py \\
  --stage1-checkpoint "\$STAGE1_CKPT" \\
  --output-path "\$TOKEN_CACHE" \\
  --dataset "$DATASET" \\
  --data-dir "$DATA_DIR" \\
  --image-size $IMAGE_SIZE \\
  --batch-size $BATCH_SIZE \\
  --num-workers $NUM_WORKERS \\
  --seed 42 \\
  --coeff-max auto \\
  --coeff-bins $coeff_bins \\
  --coeff-quantization $coeff_quantization \\
  --coeff-mu $COEFF_MU

echo ""
echo "========================================"
echo "STAGE 2: Autoregressive Prior Training"
echo "  $run_name  bins=$coeff_bins  quant=$coeff_quantization  d=$d_model"
echo "========================================"

python train_ar.py \\
  token_cache_path="\$TOKEN_CACHE" \\
  output_dir="\$STAGE2_DIR" \\
  seed=42 \\
  ar.type=sparse_spatial_depth \\
  ar.d_model=$d_model \\
  ar.n_heads=$n_heads \\
  ar.n_layers=$n_layers \\
  ar.d_ff=$d_ff \\
  ar.dropout=0.1 \\
  ar.learning_rate=$STAGE2_LR \\
  ar.warmup_steps=1000 \\
  ar.min_lr_ratio=0.01 \\
  ar.coeff_loss_type=auto \\
  ar.coeff_loss_weight=1.0 \\
  train_ar.batch_size=$STAGE2_BATCH_SIZE \\
  train_ar.max_epochs=$STAGE2_EPOCHS \\
  train_ar.accelerator=gpu \\
  train_ar.devices=1 \\
  train_ar.strategy=auto \\
  train_ar.precision=bf16-mixed \\
  train_ar.gradient_clip_val=1.0 \\
  train_ar.log_every_n_steps=50 \\
  train_ar.sample_every_n_epochs=1 \\
  train_ar.sample_log_to_wandb=true \\
  train_ar.log_recon_every_n_steps=200 \\
  train_ar.sample_num_images=8 \\
  train_ar.sample_temperature=1.0 \\
  train_ar.sample_coeff_mode=mean \\
  data.num_workers=$NUM_WORKERS \\
  wandb.project="$WANDB_PROJECT" \\
  wandb.name="${run_name}_stage2"

echo ""
echo "========================================"
echo "DONE: $run_name"
echo "========================================"
RUNNER_EOF
  chmod +x "$RUNNER"

  # -- Submit via sbatch ---------------------------------------------------
  SBATCH_ARGS=(
    --partition="$PARTITION"
    --job-name="$job_name"
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task="$CPUS"
    --gres="gpu:${GPUS}"
    --mem="${MEM_MB}"
    --time="$TIME_LIMIT"
    --chdir="$PROJECT_DIR"
    --output="$run_dir/${run_name}_%j.out"
    --error="$run_dir/${run_name}_%j.err"
    --requeue
  )

  sbatch "${SBATCH_ARGS[@]}" --wrap='#!/bin/bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
  fi
fi

if ! command -v singularity >/dev/null 2>&1; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
fi

echo "=== GPU inventory ==="
nvidia-smi
echo ""

if command -v singularity >/dev/null 2>&1; then
  singularity exec --nv \
    --bind "'"$PROJECT_DIR"'" \
    --bind "/scratch/'"$USER"'" \
    --bind "'"$DATA_DIR"'" \
    --bind "'"$run_dir"'" \
    --bind /dev/shm \
    "'"$IMAGE"'" \
    bash "'"$RUNNER"'"
else
  echo "Warning: singularity not found; running bare" >&2
  bash "'"$RUNNER"'"
fi
'

  submitted=$((submitted + 1))
done

echo ""
if ((submitted == 0)); then
  echo "No cases matched CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi
echo "[Sweep] submitted $submitted jobs to $OUT_ROOT"
