#!/bin/bash
# Sweep: Quantized coefficient AR prior over patched stage-1 checkpoints.
#
# Re-extracts tokens with quantized coefficients from existing stage-1
# checkpoints (patched and non-patched), then trains stage-2 AR models.
#
# Sweep axes:
#   - stage1 model: fast_r, p4s4_k16_d4k, p4s4_k24_d4k, p8s8_k24_d4k, p8s8_k16_d6k
#   - coeff_bins: 256, 512
#   - AR model: small (d=256) vs large (d=512)
#
# Usage:
#   ./scripts/sweep_quantized_coeffs_patched.sh
#   CASE_FILTER=p4s4_k16_q512_large ./scripts/sweep_quantized_coeffs_patched.sh
#   DRY_RUN=1 ./scripts/sweep_quantized_coeffs_patched.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -- Cluster ----------------------------------------------------------------
PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-96000}"

# -- Data -------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
DATASET="${DATASET:-celeba}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# -- Training ---------------------------------------------------------------
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
STAGE2_LR="${STAGE2_LR:-3e-4}"

# -- Output / logging ------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/src_qcoeff_patched_sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_PREFIX="${RUN_PREFIX:-src_qcp}"
JOB_PREFIX="${JOB_PREFIX:-srcqcp}"

CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

# -- Container (Amarel) ----------------------------------------------------
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_src_py311}"

# -- Stage-1 checkpoint root ------------------------------------------------
V3_ROOT="${V3_ROOT:-/scratch/$USER/runs/src_pq_sweep_v3_10ep}"

# -- Pre-flight -------------------------------------------------------------
echo "=== Login node GPU check ==="
nvidia-smi 2>/dev/null || echo "(no GPUs on login node -- compute nodes will have them)"
echo ""

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# -- Cases: name | stage1_subdir | coeff_bins | d_model | n_heads | n_layers | d_ff
#
# Each case reuses an existing stage-1 checkpoint, re-extracts with quantized
# coefficients, and trains a stage-2 AR model.
#
cases=(
  # ================================================================
  # fast_r (non-patched, emb=16, K=1024, k=8) — for comparison
  # ================================================================
  "fast_r_q256_large|src_pqv3_10ep_fast_r|256|512|8|6|2048"
  "fast_r_q512_large|src_pqv3_10ep_fast_r|512|512|8|6|2048"

  # ================================================================
  # p4s4_k16_d4k (patch=4, stride=4, k=16, K=4096, emb=4, dim=64)
  # ================================================================
  "p4s4_k16_q256_small|src_pqv3_10ep_p4s4_k16_d4k|256|256|4|6|512"
  "p4s4_k16_q256_large|src_pqv3_10ep_p4s4_k16_d4k|256|512|8|6|2048"
  "p4s4_k16_q512_small|src_pqv3_10ep_p4s4_k16_d4k|512|256|4|6|512"
  "p4s4_k16_q512_large|src_pqv3_10ep_p4s4_k16_d4k|512|512|8|6|2048"

  # ================================================================
  # p4s4_k24_d4k (patch=4, stride=4, k=24, K=4096, emb=4, dim=64)
  # ================================================================
  "p4s4_k24_q256_large|src_pqv3_10ep_p4s4_k24_d4k|256|512|8|6|2048"
  "p4s4_k24_q512_large|src_pqv3_10ep_p4s4_k24_d4k|512|512|8|6|2048"

  # ================================================================
  # p8s8_k24_d4k (patch=8, stride=8, k=24, K=4096, emb=4, dim=256)
  # ================================================================
  "p8s8_k24_q256_large|src_pqv3_10ep_p8s8_k24_d4k|256|512|8|6|2048"
  "p8s8_k24_q512_large|src_pqv3_10ep_p8s8_k24_d4k|512|512|8|6|2048"

  # ================================================================
  # p8s8_k16_d6k (patch=8, stride=8, k=16, K=6144, emb=4, dim=256)
  # ================================================================
  "p8s8_k16d6k_q256_large|src_pqv3_10ep_p8s8_k16_d6k|256|512|8|6|2048"
  "p8s8_k16d6k_q512_large|src_pqv3_10ep_p8s8_k16_d6k|512|512|8|6|2048"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name stage1_subdir coeff_bins d_model n_heads n_layers d_ff <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  # Find stage-1 checkpoint
  stage1_dir="$V3_ROOT/$stage1_subdir/stage1"
  stage1_ckpt="$(find "$stage1_dir" -name '*.ckpt' -path '*/checkpoints/*' 2>/dev/null | sort | tail -1)"
  if [[ -z "$stage1_ckpt" ]]; then
    echo "[Sweep] SKIP $case_name — no stage-1 checkpoint in $stage1_dir"
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  run_dir="$OUT_ROOT/$run_name"

  echo "[Sweep] $case_name  bins=$coeff_bins  d=$d_model  ckpt=$(basename $stage1_subdir)"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry run -- skipped)"
    submitted=$((submitted + 1))
    continue
  fi

  mkdir -p "$run_dir"

  RUNNER="$run_dir/run_${case_name}.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/bin/bash
set -euo pipefail

echo "=== GPU inventory ==="
nvidia-smi
echo ""

unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODELIST 2>/dev/null || true

export PYTHONUSERBASE="$PYDEPS"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"

pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib 2>/dev/null || true

cd "$PROJECT_DIR"

STAGE2_DIR="$run_dir/stage2"
TOKEN_CACHE="$run_dir/token_cache_q${coeff_bins}.pt"
mkdir -p "\$STAGE2_DIR"

STAGE1_CKPT="$stage1_ckpt"
echo "Using stage-1 checkpoint: \$STAGE1_CKPT"

echo ""
echo "========================================"
echo "TOKEN EXTRACTION (quantized: bins=$coeff_bins, uniform)"
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
  --coeff-quantization uniform

echo ""
echo "========================================"
echo "STAGE 2: AR Prior ($case_name, bins=$coeff_bins, d=$d_model)"
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
