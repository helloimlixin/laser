#!/bin/bash
# Sweep: Proto-matched AR prior settings over existing stage-1 checkpoints.
#
# Matches the successful proto.py run (351lokx0):
#   - 12-layer transformer (d=512, h=8, ff=1024)
#   - coef_max=3, n_bins=256 (tight quantization)
#   - 100 stage2 epochs, lr=1e-3
#   - sample_temperature=0.5
#   - coeff_loss_type=gt_atom_recon_mse, coeff_loss_weight=0.1
#
# Tests both non-patched (fast_r) and patched (p4s4, p8s8) stage-1 models.
#
# Usage:
#   ./scripts/sweep_proto_matched.sh
#   CASE_FILTER=fast_r_proto ./scripts/sweep_proto_matched.sh
#   DRY_RUN=1 ./scripts/sweep_proto_matched.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -- Cluster ----------------------------------------------------------------
PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-96000}"

# -- Data -------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
DATASET="${DATASET:-celeba}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# -- Proto-matched stage 2 defaults ----------------------------------------
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
STAGE2_LR="${STAGE2_LR:-1e-3}"
COEF_MAX="${COEF_MAX:-3}"
COEFF_BINS="${COEFF_BINS:-256}"
SAMPLE_TEMP="${SAMPLE_TEMP:-0.5}"

# -- Output / logging ------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/src_proto_matched_sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_PREFIX="${RUN_PREFIX:-src_pm}"
JOB_PREFIX="${JOB_PREFIX:-srcpm}"

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

# -- Cases: name | stage1_subdir | coef_max_override
#
# All cases use the same proto-matched AR config (12L, d=512, 100ep, temp=0.5).
# The coef_max for token extraction may differ from stage-1's coef_max.
# We use coef_max=3 (proto default) for extraction — tighter bound = finer bins.
#
cases=(
  # Non-patched baseline (same as proto.py 351lokx0)
  "fast_r_proto|src_pqv3_10ep_fast_r"

  # Patched models
  "p4s4_k16_proto|src_pqv3_10ep_p4s4_k16_d4k"
  "p4s4_k24_proto|src_pqv3_10ep_p4s4_k24_d4k"
  "p8s8_k24_proto|src_pqv3_10ep_p8s8_k24_d4k"
  "p8s8_k16d6k_proto|src_pqv3_10ep_p8s8_k16_d6k"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name stage1_subdir <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  stage1_dir="$V3_ROOT/$stage1_subdir/stage1"
  stage1_ckpt="$(find "$stage1_dir" -name '*.ckpt' -path '*/checkpoints/*' 2>/dev/null | sort | tail -1)"
  if [[ -z "$stage1_ckpt" ]]; then
    echo "[Sweep] SKIP $case_name — no stage-1 checkpoint in $stage1_dir"
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  run_dir="$OUT_ROOT/$run_name"

  echo "[Sweep] $case_name  bins=$COEFF_BINS  coef_max=$COEF_MAX  ep=$STAGE2_EPOCHS  temp=$SAMPLE_TEMP  ckpt=$(basename $stage1_subdir)"

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
TOKEN_CACHE="$run_dir/token_cache_q${COEFF_BINS}_cm${COEF_MAX}.pt"
mkdir -p "\$STAGE2_DIR"

STAGE1_CKPT="$stage1_ckpt"
echo "Using stage-1 checkpoint: \$STAGE1_CKPT"

echo ""
echo "========================================"
echo "TOKEN EXTRACTION (quantized: bins=$COEFF_BINS, coef_max=$COEF_MAX)"
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
  --coeff-max $COEF_MAX \\
  --coeff-bins $COEFF_BINS \\
  --coeff-quantization uniform

echo ""
echo "========================================"
echo "STAGE 2: Proto-matched AR ($case_name)"
echo "  12L d=512 h=8 ff=1024 | 100ep lr=1e-3 | temp=$SAMPLE_TEMP"
echo "========================================"

python train_ar.py \\
  token_cache_path="\$TOKEN_CACHE" \\
  output_dir="\$STAGE2_DIR" \\
  seed=42 \\
  ar.type=sparse_spatial_depth \\
  ar.d_model=512 \\
  ar.n_heads=8 \\
  ar.n_layers=12 \\
  ar.d_ff=1024 \\
  ar.dropout=0.1 \\
  ar.learning_rate=$STAGE2_LR \\
  ar.warmup_steps=1000 \\
  ar.min_lr_ratio=0.01 \\
  ar.coeff_loss_type=gt_atom_recon_mse \\
  ar.coeff_loss_weight=0.1 \\
  train_ar.batch_size=$STAGE2_BATCH_SIZE \\
  train_ar.max_epochs=$STAGE2_EPOCHS \\
  train_ar.accelerator=gpu \\
  train_ar.devices=1 \\
  train_ar.strategy=auto \\
  train_ar.precision=bf16-mixed \\
  train_ar.gradient_clip_val=1.0 \\
  train_ar.log_every_n_steps=50 \\
  train_ar.sample_every_n_epochs=5 \\
  train_ar.sample_log_to_wandb=true \\
  train_ar.log_recon_every_n_steps=500 \\
  train_ar.sample_num_images=8 \\
  train_ar.sample_temperature=$SAMPLE_TEMP \\
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
