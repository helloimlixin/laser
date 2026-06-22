#!/bin/bash
# Single stage-1 run: FFHQ-256 LASER with PatchGAN + higher-res latent at a fixed
# stage-2 token budget (configs/model/laser_ffhq_gan.yaml). This is the "do better"
# follow-up to the over-smoothed run zdmy0xi3.
#
# Single-GPU by default for this probe; the adversarial path now uses standard
# DDP-compatible manual optimization when launched with multiple GPUs.
#
# Usage:
#   ./scripts/submit_ffhq256_gan_stage1.sh
#   DRY_RUN=1 ./scripts/submit_ffhq256_gan_stage1.sh
#   STAGE1_EPOCHS=50 BATCH_SIZE=6 ./scripts/submit_ffhq256_gan_stage1.sh

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -- Cluster (see memory: use gpu-redhat, MEM_MB <= 250000) ------------------
PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-120000}"

# -- Data --------------------------------------------------------------------
FFHQ_DIR="${FFHQ_DIR:-/scratch/$USER/datasets/ffhq}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-2}"   # fits 24GB (ds4+attention+LPIPS OOMs at batch 4 on a 3090)
NUM_WORKERS="${NUM_WORKERS:-8}"

# -- Stage-1 knobs (match zdmy0xi3's optimizer schedule) ---------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
# Discriminator warmup (training batches before the critic engages). Lower than
# the config default so the memory-heaviest (GAN-active) phase is exercised early
# — a 24GB card that survives this survives the whole run.
DISC_START="${DISC_START:-1000}"

# -- Output / logging --------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/ffhq256_gan_stage1}"
RUN_NAME="${RUN_NAME:-ffhq256-gan-ds4-p4-k8-attn16-32-${TS}}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
WANDB_MODE="${WANDB_MODE:-online}"

DRY_RUN="${DRY_RUN:-0}"

# -- Container (Amarel) ------------------------------------------------------
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_src_py311}"

if [[ ! -d "$FFHQ_DIR" ]]; then
  echo "FFHQ directory not found: $FFHQ_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== FFHQ-256 GAN stage-1 ==="
echo "  run_name=$RUN_NAME"
echo "  partition=$PARTITION gpus=$GPUS mem=${MEM_MB} batch=$BATCH_SIZE epochs=$STAGE1_EPOCHS"
echo "  out=$RUN_DIR"
echo "  wandb=$WANDB_PROJECT (mode=$WANDB_MODE)"

# -- Inner runner (executed inside the container) ----------------------------
RUNNER="$RUN_DIR/run.sh"
cat > "$RUNNER" <<RUNNER_EOF
#!/bin/bash
set -euo pipefail
echo "=== GPU inventory ==="; nvidia-smi; echo ""

unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODELIST 2>/dev/null || true
export PYTHONUSERBASE="$PYDEPS"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"
# ds4 (16x16 latent) + attention + the GAN phase is memory-heavy; reduce
# allocator fragmentation. Pair with --constraint=adalovelace (48GB L40S).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true

cd "$PROJECT_DIR"
echo "========================================"
echo "STAGE 1: FFHQ-256 LASER + PatchGAN (laser_ffhq_gan)"
echo "========================================"

python train.py stage1 \\
  seed=42 \\
  output_dir="$RUN_DIR/stage1" \\
  model=laser_ffhq_gan \\
  model.disc_start_step=$DISC_START \\
  data=ffhq \\
  data.data_dir="$FFHQ_DIR" \\
  data.image_size=$IMAGE_SIZE \\
  data.batch_size=$BATCH_SIZE \\
  data.num_workers=$NUM_WORKERS \\
  train.learning_rate=$STAGE1_LR \\
  train.warmup_steps=$WARMUP_STEPS \\
  train.min_lr_ratio=$MIN_LR_RATIO \\
  train.max_epochs=$STAGE1_EPOCHS \\
  train.accelerator=gpu \\
  train.devices=auto \\
  train.strategy=auto \\
  train.precision=bf16-mixed \\
  train.gradient_clip_val=1.0 \\
  train.log_every_n_steps=50 \\
  train.val_check_interval=1.0 \\
  wandb.project="$WANDB_PROJECT" \\
  wandb.name="$RUN_NAME"

echo "DONE: $RUN_NAME"
RUNNER_EOF
chmod +x "$RUNNER"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run) wrote runner: $RUNNER -- not submitting"
  exit 0
fi

SBATCH_ARGS=(
  --partition="$PARTITION"
  --job-name="ffhq256gan"
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task="$CPUS"
  --gres="gpu:${GPUS}"
  --mem="${MEM_MB}"
  --time="$TIME_LIMIT"
  --chdir="$PROJECT_DIR"
  --output="$RUN_DIR/${RUN_NAME}_%j.out"
  --error="$RUN_DIR/${RUN_NAME}_%j.err"
  --requeue
)
[[ -n "${CONSTRAINT:-}" ]] && SBATCH_ARGS+=(--constraint="$CONSTRAINT")

sbatch "${SBATCH_ARGS[@]}" --wrap='#!/bin/bash
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then set +u; source /usr/share/lmod/lmod/init/bash; set -u;
  elif [[ -f /usr/share/Modules/init/bash ]]; then set +u; source /usr/share/Modules/init/bash; set -u; fi
fi
if ! command -v singularity >/dev/null 2>&1; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
fi
echo "=== GPU inventory ==="; nvidia-smi; echo ""
if command -v singularity >/dev/null 2>&1; then
  singularity exec --nv \
    --bind "'"$PROJECT_DIR"'" \
    --bind "/scratch/'"$USER"'" \
    --bind "'"$FFHQ_DIR"'" \
    --bind "'"$RUN_DIR"'" \
    --bind /dev/shm \
    "'"$IMAGE"'" \
    bash "'"$RUNNER"'"
else
  echo "Warning: singularity not found; running bare" >&2
  bash "'"$RUNNER"'"
fi
'
echo "[submitted] $RUN_NAME -> $RUN_DIR"
