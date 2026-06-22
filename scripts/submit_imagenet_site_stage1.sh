#!/bin/bash
# Single stage-1 run: ImageNet-256 LASER per-site dictionary bottleneck
# (configs/model/laser_imagenet_site.yaml). This is the fixed RELAUNCH of run
# 88qwzymn (imagenet-site-ds4-k16-a8192-prod-150k), whose dictionary was frozen
# because online_ksvd never updated on the non-adversarial path. The config here
# uses dictionary_update_mode=gradient + commitment_cost=0.25 (see commit message
# / the config header). Fresh run, NOT a resume — a resume would carry the frozen
# dictionary and inflated coefficients forward.
#
# Multi-GPU DDP: adversarial loss is OFF, so this uses Lightning AUTOMATIC
# optimization, which is DDP-safe (unlike the manual-optimization GAN path).
#
# Usage:
#   ./scripts/submit_imagenet_site_stage1.sh
#   DRY_RUN=1 ./scripts/submit_imagenet_site_stage1.sh
#   GPUS=3 BATCH_SIZE=3 MAX_STEPS=150000 ./scripts/submit_imagenet_site_stage1.sh
#   # K-SVD atoms instead of gradient atoms (now that ksvd is fixed):
#   EXTRA_OVERRIDES="model.dictionary_update_mode=online_ksvd model.dictionary_ksvd_lr=0.08" \
#     ./scripts/submit_imagenet_site_stage1.sh

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -- Cluster (see memory: use gpu-redhat, MEM_MB <= 250000) ------------------
PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPUS:-3}}"
GPUS="$GPUS_PER_NODE"       # Backward-compatible alias: GPUS means per-node GPUs.
WORLD_SIZE=$((NODES * GPUS_PER_NODE))
CPUS="${CPUS:-32}"          # per node: ranks x dataloader workers + overhead
MEM_MB="${MEM_MB:-200000}"  # <= 250000 hard cap

# -- Data --------------------------------------------------------------------
IMAGENET_DIR="${IMAGENET_DIR:-/scratch/$USER/Projects/data/imagenet}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# -- Stage-1 knobs (match 88qwzymn) ------------------------------------------
MAX_STEPS="${MAX_STEPS:-150000}"
STAGE1_LR="${STAGE1_LR:-1.0e-4}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-30000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-256}"
SAVE_TOP_K="${SAVE_TOP_K:-0}"
SAVE_LAST="${SAVE_LAST:-false}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# -- Output / logging --------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/imagenet_site_followup}"
RUN_NAME="${RUN_NAME:-imagenet-site-ds4-k16-a8192-gradient-150k-${TS}}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-1}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-0}"

DRY_RUN="${DRY_RUN:-0}"

# -- Container (Amarel) ------------------------------------------------------
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_src_py311}"

if [[ ! -d "$IMAGENET_DIR" ]]; then
  echo "ImageNet directory not found: $IMAGENET_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== ImageNet-256 site stage-1 (FIXED relaunch of 88qwzymn) ==="
echo "  run_name=$RUN_NAME"
echo "  partition=$PARTITION nodes=$NODES gpus_per_node=$GPUS_PER_NODE world_size=$WORLD_SIZE cpus_per_node=$CPUS mem=${MEM_MB} batch_per_gpu=$BATCH_SIZE max_steps=$MAX_STEPS"
echo "  out=$RUN_DIR"
echo "  wandb=$WANDB_PROJECT (mode=$WANDB_MODE media_disabled=$LASER_DISABLE_WANDB_MEDIA)"
[[ -n "$EXTRA_OVERRIDES" ]] && echo "  extra=$EXTRA_OVERRIDES"

# -- Inner runner (executed inside the container) ----------------------------
RUNNER="$RUN_DIR/run.sh"
cat > "$RUNNER" <<RUNNER_EOF
#!/bin/bash
set -euo pipefail
echo "=== GPU inventory ==="; nvidia-smi; echo ""

if [[ "$NODES" -le 1 ]]; then
  unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODELIST 2>/dev/null || true
fi
export PYTHONUSERBASE="$PYDEPS"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"
export LASER_DISABLE_WANDB_MEDIA="$LASER_DISABLE_WANDB_MEDIA"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "\$PYTHONUSERBASE"
if command -v flock >/dev/null 2>&1; then
  (
    flock 9
    pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
  ) 9>"\$PYTHONUSERBASE/.install.lock"
else
  pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
fi

cd "$PROJECT_DIR"
echo "========================================"
echo "STAGE 1: ImageNet-256 LASER site (laser_imagenet_site)"
echo "========================================"

python train_stage1_autoencoder.py \\
  seed=42 \\
  output_dir="$RUN_DIR/stage1" \\
  model=laser_imagenet_site \\
  data=imagenet \\
  data.data_dir="$IMAGENET_DIR" \\
  data.image_size=$IMAGE_SIZE \\
  data.batch_size=$BATCH_SIZE \\
  data.num_workers=$NUM_WORKERS \\
  train.learning_rate=$STAGE1_LR \\
  train.max_steps=$MAX_STEPS \\
  train.val_check_interval=$VAL_CHECK_INTERVAL \\
  train.limit_val_batches=$LIMIT_VAL_BATCHES \\
  train.accelerator=gpu \\
  train.devices=$GPUS \\
  train.num_nodes=$NODES \\
  train.strategy=ddp \\
  train.precision=bf16-mixed \\
  train.log_every_n_steps=50 \\
  train.run_test_after_fit=false \\
  checkpoint.save_top_k=$SAVE_TOP_K \\
  checkpoint.save_last=$SAVE_LAST \\
  model.log_images_every_n_steps=$LOG_IMAGES_EVERY \\
  model.enable_val_latent_visuals=false \\
  wandb.project="$WANDB_PROJECT" \\
  wandb.name="$RUN_NAME" \\
  wandb.append_timestamp=false \\
  $EXTRA_OVERRIDES

echo "DONE: $RUN_NAME"
RUNNER_EOF
chmod +x "$RUNNER"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run) wrote runner: $RUNNER -- not submitting"
  echo "--- runner contents ---"
  cat "$RUNNER"
  exit 0
fi

SBATCH_ARGS=(
  --partition="$PARTITION"
  --job-name="insite150k"
  --nodes="$NODES"
  --ntasks="$NODES"
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

SBATCH_SCRIPT="$RUN_DIR/sbatch.sh"
cat > "$SBATCH_SCRIPT" <<SBATCH_EOF
#!/bin/bash
set -euo pipefail
IMAGENET_NODES=$NODES
LAUNCH=()
if [[ "\$IMAGENET_NODES" -gt 1 ]]; then
  LAUNCH=(srun --nodes="\$IMAGENET_NODES" --ntasks="\$IMAGENET_NODES" --ntasks-per-node=1)
fi
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
  "\${LAUNCH[@]}" singularity exec --nv \
    --bind "$PROJECT_DIR" \
    --bind "/scratch/$USER" \
    --bind "$IMAGENET_DIR" \
    --bind "$RUN_DIR" \
    --bind /dev/shm \
    "$IMAGE" \
    bash "$RUNNER"
else
  echo "Warning: singularity not found; running bare" >&2
  "\${LAUNCH[@]}" bash "$RUNNER"
fi
SBATCH_EOF
chmod +x "$SBATCH_SCRIPT"

sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT"
echo "[submitted] $RUN_NAME -> $RUN_DIR"
