#!/usr/bin/env bash
# Continue the compact FFHQ LASER stage-1 checkpoint with targeted GAN/capacity
# probes. Defaults are set for W&B run igsg1pm5.

set -euo pipefail

USER_NAME="${USER:-xl598}"

SNAPSHOT_PATH="${SNAPSHOT_PATH:-/scratch/$USER_NAME/submission_snapshots/laser_laser_train_ffhq-dense-nonpatch-k2-z64-a8192-seq512-reconsharp-s1-8-adv3-s2-6_20260608_180810}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/scratch/$USER_NAME/Projects/laser/runs/vision_lsr_patch_pipeline_ffhq_20260608/laser-train-ffhq-dense-nonpatch-k2-z64-a8192-seq512-reconsharp-s1-8-adv3-s2-6-20260608_180810/ffhq}"
INIT_CKPT="${INIT_CKPT:-$SOURCE_RUN_ROOT/stage1_adv/checkpoints/run_20260608_222740/laser/final.ckpt}"
FFHQ_DIR="${FFHQ_DIR:-/cache/home/$USER_NAME/datasets/ffhq/images1024x1024_webp}"

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-120000}"
CONSTRAINT="${CONSTRAINT:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018,gpuk[005-018]}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STAGE1_LR="${STAGE1_LR:-4e-5}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-2}"
ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.03}"
USE_ADAPTIVE_DISC_WEIGHT="${USE_ADAPTIVE_DISC_WEIGHT:-false}"
DISC_LEARNING_RATE="${DISC_LEARNING_RATE:-null}"
SPARSITY_REG_WEIGHT="${SPARSITY_REG_WEIGHT:-0.01}"

WANDB_PROJECT="${WANDB_PROJECT:-laser-vision-lsr-compact-ffhq}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-ffhq-dense-nonpatch-k${SPARSITY_LEVEL}-z64-a8192-seq$((256 * SPARSITY_LEVEL))-adv${ADVERSARIAL_WEIGHT//./p}-adaptive${USE_ADAPTIVE_DISC_WEIGHT}-cont${STAGE1_EPOCHS}-${STAMP}}"
WANDB_GROUP="${WANDB_GROUP:-laser-train-${RUN_LABEL}}"
WANDB_NAME="${WANDB_NAME:-ffhq-stage1-adv-probe}"
WANDB_MODE="${WANDB_MODE:-online}"

RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER_NAME/Projects/laser/runs/vision_lsr_patch_pipeline_ffhq_20260609}"
RUN_DIR="$RUN_ROOT_BASE/$RUN_LABEL/ffhq"
RUNNER="$RUN_DIR/run_stage1_adv_probe.sh"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHONUSERBASE_DEFAULT="/scratch/$USER_NAME/.pydeps/laser_src_py311"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"

for required in "$SNAPSHOT_PATH" "$INIT_CKPT" "$FFHQ_DIR"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required path does not exist: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR"

cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -euo pipefail

export PYTHONUSERBASE="\${PYTHONUSERBASE:-$PYTHONUSERBASE_DEFAULT}"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="\$PYTHONUSERBASE/lib/python3.11/site-packages:\$PYTHONUSERBASE/lib/python3.12/site-packages:$SNAPSHOT_PATH\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-/scratch/$USER_NAME/.cache}"
export TORCH_HOME="\${TORCH_HOME:-\$XDG_CACHE_HOME/torch}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-\$XDG_CACHE_HOME/pip}"
export WANDB_CACHE_DIR="\${WANDB_CACHE_DIR:-\$XDG_CACHE_HOME/wandb}"
export WANDB_CONFIG_DIR="\${WANDB_CONFIG_DIR:-/scratch/$USER_NAME/.config/wandb}"
export MPLCONFIGDIR="\${MPLCONFIGDIR:-\$XDG_CACHE_HOME/matplotlib}"
export HYDRA_FULL_ERROR=1

mkdir -p "\$PYTHONUSERBASE" "\$XDG_CACHE_HOME" "\$TORCH_HOME" "\$PIP_CACHE_DIR" "\$WANDB_CACHE_DIR" "\$WANDB_CONFIG_DIR" "\$MPLCONFIGDIR" "$RUN_DIR/wandb"

cd "$SNAPSHOT_PATH"
"$PYTHON_BIN" train_stage1_autoencoder.py \\
  seed=42 \\
  output_dir="$RUN_DIR/stage1_adv_probe" \\
  init_ckpt_path="$INIT_CKPT" \\
  model=laser \\
  data=ffhq \\
  data.data_dir="$FFHQ_DIR" \\
  data.image_size=256 \\
  data.batch_size=$BATCH_SIZE \\
  data.num_workers=$NUM_WORKERS \\
  train.max_epochs=$STAGE1_EPOCHS \\
  train.max_steps=-1 \\
  train.limit_train_batches=1.0 \\
  train.limit_val_batches=1.0 \\
  train.limit_test_batches=1.0 \\
  train.run_test_after_fit=false \\
  train.learning_rate=$STAGE1_LR \\
  train.warmup_steps=0 \\
  train.min_lr_ratio=$MIN_LR_RATIO \\
  train.gradient_clip_val=1.0 \\
  train.log_every_n_steps=50 \\
  train.val_check_interval=1.0 \\
  train.devices=1 \\
  train.strategy=auto \\
  train.precision=bf16-mixed \\
  train.accelerator=gpu \\
  model.backbone=vqgan \\
  model.num_downsamples=4 \\
  model.channel_multipliers=[1,1,2,2,4] \\
  model.attn_resolutions=[] \\
  model.use_mid_attention=false \\
  model.decoder_extra_residual_layers=1 \\
  model.num_hiddens=128 \\
  model.num_residual_blocks=2 \\
  model.num_residual_hiddens=64 \\
  model.backbone_latent_channels=256 \\
  model.max_ch_mult=4 \\
  model.out_tanh=false \\
  model.embedding_dim=64 \\
  model.num_embeddings=8192 \\
  model.sparsity_level=$SPARSITY_LEVEL \\
  model.patch_based=false \\
  model.patch_size=1 \\
  model.patch_stride=1 \\
  model.patch_reconstruction=tile \\
  model.coef_max=8.0 \\
  model.bounded_omp_refine_steps=16 \\
  model.bottleneck_loss_weight=0.75 \\
  model.commitment_cost=1.0 \\
  model.dict_learning_rate=4e-5 \\
  model.sparsity_reg_weight=$SPARSITY_REG_WEIGHT \\
  model.recon_mse_weight=0.25 \\
  model.recon_l1_weight=1.0 \\
  model.recon_edge_weight=0.50 \\
  model.perceptual_weight=0.10 \\
  model.perceptual_start_step=0 \\
  model.perceptual_warmup_steps=0 \\
  model.adversarial_weight=$ADVERSARIAL_WEIGHT \\
  model.adversarial_start_step=0 \\
  model.adversarial_warmup_steps=0 \\
  model.disc_start_step=0 \\
  model.disc_norm=group \\
  model.disc_loss=hinge \\
  model.disc_learning_rate=$DISC_LEARNING_RATE \\
  model.use_adaptive_disc_weight=$USE_ADAPTIVE_DISC_WEIGHT \\
  model.compute_fid=true \\
  model.log_images_every_n_steps=200 \\
  wandb.project="$WANDB_PROJECT" \\
  wandb.group="$WANDB_GROUP" \\
  wandb.name="$WANDB_NAME" \\
  wandb.tags=[train,laser,ffhq,stage1,adversarial,probe] \\
  wandb.append_timestamp=false \\
  wandb.save_dir="$RUN_DIR/wandb"
RUNNER_EOF
chmod +x "$RUNNER"

echo "=== FFHQ compact adversarial probe ==="
echo "snapshot=$SNAPSHOT_PATH"
echo "init_ckpt=$INIT_CKPT"
echo "run_dir=$RUN_DIR"
echo "wandb=$WANDB_PROJECT/$WANDB_GROUP/$WANDB_NAME"
echo "sparsity=$SPARSITY_LEVEL adv_weight=$ADVERSARIAL_WEIGHT adaptive=$USE_ADAPTIVE_DISC_WEIGHT epochs=$STAGE1_EPOCHS"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "(dry run) wrote runner: $RUNNER"
  exit 0
fi

SBATCH_ARGS=(
  --partition="$PARTITION"
  --job-name="ffhqadvprobe"
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task="$CPUS"
  --gres="gpu:${GPUS}"
  --mem="$MEM_MB"
  --time="$TIME_LIMIT"
  --chdir="$SNAPSHOT_PATH"
  --output="$RUN_DIR/${RUN_LABEL}_%j.out"
  --error="$RUN_DIR/${RUN_LABEL}_%j.err"
  --requeue
)
if [[ -n "${CONSTRAINT// }" ]]; then
  SBATCH_ARGS+=(--constraint="$CONSTRAINT")
fi
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  SBATCH_ARGS+=(--exclude="$EXCLUDE_NODES")
fi

sbatch "${SBATCH_ARGS[@]}" --wrap='#!/usr/bin/env bash
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then set +u; source /usr/share/lmod/lmod/init/bash; set -u;
  elif [[ -f /usr/share/Modules/init/bash ]]; then set +u; source /usr/share/Modules/init/bash; set -u;
  elif [[ -f /etc/profile.d/modules.sh ]]; then set +u; source /etc/profile.d/modules.sh; set -u; fi
fi
CONTAINER_BIN=""
for candidate in singularity apptainer; do
  if command -v "$candidate" >/dev/null 2>&1; then CONTAINER_BIN="$candidate"; break; fi
done
if [[ -z "$CONTAINER_BIN" ]]; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
  module load apptainer 2>/dev/null || true
  for candidate in singularity apptainer; do
    if command -v "$candidate" >/dev/null 2>&1; then CONTAINER_BIN="$candidate"; break; fi
  done
fi
echo "container_bin=$CONTAINER_BIN"
nvidia-smi || true
if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec --nv \
    --bind "'"$SNAPSHOT_PATH"'" \
    --bind "/scratch/'"$USER_NAME"'" \
    --bind "'"$FFHQ_DIR"'" \
    --bind "'"$RUN_DIR"'" \
    --bind /dev/shm \
    --bind /projects \
    "'"$IMAGE"'" \
    bash "'"$RUNNER"'"
else
  bash "'"$RUNNER"'"
fi
'

echo "[submitted] $RUN_LABEL -> $RUN_DIR"
