#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE=/workspace/tmp/rqvae-rfid4483-continuation-r11-20260828
PYTHON=/workspace/tmp/laser-duplicate-venv/bin/python
CONFIG="$ROOT/configs/imagenet_laser_8x8x4_rqvae_dynamics.yaml"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ID="${WANDB_RUN_ID:-imglaser-h200-scaled-b64-${STAMP//-/}}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/imagenet-laser-h200-scaled-b64-$STAMP}"

mkdir -p "$RUN_ROOT"/{logs,wandb,wandb-cache,wandb-data,torch-cache} \
  "/workspace/tmp/$RUN_ID"
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export WANDB_MODE=online
export WANDB_ENTITY=helloimlixin-rutgers
export WANDB_PROJECT=laser
export WANDB_RUN_ID="$RUN_ID"
export WANDB_NAME="$RUN_ID"
export WANDB_RUN_GROUP=imglaser-h200-scaled
export WANDB_TAGS=stage1,imagenet,laser,scratch,upstream-config,h200-scaled,world8,local-batch64,effective-batch512,lr8e-5,no-activation-checkpointing,fp32,8x8x4,discriminator-verified
export WANDB_RESUME=allow
export WANDB_CHECKPOINT_UPLOAD=0
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb-cache"
export WANDB_DATA_DIR="$RUN_ROOT/wandb-data"
export WANDB__SERVICE_WAIT=300
export TMPDIR="/workspace/tmp/$RUN_ID"
export TORCH_HOME="$RUN_ROOT/torch-cache"
export LASER_VGG_LPIPS_DIR="$ROOT/vgg_lpips"
export LASER_VGG16_WEIGHTS=/workspace/tmp/laser-vgg/vgg16-397923af.pth

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'config=%s\n' "$CONFIG"
  printf 'dataset=%s\n' /workspace/Projects/data/imagenet2012
  printf 'world_size=8\nlocal_batch_size=64\neffective_batch_size=512\n'
  printf 'learning_rate=8.0e-5\nactivation_checkpointing=false\n'
  printf 'discriminator_learning_rate=8.0e-5\n'
} > "$RUN_ROOT/run.info"

cd "$SOURCE"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  main_stage1.py \
  -m "$CONFIG" \
  -r "$RUN_ROOT" \
  --dist-backend nccl \
  --timeout 86400 \
  experiment.batch_size=64 \
  optimizer.init_lr=8.0e-5 \
  optimizer.warmup.min_lr=8.0e-5 \
  arch.hparams.dict_learning_rate=8.0e-5 \
  arch.checkpointing=false \
  2>&1 | tee -a "$RUN_ROOT/logs/production.log"
