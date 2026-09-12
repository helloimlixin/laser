#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

ROOT="/workspace/Projects/laser"
DATA="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
CACHE="${TOKEN_CACHE:-$ROOT/outputs/swgbasnb_cached_duplicate/token_cache/imagenet_train_sparse_components.pt}"
SOURCE="${SOURCE_STAGE2_CHECKPOINT:-$ROOT/outputs/swgbasnb_cached_duplicate/stage2/checkpoints/best_fid_19.5370_epoch_045.pt}"
OUT="${OUTPUT_DIR:-$ROOT/outputs/swgbasnb_separate_tokens_cosine_from_v8yrnory_best45}"
STAGE2="$OUT/stage2"
CHECKPOINTS="$STAGE2/checkpoints"

mkdir -p "$CHECKPOINTS"

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1" \
  --token-cache "$CACHE" \
  --resume-checkpoint "$SOURCE" \
  --data "$DATA" \
  --output "$STAGE2" \
  --checkpoint-dir "$CHECKPOINTS" \
  --epochs "${TARGET_EPOCHS:-55}" \
  --batch-size 64 \
  --total-batch-size 2048 \
  --num-atoms 16384 \
  --coeff-vocab-size 2048 \
  --coeff-max 20 \
  --coeff-scale 6.4 \
  --lr 0.0005 \
  --lr-schedule cosine \
  --lr-schedule-epochs 100 \
  --min-lr 0 \
  --fid-num-samples 50000 \
  --fid-batch-size 96 \
  --fid-every 5 \
  --save-ckpt-freq 2 \
  --save-step-freq 250 \
  --sample-grid-every 500 \
  --upload-checkpoints \
  --resume \
  --wandb-name imagenet-rqtransformer-laser-separate-tokens-cosine-from-v8yrnory-best45
