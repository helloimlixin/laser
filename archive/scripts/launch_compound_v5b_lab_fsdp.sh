#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

ROOT="${LASER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${IMAGENET_ROOT:-$ROOT/../data/imagenet}"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
CACHE="${TOKEN_CACHE:-$ROOT/outputs/swgbasnb_compound_pairs_from_scratch/token_cache/imagenet_train_compound_pairs.pt}"
STAGE2="${OUTPUT_DIR:-$ROOT/outputs/swgbasnb_compound_v5b_original_cosine_from_epoch10/stage2}"
CHECKPOINTS="${CHECKPOINT_DIR:-$STAGE2/checkpoints}"
RESUME="${RESUME_CHECKPOINT:-$CHECKPOINTS/last.pt}"

for required_file in "$STAGE1" "$CACHE" "$RESUME"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done
for required_dir in "$DATA/val"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Missing ImageNet directory: $required_dir" >&2
    exit 1
  fi
done

mkdir -p "$CHECKPOINTS"

exec torchrun --standalone --nproc_per_node=2 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --distributed-backend fsdp \
  --checkpoint "$STAGE1" \
  --token-cache "$CACHE" \
  --compound-tokens \
  --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads \
  --compound-distribution-geometry \
  --geometry-top-k 4 \
  --atom-loss-weight 1.5 \
  --geometry-loss-weight 0.05 \
  --geometry-start-epoch 2 \
  --geometry-warmup-epochs 3 \
  --data "$DATA" \
  --output "$STAGE2" \
  --checkpoint-dir "$CHECKPOINTS" \
  --epochs 100 \
  --batch-size 2 \
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
  --fid-batch-size 2 \
  --fid-every 5 \
  --save-ckpt-freq 2 \
  --save-step-freq 250 \
  --sample-grid-every 0 \
  --atom-temperature 0.90 \
  --atom-top-p 0.92 \
  --coeff-temperature 1.00 \
  --coeff-top-p 0.85 \
  --upload-checkpoints \
  --resume \
  --resume-checkpoint "$RESUME" \
  --wandb-id c5cos10r \
  --wandb-name imagenet-rqtransformer-laser-compound-v5b-original-cosine-from-epoch10
