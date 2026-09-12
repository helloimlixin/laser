#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

ROOT="/workspace/Projects/laser"
DATA="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
CACHE="${TOKEN_CACHE:-$ROOT/outputs/swgbasnb_compound_pairs_from_scratch/token_cache/imagenet_train_compound_pairs.pt}"
OUT="$ROOT/outputs/swgbasnb_compound_v5b_micro2_distgeom005_warm2to5_depthheads_atom15_scratch/stage2"
CHECKPOINTS="$OUT/checkpoints"
TARGET_EPOCHS="${TARGET_EPOCHS:-30}"

mkdir -p "$CHECKPOINTS"

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1" \
  --token-cache "$CACHE" \
  --resume-checkpoint "$CHECKPOINTS/last.pt" \
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
  --output "$OUT" \
  --checkpoint-dir "$CHECKPOINTS" \
  --epochs "$TARGET_EPOCHS" \
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
  --atom-temperature 0.90 \
  --atom-top-p 0.92 \
  --coeff-temperature 1.00 \
  --coeff-top-p 0.85 \
  --upload-checkpoints \
  --resume \
  --wandb-id wu3qfqgh \
  --wandb-name imagenet-rqtransformer-laser-compound-v5b-micro2-distgeom005-warm2to5-depthheads-atom15-scratch
