#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

ROOT="/workspace/Projects/laser"
DATA="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
CACHE="${TOKEN_CACHE:-$ROOT/outputs/swgbasnb_compound_pairs_from_scratch/token_cache/imagenet_train_compound_pairs.pt}"
OUT="${OUTPUT_DIR:-$ROOT/outputs/swgbasnb_compound_v5_micro2_distgeom005_warm2to5_depthheads_atom15_scratch}"
STAGE2="$OUT/stage2"
CHECKPOINTS="$STAGE2/checkpoints"

mkdir -p "$CHECKPOINTS"

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
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
  --epochs 40 \
  --batch-size 64 \
  --total-batch-size 2048 \
  --num-atoms 16384 \
  --coeff-vocab-size 2048 \
  --coeff-max 20 \
  --coeff-scale 6.4 \
  --lr 0.0005 \
  --fid-num-samples 50000 \
  --fid-batch-size 96 \
  --fid-every 5 \
  --save-ckpt-freq 2 \
  --save-step-freq 250 \
  --sample-grid-every 500 \
  --atom-temperature 1.0 \
  --atom-top-p 0.92 \
  --coeff-temperature 1.0 \
  --coeff-top-p 0.92 \
  --upload-checkpoints \
  --no-resume \
  --wandb-name "${WANDB_NAME:-imagenet-rqtransformer-laser-compound-v5-micro2-distgeom005-warm2to5-depthheads-atom15-scratch}"
