#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
CACHE="${TOKEN_CACHE:-$ROOT/outputs/swgbasnb_compound_pairs_from_scratch/token_cache/imagenet_train_compound_pairs.pt}"
OUT="${OUTPUT_DIR:-$ROOT/outputs/swgbasnb_compound_v3_refiner3_geom_atom2_scratch}"
CHECKPOINT_DIR="$OUT/stage2/checkpoints"

test -f "$STAGE1"
test -f "$CACHE"
mkdir -p "$CHECKPOINT_DIR"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1" \
  --token-cache "$CACHE" \
  --compound-tokens \
  --compound-refiner-layers 3 \
  --atom-loss-weight 2.0 \
  --geometry-loss-weight 0.25 \
  --data "$DATA_ROOT" \
  --output "$OUT/stage2" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs "${TARGET_EPOCHS:-40}" \
  --batch-size "${STAGE2_BATCH_SIZE:-64}" \
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
  --sample-grid-every 500 \
  --upload-checkpoints \
  --no-resume \
  --wandb-name imagenet-rqtransformer-laser-compound-v3-refiner3-geom025-atom2-scratch
