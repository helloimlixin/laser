#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
CACHE="${TOKEN_CACHE:-$ROOT/outputs/swgbasnb_compound_pairs_from_scratch/token_cache/imagenet_train_compound_pairs.pt}"
OUT="${OUTPUT_DIR:-$ROOT/outputs/swgbasnb_compound_v2_from_scratch}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUT/stage2/checkpoints}"

test -f "$CACHE"
test -f "${CACHE%.pt}.validation.json"
mkdir -p "$OUT" "$CHECKPOINT_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1" --token-cache "$CACHE" --compound-tokens \
  --data "$DATA_ROOT" --output "$OUT/stage2" --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs "${TARGET_EPOCHS:-100}" --batch-size "${STAGE2_BATCH_SIZE:-64}" \
  --total-batch-size 2048 --num-atoms 16384 --coeff-vocab-size 2048 \
  --coeff-max 20 --coeff-scale 6.4 --lr 0.0005 \
  --fid-num-samples 50000 --fid-batch-size 96 --fid-every 5 \
  --save-ckpt-freq 2 --sample-grid-every 500 \
  --no-upload-checkpoints --no-resume \
  --wandb-name imagenet-rqtransformer-laser-compound-v2-rich-pairs-from-scratch
