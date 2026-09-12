#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUTPUT_DIR:-$ROOT/outputs/imagenet_x3h5cl0h_stage2_a16384_k2_c2048_m20/stage2}"
DATA_ROOT="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-2048}"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="8"
export WANDB_DIR=/workspace/wandb
export WANDB_CACHE_DIR=/workspace/.cache/wandb
export WANDB_DATA_DIR=/workspace/.local/share/wandb
export XDG_CACHE_HOME=/workspace/.cache
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
# Use native NCCL P2P/shared-memory transport on the migration host. If a host
# lacks working peer mappings, set NCCL_P2P_DISABLE=1 and NCCL_SHM_DISABLE=1
# explicitly in its environment before launching.

exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --data "$DATA_ROOT" \
  --output "$OUT" \
  --epochs 100 --batch-size 8 --total-batch-size "$TOTAL_BATCH_SIZE" \
  --num-atoms 16384 --coeff-vocab-size 2048 --coeff-max 20 --coeff-scale 6.4 --lr 0.0005 \
  --fid-num-samples 50000 --fid-batch-size 96 --fid-every 5 --save-ckpt-freq 2 \
  --upload-checkpoints \
  --wandb-id swgbasnb \
  --wandb-name imagenet-official-rqtransformer-laser-a16384-k2-c2048-m20
