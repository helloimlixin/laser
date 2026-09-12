#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="celebahq-a2048-k2-rqvae-strict-20260720-145706"
DATA_ROOT="${CELEBAHQ_ROOT:-/home/xl598/Projects/data/celeba_hq}"
OUT="$ROOT/outputs/$SOURCE_RUN/stage2-compound-v5b-official-rqtransformer-350M"
CHECKPOINT="$ROOT/outputs/$SOURCE_RUN/stage1_checkpoint/best_rfid_slot1_model.pt"
TOKEN_CACHE="$ROOT/outputs/$SOURCE_RUN/token_cache/celebahq_train_compound_pairs.pt"

for required in "$CHECKPOINT" "$TOKEN_CACHE" "$DATA_ROOT/train"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT" "$ROOT/.cache/wandb" "$ROOT/.local/share/wandb" "$ROOT/wandb"
echo "$$" > "$OUT/launcher.pid"

export WANDB_MODE=online
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

# This preserves the original FFHQ RQ-Transformer architecture and optimizer
# recipe (350M geometry, AdamW 5e-4, fixed LR, batch 128, 200 epochs, top-k
# 250), while replacing four scalar depth tokens with two compound LASER
# (atom, coefficient) events. The calibrated per-depth coefficient scales are
# read and checked directly from TOKEN_CACHE by the trainer.
exec torchrun --standalone --nproc_per_node=2 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_ROOT" \
  --dataset celebahq \
  --model-preset ffhq-350m \
  --token-cache "$TOKEN_CACHE" \
  --output "$OUT" \
  --distributed-backend ddp \
  --epochs 200 \
  --batch-size 8 \
  --total-batch-size 128 \
  --num-atoms 2048 \
  --coeff-vocab-size 1024 \
  --coeff-max 3 \
  --compound-tokens \
  --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads \
  --compound-distribution-geometry \
  --geometry-top-k 4 \
  --atom-loss-weight 1.5 \
  --geometry-loss-weight 0.05 \
  --geometry-start-epoch 2 \
  --geometry-warmup-epochs 3 \
  --lr 0.0005 \
  --lr-schedule constant \
  --atom-temperature 1.0 \
  --atom-top-k 250 \
  --atom-top-p 1.0 \
  --coeff-temperature 1.0 \
  --coeff-top-k 250 \
  --coeff-top-p 1.0 \
  --fid-real-split train \
  --fid-num-samples 50000 \
  --fid-batch-size 8 \
  --fid-every 50 \
  --save-ckpt-freq 10 \
  --save-step-freq 250 \
  --sample-grid-every 1000 \
  --sample-grid-size 64 \
  --sample-grid-batch-size 8 \
  --sample-grid-sweep \
  --sample-grid-on-start \
  --upload-checkpoints \
  --upload-token-cache \
  --resume \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id celebahq-compound-rqt350-a2048k2-20260804 \
  --wandb-name celebahq-a2048-k2-compound-v5b-official-rqtransformer-350M
