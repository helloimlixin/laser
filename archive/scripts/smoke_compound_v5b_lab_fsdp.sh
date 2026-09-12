#!/usr/bin/env bash
set -euo pipefail

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

ROOT="${LASER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
CACHE="${TOKEN_CACHE:-$ROOT/outputs/swgbasnb_compound_pairs_from_scratch/token_cache/imagenet_train_compound_pairs.pt}"
RESUME="${RESUME_CHECKPOINT:-$ROOT/outputs/swgbasnb_compound_v5b_original_cosine_from_epoch10/stage2/checkpoints/last.pt}"
BATCH_SIZE="${1:-1}"
SMOKE_MODE="${2:-train}"
SAVE_STEP_FREQ="${SMOKE_SAVE_STEP_FREQ:-0}"
SMOKE_OUTPUT="${SMOKE_OUTPUT_DIR:-$ROOT/outputs/fsdp_smoke_batch${BATCH_SIZE}}"

if [[ "$BATCH_SIZE" != "1" && "$BATCH_SIZE" != "2" && "$BATCH_SIZE" != "4" ]]; then
  echo "Usage: $0 [1|2|4] [train|generation]" >&2
  exit 2
fi
if [[ "$SMOKE_MODE" != "train" && "$SMOKE_MODE" != "generation" ]]; then
  echo "Usage: $0 [1|2|4] [train|generation]" >&2
  exit 2
fi
for required_file in "$STAGE1" "$CACHE" "$RESUME"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

mkdir -p "$SMOKE_OUTPUT/checkpoints"

SMOKE_ARGS=(--smoke-test --max-optimizer-steps 1)
if [[ "$SMOKE_MODE" == "generation" ]]; then
  SMOKE_ARGS=(--generation-smoke-test)
fi

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
  --data "$ROOT/does-not-need-imagenet" \
  --output "$SMOKE_OUTPUT" \
  --checkpoint-dir "$SMOKE_OUTPUT/checkpoints" \
  --epochs 100 \
  --batch-size "$BATCH_SIZE" \
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
  --fid-batch-size "$BATCH_SIZE" \
  --fid-every 0 \
  --save-ckpt-freq 2 \
  --save-step-freq "$SAVE_STEP_FREQ" \
  --sample-grid-every 0 \
  --atom-temperature 0.90 \
  --atom-top-p 0.92 \
  --coeff-temperature 1.00 \
  --coeff-top-p 0.85 \
  --resume \
  --resume-checkpoint "$RESUME" \
  --wandb-mode disabled \
  "${SMOKE_ARGS[@]}"
