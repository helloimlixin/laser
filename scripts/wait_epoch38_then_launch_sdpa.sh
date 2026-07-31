#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/swgbasnb_cached_duplicate/stage2"
TRAIN_LOG="$OUT/train-cache-plus5-split.log"
WATCH_LOG="$OUT/wait-epoch38-sdpa.log"
OLD_PID="${OLD_PID:-35021}"

echo "waiting for epoch 38 completion from PID $OLD_PID" >> "$WATCH_LOG"
while kill -0 "$OLD_PID" 2>/dev/null; do
  if grep -q '^Epoch 38:' "$TRAIN_LOG"; then
    echo "epoch 38 checkpoint and post-save work completed" >> "$WATCH_LOG"
    kill -INT "$OLD_PID"
    break
  fi
  sleep 30
done

for _ in $(seq 1 60); do
  if ! kill -0 "$OLD_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$OLD_PID" 2>/dev/null; then
  echo "old trainer did not stop cleanly" >> "$WATCH_LOG"
  exit 1
fi

echo "launching SDPA continuation" >> "$WATCH_LOG"
exec env WANDB_ENTITY=helloimlixin-rutgers TARGET_EPOCHS=41 FID_EVERY=41 SAMPLE_GRID_EVERY=0 \
  bash "$ROOT/scripts/launch_swgbasnb_cached_duplicate.sh" \
  >> "$OUT/train-cache-plus5-sdpa.log" 2>&1
