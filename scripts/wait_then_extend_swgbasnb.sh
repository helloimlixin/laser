#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CURRENT_TORCHRUN_PID TARGET_EPOCH" >&2
  exit 2
fi

CURRENT_PID="$1"
TARGET_EPOCH="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/swgbasnb_cached_duplicate/stage2/checkpoints}"
CHECKPOINT="$CHECKPOINT_DIR/last.pt"

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep 60
done

python - "$CHECKPOINT" "$TARGET_EPOCH" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
target = int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
epoch = int(payload.get("epoch", -1))
if epoch != target - 10:
    raise SystemExit(f"refusing extension: expected epoch {target - 10}, found {epoch}")
print(f"validated completed checkpoint: epoch={epoch}, step={payload.get('global_step')}", flush=True)
PY

exec env \
  TARGET_EPOCHS="$TARGET_EPOCH" \
  FID_EVERY="${FID_EVERY:-5}" \
  SAMPLE_GRID_EVERY="${SAMPLE_GRID_EVERY:-500}" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  bash "$ROOT/scripts/launch_swgbasnb_cached_duplicate.sh"
