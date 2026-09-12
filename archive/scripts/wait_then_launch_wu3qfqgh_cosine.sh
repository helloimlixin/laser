#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CURRENT_TORCHRUN_PID TARGET_EPOCH" >&2
  exit 2
fi

CURRENT_PID="$1"
TARGET_EPOCH="$2"
ROOT="/workspace/Projects/laser"
CHECKPOINT="$ROOT/outputs/swgbasnb_compound_v5b_micro2_distgeom005_warm2to5_depthheads_atom15_scratch/stage2/checkpoints/last.pt"

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep 30
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
    raise SystemExit(
        f"refusing cosine continuation: expected epoch {target - 10}, found {epoch}"
    )
if int(payload.get("global_step", -1)) <= 0:
    raise SystemExit("refusing cosine continuation: checkpoint has no valid global_step")
print(
    f"validated completed checkpoint: epoch={epoch}, "
    f"step={payload.get('global_step')}",
    flush=True,
)
PY

exec env TARGET_EPOCHS="$TARGET_EPOCH" \
  bash "$ROOT/scripts/launch_wu3qfqgh_cosine_resume.sh"
