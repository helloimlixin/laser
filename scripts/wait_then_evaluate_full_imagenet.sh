#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CURRENT_TORCHRUN_PID" >&2
  exit 2
fi

CURRENT_PID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/outputs/swgbasnb_cached_duplicate/stage2/checkpoints}"
CHECKPOINT="$CHECKPOINT_DIR/last.pt"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
DATA="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
OUTPUT="${EVAL_OUTPUT:-$ROOT/outputs/swgbasnb_cached_duplicate/full_imagenet_eval}"

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep 60
done

python - "$CHECKPOINT" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
if int(p.get("epoch", -1)) != 56:
    raise SystemExit(f"refusing evaluation: expected epoch 56, found {p.get('epoch')}")
print(f"validated final checkpoint: epoch=56 step={p.get('global_step')}", flush=True)
PY

mkdir -p "$OUTPUT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/evaluate_laser_full_imagenet.py" \
  --stage1 "$STAGE1" --stage2 "$CHECKPOINT" --data "$DATA" --output "$OUTPUT" \
  --batch-size 96 --num-samples 50000 \
  --temperatures 0.85 0.95 1.0 1.05 --top-ps 0.90 0.92 0.95
