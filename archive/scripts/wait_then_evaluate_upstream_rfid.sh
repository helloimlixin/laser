#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 CURRENT_PID" >&2; exit 2; fi
CURRENT_PID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while kill -0 "$CURRENT_PID" 2>/dev/null; do sleep 30; done
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
exec torchrun --standalone --nproc_per_node=4 "$ROOT/scripts/evaluate_upstream_laser_rfid.py" \
  --checkpoint "$ROOT/outputs/x3h5cl0h_lw075_upstreamloss_rfid/epoch10_model.pt.assembled" \
  --data /workspace/Projects/data/imagenet \
  --output "$ROOT/outputs/x3h5cl0h_lw075_upstreamloss_rfid/full_imagenet_val_rfid.json" \
  --batch-size 96
