#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CURRENT_TORCHRUN_PID" >&2
  exit 2
fi

CURRENT_PID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1="$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt"
STAGE2="$ROOT/outputs/swgbasnb_compound_v2_from_scratch/stage2/checkpoints/best_fid_33.2242_epoch_030.pt"
OUTPUT="$ROOT/outputs/swgbasnb_compound_v2_from_scratch/rejection_2x_epoch030"

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep 60
done

test -f "$STAGE1"
test -f "$STAGE2"
mkdir -p "$OUTPUT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/evaluate_compound_rejection.py" \
  --stage1 "$STAGE1" \
  --stage2 "$STAGE2" \
  --data /workspace/Projects/data/imagenet \
  --output "$OUTPUT" \
  --num-samples 50000 \
  --candidate-multiplier 2 \
  --batch-size 64 \
  --atom-temperature 1.0 --atom-top-p 0.92 \
  --coeff-temperature 1.0 --coeff-top-p 0.92 \
  --wandb-project laser \
  --wandb-group p9f82bsr \
  --wandb-name p9f82bsr-compound-rejection-2x-epoch030
