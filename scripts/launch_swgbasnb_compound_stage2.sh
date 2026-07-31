#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAGE1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
OUT="${OUTPUT_DIR:-$ROOT/outputs/swgbasnb_compound_pairs_from_scratch}"
CACHE="${TOKEN_CACHE:-$OUT/token_cache/imagenet_train_compound_pairs.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/laser-swgbasnb-compound-checkpoints}"
WANDB_ID="${WANDB_ID:-}"

mkdir -p "$OUT/token_cache" "$CHECKPOINT_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

if [[ ! -f "$CACHE" ]]; then
  torchrun --standalone --nproc_per_node=4 \
    "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$STAGE1" --data "$DATA_ROOT" --output "$CACHE" \
    --batch-size "${CACHE_BATCH_SIZE:-128}" --compound
fi

python - "$CACHE" <<'PY'
import json, sys
from pathlib import Path
report_path = Path(sys.argv[1]).with_suffix('.validation.json')
report = json.loads(report_path.read_text())
assert report['passed'], report
assert report['compound_sequence_length'] == 128, report
print('validated compound token cache:', report_path)
PY

WANDB_ARGS=()
if [[ -n "$WANDB_ID" ]]; then WANDB_ARGS+=(--wandb-id "$WANDB_ID"); fi

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1" --token-cache "$CACHE" --compound-tokens \
  --data "$DATA_ROOT" --output "$OUT/stage2" --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs "${TARGET_EPOCHS:-100}" --batch-size "${STAGE2_BATCH_SIZE:-64}" \
  --total-batch-size 2048 --num-atoms 16384 --coeff-vocab-size 2048 \
  --coeff-max 20 --coeff-scale 6.4 --lr 0.0005 \
  --fid-num-samples 50000 --fid-batch-size 96 --fid-every "${FID_EVERY:-5}" \
  --save-ckpt-freq 2 --sample-grid-every "${SAMPLE_GRID_EVERY:-500}" \
  --no-upload-checkpoints --resume "${WANDB_ARGS[@]}" \
  --wandb-name imagenet-rqtransformer-laser-compound-pairs-swgbasnb-from-scratch
