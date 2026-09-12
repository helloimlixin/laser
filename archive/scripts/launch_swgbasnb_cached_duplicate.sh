#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
S1="${STAGE1_CHECKPOINT:-$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
OUT="${OUTPUT_DIR:-$ROOT/outputs/swgbasnb_cached_duplicate}"
CACHE="${TOKEN_CACHE:-$OUT/token_cache/imagenet_train_sparse_components.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUT/stage2/checkpoints}"
WANDB_ID="${WANDB_ID:-v8yrnory}"
SOURCE_S2_DEFAULT="$ROOT/outputs/imagenet_x3h5cl0h_stage2_a16384_k2_c2048_m20/stage2/checkpoints/last.pt"
if [[ -f "$OUT/stage2/checkpoints/last.pt" ]]; then
  SOURCE_S2_DEFAULT="$OUT/stage2/checkpoints/last.pt"
fi
if [[ -f "$CHECKPOINT_DIR/last.pt" ]]; then
  SOURCE_S2_DEFAULT="$CHECKPOINT_DIR/last.pt"
fi
SOURCE_S2="${SOURCE_STAGE2_CHECKPOINT:-$SOURCE_S2_DEFAULT}"

UPLOAD_ARGS=(--upload-checkpoints)
if [[ "${UPLOAD_CHECKPOINTS:-true}" != "true" ]]; then
  UPLOAD_ARGS=(--no-upload-checkpoints)
fi

mkdir -p "$OUT/token_cache" "$CHECKPOINT_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

if [[ ! -f "$CACHE" ]]; then
  torchrun --standalone --nproc_per_node=4 "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$S1" --data "$DATA_ROOT" --output "$CACHE" \
    --batch-size "${CACHE_BATCH_SIZE:-128}"
fi

python - "$CACHE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]).with_suffix('.validation.json')
report = json.loads(p.read_text())
assert report['passed'], report
print('validated token cache:', p)
PY

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$S1" --resume-checkpoint "$SOURCE_S2" --token-cache "$CACHE" \
  --data "$DATA_ROOT" --output "$OUT/stage2" --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs "${TARGET_EPOCHS:-100}" --batch-size "${STAGE2_BATCH_SIZE:-64}" \
  --total-batch-size 2048 --num-atoms 16384 --coeff-vocab-size 2048 \
  --coeff-max 20 --coeff-scale 6.4 --lr 0.0005 \
  --fid-num-samples 50000 --fid-batch-size 96 --fid-every "${FID_EVERY:-5}" \
  --save-ckpt-freq 2 --sample-grid-every "${SAMPLE_GRID_EVERY:-0}" \
  "${UPLOAD_ARGS[@]}" --wandb-id "$WANDB_ID" \
  --wandb-name imagenet-official-rqtransformer-laser-cached-swgbasnb-duplicate
