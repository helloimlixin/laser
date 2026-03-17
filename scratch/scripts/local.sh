#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/runs/laser_celeba128}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
NUM_ATOMS="${NUM_ATOMS:-256}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-4}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-scratch}"

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set; falling back to WANDB_MODE=offline."
  WANDB_MODE="offline"
fi

if [[ ! -f "$PROJECT_DIR/proto.py" ]]; then
  echo "ERROR: proto.py not found under PROJECT_DIR=$PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

GPU_COUNT="$("$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except Exception:
    print(-1)
    sys.exit(0)
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

if [[ "$GPU_COUNT" == "-1" ]]; then
  echo "ERROR: failed to import torch from PYTHON_BIN=$PYTHON_BIN" >&2
  exit 1
fi
if [[ "$GPU_COUNT" -le 0 ]]; then
  echo "ERROR: no CUDA GPUs visible to $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$STAGE1_EPOCHS" == "auto" ]]; then
  if [[ -f "$OUT_DIR/stage1/ae_last.pt" ]]; then
    STAGE1_EPOCHS=0
  else
    STAGE1_EPOCHS=5
  fi
fi

if ! [[ "$STAGE1_EPOCHS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: STAGE1_EPOCHS must be a non-negative integer or 'auto'. Got: $STAGE1_EPOCHS" >&2
  exit 1
fi
if ! [[ "$STAGE2_EPOCHS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: STAGE2_EPOCHS must be a non-negative integer. Got: $STAGE2_EPOCHS" >&2
  exit 1
fi

echo "PYTHON_BIN=$PYTHON_BIN"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "GPU_COUNT=$GPU_COUNT"
echo "STAGE1_EPOCHS=$STAGE1_EPOCHS"
echo "STAGE2_EPOCHS=$STAGE2_EPOCHS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "STAGE2_BATCH_SIZE=$STAGE2_BATCH_SIZE"
echo "EMBEDDING_DIM=$EMBEDDING_DIM"
echo "NUM_ATOMS=$NUM_ATOMS"
echo "SPARSITY_LEVEL=$SPARSITY_LEVEL"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_PROJECT=$WANDB_PROJECT"

"$PYTHON_BIN" -u "$PROJECT_DIR/proto.py" \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --image_size 128 \
  --out_dir "$OUT_DIR" \
  --stage1_epochs "$STAGE1_EPOCHS" \
  --stage2_epochs "$STAGE2_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --stage2_batch_size "$STAGE2_BATCH_SIZE" \
  --embedding_dim "$EMBEDDING_DIM" \
  --num_atoms "$NUM_ATOMS" \
  --sparsity_level "$SPARSITY_LEVEL" \
  --rfid_num_samples "$FID_NUM_SAMPLES" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --quantize_sparse_coeffs false \
  "$@"
