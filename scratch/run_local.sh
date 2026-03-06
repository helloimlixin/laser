#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$BASE_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/runs/laser_celeba128}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba}"

STAGE1_STRATEGY="${STAGE1_STRATEGY:-ddp}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
NUM_ATOMS="${NUM_ATOMS:-256}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-4}"
STAGE2_ARCH="${STAGE2_ARCH:-spatial_depth}"
STAGE2_COEFF_ENERGY_LOSS_WEIGHT="${STAGE2_COEFF_ENERGY_LOSS_WEIGHT:-0.25}"
STAGE2_SITE_TOKEN_BINS="${STAGE2_SITE_TOKEN_BINS:-16}"
STAGE2_SITE_VOCAB_SIZE="${STAGE2_SITE_VOCAB_SIZE:-8192}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-0}"
STAGE2_FID_NUM_SAMPLES="${STAGE2_FID_NUM_SAMPLES:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-scratch}"

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set; falling back to WANDB_MODE=offline."
  WANDB_MODE="offline"
fi

if [[ ! -f "$PROJECT_DIR/laser.py" ]]; then
  echo "ERROR: laser.py not found under PROJECT_DIR=$PROJECT_DIR" >&2
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

STAGE1_DEVICES="${STAGE1_DEVICES:-$GPU_COUNT}"
STAGE2_DEVICES="${STAGE2_DEVICES:-$GPU_COUNT}"

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
echo "STAGE1_DEVICES=$STAGE1_DEVICES"
echo "STAGE2_DEVICES=$STAGE2_DEVICES"
echo "STAGE1_EPOCHS=$STAGE1_EPOCHS"
echo "STAGE2_EPOCHS=$STAGE2_EPOCHS"
echo "STAGE1_STRATEGY=$STAGE1_STRATEGY"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "STAGE2_BATCH_SIZE=$STAGE2_BATCH_SIZE"
echo "EMBEDDING_DIM=$EMBEDDING_DIM"
echo "NUM_ATOMS=$NUM_ATOMS"
echo "SPARSITY_LEVEL=$SPARSITY_LEVEL"
echo "STAGE2_ARCH=$STAGE2_ARCH"
echo "STAGE2_COEFF_ENERGY_LOSS_WEIGHT=$STAGE2_COEFF_ENERGY_LOSS_WEIGHT"
echo "STAGE2_SITE_TOKEN_BINS=$STAGE2_SITE_TOKEN_BINS"
echo "STAGE2_SITE_VOCAB_SIZE=$STAGE2_SITE_VOCAB_SIZE"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_PROJECT=$WANDB_PROJECT"

"$PYTHON_BIN" -u "$PROJECT_DIR/laser.py" \
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
  --stage2_arch "$STAGE2_ARCH" \
  --stage2_coeff_energy_loss_weight "$STAGE2_COEFF_ENERGY_LOSS_WEIGHT" \
  --stage2_site_token_bins "$STAGE2_SITE_TOKEN_BINS" \
  --stage2_site_vocab_size "$STAGE2_SITE_VOCAB_SIZE" \
  --fid_num_samples "$FID_NUM_SAMPLES" \
  --stage2_fid_num_samples "$STAGE2_FID_NUM_SAMPLES" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --stage1_devices "$STAGE1_DEVICES" \
  --stage2_devices "$STAGE2_DEVICES" \
  --stage1_strategy "$STAGE1_STRATEGY" \
  --no_quantize_sparse_coeffs \
  "$@"
