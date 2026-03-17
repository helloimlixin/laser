#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/research/bin/python3.12}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/xl598/.local/bin/torchrun}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
STAGE2_LR="${STAGE2_LR:-1e-3}"
STAGE2_COEFF_LOSS_WEIGHT="${STAGE2_COEFF_LOSS_WEIGHT:-0.1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-scratch}"
WANDB_NAME="${WANDB_NAME:-laser_celeba128_quantized_spatial_depth}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TORCHRUN_BIN" ]]; then
  echo "Missing torchrun entrypoint: $TORCHRUN_BIN" >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/proto.py" ]]; then
  echo "Missing proto.py under $ROOT_DIR" >&2
  exit 1
fi

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$ROOT_DIR/runs/laser_celeba128_quantized}"
mkdir -p "$EXPERIMENT_ROOT"
launch_ts="$(date +%Y%m%d_%H%M%S)"
log_path="$EXPERIMENT_ROOT/stage2_quantized_spatial_${launch_ts}.log"

echo "ROOT_DIR=$ROOT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "STAGE2_EPOCHS=$STAGE2_EPOCHS"
echo "STAGE2_BATCH_SIZE=$STAGE2_BATCH_SIZE"
echo "WANDB_MODE=$WANDB_MODE"
echo "log_path=$log_path"

cd "$ROOT_DIR"
nohup "$PYTHON_BIN" "$TORCHRUN_BIN" \
  --standalone --nproc_per_node="$NPROC_PER_NODE" \
  proto.py \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --image_size 128 \
  --out_dir "$EXPERIMENT_ROOT" \
  --stage1_epochs 0 \
  --stage2_epochs "$STAGE2_EPOCHS" \
  --batch_size 16 \
  --stage2_batch_size "$STAGE2_BATCH_SIZE" \
  --ae_num_downsamples 4 \
  --embedding_dim 16 \
  --num_atoms 1024 \
  --sparsity_level 8 \
  --patch_size 4 \
  --patch_stride 2 \
  --quantize_sparse_coeffs true \
  --stage2_coeff_loss_weight "$STAGE2_COEFF_LOSS_WEIGHT" \
  --stage2_lr "$STAGE2_LR" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$WANDB_NAME" \
  "$@" \
  >"$log_path" 2>&1 &

new_pid="$!"
echo "pid=$new_pid"
echo "log_path=$log_path"
echo ""
echo "Monitor with: tail -f $log_path"
