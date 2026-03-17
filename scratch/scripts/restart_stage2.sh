#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$ROOT_DIR/runs/laser_celeba128}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/research/bin/python3.12}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/xl598/.local/bin/torchrun}"
CURRENT_LAUNCHER_PID="${CURRENT_LAUNCHER_PID:-1754025}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_COEFF_LOSS_WEIGHT="${STAGE2_COEFF_LOSS_WEIGHT:-0.1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_NAME="${WANDB_NAME:-laser_celeba128_stage2_mse}"

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

if kill -0 "$CURRENT_LAUNCHER_PID" 2>/dev/null; then
  echo "Stopping current stage-2 launcher PID $CURRENT_LAUNCHER_PID"
  kill -INT "$CURRENT_LAUNCHER_PID"
  for _ in $(seq 1 30); do
    if ! kill -0 "$CURRENT_LAUNCHER_PID" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  if kill -0 "$CURRENT_LAUNCHER_PID" 2>/dev/null; then
    echo "Launcher still active after SIGINT; sending SIGTERM"
    kill -TERM "$CURRENT_LAUNCHER_PID"
    for _ in $(seq 1 15); do
      if ! kill -0 "$CURRENT_LAUNCHER_PID" 2>/dev/null; then
        break
      fi
      sleep 2
    done
  fi
fi

mkdir -p "$EXPERIMENT_ROOT"
launch_ts="$(date +%Y%m%d_%H%M%S)"
log_path="$EXPERIMENT_ROOT/stage2_mse_${launch_ts}.log"

cd "$ROOT_DIR"
nohup "$PYTHON_BIN" "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" proto.py \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --image_size 128 \
  --out_dir "$EXPERIMENT_ROOT" \
  --stage1_epochs 0 \
  --stage2_epochs "$STAGE2_EPOCHS" \
  --batch_size 16 \
  --stage2_batch_size 32 \
  --ae_num_downsamples 4 \
  --embedding_dim 16 \
  --num_atoms 1024 \
  --sparsity_level 8 \
  --patch_size 4 \
  --patch_stride 2 \
  --stage2_coeff_loss_type mse \
  --stage2_coeff_loss_weight "$STAGE2_COEFF_LOSS_WEIGHT" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_name "$WANDB_NAME" \
  >"$log_path" 2>&1 &

new_pid="$!"
echo "log_path=$log_path"
echo "pid=$new_pid"
