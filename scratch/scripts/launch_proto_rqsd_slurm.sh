#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${JOB_NAME:-proto-rqsd}"
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/runs/laser_celeba128_quantized}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER/.pydeps/proto_rqsd_py311}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
DEFAULT_PER_GPU_BATCH="${DEFAULT_PER_GPU_BATCH:-32}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_PER_GPU_BATCH}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-$DEFAULT_PER_GPU_BATCH}"
BASELINE_GLOBAL_BATCH="${BASELINE_GLOBAL_BATCH:-8}"
BASELINE_STAGE1_LR="${BASELINE_STAGE1_LR:-2e-4}"
BASELINE_STAGE2_LR="${BASELINE_STAGE2_LR:-1e-3}"
STAGE2_LR_SCALE_MODE="${STAGE2_LR_SCALE_MODE:-sqrt}"
STAGE2_MAX_LR="${STAGE2_MAX_LR:-4e-3}"
STAGE1_LR="${STAGE1_LR:-}"
STAGE2_LR="${STAGE2_LR:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-scratch}"
WANDB_NAME="${WANDB_NAME:-proto_rqsd_${GPUS}gpu_s${STAGE1_EPOCHS}_s2${STAGE2_EPOCHS}}"
LOG_PREFIX="${LOG_PREFIX:-proto_rqsd}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/proto.py" ]]; then
  echo "Missing proto.py under $ROOT_DIR" >&2
  exit 1
fi

if [[ -z "$STAGE1_LR" ]]; then
  STAGE1_LR="$(python3 -c 'import sys; base=float(sys.argv[1]); gpus=int(sys.argv[2]); batch=int(sys.argv[3]); baseline=max(1,int(sys.argv[4])); print(f"{base * (gpus * batch) / baseline:.10g}")' "$BASELINE_STAGE1_LR" "$GPUS" "$BATCH_SIZE" "$BASELINE_GLOBAL_BATCH")"
fi
if [[ -z "$STAGE2_LR" ]]; then
  STAGE2_LR="$(python3 -c 'import math, sys; base=float(sys.argv[1]); gpus=int(sys.argv[2]); batch=int(sys.argv[3]); baseline=max(1,int(sys.argv[4])); mode=str(sys.argv[5]).strip().lower(); max_lr=float(sys.argv[6]); scale=(gpus * batch) / baseline; scale=max(scale, 1e-12)
if mode == "linear":
    lr = base * scale
elif mode == "sqrt":
    lr = base * math.sqrt(scale)
else:
    raise SystemExit(f"unsupported STAGE2_LR_SCALE_MODE: {mode!r}")
if max_lr > 0:
    lr = min(lr, max_lr)
print(f"{lr:.10g}")' "$BASELINE_STAGE2_LR" "$GPUS" "$STAGE2_BATCH_SIZE" "$BASELINE_GLOBAL_BATCH" "$STAGE2_LR_SCALE_MODE" "$STAGE2_MAX_LR")"
fi

sbatch \
  --partition="$PARTITION" \
  --requeue \
  --job-name="$JOB_NAME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:${GPUS}" \
  --mem="$MEM_MB" \
  --time="$TIME_LIMIT" \
  --chdir="$ROOT_DIR" \
  --output="$ROOT_DIR/${LOG_PREFIX}_%j.out" \
  --error="$ROOT_DIR/${LOG_PREFIX}_%j.err" \
  --wrap='set -euo pipefail
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
USER_NAME=$(id -un)
PROJECT_DIR='"$ROOT_DIR"'
DATA_DIR='"$DATA_DIR"'
OUT_DIR='"$OUT_DIR"'
IMAGE='"$IMAGE"'
PYTHONUSERBASE_DIR='"$PYTHONUSERBASE_DIR"'
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"
STAGE1_EPOCHS='"$STAGE1_EPOCHS"'
STAGE2_EPOCHS='"$STAGE2_EPOCHS"'
STAGE1_LR='"$STAGE1_LR"'
STAGE2_LR='"$STAGE2_LR"'
NUM_WORKERS='"$NUM_WORKERS"'
TOKEN_NUM_WORKERS='"$TOKEN_NUM_WORKERS"'
BATCH_SIZE='"$BATCH_SIZE"'
STAGE2_BATCH_SIZE='"$STAGE2_BATCH_SIZE"'
WANDB_MODE='"$WANDB_MODE"'
WANDB_PROJECT='"$WANDB_PROJECT"'
WANDB_NAME='"$WANDB_NAME"'
WANDB_API_KEY_FILE='"$WANDB_API_KEY_FILE"'
GPUS='"$GPUS"'
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
  fi
fi
if ! command -v singularity >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  fi
fi
command -v singularity >/dev/null 2>&1 || { echo singularity_not_found >&2; exit 1; }
mkdir -p "$OUT_DIR" "$PYTHONUSERBASE_DIR" "$(dirname "$WANDB_API_KEY_FILE")"
if [[ -f "$WANDB_API_KEY_FILE" ]]; then
  export WANDB_API_KEY="$(tr -d "\r\n" < "$WANDB_API_KEY_FILE")"
fi
export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$OUT_DIR" "$IMAGE" python3 -c "import scipy, wandb" >/dev/null 2>&1; then
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$OUT_DIR" "$IMAGE" python3 -m pip install --user scipy wandb
fi
nvidia-smi
srun singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER_NAME" \
  --bind "$DATA_DIR" \
  --bind "$OUT_DIR" \
  "$IMAGE" \
  python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$GPUS" \
    "$PROJECT_DIR/proto.py" \
    --dataset celeba \
    --data_dir "$DATA_DIR" \
    --image_size 128 \
    --out_dir "$OUT_DIR" \
    --stage1_epochs "$STAGE1_EPOCHS" \
    --stage2_epochs "$STAGE2_EPOCHS" \
    --stage1_lr "$STAGE1_LR" \
    --stage2_lr "$STAGE2_LR" \
    --num_workers "$NUM_WORKERS" \
    --token_num_workers "$TOKEN_NUM_WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --stage2_batch_size "$STAGE2_BATCH_SIZE" \
    --rfid_num_samples 0 \
    --wandb_mode "$WANDB_MODE" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    --quantize_sparse_coeffs true'
