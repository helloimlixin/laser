#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${JOB_NAME:-proto-lightning}"
PARTITION="${PARTITION:-cgpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/runs/proto_lightning}"
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
STAGE1_LR="${STAGE1_LR:-}"
STAGE2_LR="${STAGE2_LR:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-scratch}"
WANDB_NAME="${WANDB_NAME:-proto_lightning_${GPUS}gpu_s${STAGE1_EPOCHS}_s2${STAGE2_EPOCHS}}"
LOG_PREFIX="${LOG_PREFIX:-proto_lightning}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"
STAGE1_PRECISION="${STAGE1_PRECISION:-16-mixed}"
STAGE2_PRECISION="${STAGE2_PRECISION:-16-mixed}"
STAGE1_STRATEGY="${STAGE1_STRATEGY:-ddp}"
STAGE2_STRATEGY="${STAGE2_STRATEGY:-ddp}"
TF_D_MODEL="${TF_D_MODEL:-512}"
TF_HEADS="${TF_HEADS:-8}"
TF_LAYERS="${TF_LAYERS:-12}"
TF_FF="${TF_FF:-1024}"
TF_DROPOUT="${TF_DROPOUT:-0.1}"
TF_GLOBAL_TOKENS="${TF_GLOBAL_TOKENS:-0}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-500}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.01}"
STAGE2_WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-0.01}"
SAMPLE_EVERY_STEPS="${SAMPLE_EVERY_STEPS:-0}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SAMPLE_CANDIDATE_FACTOR="${SAMPLE_CANDIDATE_FACTOR:-4}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.5}"
SAMPLE_TOP_K="${SAMPLE_TOP_K:-0}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/proto_lightning.py" ]]; then
  echo "Missing proto_lightning.py under $ROOT_DIR" >&2
  exit 1
fi

if [[ -z "$STAGE1_LR" ]]; then
  STAGE1_LR="$(python3 -c 'import sys; base=float(sys.argv[1]); gpus=int(sys.argv[2]); batch=int(sys.argv[3]); baseline=max(1,int(sys.argv[4])); print(f"{base * (gpus * batch) / baseline:.10g}")' "$BASELINE_STAGE1_LR" "$GPUS" "$BATCH_SIZE" "$BASELINE_GLOBAL_BATCH")"
fi
if [[ -z "$STAGE2_LR" ]]; then
  STAGE2_LR="$(python3 -c 'import sys; base=float(sys.argv[1]); gpus=int(sys.argv[2]); batch=int(sys.argv[3]); baseline=max(1,int(sys.argv[4])); print(f"{base * (gpus * batch) / baseline:.10g}")' "$BASELINE_STAGE2_LR" "$GPUS" "$STAGE2_BATCH_SIZE" "$BASELINE_GLOBAL_BATCH")"
fi

sbatch   --partition="$PARTITION"   --requeue   --job-name="$JOB_NAME"   --nodes=1   --ntasks=1   --cpus-per-task="$CPUS_PER_TASK"   --gres="gpu:${GPUS}"   --mem="$MEM_MB"   --time="$TIME_LIMIT"   --chdir="$ROOT_DIR"   --output="$ROOT_DIR/${LOG_PREFIX}_%j.out"   --error="$ROOT_DIR/${LOG_PREFIX}_%j.err"   --wrap='set -euo pipefail
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
BATCH_SIZE='"$BATCH_SIZE"'
STAGE2_BATCH_SIZE='"$STAGE2_BATCH_SIZE"'
WANDB_MODE='"$WANDB_MODE"'
WANDB_PROJECT='"$WANDB_PROJECT"'
WANDB_NAME='"$WANDB_NAME"'
WANDB_API_KEY_FILE='"$WANDB_API_KEY_FILE"'
GPUS='"$GPUS"'
STAGE1_PRECISION='"$STAGE1_PRECISION"'
STAGE2_PRECISION='"$STAGE2_PRECISION"'
STAGE1_STRATEGY='"$STAGE1_STRATEGY"'
STAGE2_STRATEGY='"$STAGE2_STRATEGY"'
TF_D_MODEL='"$TF_D_MODEL"'
TF_HEADS='"$TF_HEADS"'
TF_LAYERS='"$TF_LAYERS"'
TF_FF='"$TF_FF"'
TF_DROPOUT='"$TF_DROPOUT"'
TF_GLOBAL_TOKENS='"$TF_GLOBAL_TOKENS"'
STAGE2_WARMUP_STEPS='"$STAGE2_WARMUP_STEPS"'
STAGE2_MIN_LR_RATIO='"$STAGE2_MIN_LR_RATIO"'
STAGE2_WEIGHT_DECAY='"$STAGE2_WEIGHT_DECAY"'
SAMPLE_EVERY_STEPS='"$SAMPLE_EVERY_STEPS"'
SAMPLE_BATCH_SIZE='"$SAMPLE_BATCH_SIZE"'
SAMPLE_CANDIDATE_FACTOR='"$SAMPLE_CANDIDATE_FACTOR"'
SAMPLE_TEMPERATURE='"$SAMPLE_TEMPERATURE"'
SAMPLE_TOP_K='"$SAMPLE_TOP_K"'
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
  export WANDB_API_KEY="$(tr -d "
" < "$WANDB_API_KEY_FILE")"
fi
export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$OUT_DIR" "$IMAGE" python3 -c "import scipy, wandb, lightning" >/dev/null 2>&1; then
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$OUT_DIR" "$IMAGE" python3 -m pip install --user scipy wandb lightning
fi
nvidia-smi
srun singularity exec --nv   --bind "$PROJECT_DIR"   --bind "/scratch/$USER_NAME"   --bind "$DATA_DIR"   --bind "$OUT_DIR"   "$IMAGE"   python3 "$PROJECT_DIR/proto_lightning.py"     --data_dir "$DATA_DIR"     --image_size 128     --out_dir "$OUT_DIR"     --stage1_epochs "$STAGE1_EPOCHS"     --stage2_epochs "$STAGE2_EPOCHS"     --stage1_lr "$STAGE1_LR"     --stage2_lr "$STAGE2_LR"     --batch_size "$BATCH_SIZE"     --stage2_batch_size "$STAGE2_BATCH_SIZE"     --stage1_devices "$GPUS"     --stage2_devices "$GPUS"     --stage1_precision "$STAGE1_PRECISION"     --stage2_precision "$STAGE2_PRECISION"     --stage1_strategy "$STAGE1_STRATEGY"     --stage2_strategy "$STAGE2_STRATEGY"     --wandb_mode "$WANDB_MODE"     --wandb_project "$WANDB_PROJECT"     --wandb_name "$WANDB_NAME"     --tf_d_model "$TF_D_MODEL"     --tf_heads "$TF_HEADS"     --tf_layers "$TF_LAYERS"     --tf_ff "$TF_FF"     --tf_dropout "$TF_DROPOUT"     --tf_global_tokens "$TF_GLOBAL_TOKENS"     --stage2_warmup_steps "$STAGE2_WARMUP_STEPS"     --stage2_min_lr_ratio "$STAGE2_MIN_LR_RATIO"     --stage2_weight_decay "$STAGE2_WEIGHT_DECAY"     --stage2_sample_every_steps "$SAMPLE_EVERY_STEPS"     --stage2_sample_batch_size "$SAMPLE_BATCH_SIZE"     --stage2_sample_candidate_factor "$SAMPLE_CANDIDATE_FACTOR"     --stage2_sample_temperature "$SAMPLE_TEMPERATURE"     --stage2_sample_top_k "$SAMPLE_TOP_K"     --quantize_sparse_coeffs'
