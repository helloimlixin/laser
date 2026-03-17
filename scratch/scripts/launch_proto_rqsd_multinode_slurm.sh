#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${JOB_NAME:-proto-rqsd-8g-mn}"
PARTITION="${PARTITION:-gpu}"
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TOTAL_GPUS="$((NODES * GPUS_PER_NODE))"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/runs/laser_celeba128_quantized}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER/.pydeps/proto_rqsd_py311}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE1_SOURCE_CKPT="${STAGE1_SOURCE_CKPT:-}"
REBUILD_TOKEN_CACHE="${REBUILD_TOKEN_CACHE:-false}"
STAGE1_LR="${STAGE1_LR:-1.6e-3}"
STAGE2_LR="${STAGE2_LR:-8e-3}"
STAGE1_LR_SCHEDULE="${STAGE1_LR_SCHEDULE:-cosine}"
STAGE1_WARMUP_EPOCHS="${STAGE1_WARMUP_EPOCHS:-1}"
STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-0.1}"
STAGE1_DICT_OPTIMIZER="${STAGE1_DICT_OPTIMIZER:-shared_adam}"
STAGE1_DICT_LR_MULTIPLIER="${STAGE1_DICT_LR_MULTIPLIER:-1.0}"
STAGE1_DICT_LR_SCHEDULE="${STAGE1_DICT_LR_SCHEDULE:-cosine}"
STAGE1_DICT_WARMUP_EPOCHS="${STAGE1_DICT_WARMUP_EPOCHS:-1}"
STAGE1_DICT_MIN_LR_RATIO="${STAGE1_DICT_MIN_LR_RATIO:-0.05}"
STAGE1_DICT_GRAD_CLIP="${STAGE1_DICT_GRAD_CLIP:-0.1}"
STAGE1_DICT_MAX_UPDATE_NORM="${STAGE1_DICT_MAX_UPDATE_NORM:-0.0}"
STAGE1_LOSS_SPIKE_SKIP_RATIO="${STAGE1_LOSS_SPIKE_SKIP_RATIO:-0.0}"
STAGE1_LOSS_EMA_BETA="${STAGE1_LOSS_EMA_BETA:-0.98}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
STAGE1_BOTTLENECK_WEIGHT_START="${STAGE1_BOTTLENECK_WEIGHT_START:-1.0}"
STAGE1_BOTTLENECK_WARMUP_EPOCHS="${STAGE1_BOTTLENECK_WARMUP_EPOCHS:-0}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-500}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.01}"
STAGE2_WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-0}"
TOKEN_SUBSET="${TOKEN_SUBSET:-0}"
STAGE2_SAMPLE_EVERY_STEPS="${STAGE2_SAMPLE_EVERY_STEPS:-2000}"
STAGE2_SAMPLE_BATCH_SIZE="${STAGE2_SAMPLE_BATCH_SIZE:-32}"
QUANTIZE_SPARSE_COEFFS="${QUANTIZE_SPARSE_COEFFS:-true}"
AE_NUM_DOWNSAMPLES="${AE_NUM_DOWNSAMPLES:-4}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-scratch}"
WANDB_NAME="${WANDB_NAME:-proto_rqsd_${TOTAL_GPUS}gpu_mn_s${STAGE1_EPOCHS}_s2${STAGE2_EPOCHS}}"
LOG_PREFIX="${LOG_PREFIX:-proto_rqsd_${TOTAL_GPUS}g_mn}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-180}"
ENTRYPOINT="${ENTRYPOINT:-proto.py}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/$ENTRYPOINT" ]]; then
  echo "Missing $ENTRYPOINT under $ROOT_DIR" >&2
  exit 1
fi

sbatch \
  --partition="$PARTITION" \
  --requeue \
  --job-name="$JOB_NAME" \
  --nodes="$NODES" \
  --ntasks-per-node="$GPUS_PER_NODE" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:${GPUS_PER_NODE}" \
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
RUN_TIMESTAMP='"$RUN_TIMESTAMP"'
STAGE1_EPOCHS='"$STAGE1_EPOCHS"'
STAGE2_EPOCHS='"$STAGE2_EPOCHS"'
STAGE1_SOURCE_CKPT='"$STAGE1_SOURCE_CKPT"'
REBUILD_TOKEN_CACHE='"$REBUILD_TOKEN_CACHE"'
STAGE1_LR='"$STAGE1_LR"'
STAGE2_LR='"$STAGE2_LR"'
STAGE1_LR_SCHEDULE='"$STAGE1_LR_SCHEDULE"'
STAGE1_WARMUP_EPOCHS='"$STAGE1_WARMUP_EPOCHS"'
STAGE1_MIN_LR_RATIO='"$STAGE1_MIN_LR_RATIO"'
STAGE1_DICT_OPTIMIZER='"$STAGE1_DICT_OPTIMIZER"'
STAGE1_DICT_LR_MULTIPLIER='"$STAGE1_DICT_LR_MULTIPLIER"'
STAGE1_DICT_LR_SCHEDULE='"$STAGE1_DICT_LR_SCHEDULE"'
STAGE1_DICT_WARMUP_EPOCHS='"$STAGE1_DICT_WARMUP_EPOCHS"'
STAGE1_DICT_MIN_LR_RATIO='"$STAGE1_DICT_MIN_LR_RATIO"'
STAGE1_DICT_GRAD_CLIP='"$STAGE1_DICT_GRAD_CLIP"'
STAGE1_DICT_MAX_UPDATE_NORM='"$STAGE1_DICT_MAX_UPDATE_NORM"'
STAGE1_LOSS_SPIKE_SKIP_RATIO='"$STAGE1_LOSS_SPIKE_SKIP_RATIO"'
STAGE1_LOSS_EMA_BETA='"$STAGE1_LOSS_EMA_BETA"'
GRAD_CLIP='"$GRAD_CLIP"'
STAGE1_BOTTLENECK_WEIGHT_START='"$STAGE1_BOTTLENECK_WEIGHT_START"'
STAGE1_BOTTLENECK_WARMUP_EPOCHS='"$STAGE1_BOTTLENECK_WARMUP_EPOCHS"'
STAGE2_WARMUP_STEPS='"$STAGE2_WARMUP_STEPS"'
STAGE2_MIN_LR_RATIO='"$STAGE2_MIN_LR_RATIO"'
STAGE2_WEIGHT_DECAY='"$STAGE2_WEIGHT_DECAY"'
BATCH_SIZE='"$BATCH_SIZE"'
STAGE2_BATCH_SIZE='"$STAGE2_BATCH_SIZE"'
NUM_WORKERS='"$NUM_WORKERS"'
TOKEN_NUM_WORKERS='"$TOKEN_NUM_WORKERS"'
TOKEN_SUBSET='"$TOKEN_SUBSET"'
STAGE2_SAMPLE_EVERY_STEPS='"$STAGE2_SAMPLE_EVERY_STEPS"'
STAGE2_SAMPLE_BATCH_SIZE='"$STAGE2_SAMPLE_BATCH_SIZE"'
QUANTIZE_SPARSE_COEFFS='"$QUANTIZE_SPARSE_COEFFS"'
AE_NUM_DOWNSAMPLES='"$AE_NUM_DOWNSAMPLES"'
WANDB_MODE='"$WANDB_MODE"'
WANDB_PROJECT='"$WANDB_PROJECT"'
WANDB_NAME='"$WANDB_NAME"'
WANDB_API_KEY_FILE='"$WANDB_API_KEY_FILE"'
DIST_TIMEOUT_MINUTES='"$DIST_TIMEOUT_MINUTES"'
ENTRYPOINT='"$ENTRYPOINT"'
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    set +u
    source /etc/profile.d/modules.sh
    set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u
    source /usr/share/Modules/init/bash
    set -u
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
export WANDB_API_KEY_FILE
export SINGULARITYENV_WANDB_API_KEY_FILE="$WANDB_API_KEY_FILE"
export APPTAINERENV_WANDB_API_KEY_FILE="$WANDB_API_KEY_FILE"
unset WANDB_API_KEY || true
unset SINGULARITYENV_WANDB_API_KEY || true
unset APPTAINERENV_WANDB_API_KEY || true
export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
export MASTER_ADDR MASTER_PORT
if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$OUT_DIR" "$IMAGE" python3 - <<'PY_DEP' >/dev/null 2>&1
import importlib
import scipy
module = importlib.import_module("wandb")
raise SystemExit(0 if hasattr(module, "init") and hasattr(module, "Api") else 1)
PY_DEP
then
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$OUT_DIR" "$IMAGE" python3 -m pip install --user scipy wandb
fi
RUNNER="$OUT_DIR/slurm_proto_multinode_${SLURM_JOB_ID}.sh"
cat > "$RUNNER" <<EOF_RUNNER
#!/bin/bash
set -euo pipefail
export RANK="\$SLURM_PROCID"
export WORLD_SIZE="\$SLURM_NTASKS"
export LOCAL_RANK="\$SLURM_LOCALID"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export PROTO_LAUNCH_TIMESTAMP="${RUN_TIMESTAMP}"
export PYTHONUSERBASE="${PYTHONUSERBASE_DIR}"
export PYTHONNOUSERSITE=0
export PYTHONPATH="${PYTHON_SITE}\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE}"
if [[ -f "\${WANDB_API_KEY_FILE}" ]]; then
  IFS= read -r WANDB_API_KEY < "\${WANDB_API_KEY_FILE}" || true
  export WANDB_API_KEY
fi
EXTRA_ARGS=()
if [[ -n "${STAGE1_SOURCE_CKPT}" ]]; then
  EXTRA_ARGS+=(--stage1_source_ckpt "${STAGE1_SOURCE_CKPT}")
fi
if [[ "${REBUILD_TOKEN_CACHE}" == "1" || "${REBUILD_TOKEN_CACHE}" == "true" || "${REBUILD_TOKEN_CACHE}" == "TRUE" || "${REBUILD_TOKEN_CACHE}" == "yes" || "${REBUILD_TOKEN_CACHE}" == "YES" ]]; then
  EXTRA_ARGS+=(--rebuild_token_cache)
fi
exec python3 "${PROJECT_DIR}/${ENTRYPOINT}" \
  --dataset celeba \
  --data_dir "${DATA_DIR}" \
  --image_size 128 \
  --out_dir "${OUT_DIR}" \
  --dist_timeout_minutes "${DIST_TIMEOUT_MINUTES}" \
  --stage1_epochs "${STAGE1_EPOCHS}" \
  --stage2_epochs "${STAGE2_EPOCHS}" \
  --stage1_lr "${STAGE1_LR}" \
  --stage2_lr "${STAGE2_LR}" \
  --stage1_lr_schedule "${STAGE1_LR_SCHEDULE}" \
  --stage1_warmup_epochs "${STAGE1_WARMUP_EPOCHS}" \
  --stage1_min_lr_ratio "${STAGE1_MIN_LR_RATIO}" \
  --stage1_dict_optimizer "${STAGE1_DICT_OPTIMIZER}" \
  --stage1_dict_lr_multiplier "${STAGE1_DICT_LR_MULTIPLIER}" \
  --stage1_dict_lr_schedule "${STAGE1_DICT_LR_SCHEDULE}" \
  --stage1_dict_warmup_epochs "${STAGE1_DICT_WARMUP_EPOCHS}" \
  --stage1_dict_min_lr_ratio "${STAGE1_DICT_MIN_LR_RATIO}" \
  --stage1_dict_grad_clip "${STAGE1_DICT_GRAD_CLIP}" \
  --stage1_dict_max_update_norm "${STAGE1_DICT_MAX_UPDATE_NORM}" \
  --stage1_loss_spike_skip_ratio "${STAGE1_LOSS_SPIKE_SKIP_RATIO}" \
  --stage1_loss_ema_beta "${STAGE1_LOSS_EMA_BETA}" \
  --grad_clip "${GRAD_CLIP}" \
  --stage1_bottleneck_weight_start "${STAGE1_BOTTLENECK_WEIGHT_START}" \
  --stage1_bottleneck_warmup_epochs "${STAGE1_BOTTLENECK_WARMUP_EPOCHS}" \
  --stage2_warmup_steps "${STAGE2_WARMUP_STEPS}" \
  --stage2_min_lr_ratio "${STAGE2_MIN_LR_RATIO}" \
  --stage2_weight_decay "${STAGE2_WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --token_num_workers "${TOKEN_NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --stage2_batch_size "${STAGE2_BATCH_SIZE}" \
  --ae_num_downsamples "${AE_NUM_DOWNSAMPLES}" \
  --token_subset "${TOKEN_SUBSET}" \
  --rfid_num_samples 0 \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --stage2_sample_every_steps "${STAGE2_SAMPLE_EVERY_STEPS}" \
  --stage2_sample_batch_size "${STAGE2_SAMPLE_BATCH_SIZE}" \
  --quantize_sparse_coeffs "${QUANTIZE_SPARSE_COEFFS}" \\
  "\${EXTRA_ARGS[@]}"
EOF_RUNNER
chmod +x "$RUNNER"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 nvidia-smi
srun --ntasks-per-node='"$GPUS_PER_NODE"' --gpus-per-task=1 singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER_NAME" \
  --bind "$DATA_DIR" \
  --bind "$OUT_DIR" \
  "$IMAGE" \
  bash "$RUNNER"'
