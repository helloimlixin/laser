#!/bin/bash
# Frozen worker payload for an 8- or 16-GPU duplicate of
# helloimlixin-rutgers/laser/ffhqa2048k4rqvaestrict20260808-062048.
# Submit this file with explicit SLURM resource flags and LASER_GPUS_PER_NODE.

set -euo pipefail

if [[ "${1:-}" == "--worker" ]]; then
  shift
fi
if (( $# != 0 )); then
  echo "Unexpected arguments: $*" >&2
  exit 2
fi

SOURCE_RUN="helloimlixin-rutgers/laser/ffhqa2048k4rqvaestrict20260808-062048"
SOURCE_BASE_CONFIG="https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml"
SNAPSHOT="/scratch/xl598/submission_snapshots/ffhq_a2048k4_rqvae_strict_duplicate_20260808_130519"
RQ_ROOT="$SNAPSHOT/rq-vae-transformer"
MODEL_CONFIG="$RQ_ROOT/configs/ffhq/stage1/ffhq256-rqvae-laser-8x8-a2048-k4.yaml"
DATA_DIR="${LASER_FFHQ_ROOT:-/scratch/xl598/Projects/data/ffhq}"
RUN_NAME="ffhq-a2048-k4-rqvae-strict-duplicate-20260808-130519"
RUN_ID="ffhqa2048k4rqvaedup20260808-130519"
RUN_ROOT="${LASER_RUN_ROOT:-/scratch/xl598/runs/laser/$RUN_NAME}"
PYTHON_BIN="${PYTHON_BIN:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
CACHE_PYDEPS="/cache/home/xl598/.pydeps/laser_src_py311"
SCRATCH_PYDEPS="/scratch/xl598/.pydeps/laser_stage2_py311"

NNODES="${SLURM_NNODES:?This worker must run inside a SLURM allocation}"
GPUS_PER_NODE="${LASER_GPUS_PER_NODE:?LASER_GPUS_PER_NODE must match --gres=gpu:N}"
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
GLOBAL_BATCH=128
BASE_LR="4.0e-5"

case "$TOTAL_GPUS" in
  8)
    LOCAL_BATCH=16
    ;;
  16)
    LOCAL_BATCH=8
    ;;
  *)
    echo "Expected exactly 8 or 16 GPUs, got ${NNODES}x${GPUS_PER_NODE}=${TOTAL_GPUS}" >&2
    exit 2
    ;;
esac
if (( LOCAL_BATCH * TOTAL_GPUS != GLOBAL_BATCH )); then
  echo "Physical global batch invariant failed" >&2
  exit 2
fi
if (( GPUS_PER_NODE < 1 || GPUS_PER_NODE > 2 )); then
  echo "This launch supports one or two GPUs per node, got $GPUS_PER_NODE" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "Missing frozen model config: $MODEL_CONFIG" >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing FFHQ root: $DATA_DIR" >&2
  exit 2
fi
if (( $(find "$DATA_DIR" -maxdepth 1 -type f | wc -l) != 70000 )); then
  echo "Expected 70000 FFHQ images directly under $DATA_DIR" >&2
  exit 2
fi

LOCAL_TMP="${SLURM_TMPDIR:-/tmp}/xl598_${RUN_ID}_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT/wandb" "$RUN_ROOT/hardware" "$LOCAL_TMP/pycache"

export PYTHONUSERBASE="$CACHE_PYDEPS"
export PATH="$CACHE_PYDEPS/bin:$SCRATCH_PYDEPS/bin:$PATH"
export PYTHONPATH="$SNAPSHOT:$RQ_ROOT:$CACHE_PYDEPS/lib/python3.11/site-packages:$SCRATCH_PYDEPS/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$LOCAL_TMP/pycache"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SLURM_EXPORT_ENV=ALL

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
export WANDB_NAME="$RUN_NAME"
export WANDB_GROUP="ffhq-a2048-k4-rqvae-strict-duplicates"
export WANDB_RUN_ID="$RUN_ID"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_TAGS="stage1-adv,ffhq,laser,rqvae,k4,a2048,duplicate,source-ffhqa2048k4rqvaestrict20260808-062048,rqvae-config,${TOTAL_GPUS}gpu,effective-batch128,f32"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_DATA_DIR="$RUN_ROOT/wandb/data"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb/cache"
export WANDB_ARTIFACT_DIR="$RUN_ROOT/wandb/artifacts"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/cache/home/xl598/.config/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/home/xl598/.cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
export TMPDIR="$LOCAL_TMP"
export TEMP="$LOCAL_TMP"
export TMP="$LOCAL_TMP"
export LASER_STAGE1_RFID_EVERY_EPOCH="${LASER_STAGE1_RFID_EVERY_EPOCH:-1}"
export LASER_STAGE1_SAVE_LAST_EVERY_EPOCH=1
export LASER_RFID_BATCH_SIZE="${LASER_RFID_BATCH_SIZE:-32}"
export LASER_SOURCE_MODEL_RUN="$SOURCE_RUN"
export LASER_SOURCE_BASE_CONFIG="$SOURCE_BASE_CONFIG"

mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" \
  "$WANDB_CONFIG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \
  "$MPLCONFIGDIR"

GPU_REPORT="$(
  srun --overlap --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
    bash -lc 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'
)"
printf '%s\n' "$GPU_REPORT" | tee "$RUN_ROOT/hardware/job-${SLURM_JOB_ID}.txt"
GPU_NAME_LINES="$(grep -Ec '^NVIDIA (A100|L40S)' <<< "$GPU_REPORT" || true)"
if (( GPU_NAME_LINES != TOTAL_GPUS )); then
  echo "Expected $TOTAL_GPUS A100/L40S report lines, found $GPU_NAME_LINES" >&2
  exit 2
fi
if grep -E '^NVIDIA ' <<< "$GPU_REPORT" | grep -Evq '^NVIDIA (A100|L40S)'; then
  echo "Refusing a non-A100/L40S allocation" >&2
  exit 2
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="$((24000 + SLURM_JOB_ID % 20000))"
LATEST_CKPT="$(
  find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print; exit }'
)"
RESUME_ARGS=()
if [[ -n "$LATEST_CKPT" && -s "$LATEST_CKPT" ]]; then
  RESUME_ARGS=(--load-path "$LATEST_CKPT" --resume)
fi

echo "launch_start=$(date -Is)"
echo "source_run=$SOURCE_RUN"
echo "snapshot=$SNAPSHOT"
echo "snapshot_main_sha256=$(sha256sum "$RQ_ROOT/main_stage1.py" | awk '{print $1}')"
echo "config=$MODEL_CONFIG"
echo "config_sha256=$(sha256sum "$MODEL_CONFIG" | awk '{print $1}')"
echo "data_dir=$DATA_DIR images=70000"
echo "hardware_plan=${NNODES}x${GPUS_PER_NODE} world_size=$TOTAL_GPUS local_batch=$LOCAL_BATCH physical_global_batch=$GLOBAL_BATCH grad_accumulation=1"
echo "optimizer_schedule=adam lr=$BASE_LR betas=0.5,0.9 warmup_epochs=5 cosine_total_epochs=150"
echo "gan_schedule=adam lr=$BASE_LR betas=0.5,0.9 warmup_epochs=5 cosine_total_epochs=150"
echo "precision=f32"
echo "wandb=https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/runs/$WANDB_RUN_ID"
echo "resume_checkpoint=${LATEST_CKPT:-none}"

TRAIN_ARGS=(
  main_stage1.py
  --model-config="$MODEL_CONFIG"
  --result-path="$RUN_ROOT"
  --seed=0
  --timeout=86400
  dataset.root="$DATA_DIR"
  experiment.batch_size="$LOCAL_BATCH"
  experiment.total_batch_size="$GLOBAL_BATCH"
  experiment.epochs=150
  experiment.save_ckpt_freq=1
  experiment.test_freq=1
  experiment.amp=false
  optimizer.init_lr="$BASE_LR"
  optimizer.warmup.epoch=5
  optimizer.warmup.min_lr="$BASE_LR"
  optimizer.warmup.mode=fix
  optimizer.warmup.start_from_zero=true
  gan.disc.optimizer.init_lr="$BASE_LR"
  gan.disc.optimizer.warmup.epoch=5
  gan.disc.optimizer.warmup.min_lr="$BASE_LR"
  gan.disc.optimizer.warmup.mode=fix
  gan.disc.optimizer.warmup.start_from_zero=true
  arch.hparams.dict_learning_rate="$BASE_LR"
)
TRAIN_ARGS+=("${RESUME_ARGS[@]}")

cd "$RQ_ROOT"
srun --kill-on-bad-exit=1 --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  "$PYTHON_BIN" -X pycache_prefix="$PYTHONPYCACHEPREFIX" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="${SLURM_JOB_ID}-${RUN_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=0 \
    "${TRAIN_ARGS[@]}"

touch "$RUN_ROOT/STAGE1_COMPLETE"
echo "launch_complete=$(date -Is)"
