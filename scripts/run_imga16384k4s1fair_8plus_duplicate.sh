#!/bin/bash
# Dedicated allocation/worker script for the 8+ GPU duplicate of
# helloimlixin-rutgers/laser/imga16384k4s1fair-20260814213455.
# Resource shape and GPU generation are supplied explicitly by sbatch after
# probing with --test-only.

#SBATCH --partition=gpu-redhat
#SBATCH --job-name=imga16k4-fairdup
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nice=0

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
DATA_DIR="${LASER_IMAGENET_ROOT:-/scratch/xl598/Projects/data/imagenet}"
CACHE_PYDEPS="/cache/home/xl598/.pydeps/laser_src_py311"
SCRATCH_PYDEPS="/scratch/xl598/.pydeps/laser_src_py311"
SOURCE_WANDB_RUN="helloimlixin-rutgers/laser/imga16384k4s1fair-20260814213455"
SOURCE_GLOBAL_BATCH=128
SOURCE_LR="4.0e-5"
BATCH_PER_GPU=32

require_environment() {
  : "${LASER_RUN_ROOT:?Set LASER_RUN_ROOT to the persistent run directory}"
  : "${LASER_SOURCE_ROOT:?Set LASER_SOURCE_ROOT to the frozen exact source tree}"
  : "${LASER_MODEL_CONFIG:?Set LASER_MODEL_CONFIG to the frozen model config}"
  : "${LASER_RUN_ID:?Set LASER_RUN_ID to the stable W&B run ID}"
  : "${LASER_GPUS_PER_NODE:?Set LASER_GPUS_PER_NODE to the probed shape}"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python interpreter: $PYTHON_BIN" >&2
    return 2
  fi
  if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/val" ]]; then
    echo "Missing ImageNet train/val directories under $DATA_DIR" >&2
    return 2
  fi
  if [[ ! -f "$LASER_MODEL_CONFIG" ]]; then
    echo "Missing frozen config: $LASER_MODEL_CONFIG" >&2
    return 2
  fi
  if [[ ! -f "$LASER_SOURCE_ROOT/third_party/rq-vae-transformer/main_stage1.py" ]]; then
    echo "Missing frozen RQ-VAE source under $LASER_SOURCE_ROOT" >&2
    return 2
  fi
}

scaled_values() {
  local world_size="$1"
  case "$world_size" in
    8) printf '%s %s\n' 256 8.0e-5 ;;
    16) printf '%s %s\n' 512 1.6e-4 ;;
    *) echo "World size must be 8 or 16, got $world_size" >&2; return 2 ;;
  esac
}

latest_checkpoint() {
  find "$LASER_RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_worker() {
  require_environment
  local master_addr="${1:?master address required}"
  local master_port="${2:?master port required}"
  local node_rank="${SLURM_PROCID:?SLURM_PROCID is required}"
  local node_count="${SLURM_NNODES:?SLURM_NNODES is required}"
  local world_size=$((node_count * LASER_GPUS_PER_NODE))
  local scaled
  local total_batch_size
  local learning_rate
  local rq_root="$LASER_SOURCE_ROOT/third_party/rq-vae-transformer"
  local audit_script="$LASER_SOURCE_ROOT/scripts/tools/audit_imga16384k4s1fair_duplicate.py"
  local local_root="${SLURM_TMPDIR:-/tmp}/xl598_${LASER_RUN_ID}_${SLURM_JOB_ID}"
  local gpu_names
  local gpu_count
  local resume_ckpt
  local -a resume_args=()

  scaled="$(scaled_values "$world_size")"
  read -r total_batch_size learning_rate <<< "$scaled"

  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  gpu_count="$(wc -l <<< "$gpu_names")"
  if (( gpu_count != LASER_GPUS_PER_NODE )); then
    echo "Expected $LASER_GPUS_PER_NODE visible GPUs, found $gpu_count" >&2
    return 2
  fi
  if grep -Evq 'NVIDIA (A100|L40S)' <<< "$gpu_names"; then
    echo "Refusing unsupported GPU type(s):" >&2
    echo "$gpu_names" >&2
    return 2
  fi

  mkdir -p "$LASER_RUN_ROOT/hardware" "$LASER_RUN_ROOT/wandb" "$local_root/pycache"
  {
    echo "timestamp=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "node_rank=$node_rank"
    echo "world_size=$world_size"
    echo "visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "$gpu_names"
  } > "$LASER_RUN_ROOT/hardware/node-${node_rank}.txt"

  export PYTHONUSERBASE="${PYTHONUSERBASE:-$CACHE_PYDEPS}"
  export PATH="$CACHE_PYDEPS/bin:$SCRATCH_PYDEPS/bin:$PATH"
  export PYTHONPATH="$LASER_SOURCE_ROOT:$rq_root:$CACHE_PYDEPS/lib/python3.11/site-packages:$SCRATCH_PYDEPS/lib/python3.11/site-packages:$SCRATCH_PYDEPS/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONPYCACHEPREFIX="$local_root/pycache"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  export WANDB_RUN_ID="$LASER_RUN_ID"
  export WANDB_NAME="$LASER_RUN_ID"
  export WANDB_RUN_GROUP="imagenet-a16384-k4-fair-8plus-duplicate"
  export WANDB_RESUME="allow"
  export WANDB_CHECKPOINT_UPLOAD=1
  export WANDB_TAGS="stage1,imagenet,laser,rqvae,a16384,k4,8x8x4,source-imga16384k4s1fair,third-party-original-codec,third-party-original-discriminator,progressive-loss,equal-dict-commitment,scaled-lr,rfid-top3,f32,${world_size}gpu"
  export WANDB_DIR="/mnt/scratch/$USER/laser/$LASER_RUN_ID/wandb"
  export WANDB_DATA_DIR="/mnt/scratch/$USER/laser/$LASER_RUN_ID/wandb/data"
  export WANDB_CACHE_DIR="/mnt/scratch/$USER/laser/$LASER_RUN_ID/wandb/cache"
  export WANDB_ARTIFACT_DIR="/mnt/scratch/$USER/laser/$LASER_RUN_ID/wandb/artifacts"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/cache/home/xl598/.config/wandb}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/home/xl598/.cache}"
  export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
  export TMPDIR="$local_root"
  export TEMP="$local_root"
  export TMP="$local_root"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export LASER_STAGE1_RFID_EVERY_EPOCH=1
  export LASER_RFID_BATCH_SIZE="${LASER_RFID_BATCH_SIZE:-32}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

  mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" \
    "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" "$XDG_CACHE_HOME" \
    "$TORCH_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

  if (( node_rank == 0 )); then
    "$PYTHON_BIN" "$audit_script" \
      --source-root "$LASER_SOURCE_ROOT" \
      --config "$LASER_MODEL_CONFIG" \
      --world-size "$world_size" \
      --learning-rate "$learning_rate" \
      | tee "$LASER_RUN_ROOT/preflight-audit.json"
  fi

  resume_ckpt="$(latest_checkpoint)"
  if [[ -n "$resume_ckpt" && -s "$resume_ckpt" ]]; then
    resume_args=(--load-path "$resume_ckpt" --resume)
  fi

  echo "worker_start=$(date -Is) node=$(hostname) node_rank=$node_rank"
  echo "source_wandb_run=$SOURCE_WANDB_RUN"
  echo "source_commit=8e627116bd9336dbfed230c9fcb231bf3ff4fd33"
  echo "hardware_plan=${node_count}x${LASER_GPUS_PER_NODE} world_size=$world_size"
  echo "batch_per_gpu=$BATCH_PER_GPU total_batch_size=$total_batch_size grad_accumulation=1"
  echo "lr_scale=global_batch/${SOURCE_GLOBAL_BATCH} base_lr=$SOURCE_LR learning_rate=$learning_rate"
  echo "generator_lr=$learning_rate discriminator_lr=$learning_rate dictionary_lr=$learning_rate"
  echo "resume_checkpoint=${resume_ckpt:-none}"

  cd "$rq_root"
  exec "$PYTHON_BIN" -X pycache_prefix="$PYTHONPYCACHEPREFIX" -m torch.distributed.run \
    --nnodes="$node_count" \
    --nproc_per_node="$LASER_GPUS_PER_NODE" \
    --node_rank="$node_rank" \
    --rdzv_id="${SLURM_JOB_ID}-${LASER_RUN_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${master_addr}:${master_port}" \
    --max_restarts=0 \
    main_stage1.py \
      --model-config="$LASER_MODEL_CONFIG" \
      --result-path="$LASER_RUN_ROOT/results" \
      --seed=0 \
      --timeout=86400 \
      "${resume_args[@]}" \
      dataset.root="$DATA_DIR" \
      experiment.batch_size="$BATCH_PER_GPU" \
      experiment.total_batch_size="$total_batch_size" \
      experiment.compute_rfid=true \
      experiment.rfid_backend=original-rqvae \
      optimizer.init_lr="$learning_rate" \
      optimizer.warmup.min_lr="$learning_rate" \
      gan.disc.optimizer.init_lr="$learning_rate" \
      gan.disc.optimizer.warmup.min_lr="$learning_rate" \
      arch.hparams.dict_learning_rate="$learning_rate"
}

run_allocation() {
  require_environment
  local node_count="${SLURM_NNODES:?SLURM_NNODES is required}"
  local world_size=$((node_count * LASER_GPUS_PER_NODE))
  local master_addr
  local master_port=$((20000 + SLURM_JOB_ID % 30000))
  local -a nodes=()

  scaled_values "$world_size" >/dev/null
  mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  if (( ${#nodes[@]} != node_count )); then
    echo "Expected $node_count nodes, resolved ${#nodes[@]}" >&2
    return 2
  fi
  master_addr="${nodes[0]}"
  mkdir -p "$LASER_RUN_ROOT/slurm"
  {
    echo "allocation_start=$(date -Is)"
    echo "job_id=$SLURM_JOB_ID"
    echo "constraint=${LASER_GPU_CONSTRAINT:-unknown}"
    echo "nodes=$node_count"
    echo "gpus_per_node=$LASER_GPUS_PER_NODE"
    echo "world_size=$world_size"
    echo "node_list=${nodes[*]}"
    echo "source_root=$LASER_SOURCE_ROOT"
    echo "config=$LASER_MODEL_CONFIG"
    echo "slurm_nice=$(squeue -h -j "$SLURM_JOB_ID" -o %y)"
  } | tee -a "$LASER_RUN_ROOT/allocation.log"

  srun --kill-on-bad-exit=1 \
    --nodes="$node_count" \
    --ntasks="$node_count" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" \
    --gres="gpu:$LASER_GPUS_PER_NODE" \
    --cpu-bind=cores \
    --output="$LASER_RUN_ROOT/slurm/job-${SLURM_JOB_ID}-node-%t.out" \
    --error="$LASER_RUN_ROOT/slurm/job-${SLURM_JOB_ID}-node-%t.err" \
    bash "$0" --worker "$master_addr" "$master_port"
  date -Is > "$LASER_RUN_ROOT/STAGE1_COMPLETE"
}

case "${1:-}" in
  --worker)
    shift
    run_worker "$@"
    ;;
  *)
    run_allocation
    ;;
esac
