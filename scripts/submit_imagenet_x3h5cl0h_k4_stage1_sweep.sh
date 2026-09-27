#!/bin/bash
# Submit this file with one of the probed 8/16-GPU shapes using
# --gres=gpu:<count>. Amarel uses select/cons_res, so do not submit it with a
# --gpus-per-* option. Each model always
# trains on four GPUs, preserving the effective batch and optimizer dynamics of
# helloimlixin-rutgers/laser/x3h5cl0h-a16384-k2-20260719-014434.
#
# Required submission environment:
#   LASER_RUN_ROOT, LASER_SWEEP_ID, LASER_GPUS_PER_NODE, LASER_MODEL_CONFIG
# Optional: LASER_SPARSITY_LEVEL (default: 4).

#SBATCH --partition=gpu-redhat
#SBATCH --job-name=imnet-x3h5-k4
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

REFERENCE_SNAPSHOT="${LASER_REFERENCE_SNAPSHOT:-/scratch/xl598/submission_snapshots/laser_x3h5cl0h_strict_bottleneck_sweep_20260719_014434}"
RQ_ROOT="$REFERENCE_SNAPSHOT/third_party/rq-vae-transformer"
DATA_DIR="${LASER_IMAGENET_ROOT:-/scratch/xl598/Projects/data/imagenet}"
PYTHON_BIN="${PYTHON_BIN:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
CACHE_PYDEPS="/cache/home/xl598/.pydeps/laser_src_py311"
SCRATCH_PYDEPS="/scratch/xl598/.pydeps/laser_src_py311"
ATOMS=(2048 4096 8192 16384)
SPARSITY_LEVEL="${LASER_SPARSITY_LEVEL:-4}"

require_environment() {
  : "${LASER_RUN_ROOT:?Set LASER_RUN_ROOT to the frozen sweep output directory}"
  : "${LASER_SWEEP_ID:?Set LASER_SWEEP_ID to a stable timestamp or run identifier}"
  : "${LASER_GPUS_PER_NODE:?Set LASER_GPUS_PER_NODE to the probed allocation shape}"
  : "${LASER_MODEL_CONFIG:?Set LASER_MODEL_CONFIG to the frozen stage-1 config}"
  if ! [[ "$SPARSITY_LEVEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "LASER_SPARSITY_LEVEL must be a positive integer, got $SPARSITY_LEVEL" >&2
    return 2
  fi
}

latest_checkpoint() {
  local run_dir="$1"
  find "$run_dir" -type f -name last_model.pt -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

run_worker() {
  require_environment
  local variant_index="${1:?variant index required}"
  local master_addr="${2:?master address required}"
  local master_port="${3:?master port required}"
  local atoms="${ATOMS[$variant_index]}"
  local node_rank="${SLURM_PROCID:?SLURM_PROCID is required for the worker node rank}"
  local model_nodes="${LASER_MODEL_NNODES:?LASER_MODEL_NNODES is required}"
  local total_model_gpus=$((model_nodes * LASER_GPUS_PER_NODE))
  local run_dir="$LASER_RUN_ROOT/k${SPARSITY_LEVEL}-a${atoms}"
  local local_tmp="${SLURM_TMPDIR:-/tmp}/xl598_imnet_k${SPARSITY_LEVEL}_${SLURM_JOB_ID}_${atoms}"
  local gpu_names
  local gpu_count
  local resume_ckpt
  local resume_args=()

  if (( total_model_gpus != 4 )); then
    echo "Each model must receive four GPUs; got ${model_nodes}x${LASER_GPUS_PER_NODE}" >&2
    return 2
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python interpreter: $PYTHON_BIN" >&2
    return 2
  fi
  if [[ ! -f "$LASER_MODEL_CONFIG" ]]; then
    echo "Missing frozen model config: $LASER_MODEL_CONFIG" >&2
    return 2
  fi
  if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/val" ]]; then
    echo "Missing ImageNet train/val directories under $DATA_DIR" >&2
    return 2
  fi

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

  mkdir -p "$run_dir/wandb" "$run_dir/tmp" "$run_dir/hardware" "$local_tmp/pycache"
  {
    echo "timestamp=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "node_rank=$node_rank"
    echo "visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "$gpu_names"
  } > "$run_dir/hardware/node-${node_rank}.txt"

  export PYTHONUSERBASE="${PYTHONUSERBASE:-$CACHE_PYDEPS}"
  export PATH="$CACHE_PYDEPS/bin:$SCRATCH_PYDEPS/bin:$PATH"
  export PYTHONPATH="$REFERENCE_SNAPSHOT:$RQ_ROOT:$CACHE_PYDEPS/lib/python3.11/site-packages:$SCRATCH_PYDEPS/lib/python3.11/site-packages:$SCRATCH_PYDEPS/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONPYCACHEPREFIX="$local_tmp/pycache"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  export WANDB_NAME="imagenet-rqvae-laser-a${atoms}-k${SPARSITY_LEVEL}-10ep-x3h5cl0h-matched-${LASER_SWEEP_ID}"
  export WANDB_GROUP="imagenet-x3h5cl0h-k${SPARSITY_LEVEL}-stage1-sweep-${LASER_SWEEP_ID}"
  export WANDB_RUN_ID="x3h5k${SPARSITY_LEVEL}-a${atoms}-${LASER_SWEEP_ID}"
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
  export WANDB_TAGS="stage1-adv,imagenet,laser,rqvae,k${SPARSITY_LEVEL},a${atoms},source-x3h5cl0h,matched-x3h5cl0h,bottleneck-ablation,rfid-top3,effective-batch128,f32"
  export WANDB_DIR="$run_dir/wandb"
  export WANDB_DATA_DIR="$run_dir/wandb/data"
  export WANDB_CACHE_DIR="$run_dir/wandb/cache"
  export WANDB_ARTIFACT_DIR="$run_dir/wandb/artifacts"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/cache/home/xl598/.config/wandb}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/home/xl598/.cache}"
  export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
  export TMPDIR="$local_tmp"
  export TEMP="$local_tmp"
  export TMP="$local_tmp"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export LASER_STAGE1_RFID_EVERY_EPOCH="${LASER_STAGE1_RFID_EVERY_EPOCH:-1}"
  export LASER_RFID_BATCH_SIZE="${LASER_RFID_BATCH_SIZE:-32}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

  mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" \
    "$WANDB_CONFIG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \
    "$MPLCONFIGDIR"

  resume_ckpt="$(latest_checkpoint "$run_dir")"
  if [[ -n "$resume_ckpt" && -s "$resume_ckpt" ]]; then
    resume_args=(--load-path "$resume_ckpt" --resume)
  fi

  echo "worker_start=$(date -Is) node=$(hostname) node_rank=$node_rank"
  echo "variant=k${SPARSITY_LEVEL}-a${atoms} dictionary_size=$atoms code_shape=8,8,${SPARSITY_LEVEL}"
  echo "source_wandb_run=helloimlixin-rutgers/laser/x3h5cl0h-a16384-k2-20260719-014434"
  echo "hardware_plan=${model_nodes}x${LASER_GPUS_PER_NODE} world_size=$total_model_gpus batch_per_gpu=32 effective_batch=128"
  echo "resume_checkpoint=${resume_ckpt:-none}"

  cd "$RQ_ROOT"
  exec "$PYTHON_BIN" -X pycache_prefix="$PYTHONPYCACHEPREFIX" -m torch.distributed.run \
    --nnodes="$model_nodes" \
    --nproc_per_node="$LASER_GPUS_PER_NODE" \
    --node_rank="$node_rank" \
    --rdzv_id="${SLURM_JOB_ID}-${atoms}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${master_addr}:${master_port}" \
    --max_restarts=0 \
    main_stage1.py \
      --model-config="$LASER_MODEL_CONFIG" \
      --result-path="$run_dir" \
      --timeout=86400 \
      "${resume_args[@]}" \
      dataset.root="$DATA_DIR" \
      arch.hparams.n_embed="$atoms" \
      arch.hparams.code_shape="[8,8,${SPARSITY_LEVEL}]" \
      arch.hparams.sparsity_level="$SPARSITY_LEVEL"
}

run_sweep() {
  require_environment
  local job_nodes="${SLURM_NNODES:?SLURM_NNODES is required}"
  local total_gpus=$((job_nodes * LASER_GPUS_PER_NODE))
  local model_nodes
  local parallel_models
  local variant_index
  local wave_start
  local slot
  local start_node
  local master_port
  local nodelist
  local wave_failed
  local pid
  local -a allocated_nodes=()
  local -a remaining=()
  local -a wave_pids=()

  if (( LASER_GPUS_PER_NODE < 1 || LASER_GPUS_PER_NODE > 2 )); then
    echo "LASER_GPUS_PER_NODE must be 1 or 2, got $LASER_GPUS_PER_NODE" >&2
    return 2
  fi
  if (( total_gpus != 8 && total_gpus != 16 )); then
    echo "This sweep requires exactly 8 or 16 GPUs, got $total_gpus" >&2
    return 2
  fi
  if (( 4 % LASER_GPUS_PER_NODE != 0 )); then
    echo "Four model GPUs must divide evenly across the allocation nodes" >&2
    return 2
  fi

  model_nodes=$((4 / LASER_GPUS_PER_NODE))
  parallel_models=$((total_gpus / 4))
  export LASER_MODEL_NNODES="$model_nodes"
  mapfile -t allocated_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  if (( ${#allocated_nodes[@]} != job_nodes )); then
    echo "Node-list size ${#allocated_nodes[@]} does not match allocation size $job_nodes" >&2
    return 2
  fi

  mkdir -p "$LASER_RUN_ROOT"
  {
    echo "sweep_start=$(date -Is)"
    echo "job_id=$SLURM_JOB_ID"
    echo "job_nodes=$job_nodes"
    echo "gpus_per_node=$LASER_GPUS_PER_NODE"
    echo "total_gpus=$total_gpus"
    echo "model_nodes=$model_nodes"
    echo "model_world_size=4"
    echo "parallel_models=$parallel_models"
    echo "nodes=${allocated_nodes[*]}"
    echo "config=$LASER_MODEL_CONFIG"
    echo "reference_snapshot=$REFERENCE_SNAPSHOT"
  } | tee -a "$LASER_RUN_ROOT/allocation.log"

  for variant_index in "${!ATOMS[@]}"; do
    if [[ -f "$LASER_RUN_ROOT/k${SPARSITY_LEVEL}-a${ATOMS[$variant_index]}/STAGE1_COMPLETE" ]]; then
      echo "skip_complete=k${SPARSITY_LEVEL}-a${ATOMS[$variant_index]}"
    else
      remaining+=("$variant_index")
    fi
  done

  for ((wave_start = 0; wave_start < ${#remaining[@]}; wave_start += parallel_models)); do
    wave_pids=()
    echo "wave_start=$(date -Is) offset=$wave_start"
    for ((slot = 0; slot < parallel_models && wave_start + slot < ${#remaining[@]}; slot++)); do
      variant_index="${remaining[$((wave_start + slot))]}"
      start_node=$((slot * model_nodes))
      nodelist="$(IFS=,; echo "${allocated_nodes[*]:start_node:model_nodes}")"
      master_port=$((24000 + SLURM_JOB_ID % 20000 + variant_index))
      mkdir -p "$LASER_RUN_ROOT/k${SPARSITY_LEVEL}-a${ATOMS[$variant_index]}/slurm"
      echo "launch_variant=k${SPARSITY_LEVEL}-a${ATOMS[$variant_index]} nodes=$nodelist master=${allocated_nodes[$start_node]}:$master_port"
      (
        if srun --exclusive --exact --kill-on-bad-exit=1 \
          --nodes="$model_nodes" \
          --nodelist="$nodelist" \
          --ntasks="$model_nodes" \
          --ntasks-per-node=1 \
          --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
          --gres="gpu:$LASER_GPUS_PER_NODE" \
          --output="$LASER_RUN_ROOT/k${SPARSITY_LEVEL}-a${ATOMS[$variant_index]}/slurm/job-${SLURM_JOB_ID}-node-%t.out" \
          --error="$LASER_RUN_ROOT/k${SPARSITY_LEVEL}-a${ATOMS[$variant_index]}/slurm/job-${SLURM_JOB_ID}-node-%t.err" \
          bash "$0" worker "$variant_index" "${allocated_nodes[$start_node]}" "$master_port"; then
          date -Is > "$LASER_RUN_ROOT/k${SPARSITY_LEVEL}-a${ATOMS[$variant_index]}/STAGE1_COMPLETE"
        else
          status=$?
          echo "$status $(date -Is)" > "$LASER_RUN_ROOT/k${SPARSITY_LEVEL}-a${ATOMS[$variant_index]}/STAGE1_FAILED"
          exit "$status"
        fi
      ) &
      wave_pids+=("$!")
    done

    wave_failed=0
    for pid in "${wave_pids[@]}"; do
      if ! wait "$pid"; then
        wave_failed=1
      fi
    done
    if (( wave_failed != 0 )); then
      echo "At least one model failed in wave offset $wave_start" >&2
      return 1
    fi
    echo "wave_complete=$(date -Is) offset=$wave_start"
  done

  date -Is > "$LASER_RUN_ROOT/SWEEP_COMPLETE"
  echo "sweep_complete=$(date -Is)"
}

if [[ "${1:-}" == worker ]]; then
  shift
  run_worker "$@"
else
  run_sweep
fi
