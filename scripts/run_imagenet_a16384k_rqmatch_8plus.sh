#!/bin/bash
# Fresh ImageNet LASER Stage-1 run with 8x8xK codes and original RQ-VAE GAN dynamics.

#SBATCH --partition=gpu
#SBATCH --job-name=imga16k-rqmatch
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nice=0

set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
DATA_DIR="${LASER_IMAGENET_ROOT:-/scratch/xl598/Projects/data/imagenet}"
CACHE_PYDEPS="/cache/home/xl598/.pydeps/laser_src_py311"
SCRATCH_PYDEPS="/scratch/xl598/.pydeps/laser_src_py311"
TOTAL_BATCH_SIZE=128
LEARNING_RATE="4.0e-5"
TARGET_EPOCHS=10

require_environment() {
  : "${LASER_RUN_ROOT:?Set LASER_RUN_ROOT to the persistent run directory}"
  : "${LASER_SOURCE_ROOT:?Set LASER_SOURCE_ROOT to the frozen source tree}"
  : "${LASER_MODEL_CONFIG:?Set LASER_MODEL_CONFIG to the frozen model config}"
  : "${LASER_RUN_ID:?Set LASER_RUN_ID to the stable W&B run ID}"
  : "${LASER_GPUS_PER_NODE:?Set LASER_GPUS_PER_NODE to the probed shape}"
  : "${LASER_SPARSITY_LEVEL:?Set LASER_SPARSITY_LEVEL to 4 or 8}"
  if [[ "$LASER_SPARSITY_LEVEL" != "4" && "$LASER_SPARSITY_LEVEL" != "8" ]]; then
    echo "LASER_SPARSITY_LEVEL must be 4 or 8, got $LASER_SPARSITY_LEVEL" >&2
    return 2
  fi
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
  if [[ ! -f "$LASER_SOURCE_ROOT/scripts/tools/audit_imagenet_a16384k_rqmatch_8plus.py" ]]; then
    echo "Missing frozen audit script under $LASER_SOURCE_ROOT" >&2
    return 2
  fi
  if [[ ! -s "$LASER_SOURCE_ROOT/vgg_lpips/vgg.pth" ]]; then
    echo "Missing LPIPS linear weights under the frozen source tree" >&2
    return 2
  fi
  if [[ ! -s "$LASER_SOURCE_ROOT/vgg_lpips/vgg16-397923af.pth" ]]; then
    echo "Missing VGG16 feature weights under the frozen source tree" >&2
    return 2
  fi
  if [[ "$(md5sum "$LASER_SOURCE_ROOT/vgg_lpips/vgg.pth" | cut -d' ' -f1)" != "d507d7349b931f0638a25a48a722f98a" ]]; then
    echo "LPIPS linear-weight checksum mismatch" >&2
    return 2
  fi
  if [[ "$(sha256sum "$LASER_SOURCE_ROOT/vgg_lpips/vgg16-397923af.pth" | cut -d' ' -f1)" != "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0" ]]; then
    echo "VGG16 feature-weight checksum mismatch" >&2
    return 2
  fi
}

local_batch_size() {
  local world_size="$1"
  case "$world_size" in
    8) echo 16 ;;
    16) echo 8 ;;
    *) echo "World size must be 8 or 16, got $world_size" >&2; return 2 ;;
  esac
}

python_environment() {
  export PYTHONUSERBASE="${PYTHONUSERBASE:-$CACHE_PYDEPS}"
  export PATH="$CACHE_PYDEPS/bin:$SCRATCH_PYDEPS/bin:$PATH"
  export PYTHONPATH="$LASER_SOURCE_ROOT:$LASER_SOURCE_ROOT/third_party/rq-vae-transformer:$LASER_RUN_ROOT/python_deps:$CACHE_PYDEPS/lib/python3.11/site-packages:$SCRATCH_PYDEPS/lib/python3.11/site-packages:$SCRATCH_PYDEPS/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
}

latest_checkpoint() {
  find "$LASER_RUN_ROOT/results" -type f -name last_model.pt -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_worker() {
  require_environment
  local master_addr="${1:?master address required}"
  local master_port="${2:?master port required}"
  local node_rank="${SLURM_PROCID:?SLURM_PROCID is required}"
  local node_count="${SLURM_NNODES:?SLURM_NNODES is required}"
  local world_size=$((node_count * LASER_GPUS_PER_NODE))
  local local_batch
  local rq_root="$LASER_SOURCE_ROOT/third_party/rq-vae-transformer"
  local audit_script="$LASER_SOURCE_ROOT/scripts/tools/audit_imagenet_a16384k_rqmatch_8plus.py"
  local local_root="${SLURM_TMPDIR:-/mnt/scratch/$USER}/laser/$LASER_RUN_ID/$SLURM_JOB_ID"
  local local_tmp="${SLURM_TMPDIR:-/mnt/scratch/$USER}/lz/$SLURM_JOB_ID"
  local visible_devices
  local gpu_names
  local gpu_count
  local host_ipv4
  local nccl_socket_ifname
  local resume_checkpoint
  local -a resume_args=()

  local_batch="$(local_batch_size "$world_size")"
  visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "$visible_devices" ]]; then
    echo "SLURM did not set CUDA_VISIBLE_DEVICES" >&2
    return 2
  fi
  gpu_names="$(nvidia-smi --id="$visible_devices" --query-gpu=name --format=csv,noheader)"
  gpu_count="$(tr ',' '\n' <<< "$visible_devices" | sed '/^[[:space:]]*$/d' | wc -l)"
  if (( gpu_count != LASER_GPUS_PER_NODE )); then
    echo "Expected $LASER_GPUS_PER_NODE visible GPUs, found $gpu_count" >&2
    return 2
  fi
  if grep -Evq 'NVIDIA (A100|L40S)' <<< "$gpu_names"; then
    echo "Refusing unsupported GPU type(s):" >&2
    echo "$gpu_names" >&2
    return 2
  fi
  host_ipv4="$(getent ahostsv4 "$(hostname -s)" | awk 'NR == 1 {print $1}')"
  nccl_socket_ifname="$({ ip -o -4 addr show || true; } | awk -v ip="$host_ipv4" '
    { split($4, address, "/") }
    address[1] == ip { print $2; exit }
  ')"
  if [[ -z "$host_ipv4" || -z "$nccl_socket_ifname" ]]; then
    echo "Could not resolve the hostname network interface for NCCL" >&2
    return 2
  fi

  mkdir -p "$LASER_RUN_ROOT/hardware" "$local_root/pycache" "$local_tmp"
  {
    echo "timestamp=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "node_rank=$node_rank"
    echo "world_size=$world_size"
    echo "visible_devices=$visible_devices"
    echo "host_ipv4=$host_ipv4"
    echo "nccl_socket_ifname=$nccl_socket_ifname"
    echo "$gpu_names"
  } > "$LASER_RUN_ROOT/hardware/node-${node_rank}.txt"

  python_environment
  export PYTHONPYCACHEPREFIX="$local_root/pycache"
  export WANDB_MODE="online"
  export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  export WANDB_RUN_ID="$LASER_RUN_ID"
  export WANDB_NAME="${WANDB_NAME:-imagenet-a16384-k${LASER_SPARSITY_LEVEL}-rqmatch-$LASER_RUN_ID}"
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet-a16384-k${LASER_SPARSITY_LEVEL}-rqmatch-8plus}"
  export WANDB_RESUME="allow"
  export WANDB_CHECKPOINT_UPLOAD=1
  export WANDB_TAGS="stage1,imagenet,laser,rqvae,a16384,k${LASER_SPARSITY_LEVEL},8x8x${LASER_SPARSITY_LEVEL},rqvae-matched,original-rqvae-discriminator,effective-batch128,full-gan-checkpoint,original-rqvae-rfid,rfid-top3,f32,${world_size}gpu"
  export WANDB_DIR="$local_root/wandb"
  export WANDB_DATA_DIR="$local_root/wandb/data"
  export WANDB_CACHE_DIR="$local_root/wandb/cache"
  export WANDB_ARTIFACT_DIR="$local_root/wandb/artifacts"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/cache/home/xl598/.config/wandb}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/home/xl598/.cache}"
  export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
  export LASER_VGG_LPIPS_DIR="$LASER_SOURCE_ROOT/vgg_lpips"
  export LASER_VGG16_WEIGHTS="$LASER_SOURCE_ROOT/vgg_lpips/vgg16-397923af.pth"
  export TMPDIR="$local_tmp"
  export TEMP="$local_tmp"
  export TMP="$local_tmp"
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
  export NCCL_NVLS_ENABLE=0
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" \
    "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" "$XDG_CACHE_HOME" \
    "$TORCH_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

  if (( node_rank == 0 )); then
    "$PYTHON_BIN" "$audit_script" \
      --source-root "$LASER_SOURCE_ROOT" \
      --config "$LASER_MODEL_CONFIG" \
      --world-size "$world_size" \
      --local-batch-size "$local_batch" \
      --sparsity-level "$LASER_SPARSITY_LEVEL" \
      | tee "$LASER_RUN_ROOT/preflight-audit.json"
  fi

  resume_checkpoint="$(latest_checkpoint)"
  if [[ -n "$resume_checkpoint" && -s "$resume_checkpoint" ]]; then
    resume_args=(--load-path "$resume_checkpoint" --resume)
  fi

  echo "worker_start=$(date -Is) node=$(hostname) node_rank=$node_rank"
  echo "source_root=$LASER_SOURCE_ROOT"
  echo "hardware_plan=${node_count}x${LASER_GPUS_PER_NODE} world_size=$world_size"
  echo "local_batch_size=$local_batch total_batch_size=$TOTAL_BATCH_SIZE"
  echo "main_lr=$LEARNING_RATE discriminator_lr=$LEARNING_RATE dictionary_lr=$LEARNING_RATE"
  echo "code_shape=8x8x$LASER_SPARSITY_LEVEL"
  echo "discriminator_policy=original-rqvae-start-epoch0-one-update-per-generator-step"
  echo "resume_checkpoint=${resume_checkpoint:-none}"

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
      --model-config "$LASER_MODEL_CONFIG" \
      --result-path "$LASER_RUN_ROOT/results" \
      --seed 0 \
      --timeout 86400 \
      "${resume_args[@]}" \
      "dataset.root=$DATA_DIR" \
      "experiment.batch_size=$local_batch" \
      "experiment.total_batch_size=$TOTAL_BATCH_SIZE" \
      "experiment.epochs=$TARGET_EPOCHS" \
      "experiment.compute_rfid=true" \
      "experiment.rfid_backend=original-rqvae" \
      "experiment.recovery_ckpt_freq_steps=250" \
      "optimizer.init_lr=$LEARNING_RATE" \
      "optimizer.warmup.min_lr=$LEARNING_RATE" \
      "arch.hparams.dict_learning_rate=$LEARNING_RATE" \
      "gan.disc.optimizer.init_lr=$LEARNING_RATE" \
      "gan.disc.optimizer.warmup.min_lr=$LEARNING_RATE"
}

verify_completed_checkpoints() {
  local policy
  policy="$(find "$LASER_RUN_ROOT/results" -type f -name checkpoint_policy.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "$policy" || ! -s "$policy" ]]; then
    echo "Stage 1 completed without checkpoint_policy.json" >&2
    return 2
  fi
  python_environment
  "$PYTHON_BIN" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); b=d["best"]; assert len(b)==3, b; assert all((p.parent / x["path"]).is_file() for x in b); assert (p.parent / "last_model.pt").is_file(); print("validated last plus rFID-best three checkpoints")' "$policy"
}

run_allocation() {
  require_environment
  local node_count="${SLURM_NNODES:?SLURM_NNODES is required}"
  local world_size=$((node_count * LASER_GPUS_PER_NODE))
  local master_addr
  local master_port=$((20000 + SLURM_JOB_ID % 30000))
  local -a nodes=()

  local_batch_size "$world_size" >/dev/null
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
    bash "$LASER_SOURCE_ROOT/scripts/run_imagenet_a16384k_rqmatch_8plus.sh" \
      --worker "$master_addr" "$master_port"
  verify_completed_checkpoints
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
