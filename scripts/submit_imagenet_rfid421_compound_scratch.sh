#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/scratch/xl598/runs/laser/imagenet-rfid421-compound-1400m-scratch-20260919}
SELF="$RUN_DIR/submit.sh"
NODES=${NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CONSTRAINT=${CONSTRAINT:-ampere|adalovelace}
CPUS_PER_TASK=${CPUS_PER_TASK:-$((6 * GPUS_PER_NODE))}
MEMORY=${MEMORY:-96G}
TIME_LIMIT=${TIME_LIMIT:-72:00:00}
export RUN_DIR GPUS_PER_NODE
PYTHON="$RUN_DIR/runtime/bin/python"

case "${1:-}" in
  --worker)
    exec 9>"$RUN_DIR/allocation.lock"
    flock -n 9
    export MASTER_ADDR MASTER_PORT LASER_STOP_TIME_UNIX
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT=$((21000 + SLURM_JOB_ID % 20000))
    LASER_STOP_TIME_UNIX=$(python3 - <<'PY'
import datetime, os, re, subprocess
line = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'], text=True)
end = re.search(r'\bEndTime=(\S+)', line).group(1)
print(int(datetime.datetime.fromisoformat(end).timestamp()) - 1800)
PY
)
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash "$SELF" --node cache
    export MASTER_PORT=$((MASTER_PORT + 1))
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash "$SELF" --node smoke
    export MASTER_PORT=$((MASTER_PORT + 1))
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash "$SELF" --node train
    ;;
  --node)
    phase="$2"
    export LASER_LOCAL_SCRATCH="/mnt/scratch/$USER/laser/$SLURM_JOB_ID-imagenet-compound"
    mkdir -p "$LASER_LOCAL_SCRATCH"/{wandb/run,wandb/cache,wandb/data,wandb/artifacts,torch/hub/checkpoints,cache}
    export TMPDIR="/mnt/scratch/$USER/laser-tmp/$SLURM_JOB_ID"
    mkdir -p "$TMPDIR"
    chmod 700 "$TMPDIR"
    export WANDB_DIR="$LASER_LOCAL_SCRATCH/wandb/run"
    export WANDB_CACHE_DIR="$LASER_LOCAL_SCRATCH/wandb/cache"
    export WANDB_DATA_DIR="$LASER_LOCAL_SCRATCH/wandb/data"
    export WANDB_ARTIFACT_DIR="$LASER_LOCAL_SCRATCH/wandb/artifacts"
    export TORCH_HOME="$LASER_LOCAL_SCRATCH/torch"
    export PYTHONUSERBASE=/scratch/xl598/.pydeps/laser_stage2_py311
    export PYTHONPATH="$RUN_DIR/source:$RUN_DIR/source/scripts/tools:$RUN_DIR/source/third_party/rq-vae-transformer"
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    export NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
    if [[ "$phase" == cache ]]; then
      nvidia-smi --query-gpu=index,name,memory.total --format=csv
      cp "$RUN_DIR/pytorch_2.4.1.sif" "$LASER_LOCAL_SCRATCH/runtime.sif"
      cp /scratch/xl598/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth "$TORCH_HOME/hub/checkpoints/"
    fi
    container=(singularity exec --nv --bind /scratch/xl598 --bind /mnt/scratch --bind /dev/shm "$LASER_LOCAL_SCRATCH/runtime.sif" "$PYTHON")
    cd "$RUN_DIR/source"
    if [[ "$phase" == cache ]]; then
      if [[ ! -f "$RUN_DIR/cache/ready.json" ]]; then
        "${container[@]}" -m torch.distributed.run \
          --nnodes="$SLURM_JOB_NUM_NODES" --node_rank="$SLURM_PROCID" --nproc_per_node="$GPUS_PER_NODE" \
          --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
          scripts/tools/build_official_imagenet_token_cache.py \
          --checkpoint "$RUN_DIR/inputs/tokenizer.pt" --data /scratch/xl598/Projects/data/imagenet \
          --output "$RUN_DIR/cache/compound-cache.pt" --dataset imagenet --compound \
          --num-atoms 16384 --sparsity-level 4 --coeff-vocab-size 2048 --coeff-max 3 \
          --auto-coeff-scales-percentile 100 --batch-size 16 --num-workers 2 --verify-samples 256
      fi
      if [[ "$SLURM_PROCID" == 0 ]]; then
        "${container[@]}" scripts/tools/launch_imagenet_compound_scratch.py cache-check --base "$RUN_DIR"
      fi
      cp "$RUN_DIR/cache/compound-cache.pt" "$LASER_LOCAL_SCRATCH/cache/compound-cache.pt"
    else
      "${container[@]}" -m torch.distributed.run \
        --nnodes="$SLURM_JOB_NUM_NODES" --node_rank="$SLURM_PROCID" --nproc_per_node="$GPUS_PER_NODE" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        scripts/tools/launch_imagenet_compound_scratch.py "$phase" --base "$RUN_DIR"
    fi
    ;;
  *)
    [[ "$GPUS_PER_NODE" -le 2 ]]
    [[ $((NODES * GPUS_PER_NODE)) == 4 || $((NODES * GPUS_PER_NODE)) == 8 ]]
    test -f "$RUN_DIR/preflight.json"
    exec sbatch "$@" --partition=gpu --job-name=im421-compound \
      --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --cpus-per-task="$CPUS_PER_TASK" \
      --gres="gpu:$GPUS_PER_NODE" --mem="$MEMORY" --time="$TIME_LIMIT" --constraint="$CONSTRAINT" \
      --chdir="$RUN_DIR/source" --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
      --export=ALL "$SELF" --worker
    ;;
esac
