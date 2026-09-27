#!/usr/bin/env bash
set -euo pipefail

export CC3M_BASE=${CC3M_BASE:-/scratch/xl598/runs/laser/cc3m-imagenet421-compound-650m-100ep-20260919}
PROJECT=/scratch/xl598/Projects/laser
SELF="$CC3M_BASE/runtime/submit_cc3m_imagenet421_compound.sh"
IMAGE=/scratch/xl598/runs/laser/church-original-rqvae-released-tokenizer-control-20260917/pytorch_2.4.1.sif
PYTHON=/scratch/xl598/runs/laser/church-laser-ft3ep-scratch-adaptive-lr-20260916-amarel/runtime-env/bin/python
export PYTHONUSERBASE=/scratch/xl598/.pydeps/laser_stage2_py311
export PYTHONPATH="$CC3M_BASE/source:$CC3M_BASE/source/scripts"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=online
export NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

environment() {
  export CC3M_LOCAL="/mnt/scratch/$USER/laser/cc3m421-compound-${SLURM_JOB_ID:-preflight}"
  mkdir -p "$CC3M_LOCAL"/{tmp,cache,checkpoints,wandb/run,wandb/cache,wandb/data,wandb/artifacts,clip,torch/hub/checkpoints}
  export TMPDIR="$CC3M_LOCAL/tmp"
  export WANDB_DIR="$CC3M_LOCAL/wandb/run"
  export WANDB_CACHE_DIR="$CC3M_LOCAL/wandb/cache"
  export WANDB_DATA_DIR="$CC3M_LOCAL/wandb/data"
  export WANDB_ARTIFACT_DIR="$CC3M_LOCAL/wandb/artifacts"
  export TORCH_HOME="$CC3M_LOCAL/torch"
  export MPLCONFIGDIR="$CC3M_LOCAL/tmp/matplotlib"
}

python_container() {
  singularity exec --bind /scratch/xl598 --bind /mnt/scratch "$IMAGE" "$PYTHON" "$@"
}

case "${1:-submit}" in
  submit)
    NODES=${NODES:-4}; GPUS_PER_NODE=${GPUS_PER_NODE:-2}
    WORLD=$((NODES * GPUS_PER_NODE))
    [[ "$GPUS_PER_NODE" -le 3 && "$WORLD" -ge 4 && $((2048 % (16*WORLD))) -eq 0 ]]
    export GPUS_PER_NODE
    test -f "$CC3M_BASE/runtime-sha256.txt"
    if [[ -n "$(squeue -h -u "$USER" --name=cc3m421-cmp -o %i)" ]]; then
      echo 'This compound reproduction already has a Slurm job' >&2; exit 1
    fi
    sbatch --parsable --partition=gpu --constraint="${CONSTRAINT:-adalovelace}" \
      --job-name=cc3m421-cmp --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
      --cpus-per-task=8 --gres="gpu:$GPUS_PER_NODE" --mem=96000 --time=72:00:00 \
      --requeue --signal=B:TERM@1200 --chdir="$CC3M_BASE" \
      --output="$CC3M_BASE/slurm-%j.out" --error="$CC3M_BASE/slurm-%j.err" \
      --export=ALL "$SELF" --worker
    ;;
  --worker)
    exec 9>"$CC3M_BASE/allocation.lock"
    flock -n 9
    cd "$CC3M_BASE"
    sha256sum --quiet -c runtime-sha256.txt
    export MASTER_ADDR MASTER_PORT CC3M_STOP_TIME
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
    CC3M_STOP_TIME=$(python3 - <<'PY'
import datetime, os, re, subprocess
line=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],text=True)
print(int(datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)',line).group(1)).timestamp())-1800)
PY
)
    # The driver stops at a committed optimizer boundary before this signal.
    trap ':' TERM
    environment
    rm -f "$CC3M_BASE/requeue.json"
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash "$SELF" --node prepare
    if [[ ! -f "$CC3M_BASE/gpu-preflight.json" ]]; then
      srun --kill-on-bad-exit=1 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash "$SELF" --node preflight
    fi
    if [[ ! -f "$CC3M_BASE/cache/ready.json" ]]; then
      srun --kill-on-bad-exit=1 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash "$SELF" --node cache
    fi
    if [[ -f "$CC3M_BASE/requeue.json" ]]; then
      flock -u 9
      scontrol requeue "$SLURM_JOB_ID"
      exit 0
    fi
    python_container "$CC3M_BASE/runtime/train_cc3m_compound.py" resolve
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash "$SELF" --node stage
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash "$SELF" --node train
    if [[ -f "$CC3M_BASE/requeue.json" && ! -f "$CC3M_BASE/complete.json" ]]; then
      flock -u 9
      scontrol requeue "$SLURM_JOB_ID"
    fi
    ;;
  --node)
    environment
    if [[ "$2" == prepare ]]; then
      nvidia-smi --query-gpu=index,name,memory.total --format=csv
      cp "$IMAGE" "$CC3M_LOCAL/runtime.sif"
      cp /scratch/xl598/.cache/clip/ViT-B-32.pt "$CC3M_LOCAL/clip/"
      cp /scratch/xl598/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth "$TORCH_HOME/hub/checkpoints/"
    elif [[ "$2" == stage ]]; then
      cp -au "$CC3M_BASE/cache/." "$CC3M_LOCAL/cache/"
      python_container "$CC3M_BASE/runtime/train_cc3m_compound.py" stage
    else
      cd "$CC3M_BASE/source"
      exec singularity exec --nv --bind /scratch/xl598 --bind /mnt/scratch --bind /dev/shm \
        "$CC3M_LOCAL/runtime.sif" "$PYTHON" -m torch.distributed.run \
        --nnodes="$SLURM_NNODES" --node_rank="$SLURM_PROCID" --nproc_per_node="$GPUS_PER_NODE" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        "$CC3M_BASE/runtime/train_cc3m_compound.py" "$2" --batch 16
    fi
    ;;
  *) echo 'Usage: submit_cc3m_imagenet421_compound.sh [submit|--worker|--node PHASE]' >&2; exit 2 ;;
esac
