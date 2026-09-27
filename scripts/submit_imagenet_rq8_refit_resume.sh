#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/scratch/xl598/runs/laser/imagenet-rfid421-rq8-refit-480m-20260913-amarel}
PROJECT_DIR=/scratch/xl598/Projects/laser
SELF="$PROJECT_DIR/scripts/submit_imagenet_rq8_refit_resume.sh"
API_PYTHON=/scratch/xl598/.venvs/laser-rq8-resume-api/bin/python
NODES=${NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CONSTRAINT=${CONSTRAINT:-ampere|adalovelace}
CPUS_PER_TASK=${CPUS_PER_TASK:-$((6 * GPUS_PER_NODE))}
MEMORY=${MEMORY:-96G}
TIME_LIMIT=${TIME_LIMIT:-72:00:00}
export RUN_DIR GPUS_PER_NODE

case "${1:-}" in
  --worker)
    test -f "$RUN_DIR/ready.json"
    exec 9>"$RUN_DIR/allocation.lock"
    flock -n 9
    export ARTIFACT_SELECTION="$RUN_DIR/artifact-job-${SLURM_JOB_ID}.json"
    "$API_PYTHON" "$RUN_DIR/stage_artifact.py" resolve --base "$RUN_DIR" --selection "$ARTIFACT_SELECTION"
    export MASTER_ADDR
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000))
    export LASER_STOP_TIME_UNIX
    LASER_STOP_TIME_UNIX=$(python - <<'PY'
import os, subprocess, re, datetime
line = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'], text=True)
end = re.search(r'\bEndTime=(\S+)', line).group(1)
print(int(datetime.datetime.fromisoformat(end).timestamp()) - 600)
PY
)
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
      bash "$SELF" --node
    ;;
  --node)
    export LASER_LOCAL_SCRATCH="/mnt/scratch/$USER/laser/imagenet-rfid421-rq8-refit-480m-20260913/job-$SLURM_JOB_ID"
    mkdir -p "$LASER_LOCAL_SCRATCH"/{tmp,wandb/run,wandb/cache,wandb/data,wandb/artifacts,torch/hub/checkpoints}
    # Multiprocessing AF_UNIX sockets must fit Linux's 108-byte path limit.
    export TMPDIR="/mnt/scratch/$USER/laser-tmp/$SLURM_JOB_ID"
    mkdir -p "$TMPDIR"
    chmod 700 "$TMPDIR"
    export WANDB_DIR="$LASER_LOCAL_SCRATCH/wandb/run"
    export WANDB_CACHE_DIR="$LASER_LOCAL_SCRATCH/wandb/cache"
    export WANDB_DATA_DIR="$LASER_LOCAL_SCRATCH/wandb/data"
    export WANDB_ARTIFACT_DIR="$LASER_LOCAL_SCRATCH/wandb/artifacts"
    export TORCH_HOME="$LASER_LOCAL_SCRATCH/torch"
    export PYTHONUSERBASE=/scratch/xl598/.pydeps/laser_stage2_py311
    export LASER_PROJECT_ROOT="$RUN_DIR/source"
    export LASER_AMAREL_RESUME=1
    export PYTHONPATH="$RUN_DIR/source:$RUN_DIR/source/third_party/rq-vae-transformer"
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    export NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    "$API_PYTHON" "$RUN_DIR/stage_artifact.py" stage --base "$RUN_DIR" \
      --selection "$ARTIFACT_SELECTION" --local "$LASER_LOCAL_SCRATCH" --rank "$SLURM_PROCID"
    cp "$RUN_DIR/pytorch_2.4.1.sif" "$LASER_LOCAL_SCRATCH/runtime.sif"
    cp /scratch/xl598/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth \
      "$TORCH_HOME/hub/checkpoints/"
    cd "$RUN_DIR/source"
    exec singularity exec --nv --bind /scratch/xl598 --bind /mnt/scratch --bind /dev/shm \
      "$LASER_LOCAL_SCRATCH/runtime.sif" "$RUN_DIR/runtime/bin/python" -m torch.distributed.run \
      --nnodes="$SLURM_JOB_NUM_NODES" --node_rank="$SLURM_PROCID" \
      --nproc_per_node="$GPUS_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
      "$RUN_DIR/source/scripts/tools/resume_imagenet_rq8.py" \
      --run-id imagenet-rfid421-rq8-refit-480m-20260913-amarel \
      --resume "$LASER_LOCAL_SCRATCH/inputs/last.pt" \
      --tokenizer "$LASER_LOCAL_SCRATCH/inputs/tokenizer.pt" \
      --codebook "$LASER_LOCAL_SCRATCH/inputs/codebook.pt" \
      --manifests "$RUN_DIR/manifests" --reference "$RUN_DIR/imagenet_256_train.npz" \
      --output "$RUN_DIR/train" --data "$RUN_DIR/data" --batch-size 32 --workers 4
    ;;
  *)
    test -f "$RUN_DIR/ready.json"
    sbatch "${@}" --partition=gpu --job-name=rq8-refit-resume \
      --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
      --cpus-per-task="$CPUS_PER_TASK" --gres="gpu:$GPUS_PER_NODE" --mem="$MEMORY" \
      --time="$TIME_LIMIT" --constraint="$CONSTRAINT" --requeue \
      --chdir="$PROJECT_DIR" --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
      --export=ALL "$SELF" --worker
    ;;
esac
