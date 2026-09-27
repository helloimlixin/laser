#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/scratch/xl598/runs/laser/church-original-rqvae-released-tokenizer-control-20260917}
PROJECT_DIR=/scratch/xl598/Projects/laser
SELF="$PROJECT_DIR/scripts/submit_church_released_control_resume.sh"
NODES=${NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CONSTRAINT=${CONSTRAINT:-ampere}
TIME_LIMIT=${TIME_LIMIT:-72:00:00}
export RUN_DIR GPUS_PER_NODE
export PYTHONUSERBASE=/scratch/xl598/.pydeps/laser_stage2_py311
PYTHON="$RUN_DIR/runtime-env/bin/python"
IMAGE="$RUN_DIR/pytorch_2.4.1.sif"

case "${1:-}" in
  --worker)
    test -f "$RUN_DIR/ready.json"
    exec 9>"$RUN_DIR/allocation.lock"
    flock -n 9
    export CHURCH_SELECTION="$RUN_DIR/artifact-job-${SLURM_JOB_ID}.json"
    export CHURCH_LOCAL_ROOT="/mnt/scratch/$USER/laser/church-released-control/job-$SLURM_JOB_ID"
    mkdir -p "$CHURCH_LOCAL_ROOT/wandb/cache"
    export WANDB_CACHE_DIR="$CHURCH_LOCAL_ROOT/wandb/cache"
    singularity exec --bind /scratch/xl598 --bind /mnt/scratch "$IMAGE" "$PYTHON" \
      "$RUN_DIR/source/stage_church_control_resume.py" resolve --selection "$CHURCH_SELECTION"
    export MASTER_ADDR
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=$((21000 + SLURM_JOB_ID % 20000))
    export CHURCH_STOP_TIME
    CHURCH_STOP_TIME=$(python3 - <<'PY'
import datetime, os, re, subprocess
line = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'], text=True)
end = re.search(r'\bEndTime=(\S+)', line).group(1)
print(int(datetime.datetime.fromisoformat(end).timestamp()) - 1200)
PY
)
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
      bash "$SELF" --node prepare
    export MASTER_PORT=$((MASTER_PORT + 1))
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
      bash "$SELF" --node train
    ;;
  --node)
    phase="$2"
    mkdir -p "$CHURCH_LOCAL_ROOT"/{tmp,wandb/run,wandb/cache,wandb/data,wandb/artifacts,torch/hub/checkpoints}
    export TMPDIR="$CHURCH_LOCAL_ROOT/tmp"
    export WANDB_DIR="$CHURCH_LOCAL_ROOT/wandb/run"
    export WANDB_CACHE_DIR="$CHURCH_LOCAL_ROOT/wandb/cache"
    export WANDB_DATA_DIR="$CHURCH_LOCAL_ROOT/wandb/data"
    export WANDB_ARTIFACT_DIR="$CHURCH_LOCAL_ROOT/wandb/artifacts"
    export TORCH_HOME="$CHURCH_LOCAL_ROOT/torch"
    export PYTHONPATH="$RUN_DIR/source:$RUN_DIR/source/upstream"
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    export NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1
    if [[ "$phase" == prepare ]]; then
      nvidia-smi --query-gpu=index,name,memory.total --format=csv
      cp "$IMAGE" "$CHURCH_LOCAL_ROOT/runtime.sif"
      cp /scratch/xl598/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth \
        "$TORCH_HOME/hub/checkpoints/"
      singularity exec --bind /scratch/xl598 --bind /mnt/scratch "$CHURCH_LOCAL_ROOT/runtime.sif" "$PYTHON" \
        "$RUN_DIR/source/stage_church_control_resume.py" stage --selection "$CHURCH_SELECTION" \
        --local "$CHURCH_LOCAL_ROOT" --node-rank "$SLURM_PROCID"
      program=("$RUN_DIR/source/prepare_church_control_cache.py" --base "$RUN_DIR" --batch-size 16)
    else
      program=("$RUN_DIR/source/resume_church_released_control.py" --run-root "$RUN_DIR" \
        --output "$RUN_DIR/train" --run-id church-original-rqvae-released-tokenizer-control-20260917 \
        --resume "$CHURCH_LOCAL_ROOT/inputs/last.pt" --batch-size 32 --memory-fraction .95)
    fi
    cd "$RUN_DIR/source"
    exec singularity exec --nv --bind /scratch/xl598 --bind /mnt/scratch --bind /dev/shm \
      "$CHURCH_LOCAL_ROOT/runtime.sif" "$PYTHON" -m torch.distributed.run \
      --nnodes="$SLURM_JOB_NUM_NODES" --node_rank="$SLURM_PROCID" \
      --nproc_per_node="$GPUS_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
      "${program[@]}"
    ;;
  *)
    test -f "$RUN_DIR/ready.json"
    if [[ "$GPUS_PER_NODE" -gt 2 ]] || [[ $((NODES * GPUS_PER_NODE)) != 8 && $((NODES * GPUS_PER_NODE)) != 16 ]]; then
      echo 'Expected 8 or 16 GPUs, with at most 2 GPUs per node' >&2
      exit 1
    fi
    if [[ -n "$(squeue -h -u "$USER" --name=church-rq-resume -o %i)" ]]; then
      echo 'A Church resume allocation already exists; inspect it first' >&2
      exit 1
    fi
    job_id=$(sbatch --parsable "$@" --partition=gpu --job-name=church-rq-resume \
      --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --cpus-per-task=8 \
      --gres="gpu:$GPUS_PER_NODE" --mem=64000 --time="$TIME_LIMIT" --constraint="$CONSTRAINT" \
      --requeue --chdir="$PROJECT_DIR" --output="$RUN_DIR/slurm-%j.out" \
      --error="$RUN_DIR/slurm-%j.err" --export=ALL "$SELF" --worker)
    python3 - "$RUN_DIR" "$job_id" "$NODES" "$GPUS_PER_NODE" "$CONSTRAINT" <<'PY'
import datetime, json, pathlib, sys
base, job, nodes, gpn, constraint = sys.argv[1:]
record = dict(job_id=job.split(';')[0], nodes=int(nodes), gpus_per_node=int(gpn),
              constraint=constraint, submitted_at=datetime.datetime.now().isoformat(),
              run_id='church-original-rqvae-released-tokenizer-control-20260917')
path = pathlib.Path(base) / 'launch.json'
if path.exists():
    previous = json.loads(path.read_text())
    (path.parent / ('launch-' + previous['job_id'] + '.json')).write_text(path.read_text())
path.with_suffix('.tmp').write_text(json.dumps(record, indent=2) + '\n')
path.with_suffix('.tmp').replace(path)
print(json.dumps(record))
PY
    ;;
esac
