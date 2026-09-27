#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/scratch/xl598/Projects/laser
RUN_DIR=${RUN_DIR:-/scratch/xl598/runs/laser/celebahq256-var-laser-repair-20260917}
SELF="$RUN_DIR/source/scripts/submit_celebahq_var_repair.sh"
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
PARTITION=${PARTITION:-gpu}
CONSTRAINT=${CONSTRAINT:-ampere}
export RUN_DIR GPUS_PER_NODE

case "${1:-}" in
  --worker)
    test -f "$RUN_DIR/ready.json"
    test -f "$RUN_DIR/data-audit.json"
    test -f "$RUN_DIR/cpu-smoke-passed.json"
    exec 9>"$RUN_DIR/allocation.lock"
    flock -n 9
    python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
base = Path(os.environ['RUN_DIR'])
assert json.loads((base/'data-audit.json').read_text())['passed']
assert json.loads((base/'cpu-smoke-passed.json').read_text())['passed']
for name, expected in json.loads((base/'source-manifest.json').read_text()).items():
    assert hashlib.sha256((base/'source'/name).read_bytes()).hexdigest() == expected, name
print('SOURCE_AND_PREFLIGHT_VERIFIED', flush=True)
PY
    export MASTER_ADDR
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
    export LASER_STOP_TIME_UNIX
    LASER_STOP_TIME_UNIX=$(python3 - <<'PY'
import datetime, os, re, subprocess
line = subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'], text=True)
print(int(datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)',line).group(1)).timestamp())-900)
PY
)
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash "$SELF" --node
    # The Python driver requests a safe checkpoint before walltime. Requeue
    # only for that deliberate exit; actual failures remain visible.
    if [ "$(date +%s)" -ge "$LASER_STOP_TIME_UNIX" ] && ! [ -f "$RUN_DIR/train/completed.json" ]; then
      scontrol requeue "$SLURM_JOB_ID"
    fi
    ;;
  --node)
    export LASER_LOCAL_SCRATCH="/mnt/scratch/$USER/laser/celebahq256-var-laser-repair-20260917/job-$SLURM_JOB_ID"
    mkdir -p "$LASER_LOCAL_SCRATCH"/{tmp,wandb/run,wandb/cache,wandb/data,wandb/artifacts,torch/hub/checkpoints,data}
    export TMPDIR="$LASER_LOCAL_SCRATCH/tmp"
    export WANDB_DIR="$LASER_LOCAL_SCRATCH/wandb/run" WANDB_CACHE_DIR="$LASER_LOCAL_SCRATCH/wandb/cache"
    export WANDB_DATA_DIR="$LASER_LOCAL_SCRATCH/wandb/data" WANDB_ARTIFACT_DIR="$LASER_LOCAL_SCRATCH/wandb/artifacts"
    export TORCH_HOME="$LASER_LOCAL_SCRATCH/torch"
    export LASER_SOURCE_ARCHIVE="$RUN_DIR/source.tar.gz"
    export LASER_VGG16_WEIGHTS="$LASER_LOCAL_SCRATCH/vgg16-397923af.pth"
    export PYTHONUSERBASE=/scratch/xl598/.pydeps/laser_stage2_py311
    export PYTHONPATH="$RUN_DIR/source"
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    export NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    cp "$RUN_DIR/runtime.sif" "$LASER_LOCAL_SCRATCH/runtime.sif"
    cp "$RUN_DIR/vgg16-397923af.pth" "$LASER_VGG16_WEIGHTS"
    cp /scratch/xl598/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth "$TORCH_HOME/hub/checkpoints/"
    tar -xf "$RUN_DIR/data.tar" -C "$LASER_LOCAL_SCRATCH/data"
    cd "$RUN_DIR/source"
    container=(singularity exec --nv --bind /scratch/xl598 --bind /mnt/scratch --bind /dev/shm "$LASER_LOCAL_SCRATCH/runtime.sif")
    python_bin="$RUN_DIR/runtime/bin/python"
    "${container[@]}" "$python_bin" - <<'PY'
import os, torch
count = torch.cuda.device_count()
assert count == int(os.environ['GPUS_PER_NODE']), (count, os.environ['GPUS_PER_NODE'])
for i in range(count):
    name = torch.cuda.get_device_name(i)
    assert 'A100' in name or 'L40S' in name, name
    print('VERIFIED_GPU', i, name, flush=True)
PY
    launcher=("${container[@]}" "$python_bin" -m torch.distributed.run
      --nnodes="$SLURM_JOB_NUM_NODES" --node_rank="$SLURM_PROCID" --nproc_per_node="$GPUS_PER_NODE"
      --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
    config="$RUN_DIR/source/configs/experiments/celebahq-var-laser-repair.yaml"
    if ! [ -f "$RUN_DIR/gpu-smoke-passed.json" ]; then
      smoke="$RUN_DIR/smoke-$SLURM_JOB_ID"
      "${launcher[@]}" train.py --config "$config" output_dir="$smoke" data.root="$LASER_LOCAL_SCRATCH/data" \
        wandb.mode=disabled smoke_steps=3 tokenizer.adversarial_start=0 evaluation.preview_samples=32 \
        evaluation.full_samples=32 evaluation.validation_images=16 logging.every_steps=1
      "${launcher[@]}" train.py --config "$config" output_dir="$smoke" data.root="$LASER_LOCAL_SCRATCH/data" \
        wandb.mode=disabled stage=stage2 pipeline=false smoke_steps=4 tokenizer.adversarial_start=0 \
        evaluation.preview_samples=32 evaluation.full_samples=32 evaluation.validation_images=16 logging.every_steps=1
      "${container[@]}" "$python_bin" scripts/tools/verify_celebahq_smoke.py --run "$smoke" --base "$RUN_DIR"
    fi
    "${launcher[@]}" train.py --config "$config" data.root="$LASER_LOCAL_SCRATCH/data"
    ;;
  *)
    test -f "$RUN_DIR/ready.json"
    if [ "$((NODES * GPUS_PER_NODE))" -ne 4 ]; then
      echo 'This fixed global-batch recipe requires exactly four GPUs.' >&2
      exit 2
    fi
    sbatch "$@" --partition="$PARTITION" --constraint="$CONSTRAINT" --job-name=celeba-var-repair \
      --nodes="$NODES" --ntasks-per-node=1 --cpus-per-task="$((GPUS_PER_NODE * 6))" \
      --gres="gpu:$GPUS_PER_NODE" --mem="$((GPUS_PER_NODE * 32))G" --time=72:00:00 --requeue \
      --chdir="$PROJECT_DIR" --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" \
      --export=ALL "$SELF" --worker
    ;;
esac
