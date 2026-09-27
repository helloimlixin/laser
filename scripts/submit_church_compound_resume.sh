#!/usr/bin/env bash
set -euo pipefail
umask 077

CHURCH_BASE=${CHURCH_BASE:-/home/xl598/laser-church-compound-20260919}
PROJECT_DIR=/scratch/xl598/Projects/laser
SELF="$CHURCH_BASE/runtime/submit_church_compound_resume.sh"
IMAGE="$CHURCH_BASE/pytorch_2.4.1.sif"
PYTHON="$CHURCH_BASE/runtime-env/bin/python"
export CHURCH_BASE
export PYTHONUSERBASE="$CHURCH_BASE/pydeps"
export PYTHONPATH="$CHURCH_BASE/runtime:$CHURCH_BASE/source"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1

container_python() {
  singularity exec --bind /home/xl598 --bind /mnt/scratch "$IMAGE" "$PYTHON" "$@"
}
local_environment() {
  mkdir -p "$CHURCH_LOCAL_ROOT"/{tmp,cache,train,checkpoints,wandb/run,wandb/cache,wandb/data,wandb/artifacts,torch/hub/checkpoints}
  export TMPDIR="$CHURCH_LOCAL_ROOT/tmp"
  export WANDB_DIR="$CHURCH_LOCAL_ROOT/wandb/run"
  export WANDB_CACHE_DIR="$CHURCH_LOCAL_ROOT/wandb/cache"
  export WANDB_DATA_DIR="$CHURCH_LOCAL_ROOT/wandb/data"
  export WANDB_ARTIFACT_DIR="$CHURCH_LOCAL_ROOT/wandb/artifacts"
  export TORCH_HOME="$CHURCH_LOCAL_ROOT/torch"
}

case "${1:-}" in
  --cache-worker)
    exec 9>"$CHURCH_BASE/cache.lock"
    flock -n 9
    export CHURCH_LOCAL_ROOT="/mnt/scratch/$USER/laser/church-compound/cache-$SLURM_JOB_ID"
    local_environment
    cd "$CHURCH_BASE"
    sha256sum --quiet -c runtime-sha256.txt
    if [[ -f "$CHURCH_BASE/prepared/cache-ready.json" ]]; then
      container_python "$CHURCH_BASE/runtime/stage_church_compound_resume.py" cache-check --base "$CHURCH_BASE"
      exit 0
    fi
    PYTHONPATH="$PYTHONPATH:$CHURCH_BASE/test-deps" \
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 singularity exec --nv --bind /home/xl598 --bind /scratch/xl598 --bind /mnt/scratch \
      "$IMAGE" "$PYTHON" -m pytest -q \
      "$CHURCH_BASE/source/tests/test_compound_pair_autoregressive.py" \
      "$CHURCH_BASE/source/tests/test_compound_v3_objective.py" \
      > "$CHURCH_BASE/cache-gpu-tests-$SLURM_JOB_ID.log" 2>&1
    singularity exec --nv --bind /home/xl598 --bind /scratch/xl598 --bind /mnt/scratch "$IMAGE" "$PYTHON" \
      -m torch.distributed.run --standalone --nproc_per_node=1 \
      "$CHURCH_BASE/runtime/build_church_compound_cache.py" --base "$CHURCH_BASE"
    container_python "$CHURCH_BASE/runtime/stage_church_compound_resume.py" cache-finish --base "$CHURCH_BASE"
    ;;
  --train-worker)
    export CHURCH_LOCAL_ROOT="/mnt/scratch/$USER/laser/church-compound/job-$SLURM_JOB_ID"
    local_environment
    TMPDIR=/tmp srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
      bash -c 'PYTHONPATH="" PYTHONUSERBASE=/nonexistent python3 -s -c "$CHURCH_BOOTSTRAP_CODE"'
    export CHURCH_BASE="$CHURCH_LOCAL_ROOT/bundle"
    export PYTHONUSERBASE="$CHURCH_BASE/pydeps" PYTHONPATH="$CHURCH_BASE/runtime:$CHURCH_BASE/source"
    IMAGE="$CHURCH_BASE/pytorch_2.4.1.sif"
    PYTHON="$CHURCH_BASE/runtime-env/bin/python"
    mkdir -p "$CHURCH_BASE/train"
    exec 9>"$CHURCH_LOCAL_ROOT/train.lock"
    flock -n 9
    export CHURCH_SELECTION="$CHURCH_LOCAL_ROOT/selection.json"
    SELF="$CHURCH_LOCAL_ROOT/bundle/runtime/submit_church_compound_resume.sh"
    cd "$CHURCH_LOCAL_ROOT/bundle"
    sha256sum --quiet -c runtime-sha256.txt
    container_python "$CHURCH_BASE/runtime/stage_church_compound_resume.py" cache-check --base "$CHURCH_BASE"
    container_python "$CHURCH_BASE/runtime/stage_church_compound_resume.py" resolve \
      --base "$CHURCH_BASE" --selection "$CHURCH_SELECTION"
    export CHURCH_SELECTION_JSON
    CHURCH_SELECTION_JSON=$(cat "$CHURCH_SELECTION")
    export MASTER_ADDR
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=$((21000 + SLURM_JOB_ID % 20000))
    export CHURCH_STOP_TIME
    CHURCH_STOP_TIME=$(python3 - <<'PY'
import datetime, os, re, subprocess
line = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'], text=True)
end = re.search(r'\bEndTime=(\S+)', line).group(1)
print(int(datetime.datetime.fromisoformat(end).timestamp()) - 2700)
PY
)
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash "$SELF" --node prepare
    srun --kill-on-bad-exit=1 --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash "$SELF" --node train
    container_python "$CHURCH_BASE/runtime/stage_church_compound_resume.py" finish \
      --base "$CHURCH_BASE" --local "$CHURCH_LOCAL_ROOT"
    complete=$(python3 - "$CHURCH_BASE/train/completion-$SLURM_JOB_ID.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['complete']))
PY
)
    if [[ "$complete" != 1 ]]; then
      export CHURCH_MIN_STEP
      CHURCH_MIN_STEP=$(python3 - "$CHURCH_BASE/train/completion-$SLURM_JOB_ID.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['step'])
PY
)
      next_job=$(CHURCH_HANDOFF_FROM="$SLURM_JOB_ID" NODES="$SLURM_JOB_NUM_NODES" \
        TIME_LIMIT=72:00:00 TIME_MIN=01:30:00 MEMORY="${MEMORY:-$((32000 * GPUS_PER_NODE))}" CONSTRAINT='ampere|adalovelace' \
        bash "$SELF" train --dependency="afterok:$SLURM_JOB_ID" --kill-on-invalid-dep=yes)
      python3 - "$CHURCH_BASE" "$SLURM_JOB_ID" "$next_job" <<'PY'
import datetime,json,pathlib,sys
base,previous,following=sys.argv[1:];base=pathlib.Path(base)
record=dict(previous_job=previous,next_job=following.split(';')[0],
            submitted_at=datetime.datetime.now().isoformat(),reason='continue to epoch 90 after verified online checkpoint upload')
(base/'train'/f'handoff-{previous}.json').write_text(json.dumps(record,indent=2)+'\n')
launch=json.loads((base/'launch.json').read_text()) if (base/'launch.json').exists() else {}
launch.update(train_job=record['next_job'],previous_job=previous)
(base/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
    fi
    ;;
  --node)
    local_environment
    export NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [[ "$2" == prepare ]]; then
      nvidia-smi --query-gpu=index,name,memory.total --format=csv
      ln -sf "$CHURCH_LOCAL_ROOT/bundle/pytorch_2.4.1.sif" "$CHURCH_LOCAL_ROOT/runtime.sif"
      cp "$CHURCH_LOCAL_ROOT/bundle/prepared/compound-cache.pt" "$CHURCH_LOCAL_ROOT/cache/compound-cache.pt"
      cp "$CHURCH_LOCAL_ROOT/bundle/reference/weights-inception-2015-12-05-6726825d.pth" "$TORCH_HOME/hub/checkpoints/"
      printf '%s\n' "$CHURCH_SELECTION_JSON" > "$CHURCH_SELECTION"
      container_python "$CHURCH_BASE/runtime/stage_church_compound_resume.py" stage --base "$CHURCH_BASE" \
        --selection "$CHURCH_SELECTION" --local "$CHURCH_LOCAL_ROOT"
    else
      cd "$CHURCH_LOCAL_ROOT/bundle/source"
      exec singularity exec --nv --bind /home/xl598 --bind /scratch/xl598 --bind /mnt/scratch --bind /dev/shm \
        --bind "$CHURCH_LOCAL_ROOT:/workspace" "$CHURCH_LOCAL_ROOT/runtime.sif" "$PYTHON" \
        -m torch.distributed.run --nnodes="$SLURM_JOB_NUM_NODES" --node_rank="$SLURM_PROCID" \
        --nproc_per_node="$GPUS_PER_NODE" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        "$CHURCH_BASE/runtime/resume_church_compound.py" --base "$CHURCH_BASE"
    fi
    ;;
  cache|train)
    phase="$1"
    shift
    test -f "$CHURCH_BASE/preflight.json"
    test -f "$CHURCH_BASE/runtime-sha256.txt"
    if [[ "$phase" == train ]]; then
      : "${CHURCH_BUNDLE_FILE:?Missing pinned runtime file}" "${CHURCH_BUNDLE_SHA256:?Missing runtime digest}"
      if [[ -n "${CHURCH_LOCAL_ROOT:-}" && -f "$CHURCH_LOCAL_ROOT/bundle/runtime/submit_church_compound_resume.sh" ]]; then
        SELF="$CHURCH_LOCAL_ROOT/bundle/runtime/submit_church_compound_resume.sh"
      fi
      if [[ -z "${CHURCH_BOOTSTRAP_CODE:-}" ]]; then
        CHURCH_BOOTSTRAP_CODE=$(cat "$CHURCH_BASE/runtime/bootstrap_church_compound.py")
      fi
      export CHURCH_BOOTSTRAP_CODE CHURCH_BUNDLE_FILE CHURCH_BUNDLE_SHA256
    fi
    active_jobs=$(squeue -h -u "$USER" --name="church-cmp-$phase" -o %i)
    if [[ -n "${CHURCH_HANDOFF_FROM:-}" && "$active_jobs" == "$CHURCH_HANDOFF_FROM" ]]; then
      active_jobs=''
    fi
    if [[ -n "$active_jobs" ]]; then
      echo "A Church compound $phase job already exists" >&2
      exit 1
    fi
    if [[ "$phase" == cache ]]; then
      NODES=1
      GPUS_PER_NODE=1
      CPUS=4
      MEMORY=10000
      TIME_LIMIT=${TIME_LIMIT:-04:00:00}
    else
      NODES=${NODES:-2}
      GPUS_PER_NODE=${GPUS_PER_NODE:-2}
      CPUS=$((4 * GPUS_PER_NODE))
      MEMORY=${MEMORY:-$((32000 * GPUS_PER_NODE))}
      if (( MEMORY < 32000 * GPUS_PER_NODE )); then
        echo 'Church continuation requires at least 32000 MB of host RAM per GPU for generation and checkpoint uploads' >&2
        exit 1
      fi
      TIME_LIMIT=${TIME_LIMIT:-72:00:00}
      [[ ( $((NODES * GPUS_PER_NODE)) == 4 || $((NODES * GPUS_PER_NODE)) == 8 ) && "$GPUS_PER_NODE" -le 2 ]]
    fi
    export GPUS_PER_NODE
    time_options=()
    if [[ "$phase" == train ]]; then
      time_options+=("--time-min=${TIME_MIN:-01:30:00}")
    fi
    sbatch --parsable "$@" --partition=gpu --job-name="church-cmp-$phase" \
      --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --cpus-per-task="$CPUS" \
      --gres="gpu:$GPUS_PER_NODE" --mem="$MEMORY" --time="$TIME_LIMIT" "${time_options[@]}" --constraint="${CONSTRAINT:-adalovelace}" \
      --no-requeue --chdir=/tmp --output="/tmp/church-cmp-$phase-%j.out" \
      --error="/tmp/church-cmp-$phase-%j.err" --export=ALL "$SELF" "--$phase-worker"
    ;;
  *) echo "Usage: $0 cache|train [sbatch options]" >&2; exit 2 ;;
esac
