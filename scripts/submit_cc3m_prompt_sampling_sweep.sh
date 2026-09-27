#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/Projects/laser}"
SELF="${SELF:-$PROJECT_DIR/scripts/submit_cc3m_prompt_sampling_sweep.sh}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/runs/laser/cc3m-official-rqt650m-compound-a16384-k2-20260808_124922}"
SWEEP_DIR="${SWEEP_DIR:-$RUN_ROOT/stage2/sampling_sweeps/cc3m_prompt_sweep_step_0063000_20260814_1228}"
STAGE1="${STAGE1:-/scratch/$USER/runs/imagenet_x3h5cl0h_strict_bottleneck_sweep/imagenet-x3h5cl0h-strict-bottleneck-sweep-20260719_014434/k2-a16384/in256-rqvae-laser-8x8-a16384-k2/19072026_024726/best_rfid_slot3_model.pt}"
STAGE2="${STAGE2:-$SWEEP_DIR/source_checkpoint/last_step_0063000.pt}"
SOURCE_RUN="${SOURCE_RUN:-helloimlixin-rutgers/laser/cc3mcmp0808124922}"
PARTITION="${PARTITION:-cgpu-redhat}"
CONSTRAINT="${CONSTRAINT:-ampere}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
MEM_MB="${MEM_MB:-48000}"
CPUS="${CPUS:-4}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
WANDB_ID="${WANDB_ID:-cc3mcmp0808124922-sweep63000}"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/cc3mcmp0808124922-sweep63000}"

submit_job() {
  [[ -f "$STAGE1" ]] || { echo "missing stage-1 checkpoint: $STAGE1" >&2; exit 1; }
  [[ -f "$STAGE2" ]] || { echo "missing pinned stage-2 checkpoint: $STAGE2" >&2; exit 1; }
  mkdir -p "$SWEEP_DIR"
  args=(
    --partition="$PARTITION" --constraint="$CONSTRAINT"
    --job-name=cc3m-prsweep --nodes=1 --ntasks=1
    --cpus-per-task="$CPUS" --gres=gpu:1 --mem="$MEM_MB"
    --time="$TIME_LIMIT" --chdir="$PROJECT_DIR"
    --output="$SWEEP_DIR/slurm-%j.out" --error="$SWEEP_DIR/slurm-%j.err"
  )
  if [[ "${TEST_ONLY:-0}" == 1 ]]; then
    sbatch --test-only "${args[@]}" "$SELF" --worker
  else
    sbatch "${args[@]}" "$SELF" --worker
  fi
}

load_container_runtime() {
  if ! command -v module >/dev/null 2>&1; then
    set +u
    [[ -f /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash
    set -u
  fi
  command -v singularity >/dev/null 2>&1 || module load singularity 2>/dev/null || true
  command -v singularity >/dev/null 2>&1 || {
    echo "singularity is unavailable" >&2
    exit 1
  }
}

run_worker() {
  load_container_runtime
  mkdir -p "$SWEEP_DIR" "$LOCAL_SCRATCH_ROOT"
  export APPTAINERENV_PYTHONUSERBASE="$PYDEPS"
  export APPTAINERENV_PYTHONPATH="$PROJECT_DIR"
  export APPTAINERENV_PROJECT_DIR="$PROJECT_DIR"
  export APPTAINERENV_WANDB_DIR="$LOCAL_SCRATCH_ROOT/wandb/run"
  export APPTAINERENV_WANDB_CACHE_DIR="$LOCAL_SCRATCH_ROOT/wandb/cache"
  export APPTAINERENV_WANDB_DATA_DIR="$LOCAL_SCRATCH_ROOT/wandb/data"
  export APPTAINERENV_WANDB_ARTIFACT_DIR="$LOCAL_SCRATCH_ROOT/wandb/artifacts"
  export APPTAINERENV_XDG_CACHE_HOME="$LOCAL_SCRATCH_ROOT/cache"
  export APPTAINERENV_MPLCONFIGDIR="$LOCAL_SCRATCH_ROOT/cache/matplotlib"
  export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export APPTAINERENV_OMP_NUM_THREADS="$CPUS"
  mkdir -p \
    "$LOCAL_SCRATCH_ROOT/wandb/run" \
    "$LOCAL_SCRATCH_ROOT/wandb/cache" \
    "$LOCAL_SCRATCH_ROOT/wandb/data" \
    "$LOCAL_SCRATCH_ROOT/wandb/artifacts" \
    "$LOCAL_SCRATCH_ROOT/cache/matplotlib"

  singularity exec --nv \
    --bind /scratch --bind /mnt/scratch --bind /dev/shm \
    "$IMAGE" \
    python "$PROJECT_DIR/scripts/sample_cc3m_prompt_sweep.py" \
      --stage1 "$STAGE1" \
      --stage2 "$STAGE2" \
      --output "$SWEEP_DIR" \
      --source-run "$SOURCE_RUN" \
      --samples-per-prompt 8 \
      --expected-epoch 44 \
      --expected-step 63000 \
      --wandb-id "$WANDB_ID"
}

case "${1:-submit}" in
  submit) submit_job ;;
  --worker) run_worker ;;
  *) echo "usage: $SELF [submit|--worker]" >&2; exit 2 ;;
esac
