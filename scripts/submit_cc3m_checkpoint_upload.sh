#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/Projects/laser}"
SELF="${SELF:-$PROJECT_DIR/scripts/submit_cc3m_checkpoint_upload.sh}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/runs/laser/cc3m-official-rqt650m-compound-a16384-k2-20260808_124922}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$(cat "$RUN_ROOT/stage2/checkpoint_uploads/latest_snapshot_dir.txt")}"
CHECKPOINT="${CHECKPOINT:-$SNAPSHOT_DIR/last.pt}"
SOURCE_RUN="${SOURCE_RUN:-helloimlixin-rutgers/laser/cc3mcmp0808124922}"
ARTIFACT_NAME="${ARTIFACT_NAME:-cc3mcmp0808124922-checkpoint}"
UPLOADER_RUN_ID="${UPLOADER_RUN_ID:-cc3mcmp0808124922-ckpt-s65500}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/$UPLOADER_RUN_ID}"

submit_job() {
  [[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }
  mkdir -p "$SNAPSHOT_DIR"
  args=(
    --partition=main-redhat --job-name=cc3m-ckpt-up
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16000
    --time=12:00:00 --chdir="$PROJECT_DIR"
    --output="$SNAPSHOT_DIR/slurm-%j.out"
    --error="$SNAPSHOT_DIR/slurm-%j.err"
  )
  if [[ "${TEST_ONLY:-0}" == 1 ]]; then
    sbatch --test-only "${args[@]}" "$SELF" --worker
  else
    sbatch "${args[@]}" "$SELF" --worker
  fi
}

run_worker() {
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
  mkdir -p \
    "$LOCAL_SCRATCH_ROOT/wandb/run" \
    "$LOCAL_SCRATCH_ROOT/wandb/cache" \
    "$LOCAL_SCRATCH_ROOT/wandb/data" \
    "$LOCAL_SCRATCH_ROOT/wandb/artifacts" \
    "$LOCAL_SCRATCH_ROOT/cache"
  export APPTAINERENV_PYTHONUSERBASE="$PYDEPS"
  export APPTAINERENV_PYTHONPATH="$PROJECT_DIR"
  export APPTAINERENV_WANDB_MODE=online
  export APPTAINERENV_WANDB_DIR="$LOCAL_SCRATCH_ROOT/wandb/run"
  export APPTAINERENV_WANDB_CACHE_DIR="$LOCAL_SCRATCH_ROOT/wandb/cache"
  export APPTAINERENV_WANDB_DATA_DIR="$LOCAL_SCRATCH_ROOT/wandb/data"
  export APPTAINERENV_WANDB_ARTIFACT_DIR="$LOCAL_SCRATCH_ROOT/wandb/artifacts"
  export APPTAINERENV_XDG_CACHE_HOME="$LOCAL_SCRATCH_ROOT/cache"

  singularity exec --bind /scratch --bind /mnt/scratch \
    "$IMAGE" \
    python "$PROJECT_DIR/scripts/tools/upload_wandb_checkpoint_artifact.py" \
      --checkpoint "$CHECKPOINT" \
      --source-run "$SOURCE_RUN" \
      --artifact-name "$ARTIFACT_NAME" \
      --uploader-run-id "$UPLOADER_RUN_ID"
}

case "${1:-submit}" in
  submit) submit_job ;;
  --worker) run_worker ;;
  *) echo "usage: $SELF [submit|--worker]" >&2; exit 2 ;;
esac
