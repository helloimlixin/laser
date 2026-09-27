#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/Projects/laser}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-60297070}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser/cc3m-official-rqt650m-compound-a16384-k2-20260808_124922}"
SAMPLE_DIR="$RUN_DIR/stage2/samples"
TOKEN_CACHE="${TOKEN_CACHE:-$PROJECT_DIR/outputs/cc3m_x3h5cl0h_a16384k2_text2image/token_cache/cc3m_train_imagenet_x3h5cl0h_a16384k2_q128_rq_bpe16k_text32.pt}"
WANDB_RUN="${WANDB_RUN:-helloimlixin-rutgers/laser/cc3mcmp0808124922}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
HELPER="$PROJECT_DIR/scripts/tools/reformat_legacy_cc3m_text_grid.py"
WATCH_NODELIST="${WATCH_NODELIST:-hal0273}"

if [[ "${1:-}" != "--worker" ]]; then
  mkdir -p "$RUN_DIR"
  sbatch \
    --partition=main-redhat \
    --nodelist="$WATCH_NODELIST" \
    --job-name=cc3m-grid17 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
    --time=72:00:00 \
    --chdir="$PROJECT_DIR" \
    --output="$RUN_DIR/figure17-watcher-%j.out" \
    --error="$RUN_DIR/figure17-watcher-%j.err" \
    "$0" --worker
  exit 0
fi

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

processed=""
while true; do
  latest="$({ find "$SAMPLE_DIR" -maxdepth 1 -type f -name 'text_step_*.png' \
    -printf '%T@ %p\n' 2>/dev/null || true; } | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -n "$latest" && "$latest" != "$processed" ]]; then
    initial_size="$(stat -c %s "$latest")"
    sleep 10
    stable_size="$(stat -c %s "$latest")"
    if [[ "$initial_size" == "$stable_size" ]]; then
      export APPTAINERENV_PYTHONUSERBASE="$PYDEPS"
      export APPTAINERENV_PYTHONPATH="$PROJECT_DIR"
      export APPTAINERENV_MPLCONFIGDIR="/tmp/matplotlib-figure17"
      export APPTAINERENV_WANDB_DIR="/tmp/wandb-figure17"
      singularity exec --bind /scratch "$IMAGE" \
        python "$HELPER" \
          --source "$latest" \
          --token-cache "$TOKEN_CACHE" \
          --wandb-run "$WANDB_RUN" \
          --upload-wandb
      processed="$latest"
    fi
  fi

  train_state="$(squeue -h -j "$TRAIN_JOB_ID" -o '%T' 2>/dev/null || true)"
  if [[ -z "$train_state" ]]; then
    echo "training job $TRAIN_JOB_ID is no longer queued; watcher complete"
    exit 0
  fi
  sleep 50
done
