#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${JOB_NAME:-pack-celeba-128}"
PARTITION="${PARTITION:-main}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
PACK_OUT_DIR="${PACK_OUT_DIR:-/scratch/$USER/datasets/celeba_packed_128}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER/.pydeps/celeba_pack_py311}"
LOG_PREFIX="${LOG_PREFIX:-pack_celeba_128}"
FLUSH_EVERY="${FLUSH_EVERY:-512}"
OVERWRITE="${OVERWRITE:-false}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$ROOT_DIR/scripts/pack_celeba_uint8.py" ]]; then
  echo "Missing pack_celeba_uint8.py under $ROOT_DIR/scripts" >&2
  exit 1
fi

sbatch \
  --partition="$PARTITION" \
  --requeue \
  --job-name="$JOB_NAME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM_MB" \
  --time="$TIME_LIMIT" \
  --chdir="$ROOT_DIR" \
  --output="$ROOT_DIR/${LOG_PREFIX}_%j.out" \
  --error="$ROOT_DIR/${LOG_PREFIX}_%j.err" \
  --wrap='set -euo pipefail
USER_NAME=$(id -un)
PROJECT_DIR='"$ROOT_DIR"'
DATA_DIR='"$DATA_DIR"'
PACK_OUT_DIR='"$PACK_OUT_DIR"'
IMAGE='"$IMAGE"'
IMAGE_SIZE='"$IMAGE_SIZE"'
PYTHONUSERBASE_DIR='"$PYTHONUSERBASE_DIR"'
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"
FLUSH_EVERY='"$FLUSH_EVERY"'
OVERWRITE='"$OVERWRITE"'
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    set +u
    source /etc/profile.d/modules.sh
    set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u
    source /usr/share/Modules/init/bash
    set -u
  fi
fi
if ! command -v singularity >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  fi
fi
command -v singularity >/dev/null 2>&1 || { echo singularity_not_found >&2; exit 1; }
mkdir -p "$PACK_OUT_DIR" "$PYTHONUSERBASE_DIR"
export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$PACK_OUT_DIR" "$IMAGE" python3 -c "import numpy, PIL" >/dev/null 2>&1; then
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$PACK_OUT_DIR" "$IMAGE" python3 -m pip install --user numpy pillow
fi
PACK_CMD=(
  python3 "$PROJECT_DIR/scripts/pack_celeba_uint8.py"
  --src_dir "$DATA_DIR"
  --out_dir "$PACK_OUT_DIR"
  --image_size "$IMAGE_SIZE"
  --flush_every "$FLUSH_EVERY"
)
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" || "$OVERWRITE" == "yes" || "$OVERWRITE" == "YES" ]]; then
  PACK_CMD+=(--overwrite)
fi
singularity exec \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER_NAME" \
  --bind "$DATA_DIR" \
  --bind "$PACK_OUT_DIR" \
  "$IMAGE" \
  "${PACK_CMD[@]}"'
