#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${USER:-$(id -un)}"

IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER_NAME/.pydeps/laser_py311}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER_NAME/.secrets/wandb_api_key}"
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"

load_singularity() {
  if command -v singularity >/dev/null 2>&1; then
    return 0
  fi
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
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  fi
  command -v singularity >/dev/null 2>&1 || {
    echo "singularity_not_found" >&2
    exit 1
  }
}

ensure_py_deps() {
  local run_root="$1"
  mkdir -p "$run_root" "$PYTHONUSERBASE_DIR" "$(dirname "$WANDB_API_KEY_FILE")"
  load_singularity

  if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" \
    singularity exec \
      --nv \
      --bind "$ROOT_DIR" \
      --bind "/scratch/$USER_NAME" \
      --bind "$DATA_DIR" \
      --bind "$run_root" \
      "$IMAGE" \
      python3 - <<'PY_DEP' >/dev/null 2>&1
import importlib
mods = [
    "hydra",
    "omegaconf",
    "lightning",
    "wandb",
    "scipy",
    "matplotlib",
    "torchvision",
    "torchmetrics",
    "rich",
]
for name in mods:
    importlib.import_module(name)
raise SystemExit(0)
PY_DEP
  then
    PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" \
      singularity exec \
        --nv \
        --bind "$ROOT_DIR" \
        --bind "/scratch/$USER_NAME" \
        --bind "$DATA_DIR" \
        --bind "$run_root" \
        "$IMAGE" \
        python3 -m pip install --user \
          hydra-core \
          omegaconf \
          lightning \
          wandb \
          scipy \
          matplotlib \
          torchvision \
          torchmetrics \
          rich \
          torch-fidelity
  fi
}

run_in_container() {
  local run_root="$1"
  shift
  ensure_py_deps "$run_root"

  export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
  export PYTHONNOUSERSITE=0
  export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
  export WANDB_API_KEY_FILE
  export SINGULARITYENV_WANDB_API_KEY_FILE="$WANDB_API_KEY_FILE"
  export APPTAINERENV_WANDB_API_KEY_FILE="$WANDB_API_KEY_FILE"
  if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
  fi

  singularity exec \
    --nv \
    --bind "$ROOT_DIR" \
    --bind "/scratch/$USER_NAME" \
    --bind "$DATA_DIR" \
    --bind "$run_root" \
    "$IMAGE" \
    bash -c "cd '$ROOT_DIR' && export PYTHONUSERBASE='$PYTHONUSERBASE_DIR' PYTHONNOUSERSITE=0 PYTHONPATH='$PYTHON_SITE\${PYTHONPATH:+:\$PYTHONPATH}' && $*"
}
