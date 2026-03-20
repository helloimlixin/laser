#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OUT_DIR="${OUT_DIR:-/scratch/$USER/runs/laser_fast100}"
export WANDB_NAME="${WANDB_NAME:-fast100}"
export LOG_PREFIX="${LOG_PREFIX:-fast100}"
export JOB_NAME="${JOB_NAME:-laser-fast100}"

export PATCH_BASED="${PATCH_BASED:-false}"

exec "$ROOT_DIR/scripts/launch_100ep.sh" "$@"
