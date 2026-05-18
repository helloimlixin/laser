#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-main-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-32000}"
LOG_ROOT="${LOG_ROOT:-/scratch/$USER/runs/dataset_downloads}"

mkdir -p "$LOG_ROOT"

submit_one() {
  local name="$1"
  local script="$2"
  sbatch \
    --partition="$PARTITION" \
    --job-name="$name" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM_MB" \
    --time="$TIME_LIMIT" \
    --chdir="$PWD" \
    --output="$LOG_ROOT/${name}_%j.out" \
    --error="$LOG_ROOT/${name}_%j.err" \
    "$script"
}

submit_one download-ffhq scripts/download_ffhq_official.sh
submit_one download-maestro scripts/download_maestro_v3.sh
