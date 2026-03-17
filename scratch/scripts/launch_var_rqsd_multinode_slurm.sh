#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${JOB_NAME:-laser-var-rh-12g}"
PARTITION="${PARTITION:-gpu-redhat}"
NODES="${NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-500000}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
LOG_PREFIX="${LOG_PREFIX:-laser_var_rqsd_mn}"

sbatch \
  --partition="$PARTITION" \
  --requeue \
  --job-name="$JOB_NAME" \
  --nodes="$NODES" \
  --ntasks-per-node="$GPUS_PER_NODE" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --mem="$MEM_MB" \
  --time="$TIME_LIMIT" \
  --chdir="$ROOT_DIR" \
  --output="$ROOT_DIR/${LOG_PREFIX}_%j.out" \
  --error="$ROOT_DIR/${LOG_PREFIX}_%j.err" \
  "$ROOT_DIR/scripts/run_var_rqsd_multinode_job.sh"
