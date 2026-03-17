#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
JOB_NAME="${JOB_NAME:-proto-rqsd-debug}"
LOG_PREFIX="${LOG_PREFIX:-proto_rqsd_debug}"

PARTITION="${PARTITION:-gpu}" \
JOB_NAME="$JOB_NAME" \
GPUS="$GPUS" \
BATCH_SIZE="$BATCH_SIZE" \
STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
LOG_PREFIX="$LOG_PREFIX" \
"$SCRIPT_DIR/launch_proto_rqsd_slurm.sh"
