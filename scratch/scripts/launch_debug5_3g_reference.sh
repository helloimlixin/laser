#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RECIPE_PROFILE="${RECIPE_PROFILE:-415ephb2}"
export PARTITION="${PARTITION:-cgpu}"
export NODES="${NODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export MEM_MB="${MEM_MB:-128000}"
export TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
export OUT_DIR="${OUT_DIR:-/scratch/$USER/runs/laser_lighter_debug5_3g_gen_celeba128_quantized}"
export WANDB_NAME="${WANDB_NAME:-laser_lighter_debug5_3g_tok98k_s1000_bs32}"
export LOG_PREFIX="${LOG_PREFIX:-laser_lighter_debug5_3g}"
export JOB_NAME="${JOB_NAME:-laser-lighter-debug5-3g-gen}"

exec "$ROOT_DIR/scripts/launch_proto_rqsd_multinode_slurm.sh"
