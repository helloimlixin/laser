#!/bin/bash
#SBATCH --job-name=c256-ps2
#SBATCH --partition=gpu-redhat
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH --time=1-12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
unset LC_ALL || true
export LANG=C

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "$ROOT_DIR/scripts/pg.sh"
source "$ROOT_DIR/scripts/slurm_env.sh"
idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
export BIN="$(pg_bin_at "$idx")"

run_in_container "$RUN_ROOT" "bash '$ROOT_DIR/scripts/run_ps2.sh'"
