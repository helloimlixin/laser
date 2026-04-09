#!/bin/bash
#SBATCH --job-name=c256-p4g
#SBATCH --partition=gpu
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

DEBUG_ROOT="${SLURM_DEBUG_ROOT:-${RUN_ROOT:-$ROOT_DIR}/meta/slurm}"
mkdir -p "$DEBUG_ROOT"
JOB_TOKEN="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}"
DEBUG_OUT="$DEBUG_ROOT/${SLURM_JOB_NAME}_${JOB_TOKEN}.out"
DEBUG_ERR="$DEBUG_ROOT/${SLURM_JOB_NAME}_${JOB_TOKEN}.err"
exec > >(tee -a "$DEBUG_OUT")
exec 2> >(tee -a "$DEBUG_ERR" >&2)
set -x

echo "debug_root=$DEBUG_ROOT"
echo "debug_out=$DEBUG_OUT"
echo "debug_err=$DEBUG_ERR"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "start_time=$(date -Is)"
echo "slurm_job_id=${SLURM_JOB_ID:-}"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-}"
echo "root_dir=$ROOT_DIR"
echo "run_root=${RUN_ROOT:-}"
echo "partition=${PARTITION:-}"
echo "win_list=${WIN_LIST:-}"
echo "gst_list=${GST_LIST:-}"

source "$ROOT_DIR/scripts/p4g.sh"
source "$ROOT_DIR/scripts/slurm_env.sh"
idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
export WIN="${WIN:-$(p4g_window_at "$idx")}"
export GST="${GST:-0}"
if [[ -n "${GST_LIST:-}" ]]; then
  export GST="$(p4g_gst_at "$idx")"
fi

echo "resolved_win=$WIN"
echo "resolved_gst=$GST"

run_in_container "$RUN_ROOT" "bash '$ROOT_DIR/scripts/run_p4s2.sh'"
