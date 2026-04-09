#!/bin/bash
#SBATCH --job-name=c256-ps1
#SBATCH --partition=gpu-redhat
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
unset LC_ALL || true
export LANG=C

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "$ROOT_DIR/scripts/pg.sh"
source "$ROOT_DIR/scripts/slurm_env.sh"

run_in_container "$RUN_ROOT" "bash '$ROOT_DIR/scripts/run_ps1.sh'"
