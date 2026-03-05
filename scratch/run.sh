#!/bin/bash

#SBATCH --partition=gpu               # Partition (job queue)
#SBATCH --requeue                     # Return job to the queue if preempted
#SBATCH --job-name=myjob_laser        # Assign a short name to your job
#SBATCH --nodes=1                     # Number of nodes you require
#SBATCH --ntasks=1                    # Total # of tasks across all nodes
#SBATCH --cpus-per-task=16            # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:4                  # Request number of GPUs
#SBATCH --mem=128100                  # Real memory (RAM) required (MB)
#SBATCH --time=24:00:00               # Total run time limit (HH:MM:SS)
#SBATCH --output=laser.out            # STDOUT output file
#SBATCH --error=laser.err             # STDERR output file (optional)

set -euo pipefail

# Try to initialize environment modules if needed.
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
fi

# Ensure singularity is available; if not, try common module names.
if ! command -v singularity >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  fi
fi

if ! command -v singularity >/dev/null 2>&1; then
  echo "ERROR: singularity not found on PATH after module init/load attempts." >&2
  echo "Try one of these in an interactive shell, then resubmit:" >&2
  echo "  module load singularity" >&2
  echo "  module load singularityce" >&2
  echo "  module load singularity-ce" >&2
  exit 1
fi

# Set this to an existing container image on your cluster.
# Good baseline image: PyTorch + CUDA matching your GPU drivers.
IMAGE="${IMAGE:-/scratch/$USER/containers/pytorch_24.02.sif}"

if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: container image not found: $IMAGE" >&2
  echo "Set IMAGE=/path/to/your.sif before sbatch." >&2
  exit 1
fi

PROJECT_DIR="/cache/home/$USER/Projects/laser"
DATA_DIR="/scratch/$USER/data/celeba"
OUT_DIR="/scratch/$USER/runs/laser_celeba_128"

mkdir -p "$OUT_DIR"

srun singularity exec --nv \
  --bind "$PROJECT_DIR":"$PROJECT_DIR" \
  --bind "/scratch/$USER":"/scratch/$USER" \
  "$IMAGE" \
  python3 "$PROJECT_DIR/scratch/laser.py" \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --image_size 128 \
  --out_dir "$OUT_DIR" \
  --stage1_devices 4 \
  --stage2_devices 4 \
  --stage1_strategy ddp \
  --stage2_arch spatial_depth \
  --no_quantize_sparse_coeffs
