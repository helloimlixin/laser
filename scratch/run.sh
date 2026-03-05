#!/bin/bash

#SBATCH --partition=gpu               # Partition (job queue)
#SBATCH --requeue                     # Return job to the queue if preempted
#SBATCH --job-name=laser        # Assign a short name to your job
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

# Container image can be:
# 1) local path to .sif (recommended), or
# 2) URI like docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
#
# If IMAGE is unset, prefer local .sif if present, else fall back to docker:// URI.
DEFAULT_LOCAL_IMAGE="/scratch/$USER/containers/pytorch_24.02.sif"
DEFAULT_REMOTE_IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
if [[ -z "${IMAGE:-}" ]]; then
  if [[ -f "$DEFAULT_LOCAL_IMAGE" ]]; then
    IMAGE="$DEFAULT_LOCAL_IMAGE"
  else
    IMAGE="$DEFAULT_REMOTE_IMAGE"
  fi
fi

if [[ "$IMAGE" != docker://* && "$IMAGE" != library://* && "$IMAGE" != oras://* ]]; then
  if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: container image not found: $IMAGE" >&2
    echo "Set IMAGE to an existing .sif path, or use a URI, e.g.:" >&2
    echo "  IMAGE=docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime sbatch scratch/run.sh" >&2
    echo "Or pre-pull once on login node:" >&2
    echo "  mkdir -p /scratch/$USER/containers" >&2
    echo "  singularity pull /scratch/$USER/containers/pytorch_24.02.sif \\" >&2
    echo "    docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime" >&2
    exit 1
  fi
fi

# SLURM may stage the script under /var/lib/slurm/...; do NOT derive paths from BASH_SOURCE.
BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Accept overrides, but ensure all paths are absolute for singularity binds.
# PROJECT_DIR should be the folder containing laser.py.
PROJECT_DIR_INPUT="${PROJECT_DIR:-$BASE_DIR}"
DATA_DIR_INPUT="${DATA_DIR:-/scratch/$USER/data/celeba}"
OUT_DIR_INPUT="${OUT_DIR:-/scratch/$USER/runs/laser_celeba_128}"

if [[ "$PROJECT_DIR_INPUT" != /* ]]; then
  PROJECT_DIR_INPUT="$BASE_DIR/$PROJECT_DIR_INPUT"
fi
if [[ "$DATA_DIR_INPUT" != /* ]]; then
  DATA_DIR_INPUT="$BASE_DIR/$DATA_DIR_INPUT"
fi
if [[ "$OUT_DIR_INPUT" != /* ]]; then
  OUT_DIR_INPUT="$BASE_DIR/$OUT_DIR_INPUT"
fi

if [[ ! -d "$PROJECT_DIR_INPUT" ]]; then
  echo "ERROR: PROJECT_DIR does not exist: $PROJECT_DIR_INPUT" >&2
  exit 1
fi

PROJECT_DIR="$(cd "$PROJECT_DIR_INPUT" && pwd)"
DATA_DIR="$DATA_DIR_INPUT"
OUT_DIR="$OUT_DIR_INPUT"

if [[ ! -f "$PROJECT_DIR/laser.py" ]]; then
  echo "ERROR: laser.py not found under PROJECT_DIR: $PROJECT_DIR" >&2
  echo "Set PROJECT_DIR to your scratch source folder, e.g.:" >&2
  echo "  PROJECT_DIR=/cache/home/$USER/Projects/laser/scratch sbatch scratch/run.sh" >&2
  exit 1
fi

nvidia-smi

mkdir -p "$OUT_DIR"

DATA_BIND_DIR="$DATA_DIR"
if [[ ! -d "$DATA_BIND_DIR" ]]; then
  DATA_BIND_DIR="$(dirname "$DATA_DIR")"
fi
OUT_BIND_DIR="$OUT_DIR"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "IMAGE=$IMAGE"
echo "BASE_DIR=$BASE_DIR"

srun singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER" \
  --bind "$DATA_BIND_DIR" \
  --bind "$OUT_BIND_DIR" \
  "$IMAGE" \
  python3 "$PROJECT_DIR/laser.py" \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --image_size 128 \
  --out_dir "$OUT_DIR" \
  --stage1_devices 4 \
  --stage2_devices 4 \
  --stage1_strategy ddp \
  --stage2_arch spatial_depth \
  --no_quantize_sparse_coeffs
