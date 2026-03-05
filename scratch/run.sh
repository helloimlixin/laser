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

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1

STAGE1_DEVICES="${STAGE1_DEVICES:-4}"
STAGE2_DEVICES="${STAGE2_DEVICES:-4}"
STAGE1_STRATEGY="${STAGE1_STRATEGY:-ddp}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-0}"
STAGE2_FID_NUM_SAMPLES="${STAGE2_FID_NUM_SAMPLES:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-scratch}"

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set; falling back to WANDB_MODE=offline to avoid login stalls."
  WANDB_MODE="offline"
fi

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

# Use current working directory (expected: scratch/).
BASE_DIR="."

# Accept overrides, but ensure all paths are absolute for singularity binds.
# PROJECT_DIR should be the folder containing laser.py.
PROJECT_DIR_INPUT="${PROJECT_DIR:-.}"
OUT_DIR_INPUT="${OUT_DIR:-/scratch/$USER/runs/laser_celeba_128}"

if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_DIR_INPUT="$DATA_DIR"
else
  DATA_DIR_INPUT="/scratch/$USER/Projects/data/celeba"
fi

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

if [[ ! -d "$DATA_DIR" ]]; then
  for cand in \
    "$BASE_DIR/../../data/celeba" \
    "$BASE_DIR/../data/celeba" \
    "/scratch/$USER/Projects/data/celeba" \
    "/scratch/$USER/Projects/data/CelebA" \
    "/scratch/$USER/Projects/data/img_align_celeba" \
    "/scratch/$USER/data/celeba" \
    "/scratch/$USER/data/CelebA" \
    "/scratch/$USER/data/img_align_celeba"
  do
    if [[ -d "$cand" ]]; then
      DATA_DIR="$cand"
      break
    fi
  done
fi

if [[ ! -d "$DATA_DIR" ]]; then
  FOUND_DATA_DIR="$(find "/scratch/$USER" -maxdepth 6 -type d \( -iname "celeba" -o -iname "img_align_celeba" \) 2>/dev/null | head -n 1 || true)"
  if [[ -n "$FOUND_DATA_DIR" && -d "$FOUND_DATA_DIR" ]]; then
    DATA_DIR="$FOUND_DATA_DIR"
  fi
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
  echo "Tried common paths and a shallow search under /scratch/$USER." >&2
  echo "Set DATA_DIR explicitly to the folder containing CelebA images." >&2
  echo "Example:" >&2
  echo "  DATA_DIR=/scratch/$USER/Projects/data/celeba sbatch scratch/run.sh" >&2
  exit 1
fi

DATA_DIR="$(cd "$DATA_DIR" && pwd)"

IMG_COUNT="$(find "$DATA_DIR" -maxdepth 3 -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) \
  2>/dev/null | wc -l | tr -d ' ')"
echo "DATA_IMAGE_COUNT(maxdepth=3)=$IMG_COUNT"
if [[ "${IMG_COUNT:-0}" -eq 0 ]]; then
  echo "ERROR: no image files found under DATA_DIR=$DATA_DIR (maxdepth=3)." >&2
  exit 1
fi

nvidia-smi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

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
echo "PWD=$PWD"
echo "STAGE1_DEVICES=$STAGE1_DEVICES"
echo "STAGE2_DEVICES=$STAGE2_DEVICES"
echo "STAGE1_STRATEGY=$STAGE1_STRATEGY"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "STAGE2_BATCH_SIZE=$STAGE2_BATCH_SIZE"
echo "FID_NUM_SAMPLES=$FID_NUM_SAMPLES"
echo "STAGE2_FID_NUM_SAMPLES=$STAGE2_FID_NUM_SAMPLES"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_PROJECT=$WANDB_PROJECT"

srun singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER" \
  --bind "$DATA_BIND_DIR" \
  --bind "$OUT_BIND_DIR" \
  "$IMAGE" \
  python3 -u "$PROJECT_DIR/laser.py" \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --image_size 128 \
  --out_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --stage2_batch_size "$STAGE2_BATCH_SIZE" \
  --fid_num_samples "$FID_NUM_SAMPLES" \
  --stage2_fid_num_samples "$STAGE2_FID_NUM_SAMPLES" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --stage1_devices "$STAGE1_DEVICES" \
  --stage2_devices "$STAGE2_DEVICES" \
  --stage1_strategy "$STAGE1_STRATEGY" \
  --stage2_arch spatial_depth \
  --no_quantize_sparse_coeffs
