#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=laser_celeba128_4g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=200000
#SBATCH --time=24:00:00
#SBATCH --output=laser.out
#SBATCH --error=laser.err

set -euo pipefail

module purge
module load singularity

REALHOME="$(readlink -f "$HOME")"
IMG="${REALHOME}/containers/pytorch_2.5.1_cuda11.8.sif"
OVL="${REALHOME}/containers/overlays/laser_py.overlay"

# --- W&B (simple mode) ---
export SINGULARITYENV_WANDB_API_KEY="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"
export SINGULARITYENV_WANDB_MODE="online"
export SINGULARITYENV_WANDB_PROJECT="laser-scratch"
export SINGULARITYENV_WANDB_DIR="$PWD/wandb"
export SINGULARITYENV_WANDB_SILENT="true"

echo "[INFO] PWD: $(pwd)"
echo "[INFO] IMG: $IMG"
echo "[INFO] OVL: $OVL"

nvidia-smi

srun singularity exec --nv --cleanenv \
  --overlay "$OVL" --writable-tmpfs \
  "$IMG" \
    python -u proto.py \
      --dataset celeba \
      --data_dir ../../data/celeba \
      --crop_mode rcrop \
      --load_size 256 \
      --crop_size 64 \
      --stage1_epochs 200 \
      --stage2_epochs 200 \
      --token_subset 150000 \
      --stage1_devices 2 \
      --stage1_strategy ddp \
      --stage2_devices 2 \
      --stage2_strategy ddp \
      --batch_size 128 \
      --stage2_batch_size 16 \
      --stage2_sample_every_steps 5000 \
      --stage2_sample_batch_size 16 \
      --stage2_fid_num_samples 16 \
      --stage2_fid_feature 64 \
      --token_num_workers 0 \
      --n_bins 129 \
      --coef_max 3.0 \
      --stage1_lr_schedule cosine \
      --stage1_warmup_epochs 2 \
      --stage1_min_lr_ratio 0.1 \
      --stage2_lr_schedule cosine \
      --stage2_warmup_epochs 5 \
      --stage2_min_lr_ratio 0.05 \
      --fid_compute_batch_size 16 \
      --wandb_mode online \
      --wandb_project laser-scratch
