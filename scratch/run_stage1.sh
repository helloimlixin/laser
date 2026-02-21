#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=laser_s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=200000
#SBATCH --time=24:00:00
#SBATCH --output=stage1.out
#SBATCH --error=stage1.err

set -euo pipefail
module purge
module load singularity

REALHOME="$(readlink -f "$HOME")"
IMG="${REALHOME}/containers/pytorch_2.5.1_cuda11.8.sif"
OVL="${REALHOME}/containers/overlays/laser_py.overlay"

export SINGULARITYENV_WANDB_API_KEY="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"
export SINGULARITYENV_WANDB_MODE="online"
export SINGULARITYENV_WANDB_PROJECT="laser-scratch"
export SINGULARITYENV_WANDB_DIR="$PWD/wandb"
export SINGULARITYENV_WANDB_SILENT="true"

nvidia-smi

srun singularity exec --nv --cleanenv --overlay "$OVL" --writable-tmpfs "$IMG" python -u proto.py --dataset celeba --data_dir ../../data/celeba --crop_mode rcrop --load_size 256 --crop_size 64 --batch_size 64 --stage1_epochs 200 --stage2_epochs 0 --stage1_devices 4 --stage1_strategy ddp --wandb_mode online --wandb_project laser-scratch --stage2_batch_size 16
