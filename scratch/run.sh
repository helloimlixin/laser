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

cd /scratch/$USER/laser

srun python3 scratch/laser.py \
  --dataset celeba \
  --data_dir /scratch/$USER/data/celeba \
  --image_size 128 \
  --out_dir /scratch/$USER/runs/laser_celeba_128 \
  --stage1_devices 4 \
  --stage2_devices 4 \
  --stage1_strategy ddp \
  --stage2_arch spatial_depth \
  --no_quantize_sparse_coeffs
