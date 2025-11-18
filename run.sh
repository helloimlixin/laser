#!/bin/bash

#SBATCH --partition=gpu          # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --job-name=myjob_tinyvit      # Assign a short name to your job

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1    # Cores per task (>1 if multithread tasks)

#SBATCH --gres=gpu:2                # Request number of GPUs



#SBATCH --mem=128100               # Real memory (RAM) required (MB)

#SBATCH --time=5:00:00           # Total run time limit (HH:MM:SS)

#SBATCH --output=tinyvit.out  # STDOUT output file

#SBATCH --error=tinyvit.err   # STDERR output file (optional)

cd /scratch/$USER


srun /scratch/$USER/laser/train.py model=laser dataset=celeba