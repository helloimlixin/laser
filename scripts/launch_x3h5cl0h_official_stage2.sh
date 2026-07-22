#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/imagenet_x3h5cl0h_stage2_a16384_k2_c2048_m20/stage2"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/third_party/rq-vae-transformer${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="8"
# This workspace exposes all H100s but not CUDA IPC/NVLink peer mappings.
# Force NCCL onto its socket transport for single-host DDP.
export NCCL_P2P_DISABLE="1"
export NCCL_SHM_DISABLE="1"

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$ROOT/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt" \
  --data /workspace/Projects/data/imagenet \
  --output "$OUT" \
  --epochs 100 --batch-size 8 --total-batch-size 2048 \
  --num-atoms 16384 --coeff-vocab-size 2048 --coeff-max 20 --coeff-scale 6.4 --lr 0.0005 \
  --wandb-name imagenet-official-rqtransformer-laser-a16384-k2-c2048-m20
