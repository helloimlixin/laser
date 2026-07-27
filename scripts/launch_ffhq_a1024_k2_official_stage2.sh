#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="ffhq-a1024-k2-rqvae-strict-20260720-145706"
OUT="$ROOT/outputs/$SOURCE_RUN/stage2-official-rqtransformer-350M-coeffcal"
CHECKPOINT="$ROOT/outputs/$SOURCE_RUN/stage1_checkpoint/best_rfid_slot2_model.pt"
mkdir -p "$OUT"

# Keep a stable location for operators to find the torchrun process. Because
# the shell is replaced by torchrun below, this PID remains valid for the job's
# lifetime. Start this launcher with setsid/nohup so it survives terminal exit.
echo "$$" > "$OUT/launcher.pid"

export PYTHONPATH="$ROOT/third_party/rq-vae-transformer${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=online
# Keep all W&B caches, artifact staging, and run files on persistent workspace
# storage. Checkpoint artifacts are multi-GB and can exhaust the root overlay.
export WANDB_CACHE_DIR=/workspace/.cache/wandb
export WANDB_DATA_DIR=/workspace/.local/share/wandb
export WANDB_DIR=/workspace/wandb
export XDG_CACHE_HOME=/workspace/.cache
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_DIR"
export OMP_NUM_THREADS=8
# This host has NVLink between all four H100s. Let NCCL use its native P2P/SHM
# transports; forcing socket transport caused a truncated collective when two
# distributed jobs shared the GPUs.
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

exec torchrun --standalone --nproc_per_node=4 \
  "$ROOT/scripts/train_ffhq_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$CHECKPOINT" \
  --data /workspace/Projects/data/ffhq \
  --output "$OUT" \
  --epochs 200 --batch-size 32 --total-batch-size 128 \
  --num-atoms 1024 --coeff-vocab-size 1024 --coeff-max 3 \
  --coeff-scales 51.49130249 13.80611229 \
  --lr 0.0005 --fid-num-samples 50000 --fid-batch-size 64 --fid-every 50 \
  --fid-top-k 250 --fid-top-p 1.0 \
  --wandb-id ffhqa1024k2rqt350cal-fid50k-20260723 \
  --wandb-name ffhq-a1024-k2-official-rqtransformer-350M-coeffcal-fid50k-20260723
