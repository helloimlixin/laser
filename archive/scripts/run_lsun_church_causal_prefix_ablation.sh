#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/tmp/laser-h100-venv/bin/python}"
VARIANT="${1:-}"
case "$VARIANT" in
  aw15) ATOM_LOSS_WEIGHT=1.5 ;;
  aw3) ATOM_LOSS_WEIGHT=3.0 ;;
  *) echo "usage: $0 {aw15|aw3}" >&2; exit 2 ;;
esac

STAGE1="$ROOT/outputs/lsun-church-a16384-k4-fair-stage1-b256-20260815-064515/church256-rqvae-laser-8x8-a16384-k4-fair/15082026_064530/best_rfid_slot1_model.pt"
DATA="$ROOT/outputs/lsun-church-a16384-k4-compound-v5-stage2-20260816-052724/data/lsun"
CACHE="$ROOT/outputs/lsun-church-a16384-k4-causal-prefix-v2-cache-20260817/token_cache/lsun_church_train_a16384k4_causal_prefix.pt"
FID_STATS="$ROOT/third_party/rq-vae-transformer/assets/fid_stats/lsun_256_church.npz"
CONTINUOUS_RFID="$ROOT/outputs/lsun-church-a16384-k4-cache-rfid-preflight/continuous.json"
QUANTIZED_RFID="$ROOT/outputs/lsun-church-a16384-k4-cache-rfid-preflight/quantized.json"
OUTPUT="${RUN_ROOT:-$ROOT/outputs/lsun-church-a16384-k4-causal-prefix-v6-${VARIANT}-e25-20260817}"
WANDB_ID="${WANDB_RUN_ID:-lsunchurchcpv6${VARIANT}e25-20260817}"
NPROC="${NPROC:-2}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-128}"

for required in "$PYTHON_BIN" "$STAGE1" "$CACHE" "$FID_STATS" \
  "$CONTINUOUS_RFID" "$QUANTIZED_RFID"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

export PYTHONPATH="$ROOT/third_party/rq-vae-transformer:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb}"
export LASER_CHECKPOINT_STAGING_DIR="${LASER_CHECKPOINT_STAGING_DIR:-/tmp/laser-checkpoint-staging}"

exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1" --data "$DATA" --dataset lsun_church \
  --model-preset lsun-church-350m --token-cache "$CACHE" \
  --cache-rfid-preflight "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  --output "$OUTPUT" --checkpoint-dir "$OUTPUT/checkpoints" \
  --distributed-backend ddp --epochs 25 --batch-size "$STAGE2_BATCH_SIZE" \
  --total-batch-size 256 \
  --num-atoms 16384 --sparsity-level 4 --coeff-vocab-size 2048 --coeff-max 3 \
  --compound-tokens --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads --compound-distribution-geometry \
  --causal-prefix-state --causal-prefix-loss-weight 0.5 \
  --geometry-top-k 4 --atom-loss-weight "$ATOM_LOSS_WEIGHT" \
  --geometry-loss-weight 0.05 --geometry-start-epoch 2 --geometry-warmup-epochs 3 \
  --coeff-target-mode soft --coeff-target-temperature 0.5 \
  --seed 0 \
  --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
  --atom-temperature 1 --atom-top-k 250 --atom-top-p 1 \
  --coeff-temperature 1 --coeff-top-k 0 --coeff-top-p 0.92 \
  --fid-num-samples 10000 --fid-batch-size 250 --fid-every 5 \
  --metric-backend original-rqvae --fid-reference-stats "$FID_STATS" \
  --save-ckpt-freq 5 --keep-best-checkpoints 1 \
  --save-step-freq 0 --sample-grid-every 0 \
  --resume --wandb-mode online --wandb-entity helloimlixin-rutgers \
  --wandb-project laser --wandb-id "$WANDB_ID" \
  --wandb-name "lsun-church-causal-prefix-v6-${VARIANT}-e25-20260817"
