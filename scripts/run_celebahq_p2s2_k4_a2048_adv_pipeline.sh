#!/usr/bin/env bash
# CelebA-HQ p2s2/k4 quantized-token pipeline with a smaller dictionary and
# adversarially fine-tuned stage 1.
#
# Follow-up to helloimlixin-rutgers/laser/8eqmawdn:
# - half dictionary size: 4096 -> 2048 atoms
# - longer stage 1: 10 -> 30 epochs
# - conservative PatchGAN loss after reconstruction/LPIPS has warmed up
set -euo pipefail

cd /home/xl598/Projects/laser

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-2048}"
RUN_ROOT="${RUN_ROOT:-/home/xl598/Projects/laser/runs/celebahq_p2s2_k4_a${NUM_EMBEDDINGS}_adv_quant_${STAMP}}"
PIPELINE_LOG="${PIPELINE_LOG:-${RUN_ROOT}/pipeline.nohup.log}"

mkdir -p "${RUN_ROOT}"

echo "[$(date --iso-8601=seconds)] launching p2s2/k4/a${NUM_EMBEDDINGS} adversarial quant pipeline"
echo "RUN_ROOT=${RUN_ROOT}"
echo "PIPELINE_LOG=${PIPELINE_LOG}"

exec env \
  STAMP="${STAMP}" \
  RUN_ROOT="${RUN_ROOT}" \
  WANDB_GROUP="celebahq_p2s2_k4_a${NUM_EMBEDDINGS}_adv_quant_${STAMP}" \
  NUM_EMBEDDINGS="${NUM_EMBEDDINGS}" \
  STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}" \
  PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.20}" \
  PERCEPTUAL_START_STEP="${PERCEPTUAL_START_STEP:-1000}" \
  PERCEPTUAL_WARMUP_STEPS="${PERCEPTUAL_WARMUP_STEPS:-2000}" \
  ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.05}" \
  ADVERSARIAL_START_STEP="${ADVERSARIAL_START_STEP:-5000}" \
  ADVERSARIAL_WARMUP_STEPS="${ADVERSARIAL_WARMUP_STEPS:-5000}" \
  DISCRIMINATOR_LR="${DISCRIMINATOR_LR:-5.0e-5}" \
  DISCRIMINATOR_CHANNELS="${DISCRIMINATOR_CHANNELS:-64}" \
  DISCRIMINATOR_LAYERS="${DISCRIMINATOR_LAYERS:-3}" \
  bash scripts/run_celebahq_p2s2_k4_quant_pipeline.sh >"${PIPELINE_LOG}" 2>&1
