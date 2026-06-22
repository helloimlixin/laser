#!/usr/bin/env bash
# Continue the promising CelebA-HQ p2s2/k4/a4096 quantized stage-2 run.
#
# Source run: helloimlixin-rutgers/laser/8eqmawdn
# The original run stopped at max_steps=50000 during epoch 32. This resumes
# from last.ckpt, removes the step cap, and extends the total epoch target.
set -euo pipefail

cd /home/xl598/Projects/laser

RUN_ROOT="${RUN_ROOT:-/home/xl598/Projects/laser/runs/celebahq_p2s2_k4_a4096_quant_20260529_031540}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba_hq}"

STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2}"
CACHE="${CACHE:-${RUN_ROOT}/token_cache_q256.pt}"
CKPT="${CKPT:-${STAGE2_DIR}/checkpoints/s2_20260529_085657/last.ckpt}"

WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_GROUP="${WANDB_GROUP:-celebahq_p2s2_k4_a4096_quant_20260529_031540}"
WANDB_NAME="${WANDB_NAME:-celebahq_s2_p2s2_k4_a4096_q256_20260529_031540}"
WANDB_ID="${WANDB_ID:-8eqmawdn}"

STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-82}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:--1}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
STAGE2_DEVICES="${STAGE2_DEVICES:-2}"
STAGE2_STRATEGY="${STAGE2_STRATEGY:-ddp}"
STAGE2_LR="${STAGE2_LR:-3.0e-4}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-500}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.05}"
STAGE2_D_MODEL="${STAGE2_D_MODEL:-512}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-8}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-8}"
STAGE2_D_FF="${STAGE2_D_FF:-2048}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"
STAGE2_SAMPLE_EVERY_N_EPOCHS="${STAGE2_SAMPLE_EVERY_N_EPOCHS:-1}"
STAGE2_SAMPLE_NUM_IMAGES="${STAGE2_SAMPLE_NUM_IMAGES:-16}"
STAGE2_SAMPLE_TEMPERATURE="${STAGE2_SAMPLE_TEMPERATURE:-0.7}"
STAGE2_SAMPLE_TOP_K="${STAGE2_SAMPLE_TOP_K:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/home/xl598/Projects/laser
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -f "${CACHE}" ]] || { echo "Missing token cache: ${CACHE}" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }

mkdir -p "${STAGE2_DIR}/wandb"

echo "[$(date --iso-8601=seconds)] continuing p2s2/k4/a4096 quant stage 2"
echo "RUN_ROOT=${RUN_ROOT}"
echo "STAGE2_DIR=${STAGE2_DIR}"
echo "CACHE=${CACHE}"
echo "CKPT=${CKPT}"
echo "WANDB=${WANDB_PROJECT}/${WANDB_GROUP}/${WANDB_ID}"
echo "STAGE2_MAX_EPOCHS=${STAGE2_MAX_EPOCHS}"
echo "STAGE2_MAX_STEPS=${STAGE2_MAX_STEPS}"

"${PYTHON_BIN}" train.py stage2 \
  output_dir="${STAGE2_DIR}" \
  hydra.run.dir="${STAGE2_DIR}/hydra_continue_8eqmawdn" \
  token_cache_path="${CACHE}" \
  ckpt_path="${CKPT}" \
  seed=42 \
  ar.type=sparse_spatial_depth \
  ar.d_model="${STAGE2_D_MODEL}" \
  ar.n_heads="${STAGE2_N_HEADS}" \
  ar.n_layers="${STAGE2_N_LAYERS}" \
  ar.d_ff="${STAGE2_D_FF}" \
  ar.learning_rate="${STAGE2_LR}" \
  ar.warmup_steps="${STAGE2_WARMUP_STEPS}" \
  ar.max_steps="${STAGE2_MAX_STEPS}" \
  ar.min_lr_ratio="${STAGE2_MIN_LR_RATIO}" \
  ar.coeff_loss_type=auto \
  train_ar.accelerator=gpu \
  train_ar.devices="${STAGE2_DEVICES}" \
  train_ar.strategy="${STAGE2_STRATEGY}" \
  train_ar.precision=bf16-mixed \
  train_ar.max_epochs="${STAGE2_MAX_EPOCHS}" \
  train_ar.batch_size="${STAGE2_BATCH_SIZE}" \
  train_ar.gradient_clip_val=1.0 \
  train_ar.log_every_n_steps=20 \
  train_ar.val_check_interval=1.0 \
  train_ar.sample_every_n_epochs="${STAGE2_SAMPLE_EVERY_N_EPOCHS}" \
  train_ar.sample_num_images="${STAGE2_SAMPLE_NUM_IMAGES}" \
  train_ar.sample_temperature="${STAGE2_SAMPLE_TEMPERATURE}" \
  train_ar.sample_top_k="${STAGE2_SAMPLE_TOP_K}" \
  train_ar.sample_log_to_wandb=true \
  train_ar.compute_generation_fid=false \
  train_ar.generation_metric_num_samples=0 \
  train_ar.run_test_after_fit=false \
  train_ar.save_final_samples_after_fit=false \
  data.dataset=celebahq \
  data.data_dir="${DATA_DIR}" \
  data.image_size=256 \
  data.num_workers="${STAGE2_NUM_WORKERS}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="${WANDB_NAME}" \
  wandb.id="${WANDB_ID}" \
  wandb.resume=allow \
  wandb.save_dir="${STAGE2_DIR}/wandb"

echo "[$(date --iso-8601=seconds)] continuation done: ${STAGE2_DIR}"
