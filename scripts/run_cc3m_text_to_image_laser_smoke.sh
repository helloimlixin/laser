#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CC3M_DIR="${CC3M_DIR:-/workspace/Projects/data/cc3m}"
STAGE1_CKPT="${STAGE1_CKPT:-${ROOT}/ozbyadv50-last.ckpt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/outputs/cc3m_text2image_laser_smoke}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
CACHE_MAX_ITEMS="${CACHE_MAX_ITEMS:-512}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-32}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
COEFF_BINS="${COEFF_BINS:-16}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-32}"
TEXT_TOKENIZER="${TEXT_TOKENIZER:-rq_bpe16k}"

STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-100}"
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-1}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"
STAGE2_D_MODEL="${STAGE2_D_MODEL:-256}"
STAGE2_LAYERS="${STAGE2_LAYERS:-4}"
STAGE2_HEADS="${STAGE2_HEADS:-8}"
STAGE2_FF="${STAGE2_FF:-1024}"
STAGE2_SAMPLE_IMAGES="${STAGE2_SAMPLE_IMAGES:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_NAME="${WANDB_NAME:-cc3m-text2image-laser}"
WANDB_GROUP="${WANDB_GROUP:-cc3m-text2image-laser}"
WANDB_TAGS="${WANDB_TAGS:-[stage2,cc3m,text,laser,transformer]}"

CACHE_PATH="${CACHE_PATH:-${RUN_ROOT}/token_cache/cc3m__train__img${IMAGE_SIZE}__laser_cb${COEFF_BINS}_quantile_p99p5_${TEXT_TOKENIZER}_text${TEXT_MAX_LENGTH}.pt}"

mkdir -p "$(dirname "${CACHE_PATH}")" "${RUN_ROOT}"

python scripts/tools/build_token_cache.py \
  --stage1_checkpoint "${STAGE1_CKPT}" \
  --dataset cc3m \
  --data_dir "${CC3M_DIR}" \
  --split train \
  --cache_mode quantized \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${CACHE_BATCH_SIZE}" \
  --num_workers "${CACHE_NUM_WORKERS}" \
  --coeff_vocab_size "${COEFF_BINS}" \
  --coeff_quantization quantile \
  --coeff_calibration_percentile 99.5 \
  --text_max_length "${TEXT_MAX_LENGTH}" \
  --text_tokenizer "${TEXT_TOKENIZER}" \
  --output "${CACHE_PATH}" \
  --max_items "${CACHE_MAX_ITEMS}" \
  --device auto

python train.py stage2 \
  token_cache_path="${CACHE_PATH}" \
  output_dir="${RUN_ROOT}/stage2" \
  data.dataset=cc3m \
  data.num_workers="${STAGE2_NUM_WORKERS}" \
  ar.type=sparse_spatial_depth \
  ar.text_conditional=true \
  ar.text_conditioning_mode=rq_prefix \
  ar.text_prefix_length="${TEXT_MAX_LENGTH}" \
  ar.text_loss_weight=0.1 \
  ar.image_loss_weight=0.9 \
  ar.n_global_spatial_tokens=0 \
  ar.d_model="${STAGE2_D_MODEL}" \
  ar.n_heads="${STAGE2_HEADS}" \
  ar.n_layers="${STAGE2_LAYERS}" \
  ar.d_ff="${STAGE2_FF}" \
  ar.max_steps="${STAGE2_MAX_STEPS}" \
  train_ar.batch_size="${STAGE2_BATCH_SIZE}" \
  train_ar.max_epochs="${STAGE2_MAX_EPOCHS}" \
  train_ar.accelerator=gpu \
  train_ar.devices=1 \
  train_ar.strategy=auto \
  train_ar.precision=bf16-mixed \
  train_ar.deterministic=false \
  train_ar.validation_split=0.05 \
  train_ar.test_split=0.05 \
  train_ar.sample_every_n_epochs=1 \
  train_ar.sample_num_images="${STAGE2_SAMPLE_IMAGES}" \
  train_ar.sample_temperature=0.9 \
  train_ar.sample_top_k=1024 \
  train_ar.sample_log_to_wandb=false \
  train_ar.save_final_samples_after_fit=true \
  train_ar.run_test_after_fit=false \
  'train_ar.sample_text_prompts=["a river has burst its banks onto farmland","a small dog wearing sunglasses","eiffel tower on a desert","a painting by vincent van gogh"]' \
  wandb.project="${WANDB_PROJECT:-laser}" \
  wandb.name="${WANDB_NAME:-cc3m-text2image-laser}" \
  wandb.group="${WANDB_GROUP:-cc3m-text2image-laser}" \
  "wandb.tags=${WANDB_TAGS:-[stage2,cc3m,text,laser,transformer]}" \
  wandb.append_timestamp=false
