#!/usr/bin/env bash
# Smoke run: CelebA-HQ 256x256, RQ-VAE-style backbone.
# Stage 1: 5 epochs autoencoder with random crop.
# Stage 2: 50 epochs sparse-token prior on cached tokens.
set -euo pipefail

cd /home/xl598/Projects/laser

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${1:-/home/xl598/Projects/laser/runs/smoke_celebahq_rqvae_d5_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba_hq}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
MAX_ITEMS="${MAX_ITEMS:-8192}"

# Train: random-crop 128 out of full 256. With four downsamples and 4x4
# latent patches, a crop gives a small 2x2 token grid instead of compressing
# an entire face crop into one patch.
# Val: full 256 (no crop on eval transform), so reconstruction quality is reported
# against the original resolution.
IMAGE_SIZE_PRE="${IMAGE_SIZE_PRE:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"

# Cache + stage 2 use full 256 (no random crop). The encoder is convolutional so
# it handles 256 despite being trained at 128 crops. Full images have a 4x4
# patch grid with the default patch geometry below.
STAGE2_IMAGE_SIZE="${STAGE2_IMAGE_SIZE:-256}"

# Width-160 fits comfortably at batch 4. Batch 5 is the larger-batch probe that
# still keeps the crop geometry valid for the 4x4 latent patch defaults.
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-5}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE1_DICT_LR="${STAGE1_DICT_LR:-1e-4}"
STAGE1_RECON_L1_WEIGHT="${STAGE1_RECON_L1_WEIGHT:-0.5}"
STAGE1_RECON_EDGE_WEIGHT="${STAGE1_RECON_EDGE_WEIGHT:-0.05}"
STAGE1_RECON_MSE_WEIGHT="${STAGE1_RECON_MSE_WEIGHT:-0.25}"
STAGE1_PERCEPTUAL_WEIGHT="${STAGE1_PERCEPTUAL_WEIGHT:-1.0}"
STAGE1_PERCEPTUAL_START_STEP="${STAGE1_PERCEPTUAL_START_STEP:-1000}"
STAGE1_PERCEPTUAL_WARMUP_STEPS="${STAGE1_PERCEPTUAL_WARMUP_STEPS:-2000}"
STAGE1_NUM_HIDDENS="${STAGE1_NUM_HIDDENS:-160}"
STAGE1_NUM_RES_BLOCKS="${STAGE1_NUM_RES_BLOCKS:-4}"
STAGE1_NUM_RES_HIDDENS="${STAGE1_NUM_RES_HIDDENS:-160}"
STAGE1_DECODER_EXTRA_RESIDUAL_LAYERS="${STAGE1_DECODER_EXTRA_RESIDUAL_LAYERS:-2}"
STAGE1_ATTN_RESOLUTIONS="${STAGE1_ATTN_RESOLUTIONS:-[16,32]}"
STAGE1_DIAG_LOG_INTERVAL="${STAGE1_DIAG_LOG_INTERVAL:-20}"

# Bottleneck: "dictionary" (OMP) or "rq" (kakaobrain residual quantization).
BOTTLENECK_TYPE="${BOTTLENECK_TYPE:-dictionary}"
BOTTLENECK_LOSS_WEIGHT="${BOTTLENECK_LOSS_WEIGHT:-0.25}"
BOTTLENECK_COMMITMENT_COST="${BOTTLENECK_COMMITMENT_COST:-0.05}"

# OMP-only knobs (ignored when BOTTLENECK_TYPE=rq).
OMP_RESIDUAL_TOLERANCE="${OMP_RESIDUAL_TOLERANCE:-null}"
DEAD_ATOM_REVIVAL_STEPS="${DEAD_ATOM_REVIVAL_STEPS:-100}"
DICTIONARY_THROUGH_DECODER="${DICTIONARY_THROUGH_DECODER:-true}"
DATA_INIT_FROM_FIRST_BATCH="${DATA_INIT_FROM_FIRST_BATCH:-true}"

# RQ-only knobs (ignored when BOTTLENECK_TYPE=dictionary).
RQ_CODE_DEPTH="${RQ_CODE_DEPTH:-4}"
RQ_SHARED_CODEBOOK="${RQ_SHARED_CODEBOOK:-true}"
RQ_DECAY="${RQ_DECAY:-0.99}"
RQ_RESTART_UNUSED_CODES="${RQ_RESTART_UNUSED_CODES:-true}"

# Bottleneck capacity. The defaults use non-overlapping 4x4 latent patches:
# a 128 crop has a 2x2 patch grid, while full 256 cache/stage-2 examples
# have a 4x4 token grid. Coefficients are quantized for a plain GPT prior.
BOTTLENECK_NUM_EMBEDDINGS="${BOTTLENECK_NUM_EMBEDDINGS:-8192}"
BOTTLENECK_EMBEDDING_DIM="${BOTTLENECK_EMBEDDING_DIM:-64}"
PATCH_SIZE="${PATCH_SIZE:-4}"
PATCH_STRIDE="${PATCH_STRIDE:-4}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-8}"
BOTTLENECK_COEF_MAX="${BOTTLENECK_COEF_MAX:-16.0}"

STAGE2_AR_TYPE="${STAGE2_AR_TYPE:-gpt}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-256}"
STAGE2_LR="${STAGE2_LR:-5e-4}"
STAGE2_COEFF_BINS="${STAGE2_COEFF_BINS:-256}"
STAGE2_COEFF_MAX="${STAGE2_COEFF_MAX:-auto}"
STAGE2_COEFF_QUANTIZATION="${STAGE2_COEFF_QUANTIZATION:-uniform}"
STAGE2_COEFF_MU="${STAGE2_COEFF_MU:-0.0}"
STAGE2_SAMPLE_EVERY_N_EPOCHS="${STAGE2_SAMPLE_EVERY_N_EPOCHS:-1}"
STAGE2_SAMPLE_NUM_IMAGES="${STAGE2_SAMPLE_NUM_IMAGES:-8}"
STAGE2_SAMPLE_TOP_K="${STAGE2_SAMPLE_TOP_K:-128}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/home/xl598/Projects/laser
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAGE1_DIR="${RUN_ROOT}/stage1"
CACHE="${RUN_ROOT}/token_cache_${MAX_ITEMS}.pt"
STAGE2_DIR="${RUN_ROOT}/stage2"
LOG="${RUN_ROOT}/logs/pipeline.log"

mkdir -p "${RUN_ROOT}/logs" "${STAGE1_DIR}" "${STAGE2_DIR}"

echo "[$(date --iso-8601=seconds)] smoke run start" | tee -a "${LOG}"
echo "RUN_ROOT=${RUN_ROOT}" | tee -a "${LOG}"
echo "STAGE1_EPOCHS=${STAGE1_EPOCHS}  STAGE2_EPOCHS=${STAGE2_EPOCHS}  MAX_ITEMS=${MAX_ITEMS}" | tee -a "${LOG}"
echo "IMAGE_SIZE_PRE=${IMAGE_SIZE_PRE} -> IMAGE_SIZE=${IMAGE_SIZE} (train random crop)" | tee -a "${LOG}"
echo "STAGE2_IMAGE_SIZE=${STAGE2_IMAGE_SIZE} (cache + stage 2)" | tee -a "${LOG}"
echo "STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE}  STAGE1_LR=${STAGE1_LR}  STAGE1_DICT_LR=${STAGE1_DICT_LR}" | tee -a "${LOG}"
echo "STAGE1_RECON_MSE_WEIGHT=${STAGE1_RECON_MSE_WEIGHT}  STAGE1_RECON_L1_WEIGHT=${STAGE1_RECON_L1_WEIGHT}  STAGE1_RECON_EDGE_WEIGHT=${STAGE1_RECON_EDGE_WEIGHT}  STAGE1_PERCEPTUAL=${STAGE1_PERCEPTUAL_WEIGHT}@${STAGE1_PERCEPTUAL_START_STEP}+${STAGE1_PERCEPTUAL_WARMUP_STEPS}" | tee -a "${LOG}"
echo "STAGE1_NUM_HIDDENS=${STAGE1_NUM_HIDDENS}  STAGE1_NUM_RES_BLOCKS=${STAGE1_NUM_RES_BLOCKS}  STAGE1_NUM_RES_HIDDENS=${STAGE1_NUM_RES_HIDDENS}" | tee -a "${LOG}"
echo "STAGE1_DECODER_EXTRA_RESIDUAL_LAYERS=${STAGE1_DECODER_EXTRA_RESIDUAL_LAYERS}  STAGE1_ATTN_RESOLUTIONS=${STAGE1_ATTN_RESOLUTIONS}  STAGE1_DIAG_LOG_INTERVAL=${STAGE1_DIAG_LOG_INTERVAL}" | tee -a "${LOG}"
echo "BOTTLENECK_TYPE=${BOTTLENECK_TYPE}  BOTTLENECK_LOSS_WEIGHT=${BOTTLENECK_LOSS_WEIGHT}  BOTTLENECK_COMMITMENT_COST=${BOTTLENECK_COMMITMENT_COST}  OMP_RESIDUAL_TOLERANCE=${OMP_RESIDUAL_TOLERANCE}  RQ_CODE_DEPTH=${RQ_CODE_DEPTH}" | tee -a "${LOG}"
echo "BOTTLENECK_NUM_EMBEDDINGS=${BOTTLENECK_NUM_EMBEDDINGS}  BOTTLENECK_EMBEDDING_DIM=${BOTTLENECK_EMBEDDING_DIM}  PATCH=${PATCH_SIZE}/${PATCH_STRIDE}  SPARSITY_LEVEL=${SPARSITY_LEVEL}  BOTTLENECK_COEF_MAX=${BOTTLENECK_COEF_MAX}" | tee -a "${LOG}"
echo "STAGE2_AR_TYPE=${STAGE2_AR_TYPE}  STAGE2_BATCH_SIZE=${STAGE2_BATCH_SIZE}  STAGE2_LR=${STAGE2_LR}" | tee -a "${LOG}"
echo "STAGE2_COEFF_BINS=${STAGE2_COEFF_BINS}  STAGE2_COEFF_MAX=${STAGE2_COEFF_MAX}  STAGE2_COEFF_QUANTIZATION=${STAGE2_COEFF_QUANTIZATION}  STAGE2_COEFF_MU=${STAGE2_COEFF_MU}" | tee -a "${LOG}"
echo "STAGE2_SAMPLE_EVERY_N_EPOCHS=${STAGE2_SAMPLE_EVERY_N_EPOCHS}  STAGE2_SAMPLE_NUM_IMAGES=${STAGE2_SAMPLE_NUM_IMAGES}  STAGE2_SAMPLE_TOP_K=${STAGE2_SAMPLE_TOP_K}" | tee -a "${LOG}"

# Stage 1: autoencoder. VQGAN-style backbone with four downsampling stages.
"${PYTHON_BIN}" train.py stage1 \
  output_dir="${STAGE1_DIR}" \
  hydra.run.dir="${STAGE1_DIR}/hydra" \
  model=laser \
  data=celebahq \
  data.data_dir="${DATA_DIR}" \
  data.batch_size="${STAGE1_BATCH_SIZE}" \
  data.num_workers=4 \
  data.image_size="${IMAGE_SIZE_PRE}" \
  +data.train_crop_size="${IMAGE_SIZE}" \
  data.augment=true \
  train.accelerator=gpu \
  train.devices=2 \
  train.strategy=ddp \
  train.precision=32 \
  train.max_epochs="${STAGE1_EPOCHS}" \
  train.learning_rate="${STAGE1_LR}" \
  train.log_every_n_steps=20 \
  checkpoint.save_top_k=1 \
  model.backbone=vqgan \
  model.num_hiddens="${STAGE1_NUM_HIDDENS}" \
  model.num_downsamples=4 \
  "model.channel_multipliers=[1,2,4,4,4]" \
  model.max_ch_mult=4 \
  model.out_tanh=true \
  model.embedding_dim="${BOTTLENECK_EMBEDDING_DIM}" \
  model.num_embeddings="${BOTTLENECK_NUM_EMBEDDINGS}" \
  model.sparsity_level="${SPARSITY_LEVEL}" \
  model.bottleneck_loss_weight="${BOTTLENECK_LOSS_WEIGHT}" \
  model.commitment_cost="${BOTTLENECK_COMMITMENT_COST}" \
  model.coef_max="${BOTTLENECK_COEF_MAX}" \
  model.num_residual_blocks="${STAGE1_NUM_RES_BLOCKS}" \
  model.num_residual_hiddens="${STAGE1_NUM_RES_HIDDENS}" \
  model.decoder_extra_residual_layers="${STAGE1_DECODER_EXTRA_RESIDUAL_LAYERS}" \
  model.diag_log_interval="${STAGE1_DIAG_LOG_INTERVAL}" \
  model.patch_based=true \
  model.patch_size="${PATCH_SIZE}" \
  model.patch_stride="${PATCH_STRIDE}" \
  model.patch_reconstruction=tile \
  model.use_mid_attention=true \
  "model.attn_resolutions=${STAGE1_ATTN_RESOLUTIONS}" \
  +model.bottleneck_type="${BOTTLENECK_TYPE}" \
  +model.dictionary_through_decoder="${DICTIONARY_THROUGH_DECODER}" \
  +model.dead_atom_revival_steps="${DEAD_ATOM_REVIVAL_STEPS}" \
  +model.data_init_from_first_batch="${DATA_INIT_FROM_FIRST_BATCH}" \
  +model.rq_code_depth="${RQ_CODE_DEPTH}" \
  +model.rq_shared_codebook="${RQ_SHARED_CODEBOOK}" \
  +model.rq_decay="${RQ_DECAY}" \
  +model.rq_restart_unused_codes="${RQ_RESTART_UNUSED_CODES}" \
  model.dict_learning_rate="${STAGE1_DICT_LR}" \
  model.recon_mse_weight="${STAGE1_RECON_MSE_WEIGHT}" \
  model.recon_l1_weight="${STAGE1_RECON_L1_WEIGHT}" \
  model.recon_edge_weight="${STAGE1_RECON_EDGE_WEIGHT}" \
  model.perceptual_weight="${STAGE1_PERCEPTUAL_WEIGHT}" \
  model.perceptual_start_step="${STAGE1_PERCEPTUAL_START_STEP}" \
  model.perceptual_warmup_steps="${STAGE1_PERCEPTUAL_WARMUP_STEPS}" \
  model.compute_fid=false \
  model.log_images_every_n_steps=200 \
  wandb.name="s1_${STAMP}" \
  wandb.project=laser \
  2>&1 | tee -a "${LOG}"

CKPT="$(
  find "${STAGE1_DIR}/checkpoints" -type f -name "*.ckpt" ! -name "last.ckpt" -printf "%T@ %p\n" \
    | sort -nr | head -1 | cut -d" " -f2-
)"
[[ -z "${CKPT}" ]] && CKPT="$(find "${STAGE1_DIR}/checkpoints" -type f -name "last.ckpt" | head -1)"
[[ -z "${CKPT}" ]] && { echo "No stage1 checkpoint found" >&2; exit 1; }
echo "[$(date --iso-8601=seconds)] stage1 checkpoint: ${CKPT}" | tee -a "${LOG}"

# Cache tokens for stage 2.
"${PYTHON_BIN}" cache.py \
  --stage1-checkpoint "${CKPT}" \
  --output-path "${CACHE}" \
  --dataset celebahq \
  --data-dir "${DATA_DIR}" \
  --image-size "${STAGE2_IMAGE_SIZE}" \
  --batch-size 8 \
  --num-workers 4 \
  --seed 42 \
  --max-items "${MAX_ITEMS}" \
  --model-type laser \
  --coeff-bins "${STAGE2_COEFF_BINS}" \
  --coeff-max "${STAGE2_COEFF_MAX}" \
  --coeff-quantization "${STAGE2_COEFF_QUANTIZATION}" \
  --coeff-mu "${STAGE2_COEFF_MU}" \
  --device auto \
  2>&1 | tee -a "${LOG}"

# Stage 2: sparse-token prior over cached tokens.
"${PYTHON_BIN}" train.py stage2 \
  output_dir="${STAGE2_DIR}" \
  token_cache_path="${CACHE}" \
  data.dataset=celebahq \
  data.data_dir="${DATA_DIR}" \
  data.image_size="${STAGE2_IMAGE_SIZE}" \
  data.num_workers=4 \
  ar.type="${STAGE2_AR_TYPE}" \
  ar.learning_rate="${STAGE2_LR}" \
  train_ar.accelerator=gpu \
  train_ar.devices=2 \
  train_ar.strategy=ddp \
  train_ar.precision=32 \
  train_ar.max_epochs="${STAGE2_EPOCHS}" \
  train_ar.log_every_n_steps=20 \
  train_ar.batch_size="${STAGE2_BATCH_SIZE}" \
  train_ar.sample_every_n_epochs="${STAGE2_SAMPLE_EVERY_N_EPOCHS}" \
  train_ar.sample_num_images="${STAGE2_SAMPLE_NUM_IMAGES}" \
  train_ar.sample_top_k="${STAGE2_SAMPLE_TOP_K}" \
  train_ar.sample_log_to_wandb=true \
  train_ar.generation_metric_num_samples=0 \
  train_ar.save_final_samples_after_fit=false \
  wandb.name="s2_${STAMP}" \
  wandb.project=laser \
  2>&1 | tee -a "${LOG}"

echo "[$(date --iso-8601=seconds)] smoke run done -> ${RUN_ROOT}" | tee -a "${LOG}"
