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

# Train: random-crop 128 out of full 256 (data augmentation + 4x faster per iter).
# Val: full 256 (no crop on eval transform), so reconstruction quality is reported
# against the original resolution.
IMAGE_SIZE_PRE="${IMAGE_SIZE_PRE:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"

# Cache + stage 2 use full 256 (no random crop). The encoder is convolutional so
# it handles 256 just fine despite being trained at 128 crops; latent grid simply
# becomes 32x32 instead of 16x16 and the per-image token sequence grows 4x.
STAGE2_IMAGE_SIZE="${STAGE2_IMAGE_SIZE:-256}"

# 4x batch size now that input is 4x smaller spatially. LR scaled 2x with batch
# (linear-scaling rule, conservative since LPIPS+OMP can destabilize).
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE1_DICT_LR="${STAGE1_DICT_LR:-1e-4}"

# Bottleneck: "dictionary" (OMP) or "rq" (kakaobrain residual quantization).
BOTTLENECK_TYPE="${BOTTLENECK_TYPE:-dictionary}"

# OMP-only knobs (ignored when BOTTLENECK_TYPE=rq).
OMP_RESIDUAL_TOLERANCE="${OMP_RESIDUAL_TOLERANCE:-null}"
DEAD_ATOM_REVIVAL_STEPS="${DEAD_ATOM_REVIVAL_STEPS:-100}"
DICTIONARY_THROUGH_DECODER="${DICTIONARY_THROUGH_DECODER:-false}"
DATA_INIT_FROM_FIRST_BATCH="${DATA_INIT_FROM_FIRST_BATCH:-false}"

# RQ-only knobs (ignored when BOTTLENECK_TYPE=dictionary).
RQ_CODE_DEPTH="${RQ_CODE_DEPTH:-4}"
RQ_SHARED_CODEBOOK="${RQ_SHARED_CODEBOOK:-true}"
RQ_DECAY="${RQ_DECAY:-0.99}"
RQ_RESTART_UNUSED_CODES="${RQ_RESTART_UNUSED_CODES:-true}"

# Bottleneck sparsity (OMP: atoms per latent site; RQ: ignored, depth controls it).
SPARSITY_LEVEL="${SPARSITY_LEVEL:-8}"

STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE2_LR="${STAGE2_LR:-5e-4}"
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
echo "BOTTLENECK_TYPE=${BOTTLENECK_TYPE}  OMP_RESIDUAL_TOLERANCE=${OMP_RESIDUAL_TOLERANCE}  RQ_CODE_DEPTH=${RQ_CODE_DEPTH}" | tee -a "${LOG}"
echo "STAGE2_BATCH_SIZE=${STAGE2_BATCH_SIZE}  STAGE2_LR=${STAGE2_LR}" | tee -a "${LOG}"
echo "STAGE2_SAMPLE_EVERY_N_EPOCHS=${STAGE2_SAMPLE_EVERY_N_EPOCHS}  STAGE2_SAMPLE_NUM_IMAGES=${STAGE2_SAMPLE_NUM_IMAGES}  STAGE2_SAMPLE_TOP_K=${STAGE2_SAMPLE_TOP_K}" | tee -a "${LOG}"

# Stage 1: autoencoder. RQ-VAE-style backbone (vqgan), 5 downsamples,
# channel multipliers [1,1,2,2,4,4] (len = num_downsamples + 1).
"${PYTHON_BIN}" train_stage1_autoencoder.py \
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
  model.num_hiddens=128 \
  model.num_downsamples=4 \
  "model.channel_multipliers=[1,2,4,4,4]" \
  model.max_ch_mult=4 \
  model.embedding_dim=32 \
  model.num_embeddings=8192 \
  model.sparsity_level="${SPARSITY_LEVEL}" \
  model.num_residual_blocks=3 \
  model.num_residual_hiddens=128 \
  model.patch_based=true \
  model.patch_size=4 \
  model.patch_stride=4 \
  model.patch_reconstruction=tile \
  model.use_mid_attention=true \
  "model.attn_resolutions=[]" \
  +model.bottleneck_type="${BOTTLENECK_TYPE}" \
  +model.dictionary_through_decoder="${DICTIONARY_THROUGH_DECODER}" \
  +model.omp_residual_tolerance="${OMP_RESIDUAL_TOLERANCE}" \
  +model.dead_atom_revival_steps="${DEAD_ATOM_REVIVAL_STEPS}" \
  +model.data_init_from_first_batch="${DATA_INIT_FROM_FIRST_BATCH}" \
  +model.rq_code_depth="${RQ_CODE_DEPTH}" \
  +model.rq_shared_codebook="${RQ_SHARED_CODEBOOK}" \
  +model.rq_decay="${RQ_DECAY}" \
  +model.rq_restart_unused_codes="${RQ_RESTART_UNUSED_CODES}" \
  model.dict_learning_rate="${STAGE1_DICT_LR}" \
  model.perceptual_weight=1.0 \
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
  --coeff-bins 0 \
  --coeff-max 16.0 \
  --device auto \
  2>&1 | tee -a "${LOG}"

# Stage 2: sparse-token prior over cached tokens.
"${PYTHON_BIN}" train_stage2_prior.py \
  output_dir="${STAGE2_DIR}" \
  token_cache_path="${CACHE}" \
  data.dataset=celebahq \
  data.data_dir="${DATA_DIR}" \
  data.image_size="${STAGE2_IMAGE_SIZE}" \
  data.num_workers=4 \
  ar.type=sparse_spatial_depth \
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
