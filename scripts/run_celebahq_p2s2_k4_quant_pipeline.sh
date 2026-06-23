#!/usr/bin/env bash
# CelebA-HQ p2s2/k4 quantized-token pipeline.
#
# This keeps the sharper d4 DDPM-style stage-1 setup, but uses non-overlapping
# 2x2 latent patches with k=4 sparse entries, half the p4s4 dictionary size,
# and 256 quantized coefficient bins. For a 16x16 latent grid, stage 2 models
# an 8x8x8 interleaved atom/coeff token grid.
set -euo pipefail

cd /home/xl598/Projects/laser

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/xl598/Projects/laser/runs/celebahq_p2s2_k4_quant_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba_hq}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_GROUP="${WANDB_GROUP:-celebahq_p2s2_k4_quant_${STAMP}}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2}"
CACHE="${CACHE:-}"

STAGE1_CKPT="${STAGE1_CKPT:-}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-3}"
STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_DICT_LR="${STAGE1_DICT_LR:-2.5e-4}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-500}"
STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-0.05}"
STAGE1_COMPUTE_FID="${STAGE1_COMPUTE_FID:-true}"
STAGE1_FID_FEATURE="${STAGE1_FID_FEATURE:-2048}"

NUM_HIDDENS="${NUM_HIDDENS:-128}"
NUM_RESIDUAL_BLOCKS="${NUM_RESIDUAL_BLOCKS:-3}"
NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-96}"
DECODER_EXTRA_RESIDUAL_LAYERS="${DECODER_EXTRA_RESIDUAL_LAYERS:-2}"
BACKBONE_LATENT_CHANNELS="${BACKBONE_LATENT_CHANNELS:-512}"
STAGE1_DOWNSAMPLES="${STAGE1_DOWNSAMPLES:-4}"
STAGE1_CHANNEL_MULTIPLIERS="${STAGE1_CHANNEL_MULTIPLIERS:-[1,1,2,2,4]}"
STAGE1_ATTN_RESOLUTIONS="${STAGE1_ATTN_RESOLUTIONS:-[16,32]}"

NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-4096}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
PATCH_BASED="${PATCH_BASED:-true}"
PATCH_SIZE="${PATCH_SIZE:-2}"
PATCH_STRIDE="${PATCH_STRIDE:-2}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-4}"
COEF_MAX="${COEF_MAX:-auto_p99.9}"
STAGE1_COEF_MAX="${STAGE1_COEF_MAX:-null}"
CACHE_COEFF_MAX="${CACHE_COEFF_MAX:-${COEF_MAX}}"
CACHE_COEFF_MAX_PADDING="${CACHE_COEFF_MAX_PADDING:-1.05}"
COEFF_BINS="${COEFF_BINS:-256}"
CACHE_COEFF_MODE="${CACHE_COEFF_MODE:-quantized}"
CACHE_COEFF_MODE="$(printf '%s' "${CACHE_COEFF_MODE}" | tr '[:upper:]' '[:lower:]')"
case "${CACHE_COEFF_MODE}" in
  quantized|quant|q)
    CACHE_COEFF_MODE="quantized"
    CACHE_COEFF_BINS="${CACHE_COEFF_BINS:-${COEFF_BINS}}"
    CACHE_SUFFIX="q${CACHE_COEFF_BINS}"
    ;;
  continuous|real|real_valued|none)
    CACHE_COEFF_MODE="continuous"
    CACHE_COEFF_BINS=0
    CACHE_SUFFIX="realcoeff"
    ;;
  *)
    echo "ERROR: CACHE_COEFF_MODE must be quantized or continuous, got ${CACHE_COEFF_MODE}" >&2
    exit 2
    ;;
esac
if ! [[ "${CACHE_COEFF_BINS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CACHE_COEFF_BINS must be a non-negative integer, got ${CACHE_COEFF_BINS}" >&2
  exit 2
fi
if [[ "${CACHE_COEFF_MODE}" == "quantized" && "${CACHE_COEFF_BINS}" -le 0 ]]; then
  echo "ERROR: quantized cache mode requires CACHE_COEFF_BINS > 0" >&2
  exit 2
fi
if [[ -z "${CACHE}" ]]; then
  CACHE="${RUN_ROOT}/token_cache_${CACHE_SUFFIX}.pt"
fi

RECON_MSE_WEIGHT="${RECON_MSE_WEIGHT:-0.25}"
RECON_L1_WEIGHT="${RECON_L1_WEIGHT:-1.0}"
RECON_EDGE_WEIGHT="${RECON_EDGE_WEIGHT:-0.50}"
BOTTLENECK_LOSS_WEIGHT="${BOTTLENECK_LOSS_WEIGHT:-0.25}"
COMMITMENT_COST="${COMMITMENT_COST:-0.05}"
ONLINE_KSVD_ENABLED="${ONLINE_KSVD_ENABLED:-false}"
ONLINE_KSVD_START_STEP="${ONLINE_KSVD_START_STEP:-0}"
ONLINE_KSVD_INTERVAL_STEPS="${ONLINE_KSVD_INTERVAL_STEPS:-0}"
ONLINE_KSVD_STOP_STEP="${ONLINE_KSVD_STOP_STEP:-null}"
ONLINE_KSVD_MAX_SAMPLES="${ONLINE_KSVD_MAX_SAMPLES:-512}"
ONLINE_KSVD_MAX_ATOMS="${ONLINE_KSVD_MAX_ATOMS:-256}"
ONLINE_KSVD_BLEND="${ONLINE_KSVD_BLEND:-0.25}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.20}"
PERCEPTUAL_START_STEP="${PERCEPTUAL_START_STEP:-1000}"
PERCEPTUAL_WARMUP_STEPS="${PERCEPTUAL_WARMUP_STEPS:-2000}"
ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.0}"
ADVERSARIAL_START_STEP="${ADVERSARIAL_START_STEP:-5000}"
ADVERSARIAL_WARMUP_STEPS="${ADVERSARIAL_WARMUP_STEPS:-5000}"
ADVERSARIAL_START_RECON_MSE="${ADVERSARIAL_START_RECON_MSE:-null}"
ADVERSARIAL_QUALITY_EMA_DECAY="${ADVERSARIAL_QUALITY_EMA_DECAY:-0.99}"
DISCRIMINATOR_LR="${DISCRIMINATOR_LR:-5.0e-5}"
DISCRIMINATOR_CHANNELS="${DISCRIMINATOR_CHANNELS:-64}"
DISCRIMINATOR_LAYERS="${DISCRIMINATOR_LAYERS:-3}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
CACHE_MAX_ITEMS="${CACHE_MAX_ITEMS:-0}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-50000}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
STAGE2_LR="${STAGE2_LR:-3.0e-4}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-500}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.05}"
STAGE2_D_MODEL="${STAGE2_D_MODEL:-512}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-8}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-8}"
STAGE2_D_FF="${STAGE2_D_FF:-2048}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"
STAGE2_SAMPLE_EVERY_N_EPOCHS="${STAGE2_SAMPLE_EVERY_N_EPOCHS:-2}"
STAGE2_SAMPLE_NUM_IMAGES="${STAGE2_SAMPLE_NUM_IMAGES:-8}"
STAGE2_SAMPLE_TEMPERATURE="${STAGE2_SAMPLE_TEMPERATURE:-0.7}"
STAGE2_SAMPLE_TOP_K="${STAGE2_SAMPLE_TOP_K:-0}"
STAGE2_SAMPLE_VARIANTS="${STAGE2_SAMPLE_VARIANTS:-}"
STAGE2_COMPUTE_GENERATION_FID="${STAGE2_COMPUTE_GENERATION_FID:-false}"
STAGE2_GENERATION_METRIC_NUM_SAMPLES="${STAGE2_GENERATION_METRIC_NUM_SAMPLES:-0}"
STAGE2_RUN_TEST_AFTER_FIT="${STAGE2_RUN_TEST_AFTER_FIT:-false}"
STAGE2_SAVE_FINAL_SAMPLES_AFTER_FIT="${STAGE2_SAVE_FINAL_SAMPLES_AFTER_FIT:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/home/xl598/Projects/laser
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LATENT_H=$(( IMAGE_SIZE / (2 ** STAGE1_DOWNSAMPLES) ))
PATCH_BASED_LOWER="$(printf '%s' "${PATCH_BASED}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PATCH_BASED_LOWER}" == "true" || "${PATCH_BASED_LOWER}" == "1" || "${PATCH_BASED_LOWER}" == "yes" ]]; then
  PRIOR_H=$(( ((LATENT_H - PATCH_SIZE) / PATCH_STRIDE) + 1 ))
  RUN_LABEL="p${PATCH_SIZE}s${PATCH_STRIDE}_k${SPARSITY_LEVEL}_a${NUM_EMBEDDINGS}_${CACHE_SUFFIX}"
else
  PRIOR_H="${LATENT_H}"
  RUN_LABEL="site_d${STAGE1_DOWNSAMPLES}_k${SPARSITY_LEVEL}_a${NUM_EMBEDDINGS}_${CACHE_SUFFIX}"
fi
PRIOR_W="${PRIOR_H}"
if (( CACHE_COEFF_BINS <= 0 )); then
  PRIOR_D="${SPARSITY_LEVEL}"
else
  PRIOR_D=$((SPARSITY_LEVEL * 2))
fi
STAGE2_VARIANT_ARGS=()
if [[ -n "${STAGE2_SAMPLE_VARIANTS}" ]]; then
  STAGE2_VARIANT_ARGS=("+train_ar.sample_variants=[${STAGE2_SAMPLE_VARIANTS}]")
fi

mkdir -p "${RUN_ROOT}/logs" "${STAGE1_DIR}" "${STAGE2_DIR}"

echo "[$(date --iso-8601=seconds)] ${RUN_LABEL} quant pipeline start"
echo "RUN_ROOT=${RUN_ROOT}"
echo "STAGE1_DIR=${STAGE1_DIR}"
echo "STAGE2_DIR=${STAGE2_DIR}"
echo "CACHE=${CACHE}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo "IMAGE_SIZE=${IMAGE_SIZE}  LATENT_H=${LATENT_H}  DOWNSAMPLES=${STAGE1_DOWNSAMPLES}  CHANNEL_MULTIPLIERS=${STAGE1_CHANNEL_MULTIPLIERS}"
echo "PATCH_BASED=${PATCH_BASED}  PATCH=${PATCH_SIZE}/${PATCH_STRIDE}  SPARSITY_LEVEL=${SPARSITY_LEVEL}  NUM_EMBEDDINGS=${NUM_EMBEDDINGS}"
echo "CACHE_COEFF_MODE=${CACHE_COEFF_MODE}  CACHE_COEFF_BINS=${CACHE_COEFF_BINS}  STAGE1_COEF_MAX=${STAGE1_COEF_MAX}  CACHE_COEFF_MAX=${CACHE_COEFF_MAX}  CACHE_COEFF_MAX_PADDING=${CACHE_COEFF_MAX_PADDING}"
echo "BOTTLENECK_LOSS_WEIGHT=${BOTTLENECK_LOSS_WEIGHT}  COMMITMENT_COST=${COMMITMENT_COST}"
echo "STAGE1_COMPUTE_FID=${STAGE1_COMPUTE_FID}  STAGE1_FID_FEATURE=${STAGE1_FID_FEATURE}"
echo "ADVERSARIAL_WEIGHT=${ADVERSARIAL_WEIGHT}  ADV_START_STEP=${ADVERSARIAL_START_STEP}  ADV_START_RECON_MSE=${ADVERSARIAL_START_RECON_MSE}"
echo "STAGE2_COMPUTE_GENERATION_FID=${STAGE2_COMPUTE_GENERATION_FID}  STAGE2_GENERATION_METRIC_NUM_SAMPLES=${STAGE2_GENERATION_METRIC_NUM_SAMPLES}  STAGE2_SAVE_FINAL_SAMPLES_AFTER_FIT=${STAGE2_SAVE_FINAL_SAMPLES_AFTER_FIT}"
echo "ONLINE_KSVD=${ONLINE_KSVD_ENABLED} start=${ONLINE_KSVD_START_STEP} interval=${ONLINE_KSVD_INTERVAL_STEPS} stop=${ONLINE_KSVD_STOP_STEP} max_samples=${ONLINE_KSVD_MAX_SAMPLES} max_atoms=${ONLINE_KSVD_MAX_ATOMS} blend=${ONLINE_KSVD_BLEND}"
echo "EXPECTED_STAGE2_GRID=${PRIOR_H}x${PRIOR_W}x${PRIOR_D}"
if [[ -n "${STAGE2_SAMPLE_VARIANTS}" ]]; then
  echo "STAGE2_SAMPLE_VARIANTS=${STAGE2_SAMPLE_VARIANTS}"
fi

if [[ -z "${STAGE1_CKPT}" ]]; then
  "${PYTHON_BIN}" train.py stage1 \
    output_dir="${STAGE1_DIR}" \
    hydra.run.dir="${STAGE1_DIR}/hydra" \
    model=laser \
    data=celebahq \
    data.data_dir="${DATA_DIR}" \
    data.image_size="${IMAGE_SIZE}" \
    data.batch_size="${STAGE1_BATCH_SIZE}" \
    data.num_workers=4 \
    data.augment=true \
    train.accelerator=gpu \
    train.devices=2 \
    train.strategy=ddp \
    train.precision=bf16-mixed \
    train.max_epochs="${STAGE1_EPOCHS}" \
    train.learning_rate="${STAGE1_LR}" \
    train.warmup_steps="${STAGE1_WARMUP_STEPS}" \
    train.min_lr_ratio="${STAGE1_MIN_LR_RATIO}" \
    train.gradient_clip_val=1.0 \
    train.log_every_n_steps=20 \
    checkpoint.save_top_k=1 \
    model.backbone=ddpm \
    model.num_hiddens="${NUM_HIDDENS}" \
    model.num_downsamples="${STAGE1_DOWNSAMPLES}" \
    "model.channel_multipliers=${STAGE1_CHANNEL_MULTIPLIERS}" \
    model.backbone_latent_channels="${BACKBONE_LATENT_CHANNELS}" \
    model.embedding_dim="${EMBEDDING_DIM}" \
    model.num_embeddings="${NUM_EMBEDDINGS}" \
    model.sparsity_level="${SPARSITY_LEVEL}" \
    model.patch_based="${PATCH_BASED}" \
    model.patch_size="${PATCH_SIZE}" \
    model.patch_stride="${PATCH_STRIDE}" \
    model.patch_reconstruction=tile \
    model.bottleneck_loss_weight="${BOTTLENECK_LOSS_WEIGHT}" \
    model.commitment_cost="${COMMITMENT_COST}" \
    model.online_ksvd_enabled="${ONLINE_KSVD_ENABLED}" \
    model.online_ksvd_start_step="${ONLINE_KSVD_START_STEP}" \
    model.online_ksvd_interval_steps="${ONLINE_KSVD_INTERVAL_STEPS}" \
    model.online_ksvd_stop_step="${ONLINE_KSVD_STOP_STEP}" \
    model.online_ksvd_max_samples="${ONLINE_KSVD_MAX_SAMPLES}" \
    model.online_ksvd_max_atoms="${ONLINE_KSVD_MAX_ATOMS}" \
    model.online_ksvd_blend="${ONLINE_KSVD_BLEND}" \
    model.coef_max="${STAGE1_COEF_MAX}" \
    model.dict_learning_rate="${STAGE1_DICT_LR}" \
    model.num_residual_blocks="${NUM_RESIDUAL_BLOCKS}" \
    model.num_residual_hiddens="${NUM_RESIDUAL_HIDDENS}" \
    model.decoder_extra_residual_layers="${DECODER_EXTRA_RESIDUAL_LAYERS}" \
    model.use_mid_attention=true \
    "model.attn_resolutions=${STAGE1_ATTN_RESOLUTIONS}" \
    +model.dictionary_through_decoder=true \
    +model.dead_atom_revival_steps=100 \
    +model.data_init_from_first_batch=true \
    model.recon_mse_weight="${RECON_MSE_WEIGHT}" \
    model.recon_l1_weight="${RECON_L1_WEIGHT}" \
    model.recon_edge_weight="${RECON_EDGE_WEIGHT}" \
    model.perceptual_weight="${PERCEPTUAL_WEIGHT}" \
    model.perceptual_start_step="${PERCEPTUAL_START_STEP}" \
    model.perceptual_warmup_steps="${PERCEPTUAL_WARMUP_STEPS}" \
    model.adversarial_weight="${ADVERSARIAL_WEIGHT}" \
    model.adversarial_start_step="${ADVERSARIAL_START_STEP}" \
    model.adversarial_warmup_steps="${ADVERSARIAL_WARMUP_STEPS}" \
    model.adversarial_start_recon_mse="${ADVERSARIAL_START_RECON_MSE}" \
    model.adversarial_quality_ema_decay="${ADVERSARIAL_QUALITY_EMA_DECAY}" \
    model.discriminator_learning_rate="${DISCRIMINATOR_LR}" \
    model.discriminator_channels="${DISCRIMINATOR_CHANNELS}" \
    model.discriminator_layers="${DISCRIMINATOR_LAYERS}" \
    model.compute_fid="${STAGE1_COMPUTE_FID}" \
    model.fid_feature="${STAGE1_FID_FEATURE}" \
    model.log_images_every_n_steps=200 \
    wandb.project="${WANDB_PROJECT}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.name="celebahq_s1_${RUN_LABEL}_lpips_${STAMP}"

  STAGE1_CKPT="$(
    find "${STAGE1_DIR}/checkpoints" -type f -name "final.ckpt" -printf "%T@ %p\n" \
      | sort -nr | head -1 | cut -d" " -f2-
  )"
  if [[ -z "${STAGE1_CKPT}" ]]; then
    STAGE1_CKPT="$(
      find "${STAGE1_DIR}/checkpoints" -type f -name "*.ckpt" ! -name "last.ckpt" -printf "%T@ %p\n" \
        | sort -nr | head -1 | cut -d" " -f2-
    )"
  fi
  if [[ -z "${STAGE1_CKPT}" ]]; then
    STAGE1_CKPT="$(find "${STAGE1_DIR}/checkpoints" -type f -name "last.ckpt" | head -1)"
  fi
fi

[[ -n "${STAGE1_CKPT}" ]] || { echo "No stage-1 checkpoint found" >&2; exit 1; }
echo "[$(date --iso-8601=seconds)] stage-1 checkpoint: ${STAGE1_CKPT}"

"${PYTHON_BIN}" cache.py \
  --stage1-checkpoint "${STAGE1_CKPT}" \
  --model-type laser \
  --output-path "${CACHE}" \
  --dataset celebahq \
  --data-dir "${DATA_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --batch-size "${CACHE_BATCH_SIZE}" \
  --num-workers "${CACHE_NUM_WORKERS}" \
  --seed 42 \
  --max-items "${CACHE_MAX_ITEMS}" \
  --coeff-bins "${CACHE_COEFF_BINS}" \
  --coeff-max "${CACHE_COEFF_MAX}" \
  --coeff-max-padding "${CACHE_COEFF_MAX_PADDING}" \
  --device auto

echo "[$(date --iso-8601=seconds)] token cache: ${CACHE}"

"${PYTHON_BIN}" train.py stage2 \
  output_dir="${STAGE2_DIR}" \
  hydra.run.dir="${STAGE2_DIR}/hydra" \
  token_cache_path="${CACHE}" \
  seed=42 \
  ar.type=sparse_spatial_depth \
  ar.d_model="${STAGE2_D_MODEL}" \
  ar.n_heads="${STAGE2_N_HEADS}" \
  ar.n_layers="${STAGE2_N_LAYERS}" \
  ar.d_ff="${STAGE2_D_FF}" \
  ar.learning_rate="${STAGE2_LR}" \
  ar.warmup_steps="${STAGE2_WARMUP_STEPS}" \
  ar.max_steps="${STAGE2_MAX_STEPS:-50000}" \
  ar.min_lr_ratio="${STAGE2_MIN_LR_RATIO}" \
  ar.coeff_loss_type=auto \
  train_ar.accelerator=gpu \
  train_ar.devices=2 \
  train_ar.strategy=ddp \
  train_ar.precision=bf16-mixed \
  train_ar.max_epochs="${STAGE2_EPOCHS}" \
  train_ar.batch_size="${STAGE2_BATCH_SIZE}" \
  train_ar.gradient_clip_val=1.0 \
  train_ar.log_every_n_steps=20 \
  train_ar.val_check_interval=1.0 \
  train_ar.sample_every_n_epochs="${STAGE2_SAMPLE_EVERY_N_EPOCHS}" \
  train_ar.sample_num_images="${STAGE2_SAMPLE_NUM_IMAGES}" \
  train_ar.sample_temperature="${STAGE2_SAMPLE_TEMPERATURE}" \
  train_ar.sample_top_k="${STAGE2_SAMPLE_TOP_K}" \
  "${STAGE2_VARIANT_ARGS[@]}" \
  train_ar.sample_log_to_wandb=true \
  train_ar.compute_generation_fid="${STAGE2_COMPUTE_GENERATION_FID}" \
  train_ar.generation_metric_num_samples="${STAGE2_GENERATION_METRIC_NUM_SAMPLES}" \
  train_ar.run_test_after_fit="${STAGE2_RUN_TEST_AFTER_FIT}" \
  train_ar.save_final_samples_after_fit="${STAGE2_SAVE_FINAL_SAMPLES_AFTER_FIT}" \
  data.dataset=celebahq \
  data.data_dir="${DATA_DIR}" \
  data.image_size="${IMAGE_SIZE}" \
  data.num_workers="${STAGE2_NUM_WORKERS}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="celebahq_s2_${RUN_LABEL}_${STAMP}" \
  wandb.save_dir="${STAGE2_DIR}/wandb"

echo "[$(date --iso-8601=seconds)] ${RUN_LABEL} quant pipeline done: ${RUN_ROOT}"
