#!/usr/bin/env bash
# Local VCTK raw-waveform LASER two-stage pipeline.
#
# Stage 1 uses the waveform encoder/decoder in src.models.audio_codec plus a
# delayed HiFi-GAN-style MPD+MSD waveform discriminator. The 256x downsampling and k=4
# sparse depth keep stage-2 sequence length manageable:
# 32768 samples -> 128 latent sites -> 128 * (atom,coeff) * 4 = 1024 tokens.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/vctk_waveform_laser_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
VCTK_DIR="${VCTK_DIR:-/home/xl598/Projects/data/vctk}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_GROUP="${WANDB_GROUP:-vctk_waveform_laser_${STAMP}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WAIT_FOR_PGID="${WAIT_FOR_PGID:-}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"

STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2}"
CACHE="${CACHE:-${RUN_ROOT}/token_cache_q512.pt}"

STAGE1_CKPT="${STAGE1_CKPT:-}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-8}"
STAGE1_LR="${STAGE1_LR:-4.0e-4}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-750}"
STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-0.05}"

AUDIO_NUM_SAMPLES="${AUDIO_NUM_SAMPLES:-32768}"
AUDIO_DOWNSAMPLE_RATES="${AUDIO_DOWNSAMPLE_RATES:-[8,8,4]}"
NUM_HIDDENS="${NUM_HIDDENS:-160}"
NUM_RESIDUAL_BLOCKS="${NUM_RESIDUAL_BLOCKS:-3}"
NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-80}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-2048}"
EMBEDDING_DIM="${EMBEDDING_DIM:-96}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-4}"
COEF_MAX="${COEF_MAX:-12.0}"
COEFF_BINS="${COEFF_BINS:-512}"
ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.02}"
ADVERSARIAL_START_STEP="${ADVERSARIAL_START_STEP:-20000}"
ADVERSARIAL_WARMUP_STEPS="${ADVERSARIAL_WARMUP_STEPS:-10000}"
ADVERSARIAL_START_RECON_MSE="${ADVERSARIAL_START_RECON_MSE:-0.006}"
ADVERSARIAL_QUALITY_EMA_DECAY="${ADVERSARIAL_QUALITY_EMA_DECAY:-0.99}"
DISCRIMINATOR_LR="${DISCRIMINATOR_LR:-1.0e-4}"
DISCRIMINATOR_CHANNELS="${DISCRIMINATOR_CHANNELS:-32}"
DISCRIMINATOR_LAYERS="${DISCRIMINATOR_LAYERS:-3}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
CACHE_MAX_ITEMS="${CACHE_MAX_ITEMS:-0}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
STAGE2_LR="${STAGE2_LR:-4.0e-4}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-750}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.05}"
STAGE2_D_MODEL="${STAGE2_D_MODEL:-384}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-6}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-8}"
STAGE2_D_FF="${STAGE2_D_FF:-1536}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-2}"
STAGE2_SAMPLE_EVERY_N_EPOCHS="${STAGE2_SAMPLE_EVERY_N_EPOCHS:-5}"

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PROJECT_DIR}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${WAIT_FOR_PGID}" ]]; then
  echo "[$(date --iso-8601=seconds)] waiting for process group ${WAIT_FOR_PGID}"
  while kill -0 "-${WAIT_FOR_PGID}" 2>/dev/null; do
    sleep "${WAIT_INTERVAL_SECONDS}"
  done
  echo "[$(date --iso-8601=seconds)] process group ${WAIT_FOR_PGID} exited"
fi

[[ -d "${VCTK_DIR}" ]] || { echo "VCTK directory not found: ${VCTK_DIR}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}/logs" "${STAGE1_DIR}" "${STAGE2_DIR}"

audio_downsample_factor() {
  local raw="${AUDIO_DOWNSAMPLE_RATES//[\[\] ]/}"
  local factor=1
  local rate
  IFS=',' read -ra rates <<<"${raw}"
  for rate in "${rates[@]}"; do
    [[ -n "${rate}" ]] || continue
    factor=$(( factor * rate ))
  done
  echo "${factor}"
}

DOWNSAMPLE_FACTOR="$(audio_downsample_factor)"
LATENT_T=$(( AUDIO_NUM_SAMPLES / DOWNSAMPLE_FACTOR ))
EXPECTED_TOKENS=$(( LATENT_T * SPARSITY_LEVEL * 2 ))

echo "[$(date --iso-8601=seconds)] VCTK waveform LASER pipeline start"
echo "RUN_ROOT=${RUN_ROOT}"
echo "VCTK_DIR=${VCTK_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "STAGE1=epochs=${STAGE1_EPOCHS} batch=${STAGE1_BATCH_SIZE} lr=${STAGE1_LR} rates=${AUDIO_DOWNSAMPLE_RATES}"
echo "BOTTLENECK=atoms=${NUM_EMBEDDINGS} dim=${EMBEDDING_DIM} sparsity=${SPARSITY_LEVEL} coeff_bins=${COEFF_BINS} coeff_max=${COEF_MAX}"
echo "ADVERSARIAL=weight=${ADVERSARIAL_WEIGHT} start=${ADVERSARIAL_START_STEP} warmup=${ADVERSARIAL_WARMUP_STEPS} mse_gate=${ADVERSARIAL_START_RECON_MSE} disc_channels=${DISCRIMINATOR_CHANNELS} disc_layers=${DISCRIMINATOR_LAYERS}"
echo "EXPECTED_STAGE2_GRID=1x${LATENT_T}x$((SPARSITY_LEVEL * 2)) flat_tokens=${EXPECTED_TOKENS}"
echo "STAGE2=d_model=${STAGE2_D_MODEL} heads=${STAGE2_N_HEADS} layers=${STAGE2_N_LAYERS} batch=${STAGE2_BATCH_SIZE} epochs=${STAGE2_EPOCHS}"

if [[ -z "${STAGE1_CKPT}" ]]; then
  "${PYTHON_BIN}" train_stage1_autoencoder.py \
    output_dir="${STAGE1_DIR}" \
    hydra.run.dir="${STAGE1_DIR}/hydra" \
    model=laser_audio_waveform \
    data=vctk_waveform \
    data.data_dir="${VCTK_DIR}" \
    data.batch_size="${STAGE1_BATCH_SIZE}" \
    data.num_workers=4 \
    data.audio_num_samples="${AUDIO_NUM_SAMPLES}" \
    train.accelerator=gpu \
    train.devices=1 \
    train.strategy=auto \
    train.precision=bf16-mixed \
    train.max_epochs="${STAGE1_EPOCHS}" \
    train.learning_rate="${STAGE1_LR}" \
    train.warmup_steps="${STAGE1_WARMUP_STEPS}" \
    train.min_lr_ratio="${STAGE1_MIN_LR_RATIO}" \
    train.gradient_clip_val=1.0 \
    train.log_every_n_steps=50 \
    train.run_test_after_fit=false \
    model.audio_downsample_rates="${AUDIO_DOWNSAMPLE_RATES}" \
    model.num_hiddens="${NUM_HIDDENS}" \
    model.num_residual_blocks="${NUM_RESIDUAL_BLOCKS}" \
    model.num_residual_hiddens="${NUM_RESIDUAL_HIDDENS}" \
    model.num_embeddings="${NUM_EMBEDDINGS}" \
    model.embedding_dim="${EMBEDDING_DIM}" \
    model.sparsity_level="${SPARSITY_LEVEL}" \
    model.coef_max="${COEF_MAX}" \
    model.compute_fid=false \
    model.perceptual_weight=0.0 \
    model.adversarial_weight="${ADVERSARIAL_WEIGHT}" \
    model.adversarial_start_step="${ADVERSARIAL_START_STEP}" \
    model.adversarial_warmup_steps="${ADVERSARIAL_WARMUP_STEPS}" \
    model.adversarial_start_recon_mse="${ADVERSARIAL_START_RECON_MSE}" \
    model.adversarial_quality_ema_decay="${ADVERSARIAL_QUALITY_EMA_DECAY}" \
    model.discriminator_learning_rate="${DISCRIMINATOR_LR}" \
    model.discriminator_channels="${DISCRIMINATOR_CHANNELS}" \
    model.discriminator_layers="${DISCRIMINATOR_LAYERS}" \
    model.audio_waveform_l1_weight=1.0 \
    model.audio_multires_stft_loss_weight=1.0 \
    model.audio_multires_stft_fft_sizes=[512,1024,2048] \
    wandb.project="${WANDB_PROJECT}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.name="vctk_s1_laser_waveform_${STAMP}" \
    wandb.save_dir="${STAGE1_DIR}/wandb"

  STAGE1_CKPT="$(find "${STAGE1_DIR}/checkpoints" -type f -name "final.ckpt" -print -quit 2>/dev/null || true)"
  if [[ -z "${STAGE1_CKPT}" ]]; then
    STAGE1_CKPT="$(find "${STAGE1_DIR}/checkpoints" -type f -name "*.ckpt" ! -name "last.ckpt" -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "${STAGE1_CKPT}" ]]; then
    STAGE1_CKPT="$(find "${STAGE1_DIR}/checkpoints" -type f -name "last.ckpt" -print -quit 2>/dev/null || true)"
  fi
fi

[[ -n "${STAGE1_CKPT}" ]] || { echo "No stage-1 checkpoint found" >&2; exit 1; }
echo "[$(date --iso-8601=seconds)] stage-1 checkpoint: ${STAGE1_CKPT}"

"${PYTHON_BIN}" cache.py \
  --stage1-checkpoint "${STAGE1_CKPT}" \
  --model-type laser \
  --output-path "${CACHE}" \
  --dataset vctk \
  --data-dir "${VCTK_DIR}" \
  --image-size 128 \
  --audio-num-samples "${AUDIO_NUM_SAMPLES}" \
  --batch-size "${CACHE_BATCH_SIZE}" \
  --num-workers "${CACHE_NUM_WORKERS}" \
  --seed 42 \
  --max-items "${CACHE_MAX_ITEMS}" \
  --audio-representation waveform \
  --audio-dc-remove \
  --audio-peak-normalize \
  --audio-target-peak 0.95 \
  --audio-rms-normalize \
  --audio-target-rms 0.12 \
  --audio-max-gain 8.0 \
  --audio-min-crop-rms 0.03 \
  --audio-crop-attempts 64 \
  --audio-fade-samples 1024 \
  --coeff-bins "${COEFF_BINS}" \
  --coeff-max "${COEF_MAX}" \
  --coeff-quantization uniform \
  --device auto

echo "[$(date --iso-8601=seconds)] token cache: ${CACHE}"

"${PYTHON_BIN}" train_stage2_prior.py \
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
  ar.max_steps=-1 \
  ar.min_lr_ratio="${STAGE2_MIN_LR_RATIO}" \
  ar.coeff_loss_type=huber \
  ar.coeff_loss_weight=1.0 \
  ar.coeff_huber_delta=1.0 \
  train_ar.accelerator=gpu \
  train_ar.devices=1 \
  train_ar.strategy=auto \
  train_ar.precision=bf16-mixed \
  train_ar.max_epochs="${STAGE2_EPOCHS}" \
  train_ar.batch_size="${STAGE2_BATCH_SIZE}" \
  train_ar.gradient_clip_val=1.0 \
  train_ar.log_every_n_steps=50 \
  train_ar.sample_every_n_epochs="${STAGE2_SAMPLE_EVERY_N_EPOCHS}" \
  train_ar.sample_log_to_wandb=true \
  train_ar.sample_num_images=8 \
  train_ar.compute_generation_fid=false \
  train_ar.compute_audio_generation_metrics=true \
  train_ar.generation_metric_num_samples=8 \
  train_ar.sample_temperature=0.8 \
  train_ar.sample_top_k=0 \
  train_ar.sample_coeff_mode=mean \
  train_ar.run_test_after_fit=false \
  train_ar.save_final_samples_after_fit=false \
  data.dataset=vctk \
  data.data_dir="${VCTK_DIR}" \
  data.num_workers="${STAGE2_NUM_WORKERS}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="vctk_s2_laser_waveform_${STAMP}" \
  wandb.save_dir="${STAGE2_DIR}/wandb"

echo "[$(date --iso-8601=seconds)] VCTK waveform LASER pipeline done: ${RUN_ROOT}"
