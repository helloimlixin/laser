#!/usr/bin/env bash
# Submit compact LASER diffusion-prior jobs for both modalities:
#   CelebA-HQ image and VCTK waveform audio.
#
# Diffusion uses real-valued sparse coefficients, so these jobs extract caches
# with --coeff-bins 0. That is intentionally separate from transformer jobs,
# which generally use quantized coefficient tokens.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser-bimodal-diffusion}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/bimodal_short_adv_diffusion}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018,gpuk[005-018]}"
CASES="${CASES:-celebahq,vctk}"

CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/Projects/data/celeba_hq}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/Projects/data/VCTK-Corpus}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE1_ADV_EPOCHS="${STAGE1_ADV_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"

COEF_MAX="${COEF_MAX:-16.0}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-750}"
STAGE2_LR="${STAGE2_LR:-2.0e-4}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
DIFFUSION_TIMESTEPS="${DIFFUSION_TIMESTEPS:-1000}"
DIFFUSION_SAMPLE_STEPS="${DIFFUSION_SAMPLE_STEPS:-100}"
DIFFUSION_SAMPLE_NUM="${DIFFUSION_SAMPLE_NUM:-8}"
DIFFUSION_STATS_ITEMS="${DIFFUSION_STATS_ITEMS:-8192}"
DIFFUSION_SUPPORT_BANK_SIZE="${DIFFUSION_SUPPORT_BANK_SIZE:-512}"

IMAGE_GPUS="${IMAGE_GPUS:-2}"
IMAGE_CPUS_PER_TASK="${IMAGE_CPUS_PER_TASK:-12}"
IMAGE_MEM_MB="${IMAGE_MEM_MB:-240000}"
IMAGE_STAGE1_BATCH_SIZE="${IMAGE_STAGE1_BATCH_SIZE:-3}"
IMAGE_DIFFUSION_BATCH_SIZE="${IMAGE_DIFFUSION_BATCH_SIZE:-64}"
IMAGE_DIFFUSION_HIDDEN="${IMAGE_DIFFUSION_HIDDEN:-128}"
IMAGE_DIFFUSION_RES_BLOCKS="${IMAGE_DIFFUSION_RES_BLOCKS:-4}"
IMAGE_NUM_WORKERS="${IMAGE_NUM_WORKERS:-4}"
IMAGE_STAGE1_LR="${IMAGE_STAGE1_LR:-1.0e-4}"
IMAGE_DICT_LR="${IMAGE_DICT_LR:-2.5e-4}"
IMAGE_NUM_EMBEDDINGS="${IMAGE_NUM_EMBEDDINGS:-2048}"
IMAGE_EMBEDDING_DIM="${IMAGE_EMBEDDING_DIM:-128}"
IMAGE_SPARSITY_LEVEL="${IMAGE_SPARSITY_LEVEL:-4}"
IMAGE_PATCH_SIZE="${IMAGE_PATCH_SIZE:-4}"
IMAGE_PATCH_STRIDE="${IMAGE_PATCH_STRIDE:-4}"
IMAGE_ADV_WEIGHT="${IMAGE_ADV_WEIGHT:-0.05}"
IMAGE_DISC_LR="${IMAGE_DISC_LR:-5.0e-5}"

VCTK_GPUS="${VCTK_GPUS:-2}"
VCTK_CPUS_PER_TASK="${VCTK_CPUS_PER_TASK:-12}"
VCTK_MEM_MB="${VCTK_MEM_MB:-240000}"
VCTK_STAGE1_BATCH_SIZE="${VCTK_STAGE1_BATCH_SIZE:-4}"
VCTK_DIFFUSION_BATCH_SIZE="${VCTK_DIFFUSION_BATCH_SIZE:-8}"
VCTK_DIFFUSION_HIDDEN="${VCTK_DIFFUSION_HIDDEN:-128}"
VCTK_DIFFUSION_RES_BLOCKS="${VCTK_DIFFUSION_RES_BLOCKS:-4}"
VCTK_NUM_WORKERS="${VCTK_NUM_WORKERS:-4}"
VCTK_STAGE1_LR="${VCTK_STAGE1_LR:-1.5e-4}"
VCTK_DICT_LR="${VCTK_DICT_LR:-1.5e-4}"
VCTK_DOWNSAMPLE_RATES="${VCTK_DOWNSAMPLE_RATES:-[8,8]}"
VCTK_NUM_EMBEDDINGS="${VCTK_NUM_EMBEDDINGS:-8192}"
VCTK_EMBEDDING_DIM="${VCTK_EMBEDDING_DIM:-96}"
VCTK_NUM_HIDDENS="${VCTK_NUM_HIDDENS:-192}"
VCTK_NUM_RESIDUAL_HIDDENS="${VCTK_NUM_RESIDUAL_HIDDENS:-96}"
VCTK_SPARSITY_LEVEL="${VCTK_SPARSITY_LEVEL:-8}"
VCTK_ADV_WEIGHT="${VCTK_ADV_WEIGHT:-0.03}"
VCTK_DISC_LR="${VCTK_DISC_LR:-5.0e-5}"
VCTK_DISC_CHANNELS="${VCTK_DISC_CHANNELS:-32}"
VCTK_DISC_LAYERS="${VCTK_DISC_LAYERS:-3}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

case_enabled() {
  local wanted="$1"
  local raw=",${CASES// /},"
  [[ "$raw" == *",$wanted,"* ]]
}

COMMON_ARGS=(
  --full-training
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage1-adv-epochs "$STAGE1_ADV_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
  --stage2-kind diffusion
  --partition "$PARTITION"
  --time-limit "$TIME_LIMIT"
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
  --celebahq-dir "$CELEBAHQ_DIR"
  --vctk-dir "$VCTK_DIR"
)
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  COMMON_ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

REAL_COEFF_CACHE_ARGS=(
  --cache-arg=--coeff-bins
  --cache-arg=0
  --cache-arg=--coeff-max
  --cache-arg="$COEF_MAX"
)

COMMON_DIFFUSION_ARGS=(
  --diffusion-arg=--learning-rate
  --diffusion-arg="$STAGE2_LR"
  --diffusion-arg=--num-timesteps
  --diffusion-arg="$DIFFUSION_TIMESTEPS"
  --diffusion-arg=--sample-steps
  --diffusion-arg="$DIFFUSION_SAMPLE_STEPS"
  --diffusion-arg=--sample-num-images
  --diffusion-arg="$DIFFUSION_SAMPLE_NUM"
  --diffusion-arg=--stats-items
  --diffusion-arg="$DIFFUSION_STATS_ITEMS"
  --diffusion-arg=--support-bank-size
  --diffusion-arg="$DIFFUSION_SUPPORT_BANK_SIZE"
)

submit_celebahq_image() {
  if [[ ! -d "$CELEBAHQ_DIR" ]]; then
    echo "CelebA-HQ directory not found: $CELEBAHQ_DIR" >&2
    exit 1
  fi

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --gpus "$IMAGE_GPUS" \
    --cpus-per-task "$IMAGE_CPUS_PER_TASK" \
    --mem-mb "$IMAGE_MEM_MB" \
    --cases celebahq \
    --model-family laser \
    --run-label "celebahq-shortseq-adv-diffusion-p${IMAGE_PATCH_SIZE}s${IMAGE_PATCH_STRIDE}k${IMAGE_SPARSITY_LEVEL}-a${IMAGE_NUM_EMBEDDINGS}" \
    "${REAL_COEFF_CACHE_ARGS[@]}" \
    --cache-arg=--batch-size \
    --cache-arg="$IMAGE_STAGE1_BATCH_SIZE" \
    "${COMMON_DIFFUSION_ARGS[@]}" \
    --diffusion-arg=--batch-size \
    --diffusion-arg="$IMAGE_DIFFUSION_BATCH_SIZE" \
    --diffusion-arg=--hidden-channels \
    --diffusion-arg="$IMAGE_DIFFUSION_HIDDEN" \
    --diffusion-arg=--time-embed-dim \
    --diffusion-arg="$IMAGE_DIFFUSION_HIDDEN" \
    --diffusion-arg=--n-res-blocks \
    --diffusion-arg="$IMAGE_DIFFUSION_RES_BLOCKS" \
    --stage1-override model=laser \
    --stage1-override data=celebahq \
    --stage1-override data.data_dir="$CELEBAHQ_DIR" \
    --stage1-override data.image_size=256 \
    --stage1-override data.batch_size="$IMAGE_STAGE1_BATCH_SIZE" \
    --stage1-override data.num_workers="$IMAGE_NUM_WORKERS" \
    --stage1-override data.augment=true \
    --stage1-override train.learning_rate="$IMAGE_STAGE1_LR" \
    --stage1-override train.warmup_steps="$STAGE1_WARMUP_STEPS" \
    --stage1-override train.min_lr_ratio="$MIN_LR_RATIO" \
    --stage1-override train.gradient_clip_val=1.0 \
    --stage1-override train.val_check_interval=1.0 \
    --stage1-override train.limit_train_batches=1.0 \
    --stage1-override train.limit_val_batches=1.0 \
    --stage1-override train.limit_test_batches=1.0 \
    --stage1-override train.run_test_after_fit=false \
    --stage1-override model.backbone=vqgan \
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4] \
    --stage1-override model.backbone_latent_channels=512 \
    --stage1-override model.max_ch_mult=4 \
    --stage1-override model.embedding_dim="$IMAGE_EMBEDDING_DIM" \
    --stage1-override model.num_embeddings="$IMAGE_NUM_EMBEDDINGS" \
    --stage1-override model.sparsity_level="$IMAGE_SPARSITY_LEVEL" \
    --stage1-override model.patch_based=true \
    --stage1-override model.patch_size="$IMAGE_PATCH_SIZE" \
    --stage1-override model.patch_stride="$IMAGE_PATCH_STRIDE" \
    --stage1-override model.patch_reconstruction=tile \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.decoder_extra_residual_layers=2 \
    --stage1-override model.bottleneck_loss_weight=0.35 \
    --stage1-override model.commitment_cost=0.20 \
    --stage1-override model.dict_learning_rate="$IMAGE_DICT_LR" \
    --stage1-override model.coef_max="$COEF_MAX" \
    --stage1-override model.recon_mse_weight=0.25 \
    --stage1-override model.recon_l1_weight=1.0 \
    --stage1-override model.recon_edge_weight=0.5 \
    --stage1-override model.perceptual_weight=0.1 \
    --stage1-override model.compute_fid=false \
    --stage1-override model.out_tanh=true \
    --stage1-override model.adversarial_weight=0.0 \
    --stage1-adv-override model.adversarial_weight="$IMAGE_ADV_WEIGHT" \
    --stage1-adv-override model.adversarial_start_step=0 \
    --stage1-adv-override model.adversarial_warmup_steps=0 \
    --stage1-adv-override model.disc_start_step=0 \
    --stage1-adv-override model.disc_learning_rate="$IMAGE_DISC_LR" \
    --stage1-adv-override model.disc_channels=64 \
    --stage1-adv-override model.disc_num_layers=3 \
    --stage1-adv-override model.disc_norm=group \
    --stage1-adv-override model.disc_loss=hinge
}

submit_vctk_audio() {
  if [[ ! -d "$VCTK_DIR" ]]; then
    echo "VCTK directory not found: $VCTK_DIR" >&2
    exit 1
  fi

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --gpus "$VCTK_GPUS" \
    --cpus-per-task "$VCTK_CPUS_PER_TASK" \
    --mem-mb "$VCTK_MEM_MB" \
    --cases vctk \
    --model-family laser \
    --run-label "vctk-shortseq-adv-diffusion-ds64-a${VCTK_NUM_EMBEDDINGS}-d${VCTK_EMBEDDING_DIM}-s${VCTK_SPARSITY_LEVEL}" \
    "${REAL_COEFF_CACHE_ARGS[@]}" \
    --cache-arg=--batch-size \
    --cache-arg="$VCTK_STAGE1_BATCH_SIZE" \
    --cache-arg=--audio-representation \
    --cache-arg=waveform \
    --cache-arg=--audio-dc-remove \
    --cache-arg=--audio-peak-normalize \
    --cache-arg=--audio-target-peak \
    --cache-arg=0.95 \
    --cache-arg=--audio-rms-normalize \
    --cache-arg=--audio-target-rms \
    --cache-arg=0.12 \
    --cache-arg=--audio-max-gain \
    --cache-arg=8.0 \
    --cache-arg=--audio-min-crop-rms \
    --cache-arg=0.03 \
    --cache-arg=--audio-crop-attempts \
    --cache-arg=64 \
    --cache-arg=--audio-fade-samples \
    --cache-arg=1024 \
    "${COMMON_DIFFUSION_ARGS[@]}" \
    --diffusion-arg=--batch-size \
    --diffusion-arg="$VCTK_DIFFUSION_BATCH_SIZE" \
    --diffusion-arg=--hidden-channels \
    --diffusion-arg="$VCTK_DIFFUSION_HIDDEN" \
    --diffusion-arg=--time-embed-dim \
    --diffusion-arg="$VCTK_DIFFUSION_HIDDEN" \
    --diffusion-arg=--n-res-blocks \
    --diffusion-arg="$VCTK_DIFFUSION_RES_BLOCKS" \
    --stage1-override model=laser_audio_waveform \
    --stage1-override data=vctk_waveform \
    --stage1-override data.data_dir="$VCTK_DIR" \
    --stage1-override data.batch_size="$VCTK_STAGE1_BATCH_SIZE" \
    --stage1-override data.num_workers="$VCTK_NUM_WORKERS" \
    --stage1-override model.compute_fid=false \
    --stage1-override model.audio_downsample_rates="$VCTK_DOWNSAMPLE_RATES" \
    --stage1-override model.num_embeddings="$VCTK_NUM_EMBEDDINGS" \
    --stage1-override model.embedding_dim="$VCTK_EMBEDDING_DIM" \
    --stage1-override model.num_hiddens="$VCTK_NUM_HIDDENS" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens="$VCTK_NUM_RESIDUAL_HIDDENS" \
    --stage1-override model.sparsity_level="$VCTK_SPARSITY_LEVEL" \
    --stage1-override model.commitment_cost=1.0 \
    --stage1-override model.bottleneck_loss_weight=0.75 \
    --stage1-override model.dict_learning_rate="$VCTK_DICT_LR" \
    --stage1-override model.coef_max="$COEF_MAX" \
    --stage1-override model.sparsity_reg_weight=0.0 \
    --stage1-override model.recon_mse_weight=0.5 \
    --stage1-override model.recon_l1_weight=0.5 \
    --stage1-override model.recon_edge_weight=0.0 \
    --stage1-override model.perceptual_weight=0.0 \
    --stage1-override model.audio_waveform_l1_weight=1.0 \
    --stage1-override model.audio_multires_stft_loss_weight=1.0 \
    --stage1-override model.audio_multires_stft_fft_sizes=[512,1024,2048] \
    --stage1-override model.out_tanh=true \
    --stage1-override data.audio_dc_remove=true \
    --stage1-override data.audio_peak_normalize=true \
    --stage1-override data.audio_target_peak=0.95 \
    --stage1-override data.audio_rms_normalize=true \
    --stage1-override data.audio_target_rms=0.12 \
    --stage1-override data.audio_max_gain=8.0 \
    --stage1-override data.audio_min_crop_rms=0.03 \
    --stage1-override data.audio_crop_attempts=64 \
    --stage1-override data.audio_fade_samples=1024 \
    --stage1-override train.learning_rate="$VCTK_STAGE1_LR" \
    --stage1-override train.warmup_steps="$STAGE1_WARMUP_STEPS" \
    --stage1-override train.min_lr_ratio="$MIN_LR_RATIO" \
    --stage1-override train.gradient_clip_val=1.0 \
    --stage1-override train.val_check_interval=1.0 \
    --stage1-override train.limit_train_batches=1.0 \
    --stage1-override train.limit_val_batches=1.0 \
    --stage1-override train.limit_test_batches=1.0 \
    --stage1-override train.run_test_after_fit=false \
    --stage1-override model.adversarial_weight=0.0 \
    --stage1-adv-override model.adversarial_weight="$VCTK_ADV_WEIGHT" \
    --stage1-adv-override model.adversarial_start_step=0 \
    --stage1-adv-override model.adversarial_warmup_steps=0 \
    --stage1-adv-override model.disc_start_step=0 \
    --stage1-adv-override model.audio_adversarial_type=hifigan \
    --stage1-adv-override model.audio_disc_periods=[2,3,5,7,11] \
    --stage1-adv-override model.audio_disc_num_scales=3 \
    --stage1-adv-override model.audio_disc_max_channels=512 \
    --stage1-adv-override model.disc_channels="$VCTK_DISC_CHANNELS" \
    --stage1-adv-override model.disc_num_layers="$VCTK_DISC_LAYERS" \
    --stage1-adv-override model.disc_learning_rate="$VCTK_DISC_LR" \
    --stage1-adv-override model.disc_loss=hinge
}

echo "=== bimodal short-sequence adversarial diffusion sweep ==="
echo "CASES=$CASES PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT DRY_RUN=$DRY_RUN"
echo "PYTHON_SUBMIT=$PYTHON_SUBMIT"

case_enabled celebahq && submit_celebahq_image
case_enabled vctk && submit_vctk_audio

echo "=== submissions complete ==="
