#!/usr/bin/env bash
# Full-pipeline LASER sweep for per-site dictionary learning with a 5-downsample
# VQGAN/RQ-VAE-style stage-1 backbone.  The CelebA-HQ stage-2 defaults mirror
# the successful W&B run helloimlixin-rutgers/laser/rq3ivx3d: 8x8 spatial sites,
# sparse depth 8, q256 coefficients, and a d768/l18/h12 transformer.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
RUN_TAG="${RUN_TAG:-nonpatch-d5-rq3ivx3d-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_nonpatch_d5_multidata}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
CASES="${CASES:-celebahq,vctk}"

CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/Projects/data/celeba_hq}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/Projects/data/VCTK-Corpus}"

IMAGE_GPUS="${IMAGE_GPUS:-2}"
AUDIO_GPUS="${AUDIO_GPUS:-2}"
IMAGE_CPUS_PER_TASK="${IMAGE_CPUS_PER_TASK:-12}"
AUDIO_CPUS_PER_TASK="${AUDIO_CPUS_PER_TASK:-12}"
IMAGE_MEM_MB="${IMAGE_MEM_MB:-240000}"
AUDIO_MEM_MB="${AUDIO_MEM_MB:-240000}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018,gpuk[005-018]}"

# Longer than the earlier 30/150 recipe; each job runs:
#   stage1 reconstruction -> stage1 adversarial continuation -> token cache -> stage2.
STAGE1_EPOCHS="${STAGE1_EPOCHS:-75}"
STAGE1_ADV_EPOCHS="${STAGE1_ADV_EPOCHS:-25}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-300}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-300000}"

# Image d5 non-patch shape: 256 / 2^5 = 8, so 8x8 per-site sparse codes.
# k=4 gives atom+coeff sparse depth 8, matching rq3ivx3d's prior_D=8.
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-4096}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-4}"
COEFF_BINS="${COEFF_BINS:-256}"
COEF_MAX="${COEF_MAX:-16.0}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
BOTTLENECK_LOSS_WEIGHT="${BOTTLENECK_LOSS_WEIGHT:-0.75}"
STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_DICT_LR="${STAGE1_DICT_LR:-2.5e-4}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-500}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.03}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.20}"
PERCEPTUAL_START_STEP="${PERCEPTUAL_START_STEP:-1000}"
PERCEPTUAL_WARMUP_STEPS="${PERCEPTUAL_WARMUP_STEPS:-2000}"
ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.05}"
DISCRIMINATOR_LR="${DISCRIMINATOR_LR:-5.0e-5}"
DISCRIMINATOR_CHANNELS="${DISCRIMINATOR_CHANNELS:-64}"
DISCRIMINATOR_LAYERS="${DISCRIMINATOR_LAYERS:-3}"
USE_ADAPTIVE_DISC_WEIGHT="${USE_ADAPTIVE_DISC_WEIGHT:-true}"
BOUNDED_OMP_REFINE_STEPS="${BOUNDED_OMP_REFINE_STEPS:-16}"
VIS_LOG_EVERY_N_STEPS="${VIS_LOG_EVERY_N_STEPS:-100}"
DIAG_LOG_INTERVAL="${DIAG_LOG_INTERVAL:-100}"
DICTIONARY_VIS_MAX_VECTORS="${DICTIONARY_VIS_MAX_VECTORS:-4096}"

IMAGE_STAGE1_BATCH_SIZE="${IMAGE_STAGE1_BATCH_SIZE:-3}"
IMAGE_STAGE2_BATCH_SIZE="${IMAGE_STAGE2_BATCH_SIZE:-4}"
IMAGE_CACHE_BATCH_SIZE="${IMAGE_CACHE_BATCH_SIZE:-8}"
IMAGE_NUM_WORKERS="${IMAGE_NUM_WORKERS:-4}"
IMAGE_VAL_CHECK_INTERVAL="${IMAGE_VAL_CHECK_INTERVAL:-0.25}"
IMAGE_LIMIT_VAL_BATCHES="${IMAGE_LIMIT_VAL_BATCHES:-256}"
IMAGE_LIMIT_TEST_BATCHES="${IMAGE_LIMIT_TEST_BATCHES:-256}"
IMAGE_STAGE2_LR="${IMAGE_STAGE2_LR:-2.5e-4}"

# Capable transformer defaults from rq3ivx3d.
STAGE2_D_MODEL="${STAGE2_D_MODEL:-768}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-12}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-18}"
STAGE2_D_FF="${STAGE2_D_FF:-3072}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-1500}"
STAGE2_GLOBAL_TOKENS="${STAGE2_GLOBAL_TOKENS:-16}"
STAGE2_COEFF_HUBER_DELTA="${STAGE2_COEFF_HUBER_DELTA:-0.25}"

# VCTK waveform analogue: five strided audio stages with total downsample 512,
# so 32768 samples become 64 per-site latent positions.
VCTK_STAGE1_BATCH_SIZE="${VCTK_STAGE1_BATCH_SIZE:-4}"
VCTK_STAGE2_BATCH_SIZE="${VCTK_STAGE2_BATCH_SIZE:-4}"
VCTK_CACHE_BATCH_SIZE="${VCTK_CACHE_BATCH_SIZE:-8}"
VCTK_NUM_WORKERS="${VCTK_NUM_WORKERS:-4}"
VCTK_DOWNSAMPLE_RATES="${VCTK_DOWNSAMPLE_RATES:-[4,4,4,2,2]}"
VCTK_NUM_EMBEDDINGS="${VCTK_NUM_EMBEDDINGS:-4096}"
VCTK_EMBEDDING_DIM="${VCTK_EMBEDDING_DIM:-128}"
VCTK_NUM_HIDDENS="${VCTK_NUM_HIDDENS:-224}"
VCTK_NUM_RESIDUAL_HIDDENS="${VCTK_NUM_RESIDUAL_HIDDENS:-112}"
VCTK_SPARSITY_LEVEL="${VCTK_SPARSITY_LEVEL:-4}"
VCTK_COMMITMENT_COST="${VCTK_COMMITMENT_COST:-1.0}"
VCTK_BOTTLENECK_LOSS_WEIGHT="${VCTK_BOTTLENECK_LOSS_WEIGHT:-0.75}"
VCTK_STAGE1_LR="${VCTK_STAGE1_LR:-1.5e-4}"
VCTK_DICT_LR="${VCTK_DICT_LR:-1.5e-4}"
VCTK_ADVERSARIAL_WEIGHT="${VCTK_ADVERSARIAL_WEIGHT:-0.03}"
VCTK_DISCRIMINATOR_CHANNELS="${VCTK_DISCRIMINATOR_CHANNELS:-32}"
VCTK_DISCRIMINATOR_LAYERS="${VCTK_DISCRIMINATOR_LAYERS:-3}"
VCTK_DISCRIMINATOR_LR="${VCTK_DISCRIMINATOR_LR:-5.0e-5}"
VCTK_AUDIO_DISC_MAX_CHANNELS="${VCTK_AUDIO_DISC_MAX_CHANNELS:-512}"
VCTK_STAGE2_LR="${VCTK_STAGE2_LR:-2.5e-4}"
VCTK_GENERATION_METRIC_NUM_SAMPLES="${VCTK_GENERATION_METRIC_NUM_SAMPLES:-16}"

case_enabled() {
  local wanted="$1"
  local raw=",${CASES// /},"
  [[ "$raw" == *",$wanted,"* ]]
}

if case_enabled celebahq && [[ ! -d "$CELEBAHQ_DIR" ]]; then
  echo "ERROR: CELEBAHQ_DIR not found: $CELEBAHQ_DIR" >&2
  exit 1
fi
if case_enabled vctk && [[ ! -d "$VCTK_DIR" ]]; then
  echo "ERROR: VCTK_DIR not found: $VCTK_DIR" >&2
  exit 1
fi
if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: PYTHON_SUBMIT must be Python >= 3.8; got $PYTHON_SUBMIT" >&2
  exit 2
fi

COMMON_ARGS=(
  --full-training
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage1-adv-epochs "$STAGE1_ADV_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
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

COMMON_CACHE_ARGS=(
  --cache-arg=--coeff-bins
  --cache-arg="$COEFF_BINS"
  --cache-arg=--coeff-max
  --cache-arg="$COEF_MAX"
  --cache-arg=--coeff-quantization
  --cache-arg=uniform
)

COMMON_STAGE2=(
  --stage2-override ar.max_steps="$STAGE2_MAX_STEPS"
  --stage2-override ar.d_model="$STAGE2_D_MODEL"
  --stage2-override ar.n_heads="$STAGE2_N_HEADS"
  --stage2-override ar.n_layers="$STAGE2_N_LAYERS"
  --stage2-override ar.d_ff="$STAGE2_D_FF"
  --stage2-override ar.warmup_steps="$STAGE2_WARMUP_STEPS"
  --stage2-override ar.min_lr_ratio="$MIN_LR_RATIO"
  --stage2-override ar.n_global_spatial_tokens="$STAGE2_GLOBAL_TOKENS"
  --stage2-override ar.coeff_loss_type=auto
  --stage2-override ar.coeff_huber_delta="$STAGE2_COEFF_HUBER_DELTA"
  --stage2-override ar.sample_coeff_mode=gaussian
  --stage2-override train_ar.sample_top_k=0
  --stage2-override train_ar.sample_coeff_mode=gaussian
  --stage2-override train_ar.sample_every_n_epochs=2
  --stage2-override train_ar.sample_num_images=8
  --stage2-override train_ar.run_test_after_fit=false
  --stage2-override train_ar.save_final_samples_after_fit=false
)

submit_celebahq() {
  local recipe_label="site-d5-k${SPARSITY_LEVEL}-a${NUM_EMBEDDINGS}-q${COEFF_BINS}-d${STAGE2_D_MODEL}l${STAGE2_N_LAYERS}-s1e${STAGE1_EPOCHS}-adve${STAGE1_ADV_EPOCHS}-s2e${STAGE2_EPOCHS}"
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --gpus "$IMAGE_GPUS" \
    --cpus-per-task "$IMAGE_CPUS_PER_TASK" \
    --mem-mb "$IMAGE_MEM_MB" \
    --cases celebahq \
    --model-family laser \
    --run-label "${RUN_TAG}-celebahq-${recipe_label}" \
    "${COMMON_CACHE_ARGS[@]}" \
    --cache-arg=--batch-size \
    --cache-arg="$IMAGE_CACHE_BATCH_SIZE" \
    "${COMMON_STAGE2[@]}" \
    --stage2-override ar.learning_rate="$IMAGE_STAGE2_LR" \
    --stage2-override train_ar.batch_size="$IMAGE_STAGE2_BATCH_SIZE" \
    --stage2-override train_ar.sample_temperature=0.7 \
    --stage2-override train_ar.compute_generation_fid=true \
    --stage2-override train_ar.compute_audio_generation_metrics=false \
    --stage2-override train_ar.generation_metric_num_samples=32 \
    --stage1-override data.image_size=256 \
    --stage1-override data.batch_size="$IMAGE_STAGE1_BATCH_SIZE" \
    --stage1-override data.num_workers="$IMAGE_NUM_WORKERS" \
    --stage1-override data.augment=true \
    --stage1-override train.learning_rate="$STAGE1_LR" \
    --stage1-override train.warmup_steps="$STAGE1_WARMUP_STEPS" \
    --stage1-override train.min_lr_ratio="$MIN_LR_RATIO" \
    --stage1-override train.gradient_clip_val=1.0 \
    --stage1-override train.val_check_interval="$IMAGE_VAL_CHECK_INTERVAL" \
    --stage1-override train.limit_val_batches="$IMAGE_LIMIT_VAL_BATCHES" \
    --stage1-override train.limit_test_batches="$IMAGE_LIMIT_TEST_BATCHES" \
    --stage1-override train.log_every_n_steps=20 \
    --stage1-override train.run_test_after_fit=false \
    --stage1-override checkpoint.save_top_k=1 \
    --stage1-override model=laser \
    --stage1-override model.backbone=vqgan \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_downsamples=5 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4,4] \
    --stage1-override model.backbone_latent_channels=512 \
    --stage1-override model.max_ch_mult=4 \
    --stage1-override model.embedding_dim="$EMBEDDING_DIM" \
    --stage1-override model.num_embeddings="$NUM_EMBEDDINGS" \
    --stage1-override model.sparsity_level="$SPARSITY_LEVEL" \
    --stage1-override model.patch_based=false \
    --stage1-override model.patch_size=1 \
    --stage1-override model.patch_stride=1 \
    --stage1-override model.patch_reconstruction=tile \
    --stage1-override model.bottleneck_loss_weight="$BOTTLENECK_LOSS_WEIGHT" \
    --stage1-override model.commitment_cost="$COMMITMENT_COST" \
    --stage1-override model.coef_max="$COEF_MAX" \
    --stage1-override model.dict_learning_rate="$STAGE1_DICT_LR" \
    --stage1-override model.bounded_omp_refine_steps="$BOUNDED_OMP_REFINE_STEPS" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.decoder_extra_residual_layers=2 \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.attn_resolutions=[8,16] \
    --stage1-override model.data_init_from_first_batch=true \
    --stage1-override model.out_tanh=true \
    --stage1-override model.recon_mse_weight=0.25 \
    --stage1-override model.recon_l1_weight=1.0 \
    --stage1-override model.recon_edge_weight=0.50 \
    --stage1-override model.perceptual_weight="$PERCEPTUAL_WEIGHT" \
    --stage1-override model.perceptual_start_step="$PERCEPTUAL_START_STEP" \
    --stage1-override model.perceptual_warmup_steps="$PERCEPTUAL_WARMUP_STEPS" \
    --stage1-override model.adversarial_weight=0.0 \
    --stage1-override model.adversarial_start_step=1000000000 \
    --stage1-override model.adversarial_warmup_steps=0 \
    --stage1-override model.disc_start_step=1000000000 \
    --stage1-override model.compute_fid=true \
    --stage1-override model.log_images_every_n_steps="$VIS_LOG_EVERY_N_STEPS" \
    --stage1-override model.diag_log_interval="$DIAG_LOG_INTERVAL" \
    --stage1-override model.enable_val_latent_visuals=true \
    --stage1-override model.codebook_visual_max_vectors="$DICTIONARY_VIS_MAX_VECTORS" \
    --stage1-adv-override model.adversarial_weight="$ADVERSARIAL_WEIGHT" \
    --stage1-adv-override model.adversarial_start_step=0 \
    --stage1-adv-override model.adversarial_warmup_steps=0 \
    --stage1-adv-override model.disc_start_step=0 \
    --stage1-adv-override model.disc_learning_rate="$DISCRIMINATOR_LR" \
    --stage1-adv-override model.disc_channels="$DISCRIMINATOR_CHANNELS" \
    --stage1-adv-override model.disc_num_layers="$DISCRIMINATOR_LAYERS" \
    --stage1-adv-override model.disc_norm=group \
    --stage1-adv-override model.disc_loss=hinge \
    --stage1-adv-override model.use_adaptive_disc_weight="$USE_ADAPTIVE_DISC_WEIGHT"
}

submit_vctk() {
  local recipe_label="wave-d5site-k${VCTK_SPARSITY_LEVEL}-a${VCTK_NUM_EMBEDDINGS}-q${COEFF_BINS}-d${STAGE2_D_MODEL}l${STAGE2_N_LAYERS}-s1e${STAGE1_EPOCHS}-adve${STAGE1_ADV_EPOCHS}-s2e${STAGE2_EPOCHS}"
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --gpus "$AUDIO_GPUS" \
    --cpus-per-task "$AUDIO_CPUS_PER_TASK" \
    --mem-mb "$AUDIO_MEM_MB" \
    --cases vctk \
    --model-family laser \
    --run-label "${RUN_TAG}-vctk-${recipe_label}" \
    "${COMMON_CACHE_ARGS[@]}" \
    --cache-arg=--batch-size \
    --cache-arg="$VCTK_CACHE_BATCH_SIZE" \
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
    "${COMMON_STAGE2[@]}" \
    --stage2-override ar.learning_rate="$VCTK_STAGE2_LR" \
    --stage2-override train_ar.batch_size="$VCTK_STAGE2_BATCH_SIZE" \
    --stage2-override train_ar.sample_temperature=0.8 \
    --stage2-override train_ar.compute_generation_fid=false \
    --stage2-override train_ar.compute_audio_generation_metrics=true \
    --stage2-override train_ar.generation_metric_num_samples="$VCTK_GENERATION_METRIC_NUM_SAMPLES" \
    --stage1-override model=laser_audio_waveform \
    --stage1-override data=vctk_waveform \
    --stage1-override data.data_dir="$VCTK_DIR" \
    --stage1-override data.batch_size="$VCTK_STAGE1_BATCH_SIZE" \
    --stage1-override data.num_workers="$VCTK_NUM_WORKERS" \
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
    --stage1-override train.warmup_steps=750 \
    --stage1-override train.min_lr_ratio="$MIN_LR_RATIO" \
    --stage1-override train.gradient_clip_val=1.0 \
    --stage1-override train.deterministic=false \
    --stage1-override train.log_every_n_steps=20 \
    --stage1-override train.run_test_after_fit=false \
    --stage1-override model.audio_downsample_rates="$VCTK_DOWNSAMPLE_RATES" \
    --stage1-override model.num_embeddings="$VCTK_NUM_EMBEDDINGS" \
    --stage1-override model.embedding_dim="$VCTK_EMBEDDING_DIM" \
    --stage1-override model.num_hiddens="$VCTK_NUM_HIDDENS" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens="$VCTK_NUM_RESIDUAL_HIDDENS" \
    --stage1-override model.patch_based=false \
    --stage1-override model.patch_size=1 \
    --stage1-override model.patch_stride=1 \
    --stage1-override model.patch_reconstruction=tile \
    --stage1-override model.sparsity_level="$VCTK_SPARSITY_LEVEL" \
    --stage1-override model.commitment_cost="$VCTK_COMMITMENT_COST" \
    --stage1-override model.bottleneck_loss_weight="$VCTK_BOTTLENECK_LOSS_WEIGHT" \
    --stage1-override model.dict_learning_rate="$VCTK_DICT_LR" \
    --stage1-override model.coef_max="$COEF_MAX" \
    --stage1-override model.bounded_omp_refine_steps="$BOUNDED_OMP_REFINE_STEPS" \
    --stage1-override model.sparsity_reg_weight=0.0 \
    --stage1-override model.recon_mse_weight=0.5 \
    --stage1-override model.recon_l1_weight=0.5 \
    --stage1-override model.recon_edge_weight=0.0 \
    --stage1-override model.audio_waveform_l1_weight=1.0 \
    --stage1-override model.audio_multires_stft_loss_weight=1.0 \
    --stage1-override model.audio_multires_stft_fft_sizes=[512,1024,2048] \
    --stage1-override model.data_init_from_first_batch=true \
    --stage1-override model.out_tanh=true \
    --stage1-override model.compute_fid=false \
    --stage1-override model.perceptual_weight=0.0 \
    --stage1-override model.adversarial_weight=0.0 \
    --stage1-override model.adversarial_start_step=1000000000 \
    --stage1-override model.adversarial_warmup_steps=0 \
    --stage1-override model.disc_start_step=1000000000 \
    --stage1-override model.log_images_every_n_steps="$VIS_LOG_EVERY_N_STEPS" \
    --stage1-override model.diag_log_interval="$DIAG_LOG_INTERVAL" \
    --stage1-override model.enable_val_latent_visuals=true \
    --stage1-override model.codebook_visual_max_vectors="$DICTIONARY_VIS_MAX_VECTORS" \
    --stage1-adv-override model.adversarial_weight="$VCTK_ADVERSARIAL_WEIGHT" \
    --stage1-adv-override model.adversarial_start_step=0 \
    --stage1-adv-override model.adversarial_warmup_steps=0 \
    --stage1-adv-override model.disc_start_step=0 \
    --stage1-adv-override model.audio_adversarial_type=hifigan \
    --stage1-adv-override model.audio_disc_periods=[2,3,5,7,11] \
    --stage1-adv-override model.audio_disc_num_scales=3 \
    --stage1-adv-override model.audio_disc_max_channels="$VCTK_AUDIO_DISC_MAX_CHANNELS" \
    --stage1-adv-override model.disc_channels="$VCTK_DISCRIMINATOR_CHANNELS" \
    --stage1-adv-override model.disc_num_layers="$VCTK_DISCRIMINATOR_LAYERS" \
    --stage1-adv-override model.disc_learning_rate="$VCTK_DISCRIMINATOR_LR" \
    --stage1-adv-override model.disc_loss=hinge \
    --stage1-adv-override model.use_adaptive_disc_weight="$USE_ADAPTIVE_DISC_WEIGHT"
}

echo "=== non-patch d5 full-pipeline sweep ==="
echo "RUN_TAG=$RUN_TAG"
echo "CASES=$CASES"
echo "PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT DRY_RUN=$DRY_RUN"
echo "epochs: stage1=$STAGE1_EPOCHS stage1_adv=$STAGE1_ADV_EPOCHS stage2=$STAGE2_EPOCHS max_steps=$STAGE2_MAX_STEPS"
echo "image: 256 -> 8x8, nonpatch k${SPARSITY_LEVEL}, atoms=${NUM_EMBEDDINGS}, q=${COEFF_BINS}, transformer d${STAGE2_D_MODEL} l${STAGE2_N_LAYERS}"
echo "vctk: downsample=$VCTK_DOWNSAMPLE_RATES, nonpatch k${VCTK_SPARSITY_LEVEL}, atoms=${VCTK_NUM_EMBEDDINGS}, q=${COEFF_BINS}"

case_enabled celebahq && submit_celebahq
case_enabled vctk && submit_vctk

echo "=== submissions complete ==="
