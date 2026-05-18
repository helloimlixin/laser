#!/usr/bin/env bash
# Submit 10/10 LASER capacity sanity sweeps for VCTK, COCO512, and CelebA-HQ256.
#
# These runs intentionally do not increase sparse depth. Bottleneck capacity is
# raised through dictionary size, embedding/channel width, decoder depth, and
# optional variational continuous coefficients.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-384000}"
PROJECT="${PROJECT:-laser}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_capacity_sanity}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/datasets/VCTK-Corpus-0.92}"
COCO_DIR="${COCO_DIR:-/scratch/$USER/data/coco}"
CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"

# Keep these sanity runs bounded while still using the real datasets.
SANITY_STAGE1_TRAIN_BATCHES="${SANITY_STAGE1_TRAIN_BATCHES:-200}"
SANITY_STAGE1_VAL_BATCHES="${SANITY_STAGE1_VAL_BATCHES:-20}"
SANITY_CACHE_MAX_ITEMS="${SANITY_CACHE_MAX_ITEMS:-4096}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

if [[ ! -d "$VCTK_DIR" ]]; then
  echo "VCTK directory not found: $VCTK_DIR" >&2
  exit 1
fi
if [[ ! -d "$COCO_DIR/train2017" || ! -d "$COCO_DIR/val2017" ]]; then
  echo "COCO directory must contain train2017/ and val2017/: $COCO_DIR" >&2
  exit 1
fi
if [[ ! -d "$CELEBAHQ_DIR" ]]; then
  echo "CelebA-HQ packed directory not found: $CELEBAHQ_DIR" >&2
  exit 1
fi

COMMON_ARGS=(
  --model-family laser
  --full-training
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
  --partition "$PARTITION"
  --time-limit "$TIME_LIMIT"
  --gpus "$GPUS"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem-mb "$MEM_MB"
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
)
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

COMMON_STAGE1_OVERRIDES=(
  --stage1-override train.limit_train_batches="$SANITY_STAGE1_TRAIN_BATCHES"
  --stage1-override train.limit_val_batches="$SANITY_STAGE1_VAL_BATCHES"
  --stage1-override train.limit_test_batches="$SANITY_STAGE1_VAL_BATCHES"
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override train.precision=bf16-mixed
  --stage1-override model.compute_fid=false
  --stage1-override model.perceptual_weight=0.0
  --stage1-override model.patch_based=false
  --stage1-override model.out_tanh=true
)

COMMON_STAGE2_OVERRIDES=(
  --stage2-override train_ar.max_items="$SANITY_CACHE_MAX_ITEMS"
  --stage2-override train_ar.sample_every_n_epochs=2
  --stage2-override train_ar.sample_num_images=8
  --stage2-override train_ar.generation_metric_num_samples=8
  --stage2-override train_ar.compute_generation_fid=false
  --stage2-override train_ar.sample_temperature=0.7
  --stage2-override train_ar.sample_top_k=0
  --stage2-override train_ar.sample_coeff_mode=mean
  --stage2-override ar.d_model=512
  --stage2-override ar.n_heads=8
  --stage2-override ar.n_layers=8
  --stage2-override ar.d_ff=2048
  --stage2-override ar.learning_rate=3.0e-4
  --stage2-override ar.warmup_steps=250
  --stage2-override ar.min_lr_ratio=0.05
)

COMMON_CACHE_ARGS=(
  --cache-arg=--max-items
  --cache-arg="$SANITY_CACHE_MAX_ITEMS"
)

submit_vctk() {
  local label="$1"
  local coeff_variant="$2"
  shift 2

  local coeff_overrides=()
  if [[ "$coeff_variant" == "var" ]]; then
    coeff_overrides=(
      --stage1-override model.variational_coeffs=true
      --stage1-override model.variational_coeff_kl_weight=1.0e-4
      --stage1-override model.variational_coeff_prior_std=0.5
      --stage1-override model.variational_coeff_min_std=0.03
      --stage2-override ar.coeff_loss_type=gaussian_nll
      --stage2-override ar.coeff_loss_weight=1.0
      --stage2-override train_ar.sample_coeff_temperature=0.4
    )
  else
    coeff_overrides=(
      --stage1-override model.variational_coeffs=false
      --stage2-override ar.coeff_loss_type=huber
      --stage2-override ar.coeff_loss_weight=2.0
      --stage2-override ar.coeff_huber_delta=0.5
    )
  fi

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases vctk \
    --vctk-dir "$VCTK_DIR" \
    --run-label "$label" \
    "${COMMON_STAGE1_OVERRIDES[@]}" \
    "${COMMON_STAGE2_OVERRIDES[@]}" \
    "${COMMON_CACHE_ARGS[@]}" \
    --stage1-override model=laser_audio_waveform \
    --stage1-override data=vctk_waveform \
    --stage1-override data.num_workers=4 \
    --stage1-override data.batch_size=8 \
    --stage1-override model.audio_downsample_rates=[4,4,4] \
    --stage1-override model.audio_dilation_cycle=[1,3,9] \
    --stage1-override model.num_hiddens=192 \
    --stage1-override model.num_residual_blocks=4 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.num_embeddings=2048 \
    --stage1-override model.embedding_dim=128 \
    --stage1-override model.sparsity_level=8 \
    --stage1-override model.bottleneck_loss_weight=0.35 \
    --stage1-override model.commitment_cost=0.08 \
    --stage1-override model.dict_learning_rate=5.0e-5 \
    --stage1-override model.coef_max=8.0 \
    --stage1-override model.audio_waveform_l1_weight=1.0 \
    --stage1-override model.audio_multires_stft_loss_weight=2.0 \
    --stage1-override train.learning_rate=2.0e-4 \
    --stage2-override data.num_workers=2 \
    --stage2-override train_ar.batch_size=4 \
    --stage2-override train_ar.compute_audio_generation_metrics=true \
    --stage2-override ar.n_layers=10 \
    --cache-arg=--audio-representation \
    --cache-arg=waveform \
    --cache-arg=--num-workers \
    --cache-arg=4 \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg=8.0 \
    "${coeff_overrides[@]}" \
    "$@"
}

submit_coco() {
  local label="$1"
  local coeff_variant="$2"
  local atoms="$3"
  local dim="$4"
  shift 4

  local coeff_overrides=()
  if [[ "$coeff_variant" == "var" ]]; then
    coeff_overrides=(
      --stage1-override model.variational_coeffs=true
      --stage1-override model.variational_coeff_kl_weight=1.0e-4
      --stage1-override model.variational_coeff_prior_std=0.5
      --stage1-override model.variational_coeff_min_std=0.03
      --stage2-override ar.coeff_loss_type=gaussian_nll
      --stage2-override ar.coeff_loss_weight=1.0
      --stage2-override train_ar.sample_coeff_temperature=0.4
    )
  else
    coeff_overrides=(
      --stage1-override model.variational_coeffs=false
      --stage2-override ar.coeff_loss_type=huber
      --stage2-override ar.coeff_loss_weight=1.0
      --stage2-override ar.coeff_huber_delta=0.5
    )
  fi

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases coco \
    --coco-dir "$COCO_DIR" \
    --run-label "$label" \
    "${COMMON_STAGE1_OVERRIDES[@]}" \
    "${COMMON_STAGE2_OVERRIDES[@]}" \
    "${COMMON_CACHE_ARGS[@]}" \
    --stage1-override data.num_workers=4 \
    --stage1-override data.batch_size=1 \
    --stage1-override model.backbone=vqgan \
    --stage1-override model.num_downsamples=5 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4,4] \
    --stage1-override model.backbone_latent_channels=768 \
    --stage1-override model.max_ch_mult=4 \
    --stage1-override model.decoder_extra_residual_layers=2 \
    --stage1-override model.use_mid_attention=false \
    --stage1-override model.num_hiddens=192 \
    --stage1-override model.num_residual_blocks=4 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.num_embeddings="$atoms" \
    --stage1-override model.embedding_dim="$dim" \
    --stage1-override model.sparsity_level=16 \
    --stage1-override model.bottleneck_loss_weight=0.2 \
    --stage1-override model.commitment_cost=0.05 \
    --stage1-override model.coef_max=8.0 \
    --stage1-override model.recon_mse_weight=0.25 \
    --stage1-override model.recon_l1_weight=1.0 \
    --stage1-override model.recon_edge_weight=0.25 \
    --stage1-override train.learning_rate=1.0e-4 \
    --stage2-override data.num_workers=2 \
    --stage2-override train_ar.batch_size=2 \
    --stage2-override ar.n_layers=8 \
    --cache-arg=--num-workers \
    --cache-arg=4 \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg=8.0 \
    "${coeff_overrides[@]}" \
    "$@"
}

submit_celebahq() {
  local label="$1"
  local coeff_variant="$2"
  local atoms="$3"
  shift 3

  local coeff_overrides=()
  if [[ "$coeff_variant" == "var" ]]; then
    coeff_overrides=(
      --stage1-override model.variational_coeffs=true
      --stage1-override model.variational_coeff_kl_weight=1.0e-4
      --stage1-override model.variational_coeff_prior_std=0.5
      --stage1-override model.variational_coeff_min_std=0.03
      --stage2-override ar.coeff_loss_type=gaussian_nll
      --stage2-override ar.coeff_loss_weight=1.0
      --stage2-override train_ar.sample_coeff_temperature=0.4
    )
  else
    coeff_overrides=(
      --stage1-override model.variational_coeffs=false
      --stage2-override ar.coeff_loss_type=huber
      --stage2-override ar.coeff_loss_weight=1.0
      --stage2-override ar.coeff_huber_delta=0.5
    )
  fi

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases celebahq \
    --run-label "$label" \
    "${COMMON_STAGE1_OVERRIDES[@]}" \
    "${COMMON_STAGE2_OVERRIDES[@]}" \
    "${COMMON_CACHE_ARGS[@]}" \
    --stage1-override data.data_dir="$CELEBAHQ_DIR" \
    --stage1-override data.num_workers=0 \
    --stage1-override data.batch_size=4 \
    --stage1-override model.backbone=vqgan \
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4] \
    --stage1-override model.backbone_latent_channels=512 \
    --stage1-override model.max_ch_mult=4 \
    --stage1-override model.decoder_extra_residual_layers=2 \
    --stage1-override model.use_mid_attention=false \
    --stage1-override model.num_hiddens=192 \
    --stage1-override model.num_residual_blocks=4 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.num_embeddings="$atoms" \
    --stage1-override model.embedding_dim=128 \
    --stage1-override model.sparsity_level=8 \
    --stage1-override model.bottleneck_loss_weight=0.25 \
    --stage1-override model.commitment_cost=0.05 \
    --stage1-override model.coef_max=8.0 \
    --stage1-override model.recon_mse_weight=0.25 \
    --stage1-override model.recon_l1_weight=1.0 \
    --stage1-override model.recon_edge_weight=0.25 \
    --stage1-override train.learning_rate=1.5e-4 \
    --stage2-override data.num_workers=0 \
    --stage2-override train_ar.batch_size=4 \
    --stage2-override ar.n_layers=8 \
    --cache-arg=--num-workers \
    --cache-arg=0 \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg=8.0 \
    "${coeff_overrides[@]}" \
    "$@"
}

submit_vctk "vctk-waveform-laser-cap-a2048-d128-h192-s8-det-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" det
submit_vctk "vctk-waveform-laser-cap-a2048-d128-h192-s8-var-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" var

submit_coco "coco512-laser-cap-a16384-d128-lat768-s16-det-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" det 16384 128
submit_coco "coco512-laser-cap-a8192-d192-lat768-s16-var-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" var 8192 192

submit_celebahq "celebahq256-laser-cap-a2048-d128-lat512-s8-det-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" det 2048
submit_celebahq "celebahq256-laser-cap-a4096-d128-lat512-s8-var-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" var 4096
