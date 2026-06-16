#!/usr/bin/env bash
# VCTK arm of the "improved prior" sweep — the audio analog of Sweep A
# (scripts/launch_improved_spatial_prior_sweep.sh), targeting the same prior
# failure diagnosed for run m4cxjsu9 (vctk-stage2-garbage-audio): an undertrained
# AR prior over (atom, coeff) tokens whose errors compound, plus a railed coeff
# head and a smooth codec ceiling.
#
# This copies the proven p4s4/k8 VCTK waveform recipe
# (scripts/launch_p4s4k8_multidata_transformer_sweep.sh::submit_vctk_audio) and
# applies the three shipped fixes:
#   1. real-valued coefficients  (--coeff-bins 0) -> drops the 256-way coeff head
#      (the failing run had coeff perplexity ~36); spatial_depth only.
#   2. magnitude-ordered support (--support-order magnitude) -> largest-|coeff|
#      atom first (matching-pursuit order), an easier AR target.
#   3. mean coeff sampling        -> non-variational real cache, avoid the silent
#      gaussian->mean fallback.
# The adversarial HiFi-GAN stage-1 leg (raises the codec ceiling vs the smooth
# non-adversarial leg) and the big d_model=1024/24L prior carry over from p4s4k8.
# coef_max raised to 16 (failing run clamped at 8 and saturated the rail).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
RUN_TAG="${RUN_TAG:-impvctk-$(date +%Y%m%d)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_improved_vctk_prior}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

VCTK_DIR="${VCTK_DIR:-/scratch/$USER/Projects/data/VCTK-Corpus}"

AUDIO_GPUS="${AUDIO_GPUS:-2}"
AUDIO_CPUS_PER_TASK="${AUDIO_CPUS_PER_TASK:-12}"
AUDIO_MEM_MB="${AUDIO_MEM_MB:-240000}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018,gpuk[005-018]}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE1_ADV_EPOCHS="${STAGE1_ADV_EPOCHS:-25}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-250}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-240000}"

# Real-valued coeffs + magnitude-ordered support. coef_max 16 (codec headroom).
COEFF_BINS="${COEFF_BINS:-0}"
COEF_MAX="${COEF_MAX:-16.0}"
SUPPORT_ORDER="${SUPPORT_ORDER:-magnitude}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"

BOUNDED_OMP_REFINE_STEPS="${BOUNDED_OMP_REFINE_STEPS:-16}"
VIS_LOG_EVERY_N_STEPS="${VIS_LOG_EVERY_N_STEPS:-100}"
DIAG_LOG_INTERVAL="${DIAG_LOG_INTERVAL:-100}"
DICTIONARY_VIS_MAX_VECTORS="${DICTIONARY_VIS_MAX_VECTORS:-4096}"


# VCTK waveform: 32768 samples / (8*8*8) = 64 latent steps. p4s4/k8 -> 16 patch
# sites. Real-valued depth D = sparsity_level = 8.
VCTK_STAGE1_BATCH_SIZE="${VCTK_STAGE1_BATCH_SIZE:-4}"
VCTK_STAGE2_BATCH_SIZE="${VCTK_STAGE2_BATCH_SIZE:-4}"
VCTK_CACHE_BATCH_SIZE="${VCTK_CACHE_BATCH_SIZE:-8}"
VCTK_NUM_WORKERS="${VCTK_NUM_WORKERS:-4}"
VCTK_DOWNSAMPLE_RATES="${VCTK_DOWNSAMPLE_RATES:-[8,8,8]}"
VCTK_NUM_EMBEDDINGS="${VCTK_NUM_EMBEDDINGS:-8192}"
VCTK_EMBEDDING_DIM="${VCTK_EMBEDDING_DIM:-128}"
VCTK_NUM_HIDDENS="${VCTK_NUM_HIDDENS:-224}"
VCTK_NUM_RESIDUAL_HIDDENS="${VCTK_NUM_RESIDUAL_HIDDENS:-112}"
VCTK_SPARSITY_LEVEL="${VCTK_SPARSITY_LEVEL:-8}"
VCTK_PATCH_SIZE="${VCTK_PATCH_SIZE:-4}"
VCTK_PATCH_STRIDE="${VCTK_PATCH_STRIDE:-4}"
# tile = non-overlapping stitching (boundary clicks); hann = overlap-add (smooth,
# needs patch_stride < patch_size). The dictionary forces tile when stride==size.
VCTK_PATCH_RECON="${VCTK_PATCH_RECON:-tile}"
# false = reach the small token grid purely by audio downsampling (coarse latent).
VCTK_PATCH_BASED="${VCTK_PATCH_BASED:-true}"
VCTK_COMMITMENT_COST="${VCTK_COMMITMENT_COST:-1.0}"
VCTK_BOTTLENECK_LOSS_WEIGHT="${VCTK_BOTTLENECK_LOSS_WEIGHT:-0.75}"
VCTK_STAGE1_LR="${VCTK_STAGE1_LR:-1.5e-4}"
VCTK_DICT_LR="${VCTK_DICT_LR:-1.5e-4}"
VCTK_ADVERSARIAL_WEIGHT="${VCTK_ADVERSARIAL_WEIGHT:-0.03}"
VCTK_DISCRIMINATOR_CHANNELS="${VCTK_DISCRIMINATOR_CHANNELS:-32}"
VCTK_DISCRIMINATOR_LAYERS="${VCTK_DISCRIMINATOR_LAYERS:-3}"
VCTK_DISCRIMINATOR_LR="${VCTK_DISCRIMINATOR_LR:-5.0e-5}"
VCTK_AUDIO_DISC_MAX_CHANNELS="${VCTK_AUDIO_DISC_MAX_CHANNELS:-512}"
VCTK_STAGE2_LR="${VCTK_STAGE2_LR:-2.0e-4}"
VCTK_GENERATION_METRIC_NUM_SAMPLES="${VCTK_GENERATION_METRIC_NUM_SAMPLES:-16}"

# Big transformer prior (matches p4s4k8).
STAGE2_D_MODEL="${STAGE2_D_MODEL:-1024}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-16}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-24}"
STAGE2_D_FF="${STAGE2_D_FF:-4096}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-2000}"

if [[ ! -d "$VCTK_DIR" ]]; then
  echo "ERROR: required dataset directory not found: $VCTK_DIR" >&2
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
  --vctk-dir "$VCTK_DIR"
)
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  COMMON_ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

submit_vctk_audio() {
  local recipe_label="realcoeff-${SUPPORT_ORDER}-ds512-p${VCTK_PATCH_SIZE}s${VCTK_PATCH_STRIDE}k${VCTK_SPARSITY_LEVEL}-a${VCTK_NUM_EMBEDDINGS}"
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --gpus "$AUDIO_GPUS" \
    --cpus-per-task "$AUDIO_CPUS_PER_TASK" \
    --mem-mb "$AUDIO_MEM_MB" \
    --cases vctk \
    --model-family laser \
    --run-label "${RUN_TAG}-vctk-${recipe_label}" \
    --cache-arg=--coeff-bins \
    --cache-arg="$COEFF_BINS" \
    --cache-arg=--coeff-max \
    --cache-arg="$COEF_MAX" \
    --cache-arg=--support-order \
    --cache-arg="$SUPPORT_ORDER" \
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
    --stage1-override model.patch_based="$VCTK_PATCH_BASED" \
    --stage1-override model.patch_size="$VCTK_PATCH_SIZE" \
    --stage1-override model.patch_stride="$VCTK_PATCH_STRIDE" \
    --stage1-override model.patch_reconstruction="$VCTK_PATCH_RECON" \
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
    --stage1-adv-override model.use_adaptive_disc_weight=true \
    --stage2-override ar.type=sparse_spatial_depth \
    --stage2-override ar.max_steps="$STAGE2_MAX_STEPS" \
    --stage2-override ar.d_model="$STAGE2_D_MODEL" \
    --stage2-override ar.n_heads="$STAGE2_N_HEADS" \
    --stage2-override ar.n_layers="$STAGE2_N_LAYERS" \
    --stage2-override ar.d_ff="$STAGE2_D_FF" \
    --stage2-override ar.learning_rate="$VCTK_STAGE2_LR" \
    --stage2-override ar.warmup_steps="$STAGE2_WARMUP_STEPS" \
    --stage2-override ar.min_lr_ratio="$MIN_LR_RATIO" \
    --stage2-override ar.coeff_loss_type=auto \
    --stage2-override ar.sample_coeff_mode=mean \
    --stage2-override train_ar.batch_size="$VCTK_STAGE2_BATCH_SIZE" \
    --stage2-override train_ar.sample_temperature=0.8 \
    --stage2-override train_ar.sample_top_k=0 \
    --stage2-override train_ar.sample_coeff_mode=mean \
    --stage2-override train_ar.sample_every_n_epochs=2 \
    --stage2-override train_ar.sample_num_images=8 \
    --stage2-override train_ar.compute_generation_fid=false \
    --stage2-override train_ar.compute_audio_generation_metrics=true \
    --stage2-override train_ar.generation_metric_num_samples="$VCTK_GENERATION_METRIC_NUM_SAMPLES" \
    --stage2-override train_ar.run_test_after_fit=false \
    --stage2-override train_ar.save_final_samples_after_fit=true
}

echo "=== Improved VCTK prior sweep (audio analog of Sweep A) ==="
echo "RUN_TAG=$RUN_TAG"
echo "PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT DRY_RUN=$DRY_RUN"
echo "epochs: stage1=$STAGE1_EPOCHS stage1_adv=$STAGE1_ADV_EPOCHS stage2=$STAGE2_EPOCHS max_steps=$STAGE2_MAX_STEPS"
echo "recipe: ds512 p${VCTK_PATCH_SIZE}s${VCTK_PATCH_STRIDE} k${VCTK_SPARSITY_LEVEL} a${VCTK_NUM_EMBEDDINGS} REAL-VALUED (coeff-bins=$COEFF_BINS) cm${COEF_MAX} support_order=$SUPPORT_ORDER; hifigan adv stage-1; spatial_depth d_model=$STAGE2_D_MODEL/$STAGE2_N_LAYERS L; sample mode=mean"

submit_vctk_audio

echo "=== VCTK submission complete ==="
