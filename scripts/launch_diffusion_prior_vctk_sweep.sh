#!/usr/bin/env bash
# VCTK diffusion-prior sweep — stage-2 = a sparse-coefficient DDPM instead of the
# AR transformer. Same hifigan-adversarial waveform stage-1 + real-valued cache as
# scripts/launch_improved_vctk_prior_sweep.sh, so it is a clean stage-2 swap.
#
# IMPORTANT LIMITATION: the sparse diffusion prior (src/models/sparse_diffusion_prior.py)
# and its trainer have NO audio decode path — `decode_stage2_outputs` is image-only.
# So this run TRAINS the coefficient DDPM on the VCTK real-valued cache (loss/val
# curves + a checkpoint) but CANNOT synthesize audio in-pipeline. We pass
# --sample-num-images 0 so the (image-only) sampling/decode step is skipped
# (train_stage2_diffusion_prior.py:146) and the job does not crash. To get
# listenable VCTK diffusion samples, an audio decode for the diffusion sampler must
# be added (follow-up). Reusing the AR-prior fix for [[vctk-stage2-garbage-audio]].

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
RUN_TAG="${RUN_TAG:-diffvctk-$(date +%Y%m%d)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_diffusion_prior_vctk}"
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

# Diffusion coeff-DDPM hyperparameters (bigger proven variant h192/b8).
DIFF_HIDDEN_CHANNELS="${DIFF_HIDDEN_CHANNELS:-192}"
DIFF_N_RES_BLOCKS="${DIFF_N_RES_BLOCKS:-8}"
DIFF_NUM_TIMESTEPS="${DIFF_NUM_TIMESTEPS:-1000}"
DIFF_LR="${DIFF_LR:-2.0e-4}"
DIFF_SUPPORT_BANK="${DIFF_SUPPORT_BANK:-1024}"
DIFF_BATCH_SIZE="${DIFF_BATCH_SIZE:-256}"
# Audio decode for the diffusion sampler is not implemented -> skip sampling.
DIFF_SAMPLE_NUM_IMAGES="${DIFF_SAMPLE_NUM_IMAGES:-0}"

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
  local recipe_label="diff-h${DIFF_HIDDEN_CHANNELS}b${DIFF_N_RES_BLOCKS}-rc-ds512-p${VCTK_PATCH_SIZE}s${VCTK_PATCH_STRIDE}k${VCTK_SPARSITY_LEVEL}-a${VCTK_NUM_EMBEDDINGS}"
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
    --stage1-override model.patch_based=true \
    --stage1-override model.patch_size="$VCTK_PATCH_SIZE" \
    --stage1-override model.patch_stride="$VCTK_PATCH_STRIDE" \
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
    --stage1-adv-override model.use_adaptive_disc_weight=true \
    --stage2-kind diffusion \
    --diffusion-arg=--hidden-channels \
    --diffusion-arg="$DIFF_HIDDEN_CHANNELS" \
    --diffusion-arg=--n-res-blocks \
    --diffusion-arg="$DIFF_N_RES_BLOCKS" \
    --diffusion-arg=--atom-embed-dim \
    --diffusion-arg=16 \
    --diffusion-arg=--time-embed-dim \
    --diffusion-arg="$DIFF_HIDDEN_CHANNELS" \
    --diffusion-arg=--num-timesteps \
    --diffusion-arg="$DIFF_NUM_TIMESTEPS" \
    --diffusion-arg=--learning-rate \
    --diffusion-arg="$DIFF_LR" \
    --diffusion-arg=--weight-decay \
    --diffusion-arg=1.0e-2 \
    --diffusion-arg=--stats-items \
    --diffusion-arg=8192 \
    --diffusion-arg=--support-bank-size \
    --diffusion-arg="$DIFF_SUPPORT_BANK" \
    --diffusion-arg=--batch-size \
    --diffusion-arg="$DIFF_BATCH_SIZE" \
    --diffusion-arg=--devices \
    --diffusion-arg=1 \
    --diffusion-arg=--sample-num-images \
    --diffusion-arg="$DIFF_SAMPLE_NUM_IMAGES"
}

echo "=== VCTK diffusion-prior sweep (stage-2 = sparse coeff DDPM; NO audio decode) ==="
echo "RUN_TAG=$RUN_TAG"
echo "PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT DRY_RUN=$DRY_RUN"
echo "epochs: stage1=$STAGE1_EPOCHS stage1_adv=$STAGE1_ADV_EPOCHS stage2=$STAGE2_EPOCHS max_steps=$STAGE2_MAX_STEPS"
echo "recipe: ds512 p${VCTK_PATCH_SIZE}s${VCTK_PATCH_STRIDE} k${VCTK_SPARSITY_LEVEL} a${VCTK_NUM_EMBEDDINGS} REAL-VALUED (coeff-bins=$COEFF_BINS) cm${COEF_MAX} support_order=$SUPPORT_ORDER; hifigan adv stage-1"
echo "stage2: DIFFUSION coeff-DDPM h=$DIFF_HIDDEN_CHANNELS res-blocks=$DIFF_N_RES_BLOCKS T=$DIFF_NUM_TIMESTEPS support_bank=$DIFF_SUPPORT_BANK; sample-num-images=$DIFF_SAMPLE_NUM_IMAGES (0=skip; no audio decode in diffusion sampler)"

submit_vctk_audio

echo "=== VCTK submission complete ==="
