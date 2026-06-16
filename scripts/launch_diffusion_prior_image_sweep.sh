#!/usr/bin/env bash
# Diffusion-prior IMAGE sweep — stage-2 = a sparse-coefficient DDPM instead of
# the autoregressive transformer. Same stage-1 codec + real-valued cache as
# scripts/launch_improved_spatial_prior_sweep.sh (Sweep A), so this is a clean
# stage-2 swap (AR transformer -> diffusion) for an apples-to-apples comparison
# against de06l508 (celebahq AR prior, FID 66.7).
#
# Mechanics: `--stage2-kind diffusion` makes submit_multimodal_sweep's run.sh
# branch to train_stage2_diffusion_prior.py after the cache step. The diffusion
# prior denoises the REAL-VALUED coefficients (requires --coeff-bins 0); atoms are
# drawn from an empirical support bank at sampling time
# (src/models/sparse_diffusion_prior.py).
#
# CAVEAT: the diffusion trainer has NO built-in FID — it only saves qualitative
# sample grids. Compute FID post-hoc from the saved checkpoint to compare to 66.7.
# It also runs single-GPU (proven config) on one of the 2 allocated GPUs; stage-1
# still uses both.
#
# Datasets: CelebA-HQ, FFHQ, ImageNet (all staged on /scratch). Full pipeline:
# stage1 -> stage1_adv -> cache.py -> diffusion, one 2-GPU job per dataset.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
# Keep RUN_TAG short: the W&B group name is "laser-train-<RUN_TAG>-<dataset>-<recipe>-<stamp>"
# and W&B rejects group names over 128 chars. The full recipe is logged in the run config.
RUN_TAG="${RUN_TAG:-diffimg-$(date +%Y%m%d)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_diffusion_prior_image}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
# Runtime commands execute inside the PyTorch container; use its Python unless
# the caller deliberately overrides it.
export PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
CASES="${CASES:-celebahq,ffhq,imagenet}"

IMAGENET_DIR="${IMAGENET_DIR:-/scratch/$USER/Projects/data/imagenet}"
FFHQ_DIR="${FFHQ_DIR:-/scratch/$USER/Projects/data/ffhq}"
CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/Projects/data/celeba_hq}"

IMAGE_GPUS="${IMAGE_GPUS:-2}"
IMAGE_CPUS_PER_TASK="${IMAGE_CPUS_PER_TASK:-12}"
IMAGE_MEM_MB="${IMAGE_MEM_MB:-240000}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018,gpuk[005-018]}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE1_ADV_EPOCHS="${STAGE1_ADV_EPOCHS:-25}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-250}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-240000}"

# Image sparse bottleneck: p4s4/k8 gives a 4x4 patch grid on a 16x16 latent.
# With real-valued coeffs the spatial_depth token depth D = sparsity_level (8),
# spatial sites = 4x4 = 16.
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-4096}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-8}"
PATCH_SIZE="${PATCH_SIZE:-4}"
PATCH_STRIDE="${PATCH_STRIDE:-4}"
COEF_MAX="${COEF_MAX:-32.0}"
# Real-valued coeffs: pass --coeff-bins 0 so cache.py stores raw coeffs and the
# prior drops the categorical coeff head. COEFF_BINS is kept only for the run label.
COEFF_BINS="${COEFF_BINS:-0}"
SUPPORT_ORDER="${SUPPORT_ORDER:-magnitude}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
BOTTLENECK_LOSS_WEIGHT="${BOTTLENECK_LOSS_WEIGHT:-0.75}"
STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_DICT_LR="${STAGE1_DICT_LR:-2.5e-4}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-500}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.20}"
PERCEPTUAL_START_STEP="${PERCEPTUAL_START_STEP:-1000}"
PERCEPTUAL_WARMUP_STEPS="${PERCEPTUAL_WARMUP_STEPS:-2000}"
ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.05}"
DISCRIMINATOR_LR="${DISCRIMINATOR_LR:-5.0e-5}"
DISCRIMINATOR_CHANNELS="${DISCRIMINATOR_CHANNELS:-64}"
DISCRIMINATOR_LAYERS="${DISCRIMINATOR_LAYERS:-3}"
USE_ADAPTIVE_DISC_WEIGHT="${USE_ADAPTIVE_DISC_WEIGHT:-true}"

BOUNDED_OMP_REFINE_STEPS="${BOUNDED_OMP_REFINE_STEPS:-16}"
export LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-1}"
VIS_LOG_EVERY_N_STEPS="${VIS_LOG_EVERY_N_STEPS:-0}"
DIAG_LOG_INTERVAL="${DIAG_LOG_INTERVAL:-100}"
DICTIONARY_VIS_MAX_VECTORS="${DICTIONARY_VIS_MAX_VECTORS:-4096}"


IMAGE_STAGE1_BATCH_SIZE="${IMAGE_STAGE1_BATCH_SIZE:-3}"
IMAGE_STAGE2_BATCH_SIZE="${IMAGE_STAGE2_BATCH_SIZE:-4}"
IMAGE_CACHE_BATCH_SIZE="${IMAGE_CACHE_BATCH_SIZE:-8}"
IMAGE_NUM_WORKERS="${IMAGE_NUM_WORKERS:-4}"
IMAGE_VAL_CHECK_INTERVAL="${IMAGE_VAL_CHECK_INTERVAL:-0.25}"
IMAGE_LIMIT_VAL_BATCHES="${IMAGE_LIMIT_VAL_BATCHES:-256}"
IMAGE_LIMIT_TEST_BATCHES="${IMAGE_LIMIT_TEST_BATCHES:-256}"

# Diffusion coeff-DDPM hyperparameters (the bigger proven variant h192/b8).
DIFF_HIDDEN_CHANNELS="${DIFF_HIDDEN_CHANNELS:-192}"
DIFF_N_RES_BLOCKS="${DIFF_N_RES_BLOCKS:-8}"
DIFF_NUM_TIMESTEPS="${DIFF_NUM_TIMESTEPS:-1000}"
DIFF_LR="${DIFF_LR:-2.0e-4}"
DIFF_SUPPORT_BANK="${DIFF_SUPPORT_BANK:-1024}"
DIFF_BATCH_SIZE="${DIFF_BATCH_SIZE:-256}"
DIFF_SAMPLE_NUM_IMAGES="${DIFF_SAMPLE_NUM_IMAGES:-16}"
DIFF_SAMPLE_STEPS="${DIFF_SAMPLE_STEPS:-100}"

for path in "$IMAGENET_DIR" "$FFHQ_DIR" "$CELEBAHQ_DIR"; do
  if [[ ! -d "$path" ]]; then
    echo "ERROR: required dataset directory not found: $path" >&2
    exit 1
  fi
done

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
  --imagenet-dir "$IMAGENET_DIR"
  --ffhq-dir "$FFHQ_DIR"
  --celebahq-dir "$CELEBAHQ_DIR"
)
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  COMMON_ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

# Real-valued (coeff-bins 0) + magnitude-ordered support cache. coeff-max is still
# recorded in meta and used by the prior; quantization is ignored for real coeffs.
COMMON_CACHE_ARGS=(
  --cache-arg=--coeff-bins
  --cache-arg="$COEFF_BINS"
  --cache-arg=--coeff-max
  --cache-arg="$COEF_MAX"
  --cache-arg=--support-order
  --cache-arg="$SUPPORT_ORDER"
)

# Stage-2 = sparse coefficient DDPM. `--stage2-kind diffusion` routes the run.sh
# to train_stage2_diffusion_prior.py; these --diffusion-arg pairs are appended to
# that command (later values win over submit_multimodal_sweep's defaults, so
# batch-size/devices below override its per-process batch=4 / devices=2).
DIFFUSION_ARGS=(
  --diffusion-arg=--hidden-channels
  --diffusion-arg="$DIFF_HIDDEN_CHANNELS"
  --diffusion-arg=--n-res-blocks
  --diffusion-arg="$DIFF_N_RES_BLOCKS"
  --diffusion-arg=--atom-embed-dim
  --diffusion-arg=16
  --diffusion-arg=--time-embed-dim
  --diffusion-arg="$DIFF_HIDDEN_CHANNELS"
  --diffusion-arg=--num-timesteps
  --diffusion-arg="$DIFF_NUM_TIMESTEPS"
  --diffusion-arg=--learning-rate
  --diffusion-arg="$DIFF_LR"
  --diffusion-arg=--weight-decay
  --diffusion-arg=1.0e-2
  --diffusion-arg=--stats-items
  --diffusion-arg=8192
  --diffusion-arg=--support-bank-size
  --diffusion-arg="$DIFF_SUPPORT_BANK"
  --diffusion-arg=--batch-size
  --diffusion-arg="$DIFF_BATCH_SIZE"
  --diffusion-arg=--devices
  --diffusion-arg=1
  --diffusion-arg=--sample-num-images
  --diffusion-arg="$DIFF_SAMPLE_NUM_IMAGES"
  --diffusion-arg=--sample-steps
  --diffusion-arg="$DIFF_SAMPLE_STEPS"
)

submit_image_dataset() {
  local dataset="$1"
  # Short label (W&B group-name 128-char cap). Full recipe lives in the run config.
  local recipe_label="diff-h${DIFF_HIDDEN_CHANNELS}b${DIFF_N_RES_BLOCKS}-rc-p${PATCH_SIZE}s${PATCH_STRIDE}k${SPARSITY_LEVEL}-a${NUM_EMBEDDINGS}"
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --gpus "$IMAGE_GPUS" \
    --cpus-per-task "$IMAGE_CPUS_PER_TASK" \
    --mem-mb "$IMAGE_MEM_MB" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "${RUN_TAG}-${dataset}-${recipe_label}" \
    "${COMMON_CACHE_ARGS[@]}" \
    --cache-arg=--batch-size \
    --cache-arg="$IMAGE_CACHE_BATCH_SIZE" \
    --stage2-kind diffusion \
    "${DIFFUSION_ARGS[@]}" \
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
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4] \
    --stage1-override model.backbone_latent_channels=512 \
    --stage1-override model.max_ch_mult=4 \
    --stage1-override model.embedding_dim="$EMBEDDING_DIM" \
    --stage1-override model.num_embeddings="$NUM_EMBEDDINGS" \
    --stage1-override model.sparsity_level="$SPARSITY_LEVEL" \
    --stage1-override model.patch_based=true \
    --stage1-override model.patch_size="$PATCH_SIZE" \
    --stage1-override model.patch_stride="$PATCH_STRIDE" \
    --stage1-override model.patch_reconstruction=tile \
    --stage1-override model.bottleneck_loss_weight="$BOTTLENECK_LOSS_WEIGHT" \
    --stage1-override model.commitment_cost="$COMMITMENT_COST" \
    --stage1-override model.bounded_omp_refine_steps="$BOUNDED_OMP_REFINE_STEPS" \
    --stage1-override model.coef_max="$COEF_MAX" \
    --stage1-override model.dict_learning_rate="$STAGE1_DICT_LR" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.decoder_extra_residual_layers=2 \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.attn_resolutions=[16,32] \
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
    --stage1-override model.enable_val_latent_visuals=false \
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

case_enabled() {
  local wanted="$1"
  local raw=",${CASES// /},"
  [[ "$raw" == *",$wanted,"* ]]
}

echo "=== Diffusion-prior IMAGE sweep (stage-2 = sparse coeff DDPM) ==="
echo "RUN_TAG=$RUN_TAG"
echo "CASES=$CASES"
echo "PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT DRY_RUN=$DRY_RUN"
echo "epochs: stage1=$STAGE1_EPOCHS stage1_adv=$STAGE1_ADV_EPOCHS stage2=$STAGE2_EPOCHS max_steps=$STAGE2_MAX_STEPS"
echo "image recipe: p${PATCH_SIZE}s${PATCH_STRIDE} k${SPARSITY_LEVEL} a${NUM_EMBEDDINGS} REAL-VALUED coeffs (coeff-bins=$COEFF_BINS) support_order=$SUPPORT_ORDER"
echo "stage2: DIFFUSION coeff-DDPM h=$DIFF_HIDDEN_CHANNELS res-blocks=$DIFF_N_RES_BLOCKS T=$DIFF_NUM_TIMESTEPS support_bank=$DIFF_SUPPORT_BANK sample=${DIFF_SAMPLE_NUM_IMAGES}@${DIFF_SAMPLE_STEPS} (NO built-in FID; qualitative grids)"

case_enabled celebahq && submit_image_dataset celebahq
case_enabled ffhq && submit_image_dataset ffhq
case_enabled imagenet && submit_image_dataset imagenet

echo "=== Sweep A submissions complete ==="
