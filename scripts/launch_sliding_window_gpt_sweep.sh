#!/usr/bin/env bash
# Sweep B — "sliding-window gpt prior" on a finer 256px latent.
#
# Tests whether the sliding-window attention mechanism (ar.window_sites, the gpt
# prior in src/models/mingpt_prior.py) lets us use a FINER latent grid at 256x256
# for better generation than de06l508's coarse 8x8 grid.
#
# Stage-1: num_downsamples=3 + patch_based=false -> a 32x32 token grid (1024
# spatial sites) at 256px. Quantized coeffs (256 bins) interleave atom/coeff
# tokens to depth D = 2*sparsity_level = 16, so the flattened gpt sequence is
# 32*32*16 = 16384 tokens. That TRIPS the seq_len>8192 guard in
# train.py stage2:144-152, which REQUIRES ar.window_sites>0 -- i.e. full
# attention is impractical and windowed local attention is what makes the fine
# latent trainable. window_sites=64 (two 32-wide latent rows) -> 64*16 = 1024
# token local span, plus the learned global spatial prefix.
#
# gpt requires quantized coeffs (real-valued is spatial_depth-only) and even
# token depth (always true for interleaved quantized streams). Magnitude-ordered
# support + adversarial PatchGAN stage-1 carry over from Sweep A.
#
# Datasets: CelebA-HQ, FFHQ, ImageNet (all staged on /scratch). Full pipeline:
# stage1 -> stage1_adv -> cache.py -> stage2, one 2-GPU job per dataset.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
# Keep RUN_TAG short: the W&B group name is "laser-train-<RUN_TAG>-<dataset>-<recipe>-<stamp>"
# and W&B rejects group names over 128 chars. The full recipe is logged in the run config.
RUN_TAG="${RUN_TAG:-slidewin-$(date +%Y%m%d)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_sliding_window_gpt}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
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

# Finer latent: num_downsamples=3 -> 32x32 latent at 256px, non-patch (per-site)
# sparse coding -> 32x32 = 1024 token sites. k=8 -> interleaved quantized depth 16.
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-4096}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-8}"
NUM_DOWNSAMPLES="${NUM_DOWNSAMPLES:-3}"
CHANNEL_MULTIPLIERS="${CHANNEL_MULTIPLIERS:-[1,2,2,4]}"
ATTN_RESOLUTIONS="${ATTN_RESOLUTIONS:-[32]}"
COEF_MAX="${COEF_MAX:-16.0}"
COEFF_BINS="${COEFF_BINS:-256}"        # quantized (gpt requires it)
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
VIS_LOG_EVERY_N_STEPS="${VIS_LOG_EVERY_N_STEPS:-100}"
DIAG_LOG_INTERVAL="${DIAG_LOG_INTERVAL:-100}"
DICTIONARY_VIS_MAX_VECTORS="${DICTIONARY_VIS_MAX_VECTORS:-4096}"


# Smaller batches than Sweep A: the 32x32 latent (fewer downsamples) makes the
# stage-1 decoder feature maps larger, and the 16384-token gpt is memory heavy.
IMAGE_STAGE1_BATCH_SIZE="${IMAGE_STAGE1_BATCH_SIZE:-2}"
IMAGE_STAGE2_BATCH_SIZE="${IMAGE_STAGE2_BATCH_SIZE:-2}"
IMAGE_CACHE_BATCH_SIZE="${IMAGE_CACHE_BATCH_SIZE:-4}"
IMAGE_NUM_WORKERS="${IMAGE_NUM_WORKERS:-4}"
IMAGE_VAL_CHECK_INTERVAL="${IMAGE_VAL_CHECK_INTERVAL:-0.25}"
IMAGE_LIMIT_VAL_BATCHES="${IMAGE_LIMIT_VAL_BATCHES:-256}"
IMAGE_LIMIT_TEST_BATCHES="${IMAGE_LIMIT_TEST_BATCHES:-256}"

# Lighter transformer than Sweep A so the 16384-token windowed gpt fits in memory.
IMAGE_STAGE2_LR="${IMAGE_STAGE2_LR:-2.0e-4}"
STAGE2_D_MODEL="${STAGE2_D_MODEL:-768}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-12}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-12}"
STAGE2_D_FF="${STAGE2_D_FF:-3072}"
STAGE2_WINDOW_SITES="${STAGE2_WINDOW_SITES:-64}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-2000}"

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

# Quantized + magnitude-ordered support cache (gpt requires quantized coeffs).
COMMON_CACHE_ARGS=(
  --cache-arg=--coeff-bins
  --cache-arg="$COEFF_BINS"
  --cache-arg=--coeff-max
  --cache-arg="$COEF_MAX"
  --cache-arg=--coeff-quantization
  --cache-arg=uniform
  --cache-arg=--support-order
  --cache-arg="$SUPPORT_ORDER"
)

# Stage-2: gpt prior with sliding-window local attention. Vocab sizes auto-resolve
# from the (quantized) cache, so do NOT pin ar.*_vocab_size.
COMMON_STAGE2=(
  --stage2-override ar.type=gpt
  --stage2-override ar.window_sites="$STAGE2_WINDOW_SITES"
  --stage2-override ar.max_steps="$STAGE2_MAX_STEPS"
  --stage2-override ar.d_model="$STAGE2_D_MODEL"
  --stage2-override ar.n_heads="$STAGE2_N_HEADS"
  --stage2-override ar.n_layers="$STAGE2_N_LAYERS"
  --stage2-override ar.d_ff="$STAGE2_D_FF"
  --stage2-override ar.learning_rate="$IMAGE_STAGE2_LR"
  --stage2-override ar.warmup_steps="$STAGE2_WARMUP_STEPS"
  --stage2-override ar.min_lr_ratio="$MIN_LR_RATIO"
  --stage2-override ar.coeff_loss_type=auto
  --stage2-override ar.sample_coeff_mode=gaussian
  --stage2-override train_ar.batch_size="$IMAGE_STAGE2_BATCH_SIZE"
  --stage2-override train_ar.sample_temperature=0.7
  --stage2-override train_ar.sample_top_k=0
  --stage2-override train_ar.sample_coeff_mode=gaussian
  --stage2-override train_ar.sample_every_n_epochs=2
  --stage2-override train_ar.sample_num_images=8
  --stage2-override train_ar.compute_generation_fid=true
  --stage2-override train_ar.generation_metric_num_samples=32
  --stage2-override train_ar.run_test_after_fit=false
  --stage2-override train_ar.save_final_samples_after_fit=true
)

submit_image_dataset() {
  local dataset="$1"
  # Short label (W&B group-name 128-char cap). Full recipe lives in the run config.
  local recipe_label="gpt-w${STAGE2_WINDOW_SITES}-nonpatch-nd${NUM_DOWNSAMPLES}-k${SPARSITY_LEVEL}-a${NUM_EMBEDDINGS}"
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
    "${COMMON_STAGE2[@]}" \
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
    --stage1-override model.backbone=ddpm \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_downsamples="$NUM_DOWNSAMPLES" \
    --stage1-override model.channel_multipliers="$CHANNEL_MULTIPLIERS" \
    --stage1-override model.backbone_latent_channels=512 \
    --stage1-override model.embedding_dim="$EMBEDDING_DIM" \
    --stage1-override model.num_embeddings="$NUM_EMBEDDINGS" \
    --stage1-override model.sparsity_level="$SPARSITY_LEVEL" \
    --stage1-override model.patch_based=false \
    --stage1-override model.bottleneck_loss_weight="$BOTTLENECK_LOSS_WEIGHT" \
    --stage1-override model.commitment_cost="$COMMITMENT_COST" \
    --stage1-override model.coef_max="$COEF_MAX" \
    --stage1-override model.dict_learning_rate="$STAGE1_DICT_LR" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.decoder_extra_residual_layers=2 \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.attn_resolutions="$ATTN_RESOLUTIONS" \
    --stage1-override model.data_init_from_first_batch=true \
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

case_enabled() {
  local wanted="$1"
  local raw=",${CASES// /},"
  [[ "$raw" == *",$wanted,"* ]]
}

echo "=== Sweep B: sliding-window gpt prior (finer 32x32 latent @256px) ==="
echo "RUN_TAG=$RUN_TAG"
echo "CASES=$CASES"
echo "PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT DRY_RUN=$DRY_RUN"
echo "epochs: stage1=$STAGE1_EPOCHS stage1_adv=$STAGE1_ADV_EPOCHS stage2=$STAGE2_EPOCHS max_steps=$STAGE2_MAX_STEPS"
echo "stage1: non-patch nd${NUM_DOWNSAMPLES} ch=${CHANNEL_MULTIPLIERS} k${SPARSITY_LEVEL} a${NUM_EMBEDDINGS} -> 32x32 sites; quantized q${COEFF_BINS} cm${COEF_MAX} support_order=${SUPPORT_ORDER}"
echo "stage2: gpt d_model=$STAGE2_D_MODEL n_layers=$STAGE2_N_LAYERS window_sites=$STAGE2_WINDOW_SITES (seq=32*32*$((2*SPARSITY_LEVEL))=$((32*32*2*SPARSITY_LEVEL)) tokens)"

case_enabled celebahq && submit_image_dataset celebahq
case_enabled ffhq && submit_image_dataset ffhq
case_enabled imagenet && submit_image_dataset imagenet

echo "=== Sweep B submissions complete ==="
