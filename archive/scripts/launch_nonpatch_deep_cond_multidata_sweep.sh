#!/usr/bin/env bash
# Full-pipeline non-patch LASER sweep for deeper per-site dictionary learning.
# Matrix:
#   image: 256 -> 8x8 (5 downsamples) or 4x4 (6 downsamples)
#   vctk: 32768 -> 64 sites (5 audio strides) or 32 sites (6 audio strides)
#   sparsity: k=3 and k=4
# ImageNet stage 2 is class-conditional; VCTK stage 2 is text-conditional.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
RUN_TAG="${RUN_TAG:-nonpatch-deepcond-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_nonpatch_deep_cond_multidata}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-0}"

DRY_RUN="${DRY_RUN:-0}"
CASES="${CASES:-celebahq,ffhq,imagenet,vctk}"
DOWNSAMPLE_LAYERS="${DOWNSAMPLE_LAYERS:-5,6}"
SPARSITY_LEVELS="${SPARSITY_LEVELS:-3,4}"

CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/Projects/data/celeba_hq}"
FFHQ_DIR="${FFHQ_DIR:-/scratch/$USER/Projects/data/ffhq}"
IMAGENET_DIR="${IMAGENET_DIR:-/scratch/$USER/Projects/data/imagenet}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/Projects/data/VCTK-Corpus/VCTK-Corpus}"

# Amarel GPU nodes expose 2 GPUs per node. Full-pipeline jobs are currently
# single-node because cache extraction and stage 2 are single-rank orchestrated.
IMAGE_GPUS="${IMAGE_GPUS:-2}"
AUDIO_GPUS="${AUDIO_GPUS:-2}"
IMAGE_NODES="${IMAGE_NODES:-1}"
AUDIO_NODES="${AUDIO_NODES:-1}"
IMAGE_CPUS_PER_TASK="${IMAGE_CPUS_PER_TASK:-12}"
AUDIO_CPUS_PER_TASK="${AUDIO_CPUS_PER_TASK:-12}"
IMAGE_MEM_MB="${IMAGE_MEM_MB:-240000}"
AUDIO_MEM_MB="${AUDIO_MEM_MB:-240000}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018,gpu031,gpuk[005-018]}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-75}"
STAGE1_ADV_EPOCHS="${STAGE1_ADV_EPOCHS:-25}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:--1}"
STAGE1_ADV_MAX_STEPS="${STAGE1_ADV_MAX_STEPS:--1}"
STAGE1_ONLY="${STAGE1_ONLY:-0}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-300}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-300000}"
CACHE_MAX_ITEMS="${CACHE_MAX_ITEMS:-0}"

IMAGE_NUM_EMBEDDINGS="${IMAGE_NUM_EMBEDDINGS:-8192}"
IMAGE_EMBEDDING_DIM="${IMAGE_EMBEDDING_DIM:-128}"
IMAGE_CODEC_CAPACITY="${IMAGE_CODEC_CAPACITY:-strong}"
IMAGENET_NUM_EMBEDDINGS="${IMAGENET_NUM_EMBEDDINGS:-16384}"
IMAGENET_EMBEDDING_DIM="${IMAGENET_EMBEDDING_DIM:-192}"
IMAGENET_CODEC_CAPACITY="${IMAGENET_CODEC_CAPACITY:-xl}"
IMAGE_ATTENTION_PROFILE="${IMAGE_ATTENTION_PROFILE:-standard}"
IMAGE_USE_MID_ATTENTION="${IMAGE_USE_MID_ATTENTION:-true}"
IMAGE_DROPOUT="${IMAGE_DROPOUT:-0.0}"
COEFF_BINS="${COEFF_BINS:-256}"
COEF_MAX="${COEF_MAX:-auto_p99.9}"
CACHE_COEFF_MAX_PADDING="${CACHE_COEFF_MAX_PADDING:-1.05}"
STAGE1_COEF_MAX="${STAGE1_COEF_MAX:-null}"
IMAGE_COEFF_BINS="${IMAGE_COEFF_BINS:-128}"
IMAGE_CACHE_COEF_MAX="${IMAGE_CACHE_COEF_MAX:-$COEF_MAX}"
IMAGE_CACHE_COEFF_MAX_PADDING="${IMAGE_CACHE_COEFF_MAX_PADDING:-$CACHE_COEFF_MAX_PADDING}"
IMAGE_COEFF_QUANTIZATION="${IMAGE_COEFF_QUANTIZATION:-mu_law}"
IMAGE_COEFF_MU="${IMAGE_COEFF_MU:-255.0}"
IMAGE_SUPPORT_ORDER="${IMAGE_SUPPORT_ORDER:-magnitude}"
VCTK_COEFF_BINS="${VCTK_COEFF_BINS:-$COEFF_BINS}"
VCTK_CACHE_COEF_MAX="${VCTK_CACHE_COEF_MAX:-$COEF_MAX}"
VCTK_CACHE_COEFF_MAX_PADDING="${VCTK_CACHE_COEFF_MAX_PADDING:-$CACHE_COEFF_MAX_PADDING}"
VCTK_COEFF_QUANTIZATION="${VCTK_COEFF_QUANTIZATION:-uniform}"
VCTK_COEFF_MU="${VCTK_COEFF_MU:-0.0}"
VCTK_SUPPORT_ORDER="${VCTK_SUPPORT_ORDER:-magnitude}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
BOTTLENECK_LOSS_WEIGHT="${BOTTLENECK_LOSS_WEIGHT:-0.75}"
IMAGENET_BOTTLENECK_LOSS_WEIGHT="${IMAGENET_BOTTLENECK_LOSS_WEIGHT:-$BOTTLENECK_LOSS_WEIGHT}"
STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_DICT_LR="${STAGE1_DICT_LR:-2.5e-4}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-500}"
TRAIN_PRECISION="${TRAIN_PRECISION:-bf16-mixed}"
STAGE2_PRECISION="${STAGE2_PRECISION:-$TRAIN_PRECISION}"
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
VIS_LOG_EVERY_N_STEPS="${VIS_LOG_EVERY_N_STEPS:-1000}"
DIAG_LOG_INTERVAL="${DIAG_LOG_INTERVAL:-100}"
DICTIONARY_VIS_MAX_VECTORS="${DICTIONARY_VIS_MAX_VECTORS:-1024}"

IMAGE_CACHE_COEFF_MODE="${IMAGE_CACHE_COEFF_MODE:-quantized}"
IMAGE_CACHE_COEFF_MODE="$(printf '%s' "$IMAGE_CACHE_COEFF_MODE" | tr '[:upper:]' '[:lower:]')"
case "$IMAGE_CACHE_COEFF_MODE" in
  quantized|quant|q)
    IMAGE_CACHE_COEFF_MODE="quantized"
    IMAGE_CACHE_COEFF_BINS="${IMAGE_CACHE_COEFF_BINS:-$IMAGE_COEFF_BINS}"
    IMAGE_CACHE_COEFF_LABEL="q${IMAGE_CACHE_COEFF_BINS}"
    ;;
  continuous|real|real_valued|none)
    IMAGE_CACHE_COEFF_MODE="continuous"
    IMAGE_CACHE_COEFF_BINS=0
    IMAGE_CACHE_COEFF_LABEL="realcoeff"
    ;;
  *)
    echo "ERROR: IMAGE_CACHE_COEFF_MODE must be quantized or continuous, got $IMAGE_CACHE_COEFF_MODE" >&2
    exit 2
    ;;
esac
if ! [[ "$IMAGE_CACHE_COEFF_BINS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: IMAGE_CACHE_COEFF_BINS must be a non-negative integer, got $IMAGE_CACHE_COEFF_BINS" >&2
  exit 2
fi
if [[ "$IMAGE_CACHE_COEFF_MODE" == "quantized" && "$IMAGE_CACHE_COEFF_BINS" -le 0 ]]; then
  echo "ERROR: quantized image cache mode requires IMAGE_CACHE_COEFF_BINS > 0" >&2
  exit 2
fi

case "$IMAGE_CODEC_CAPACITY" in
  base)
    DEFAULT_IMAGE_STAGE1_BATCH_SIZE=3
    DEFAULT_IMAGE_CACHE_BATCH_SIZE=8
    ;;
  strong)
    DEFAULT_IMAGE_STAGE1_BATCH_SIZE=2
    DEFAULT_IMAGE_CACHE_BATCH_SIZE=6
    ;;
  xl)
    DEFAULT_IMAGE_STAGE1_BATCH_SIZE=1
    DEFAULT_IMAGE_CACHE_BATCH_SIZE=4
    ;;
  *)
    echo "Unsupported IMAGE_CODEC_CAPACITY: $IMAGE_CODEC_CAPACITY (expected base, strong, or xl)" >&2
    exit 2
    ;;
esac

case "$IMAGENET_CODEC_CAPACITY" in
  base)
    DEFAULT_IMAGENET_STAGE1_BATCH_SIZE=2
    DEFAULT_IMAGENET_CACHE_BATCH_SIZE=8
    ;;
  strong)
    DEFAULT_IMAGENET_STAGE1_BATCH_SIZE=1
    DEFAULT_IMAGENET_CACHE_BATCH_SIZE=6
    ;;
  xl)
    DEFAULT_IMAGENET_STAGE1_BATCH_SIZE=1
    DEFAULT_IMAGENET_CACHE_BATCH_SIZE=4
    ;;
  *)
    echo "Unsupported IMAGENET_CODEC_CAPACITY: $IMAGENET_CODEC_CAPACITY (expected base, strong, or xl)" >&2
    exit 2
    ;;
esac

IMAGE_STAGE1_BATCH_SIZE="${IMAGE_STAGE1_BATCH_SIZE:-$DEFAULT_IMAGE_STAGE1_BATCH_SIZE}"
IMAGENET_STAGE1_BATCH_SIZE="${IMAGENET_STAGE1_BATCH_SIZE:-$DEFAULT_IMAGENET_STAGE1_BATCH_SIZE}"
IMAGE_STAGE2_BATCH_SIZE="${IMAGE_STAGE2_BATCH_SIZE:-4}"
IMAGENET_STAGE2_BATCH_SIZE="${IMAGENET_STAGE2_BATCH_SIZE:-2}"
IMAGE_CACHE_BATCH_SIZE="${IMAGE_CACHE_BATCH_SIZE:-$DEFAULT_IMAGE_CACHE_BATCH_SIZE}"
IMAGENET_CACHE_BATCH_SIZE="${IMAGENET_CACHE_BATCH_SIZE:-$DEFAULT_IMAGENET_CACHE_BATCH_SIZE}"
IMAGE_CACHE_MAX_ITEMS="${IMAGE_CACHE_MAX_ITEMS:-$CACHE_MAX_ITEMS}"
VCTK_CACHE_MAX_ITEMS="${VCTK_CACHE_MAX_ITEMS:-$CACHE_MAX_ITEMS}"
IMAGE_NUM_WORKERS="${IMAGE_NUM_WORKERS:-4}"
IMAGENET_NUM_WORKERS="${IMAGENET_NUM_WORKERS:-8}"
IMAGE_VAL_CHECK_INTERVAL="${IMAGE_VAL_CHECK_INTERVAL:-0.25}"
IMAGE_LIMIT_VAL_BATCHES="${IMAGE_LIMIT_VAL_BATCHES:-256}"
IMAGE_LIMIT_TEST_BATCHES="${IMAGE_LIMIT_TEST_BATCHES:-256}"
IMAGENET_LIMIT_VAL_BATCHES="${IMAGENET_LIMIT_VAL_BATCHES:-512}"
IMAGENET_LIMIT_TEST_BATCHES="${IMAGENET_LIMIT_TEST_BATCHES:-512}"
IMAGE_STAGE2_LR="${IMAGE_STAGE2_LR:-2.5e-4}"

STAGE2_D_MODEL="${STAGE2_D_MODEL:-768}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-12}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-18}"
STAGE2_D_FF="${STAGE2_D_FF:-3072}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-1500}"
STAGE2_GLOBAL_TOKENS="${STAGE2_GLOBAL_TOKENS:-16}"
STAGE2_COEFF_HUBER_DELTA="${STAGE2_COEFF_HUBER_DELTA:-0.25}"

VCTK_STAGE2_BATCH_SIZE="${VCTK_STAGE2_BATCH_SIZE:-2}"
VCTK_NUM_WORKERS="${VCTK_NUM_WORKERS:-4}"
VCTK_NUM_EMBEDDINGS="${VCTK_NUM_EMBEDDINGS:-8192}"
VCTK_EMBEDDING_DIM="${VCTK_EMBEDDING_DIM:-128}"
VCTK_CODEC_CAPACITY="${VCTK_CODEC_CAPACITY:-strong}"
VCTK_DILATION_CYCLE="${VCTK_DILATION_CYCLE:-}"
VCTK_AUDIO_NUM_SAMPLES="${VCTK_AUDIO_NUM_SAMPLES:-65536}"
VCTK_AUDIO_MAX_DURATION_SECONDS="${VCTK_AUDIO_MAX_DURATION_SECONDS:-4.096}"
VCTK_AUDIO_MIN_DURATION_SECONDS="${VCTK_AUDIO_MIN_DURATION_SECONDS:-0.0}"
VCTK_REQUIRE_TEXT="${VCTK_REQUIRE_TEXT:-true}"
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
VCTK_TEXT_MAX_LENGTH="${VCTK_TEXT_MAX_LENGTH:-160}"
VCTK_TEXT_PREFIX_LENGTH="${VCTK_TEXT_PREFIX_LENGTH:-16}"
VCTK_SAMPLE_PROMPTS="${VCTK_SAMPLE_PROMPTS:-[\"The quick brown fox jumps over the lazy dog.\",\"A calm voice reads this sentence clearly.\",\"Please bring the warm tea to the table.\",\"The train arrived before sunrise.\",\"Several people waited outside the station.\",\"She opened the window and listened to the rain.\",\"This recording should follow the written text.\",\"A small boat moved slowly across the lake.\"]}"

case "$VCTK_CODEC_CAPACITY" in
  base)
    DEFAULT_VCTK_STAGE1_BATCH_SIZE=4
    DEFAULT_VCTK_CACHE_BATCH_SIZE=8
    DEFAULT_VCTK_DILATION_CYCLE='[1,3,9]'
    ;;
  strong)
    DEFAULT_VCTK_STAGE1_BATCH_SIZE=2
    DEFAULT_VCTK_CACHE_BATCH_SIZE=6
    DEFAULT_VCTK_DILATION_CYCLE='[1,3,9,27]'
    ;;
  xl)
    DEFAULT_VCTK_STAGE1_BATCH_SIZE=1
    DEFAULT_VCTK_CACHE_BATCH_SIZE=4
    DEFAULT_VCTK_DILATION_CYCLE='[1,3,9,27]'
    ;;
  *)
    echo "Unsupported VCTK_CODEC_CAPACITY: $VCTK_CODEC_CAPACITY (expected base, strong, or xl)" >&2
    exit 2
    ;;
esac
VCTK_STAGE1_BATCH_SIZE="${VCTK_STAGE1_BATCH_SIZE:-$DEFAULT_VCTK_STAGE1_BATCH_SIZE}"
VCTK_CACHE_BATCH_SIZE="${VCTK_CACHE_BATCH_SIZE:-$DEFAULT_VCTK_CACHE_BATCH_SIZE}"
VCTK_DILATION_CYCLE="${VCTK_DILATION_CYCLE:-$DEFAULT_VCTK_DILATION_CYCLE}"

case_enabled() {
  local wanted="$1"
  local raw=",${CASES// /},"
  [[ "$raw" == *",$wanted,"* ]]
}

ensure_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: $label not found: $path" >&2
    exit 1
  fi
}

image_channels() {
  case "$1" in
    5) printf '[1,1,2,2,4,4]' ;;
    6) printf '[1,1,2,2,4,4,4]' ;;
    *) echo "Unsupported image downsample depth: $1" >&2; exit 2 ;;
  esac
}

image_attn() {
  case "$IMAGE_ATTENTION_PROFILE:$1" in
    mid_only:5|mid_only:6) printf '[]' ;;
    standard:5) printf '[8,16]' ;;
    standard:6) printf '[4,8,16]' ;;
    wide:5) printf '[8,16,32]' ;;
    wide:6) printf '[4,8,16,32]' ;;
    *) echo "Unsupported image attention profile/depth: $IMAGE_ATTENTION_PROFILE d$1" >&2; exit 2 ;;
  esac
}

image_num_hiddens() {
  local capacity="${2:-$IMAGE_CODEC_CAPACITY}"
  case "$capacity:$1" in
    base:5) printf '128' ;;
    base:6) printf '160' ;;
    strong:5) printf '192' ;;
    strong:6) printf '224' ;;
    xl:5) printf '224' ;;
    xl:6) printf '256' ;;
    *) echo "Unsupported image capacity/depth: $capacity d$1" >&2; exit 2 ;;
  esac
}

image_num_residual_blocks() {
  local capacity="${2:-$IMAGE_CODEC_CAPACITY}"
  case "$capacity:$1" in
    base:5|base:6) printf '3' ;;
    strong:5|strong:6) printf '4' ;;
    xl:5|xl:6) printf '4' ;;
    *) echo "Unsupported image capacity/depth: $capacity d$1" >&2; exit 2 ;;
  esac
}

image_residual_hiddens() {
  local capacity="${2:-$IMAGE_CODEC_CAPACITY}"
  case "$capacity:$1" in
    base:5) printf '96' ;;
    base:6) printf '112' ;;
    strong:5) printf '128' ;;
    strong:6) printf '160' ;;
    xl:5) printf '160' ;;
    xl:6) printf '192' ;;
    *) echo "Unsupported image capacity/depth: $capacity d$1" >&2; exit 2 ;;
  esac
}

image_extra_decoder_layers() {
  local capacity="${2:-$IMAGE_CODEC_CAPACITY}"
  case "$capacity:$1" in
    base:5) printf '2' ;;
    base:6) printf '3' ;;
    strong:5) printf '3' ;;
    strong:6) printf '4' ;;
    xl:5) printf '4' ;;
    xl:6) printf '5' ;;
    *) echo "Unsupported image capacity/depth: $capacity d$1" >&2; exit 2 ;;
  esac
}

image_backbone_latent_channels() {
  local capacity="${1:-$IMAGE_CODEC_CAPACITY}"
  case "$capacity" in
    base) printf '512' ;;
    strong) printf '768' ;;
    xl) printf '1024' ;;
    *) echo "Unsupported image codec capacity: $capacity" >&2; exit 2 ;;
  esac
}

image_latent_hw() {
  case "$1" in
    5) printf '8' ;;
    6) printf '4' ;;
    *) echo "Unsupported image downsample depth: $1" >&2; exit 2 ;;
  esac
}

vctk_rates() {
  case "$1" in
    5) printf '[4,4,4,2,2]' ;;
    6) printf '[4,4,4,2,2,2]' ;;
    *) echo "Unsupported VCTK downsample depth: $1" >&2; exit 2 ;;
  esac
}

vctk_downsample_factor() {
  case "$1" in
    5) printf '256' ;;
    6) printf '512' ;;
    *) echo "Unsupported VCTK downsample depth: $1" >&2; exit 2 ;;
  esac
}

vctk_sites() {
  local factor
  factor="$(vctk_downsample_factor "$1")"
  if (( VCTK_AUDIO_NUM_SAMPLES % factor != 0 )); then
    echo "VCTK_AUDIO_NUM_SAMPLES=$VCTK_AUDIO_NUM_SAMPLES must be divisible by d$1 factor=$factor" >&2
    exit 2
  fi
  printf '%d' "$(( VCTK_AUDIO_NUM_SAMPLES / factor ))"
}

vctk_num_hiddens() {
  case "$VCTK_CODEC_CAPACITY:$1" in
    base:5) printf '224' ;;
    base:6) printf '256' ;;
    strong:5) printf '288' ;;
    strong:6) printf '320' ;;
    xl:5) printf '352' ;;
    xl:6) printf '384' ;;
    *) echo "Unsupported VCTK capacity/depth: $VCTK_CODEC_CAPACITY d$1" >&2; exit 2 ;;
  esac
}

vctk_num_residual_blocks() {
  case "$VCTK_CODEC_CAPACITY:$1" in
    base:5|base:6) printf '3' ;;
    strong:5|strong:6) printf '4' ;;
    xl:5|xl:6) printf '5' ;;
    *) echo "Unsupported VCTK capacity/depth: $VCTK_CODEC_CAPACITY d$1" >&2; exit 2 ;;
  esac
}

vctk_residual_hiddens() {
  case "$VCTK_CODEC_CAPACITY:$1" in
    base:5) printf '112' ;;
    base:6) printf '128' ;;
    strong:5) printf '144' ;;
    strong:6) printf '160' ;;
    xl:5) printf '176' ;;
    xl:6) printf '192' ;;
    *) echo "Unsupported VCTK capacity/depth: $VCTK_CODEC_CAPACITY d$1" >&2; exit 2 ;;
  esac
}

if case_enabled celebahq; then ensure_dir CELEBAHQ_DIR "$CELEBAHQ_DIR"; fi
if case_enabled ffhq; then ensure_dir FFHQ_DIR "$FFHQ_DIR"; fi
if case_enabled imagenet; then ensure_dir IMAGENET_DIR "$IMAGENET_DIR"; fi
if case_enabled vctk; then ensure_dir VCTK_DIR "$VCTK_DIR"; fi
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
  --ffhq-dir "$FFHQ_DIR"
  --imagenet-dir "$IMAGENET_DIR"
  --vctk-dir "$VCTK_DIR"
)
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  COMMON_ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi
if [[ "$STAGE1_ONLY" == "1" || "$STAGE1_ONLY" == "true" ]]; then
  COMMON_ARGS+=(--stage1-only)
fi

IMAGE_CACHE_ARGS=(
  --cache-arg=--coeff-bins
  --cache-arg="$IMAGE_CACHE_COEFF_BINS"
  --cache-arg=--coeff-max
  --cache-arg="$IMAGE_CACHE_COEF_MAX"
  --cache-arg=--coeff-max-padding
  --cache-arg="$IMAGE_CACHE_COEFF_MAX_PADDING"
  --cache-arg=--coeff-quantization
  --cache-arg="$IMAGE_COEFF_QUANTIZATION"
  --cache-arg=--coeff-mu
  --cache-arg="$IMAGE_COEFF_MU"
  --cache-arg=--support-order
  --cache-arg="$IMAGE_SUPPORT_ORDER"
)
if [[ "$IMAGE_CACHE_MAX_ITEMS" != "0" ]]; then
  IMAGE_CACHE_ARGS+=(--cache-arg=--max-items --cache-arg="$IMAGE_CACHE_MAX_ITEMS")
fi

VCTK_CACHE_ARGS=(
  --cache-arg=--coeff-bins
  --cache-arg="$VCTK_COEFF_BINS"
  --cache-arg=--coeff-max
  --cache-arg="$VCTK_CACHE_COEF_MAX"
  --cache-arg=--coeff-max-padding
  --cache-arg="$VCTK_CACHE_COEFF_MAX_PADDING"
  --cache-arg=--coeff-quantization
  --cache-arg="$VCTK_COEFF_QUANTIZATION"
  --cache-arg=--coeff-mu
  --cache-arg="$VCTK_COEFF_MU"
  --cache-arg=--support-order
  --cache-arg="$VCTK_SUPPORT_ORDER"
)
if [[ "$VCTK_CACHE_MAX_ITEMS" != "0" ]]; then
  VCTK_CACHE_ARGS+=(--cache-arg=--max-items --cache-arg="$VCTK_CACHE_MAX_ITEMS")
fi

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
  --stage2-override ar.sample_coeff_mode=mean
  --stage2-override train_ar.sample_top_k=0
  --stage2-override train_ar.sample_coeff_mode=mean
  --stage2-override train_ar.sample_every_n_epochs=2
  --stage2-override train_ar.sample_log_to_wandb=true
  --stage2-override train_ar.run_test_after_fit=false
  --stage2-override train_ar.save_final_samples_after_fit=true
  --stage2-override train_ar.deterministic=false
  --stage2-override train_ar.precision="$STAGE2_PRECISION"
)

submit_image_dataset() {
  local dataset="$1"
  local downsample="$2"
  local sparsity="$3"
  local latent_hw
  latent_hw="$(image_latent_hw "$downsample")"
  local channels
  channels="$(image_channels "$downsample")"
  local attn
  attn="$(image_attn "$downsample")"
  local codec_capacity="$IMAGE_CODEC_CAPACITY"
  local num_embeddings="$IMAGE_NUM_EMBEDDINGS"
  local embedding_dim="$IMAGE_EMBEDDING_DIM"
  local bottleneck_loss_weight="$BOTTLENECK_LOSS_WEIGHT"
  local stage1_batch="$IMAGE_STAGE1_BATCH_SIZE"
  local stage2_batch="$IMAGE_STAGE2_BATCH_SIZE"
  local cache_batch="$IMAGE_CACHE_BATCH_SIZE"
  local num_workers="$IMAGE_NUM_WORKERS"
  local limit_val="$IMAGE_LIMIT_VAL_BATCHES"
  local limit_test="$IMAGE_LIMIT_TEST_BATCHES"
  local temperature="0.7"
  local conditional_label="uncond"
  local token_depth="$sparsity"
  if (( IMAGE_CACHE_COEFF_BINS > 0 )); then
    token_depth=$((sparsity * 2))
  fi
  local token_length=$((latent_hw * latent_hw * token_depth))
  local class_args=(
    --stage2-override ar.class_conditional=false
    --stage2-override ar.num_classes=0
    --stage2-override train_ar.compute_generation_fid=false
    --stage2-override train_ar.compute_audio_generation_metrics=false
    --stage2-override train_ar.generation_metric_num_samples=32
  )

  if [[ "$dataset" == "imagenet" ]]; then
    codec_capacity="$IMAGENET_CODEC_CAPACITY"
    num_embeddings="$IMAGENET_NUM_EMBEDDINGS"
    embedding_dim="$IMAGENET_EMBEDDING_DIM"
    bottleneck_loss_weight="$IMAGENET_BOTTLENECK_LOSS_WEIGHT"
    stage1_batch="$IMAGENET_STAGE1_BATCH_SIZE"
    stage2_batch="$IMAGENET_STAGE2_BATCH_SIZE"
    cache_batch="$IMAGENET_CACHE_BATCH_SIZE"
    num_workers="$IMAGENET_NUM_WORKERS"
    limit_val="$IMAGENET_LIMIT_VAL_BATCHES"
    limit_test="$IMAGENET_LIMIT_TEST_BATCHES"
    temperature="0.9"
    conditional_label="classcond"
    class_args=(
      --stage2-override ar.class_conditional=true
      --stage2-override ar.num_classes=1000
      --stage2-override train_ar.sample_class_labels=[0,1,2,3,4,5,6,7]
      --stage2-override train_ar.compute_generation_fid=false
      --stage2-override train_ar.compute_audio_generation_metrics=false
      --stage2-override train_ar.generation_metric_num_samples=32
    )
  fi

  local hidden
  hidden="$(image_num_hiddens "$downsample" "$codec_capacity")"
  local residual_blocks
  residual_blocks="$(image_num_residual_blocks "$downsample" "$codec_capacity")"
  local residual_hidden
  residual_hidden="$(image_residual_hiddens "$downsample" "$codec_capacity")"
  local extra_decoder_layers
  extra_decoder_layers="$(image_extra_decoder_layers "$downsample" "$codec_capacity")"
  local backbone_latent_channels
  backbone_latent_channels="$(image_backbone_latent_channels "$codec_capacity")"

  local recipe_label="nonpatch-d${downsample}k${sparsity}-${conditional_label}-${codec_capacity}-attn${IMAGE_ATTENTION_PROFILE}-drop${IMAGE_DROPOUT}-a${num_embeddings}-e${embedding_dim}-${IMAGE_COEFF_QUANTIZATION}-${IMAGE_CACHE_COEFF_LABEL}-ord${IMAGE_SUPPORT_ORDER}-seq${token_length}-s1e${STAGE1_EPOCHS}-adve${STAGE1_ADV_EPOCHS}-s2e${STAGE2_EPOCHS}"
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --gpus "$IMAGE_GPUS" \
    --nodes "$IMAGE_NODES" \
    --cpus-per-task "$IMAGE_CPUS_PER_TASK" \
    --mem-mb "$IMAGE_MEM_MB" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "${RUN_TAG}-${dataset}-${recipe_label}" \
    "${IMAGE_CACHE_ARGS[@]}" \
    --cache-arg=--batch-size \
    --cache-arg="$cache_batch" \
    "${COMMON_STAGE2[@]}" \
    "${class_args[@]}" \
    --stage2-override ar.learning_rate="$IMAGE_STAGE2_LR" \
    --stage2-override train_ar.batch_size="$stage2_batch" \
    --stage2-override train_ar.sample_temperature="$temperature" \
    --stage2-override train_ar.sample_num_images=8 \
    --stage1-override data.image_size=256 \
    --stage1-override data.batch_size="$stage1_batch" \
    --stage1-override data.num_workers="$num_workers" \
    --stage1-override data.augment=true \
    --stage1-override train.learning_rate="$STAGE1_LR" \
    --stage1-override train.precision="$TRAIN_PRECISION" \
    --stage1-override train.max_steps="$STAGE1_MAX_STEPS" \
    --stage1-override train.warmup_steps="$STAGE1_WARMUP_STEPS" \
    --stage1-override train.min_lr_ratio="$MIN_LR_RATIO" \
    --stage1-override train.gradient_clip_val=1.0 \
    --stage1-override train.val_check_interval="$IMAGE_VAL_CHECK_INTERVAL" \
    --stage1-override train.limit_val_batches="$limit_val" \
    --stage1-override train.limit_test_batches="$limit_test" \
    --stage1-override train.log_every_n_steps=20 \
    --stage1-override train.run_test_after_fit=false \
    --stage1-override checkpoint.save_top_k=1 \
    --stage1-override model=laser \
    --stage1-override model.backbone=ddpm \
    --stage1-override model.num_hiddens="$hidden" \
    --stage1-override model.num_downsamples="$downsample" \
    --stage1-override model.dropout="$IMAGE_DROPOUT" \
    --stage1-override model.channel_multipliers="$channels" \
    --stage1-override model.backbone_latent_channels="$backbone_latent_channels" \
    --stage1-override model.embedding_dim="$embedding_dim" \
    --stage1-override model.num_embeddings="$num_embeddings" \
    --stage1-override model.sparsity_level="$sparsity" \
    --stage1-override model.patch_based=false \
    --stage1-override model.patch_size=1 \
    --stage1-override model.patch_stride=1 \
    --stage1-override model.patch_reconstruction=tile \
    --stage1-override model.bottleneck_loss_weight="$bottleneck_loss_weight" \
    --stage1-override model.commitment_cost="$COMMITMENT_COST" \
    --stage1-override model.coef_max="$STAGE1_COEF_MAX" \
    --stage1-override model.dict_learning_rate="$STAGE1_DICT_LR" \
    --stage1-override model.num_residual_blocks="$residual_blocks" \
    --stage1-override model.num_residual_hiddens="$residual_hidden" \
    --stage1-override model.decoder_extra_residual_layers="$extra_decoder_layers" \
    --stage1-override model.use_mid_attention="$IMAGE_USE_MID_ATTENTION" \
    --stage1-override model.attn_resolutions="$attn" \
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
    --stage1-adv-override model.use_adaptive_disc_weight="$USE_ADAPTIVE_DISC_WEIGHT" \
    --stage1-adv-override train.max_steps="$STAGE1_ADV_MAX_STEPS"
}

submit_vctk() {
  local downsample="$1"
  local sparsity="$2"
  local rates
  rates="$(vctk_rates "$downsample")"
  local sites
  sites="$(vctk_sites "$downsample")"
  local hidden
  hidden="$(vctk_num_hiddens "$downsample")"
  local residual_blocks
  residual_blocks="$(vctk_num_residual_blocks "$downsample")"
  local residual_hidden
  residual_hidden="$(vctk_residual_hiddens "$downsample")"
  local require_text_cache_arg="--audio-require-text"
  if [[ "$VCTK_REQUIRE_TEXT" == "0" || "$VCTK_REQUIRE_TEXT" == "false" ]]; then
    require_text_cache_arg="--no-audio-require-text"
  fi
  local token_length=$((sites * sparsity * 2))
  local recipe_label="wave-d${downsample}k${sparsity}-textcond-${VCTK_CODEC_CAPACITY}-a${VCTK_NUM_EMBEDDINGS}-e${VCTK_EMBEDDING_DIM}-${VCTK_COEFF_QUANTIZATION}-q${VCTK_COEFF_BINS}-ord${VCTK_SUPPORT_ORDER}-seq${token_length}-s1e${STAGE1_EPOCHS}-adve${STAGE1_ADV_EPOCHS}-s2e${STAGE2_EPOCHS}"
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --gpus "$AUDIO_GPUS" \
    --nodes "$AUDIO_NODES" \
    --cpus-per-task "$AUDIO_CPUS_PER_TASK" \
    --mem-mb "$AUDIO_MEM_MB" \
    --cases vctk \
    --model-family laser \
    --run-label "${RUN_TAG}-vctk-${recipe_label}" \
    "${VCTK_CACHE_ARGS[@]}" \
    --cache-arg=--batch-size \
    --cache-arg="$VCTK_CACHE_BATCH_SIZE" \
    --cache-arg=--audio-representation \
    --cache-arg=waveform \
    --cache-arg=--audio-num-samples \
    --cache-arg="$VCTK_AUDIO_NUM_SAMPLES" \
    --cache-arg=--audio-max-duration-seconds \
    --cache-arg="$VCTK_AUDIO_MAX_DURATION_SECONDS" \
    --cache-arg=--audio-min-duration-seconds \
    --cache-arg="$VCTK_AUDIO_MIN_DURATION_SECONDS" \
    --cache-arg="$require_text_cache_arg" \
    --cache-arg=--text-max-length \
    --cache-arg="$VCTK_TEXT_MAX_LENGTH" \
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
    --stage2-override ar.text_conditional=true \
    --stage2-override ar.text_prefix_length="$VCTK_TEXT_PREFIX_LENGTH" \
    --stage2-override train_ar.sample_text_prompts="$VCTK_SAMPLE_PROMPTS" \
    --stage2-override train_ar.batch_size="$VCTK_STAGE2_BATCH_SIZE" \
    --stage2-override train_ar.sample_temperature=0.8 \
    --stage2-override train_ar.sample_num_images=8 \
    --stage2-override train_ar.compute_generation_fid=false \
    --stage2-override train_ar.compute_audio_generation_metrics=true \
    --stage2-override train_ar.generation_metric_num_samples="$VCTK_GENERATION_METRIC_NUM_SAMPLES" \
    --stage1-override model=laser_audio_waveform \
    --stage1-override data=vctk_waveform \
    --stage1-override data.data_dir="$VCTK_DIR" \
    --stage1-override data.batch_size="$VCTK_STAGE1_BATCH_SIZE" \
    --stage1-override data.num_workers="$VCTK_NUM_WORKERS" \
    --stage1-override data.audio_dc_remove=true \
    --stage1-override data.audio_num_samples="$VCTK_AUDIO_NUM_SAMPLES" \
    --stage1-override data.audio_max_duration_seconds="$VCTK_AUDIO_MAX_DURATION_SECONDS" \
    --stage1-override data.audio_min_duration_seconds="$VCTK_AUDIO_MIN_DURATION_SECONDS" \
    --stage1-override data.audio_require_text="$VCTK_REQUIRE_TEXT" \
    --stage1-override data.audio_peak_normalize=true \
    --stage1-override data.audio_target_peak=0.95 \
    --stage1-override data.audio_rms_normalize=true \
    --stage1-override data.audio_target_rms=0.12 \
    --stage1-override data.audio_max_gain=8.0 \
    --stage1-override data.audio_min_crop_rms=0.03 \
    --stage1-override data.audio_crop_attempts=64 \
    --stage1-override data.audio_fade_samples=1024 \
    --stage1-override train.learning_rate="$VCTK_STAGE1_LR" \
    --stage1-override train.precision="$TRAIN_PRECISION" \
    --stage1-override train.max_steps="$STAGE1_MAX_STEPS" \
    --stage1-override train.warmup_steps=750 \
    --stage1-override train.min_lr_ratio="$MIN_LR_RATIO" \
    --stage1-override train.gradient_clip_val=1.0 \
    --stage1-override train.deterministic=false \
    --stage1-override train.log_every_n_steps=20 \
    --stage1-override train.run_test_after_fit=false \
    --stage1-override checkpoint.save_top_k=1 \
    --stage1-override model.audio_downsample_rates="$rates" \
    --stage1-override model.audio_dilation_cycle="$VCTK_DILATION_CYCLE" \
    --stage1-override model.num_embeddings="$VCTK_NUM_EMBEDDINGS" \
    --stage1-override model.embedding_dim="$VCTK_EMBEDDING_DIM" \
    --stage1-override model.num_hiddens="$hidden" \
    --stage1-override model.num_residual_blocks="$residual_blocks" \
    --stage1-override model.num_residual_hiddens="$residual_hidden" \
    --stage1-override model.patch_based=false \
    --stage1-override model.patch_size=1 \
    --stage1-override model.patch_stride=1 \
    --stage1-override model.patch_reconstruction=tile \
    --stage1-override model.sparsity_level="$sparsity" \
    --stage1-override model.commitment_cost="$VCTK_COMMITMENT_COST" \
    --stage1-override model.bottleneck_loss_weight="$VCTK_BOTTLENECK_LOSS_WEIGHT" \
    --stage1-override model.dict_learning_rate="$VCTK_DICT_LR" \
    --stage1-override model.coef_max="$STAGE1_COEF_MAX" \
    --stage1-override model.sparsity_reg_weight=0.0 \
    --stage1-override model.recon_mse_weight=0.5 \
    --stage1-override model.recon_l1_weight=0.5 \
    --stage1-override model.recon_edge_weight=0.0 \
    --stage1-override model.audio_waveform_l1_weight=1.0 \
    --stage1-override model.audio_multires_stft_loss_weight=1.0 \
    --stage1-override model.audio_multires_stft_fft_sizes=[512,1024,2048] \
    --stage1-override model.data_init_from_first_batch=true \
    --stage1-override model.compute_fid=false \
    --stage1-override model.perceptual_weight=0.0 \
    --stage1-override model.adversarial_weight=0.0 \
    --stage1-override model.adversarial_start_step=1000000000 \
    --stage1-override model.adversarial_warmup_steps=0 \
    --stage1-override model.disc_start_step=1000000000 \
    --stage1-override model.log_images_every_n_steps="$VIS_LOG_EVERY_N_STEPS" \
    --stage1-override model.diag_log_interval="$DIAG_LOG_INTERVAL" \
    --stage1-override model.enable_val_latent_visuals=false \
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
    --stage1-adv-override model.use_adaptive_disc_weight="$USE_ADAPTIVE_DISC_WEIGHT" \
    --stage1-adv-override train.max_steps="$STAGE1_ADV_MAX_STEPS"
}

echo "=== non-patch deep conditional full-pipeline sweep ==="
echo "RUN_TAG=$RUN_TAG"
echo "CASES=$CASES"
echo "DOWNSAMPLE_LAYERS=$DOWNSAMPLE_LAYERS SPARSITY_LEVELS=$SPARSITY_LEVELS"
echo "PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT DRY_RUN=$DRY_RUN"
echo "epochs: stage1=$STAGE1_EPOCHS stage1_adv=$STAGE1_ADV_EPOCHS stage2=$STAGE2_EPOCHS stage1_max_steps=$STAGE1_MAX_STEPS stage1_adv_max_steps=$STAGE1_ADV_MAX_STEPS stage2_max_steps=$STAGE2_MAX_STEPS stage1_only=$STAGE1_ONLY"
echo "precision: train=$TRAIN_PRECISION stage2=$STAGE2_PRECISION"
echo "coeffs: stage1_coef_max=$STAGE1_COEF_MAX cache_coef_max_padding=$CACHE_COEFF_MAX_PADDING"
echo "image codec_capacity=$IMAGE_CODEC_CAPACITY attention_profile=$IMAGE_ATTENTION_PROFILE use_mid_attention=$IMAGE_USE_MID_ATTENTION dropout=$IMAGE_DROPOUT atoms=$IMAGE_NUM_EMBEDDINGS emb_dim=$IMAGE_EMBEDDING_DIM cache_mode=$IMAGE_CACHE_COEFF_MODE cache_bins=$IMAGE_CACHE_COEFF_BINS coeff_quant=$IMAGE_COEFF_QUANTIZATION mu=$IMAGE_COEFF_MU cache_coef_max=$IMAGE_CACHE_COEF_MAX cache_coef_max_padding=$IMAGE_CACHE_COEFF_MAX_PADDING cache_max_items=$IMAGE_CACHE_MAX_ITEMS support_order=$IMAGE_SUPPORT_ORDER cache_batch=$IMAGE_CACHE_BATCH_SIZE stage2_batch=$IMAGE_STAGE2_BATCH_SIZE nodes/job=$IMAGE_NODES gpus/node=$IMAGE_GPUS"
echo "imagenet codec_capacity=$IMAGENET_CODEC_CAPACITY atoms=$IMAGENET_NUM_EMBEDDINGS emb_dim=$IMAGENET_EMBEDDING_DIM bottleneck_loss_weight=$IMAGENET_BOTTLENECK_LOSS_WEIGHT cache_batch=$IMAGENET_CACHE_BATCH_SIZE stage2_batch=$IMAGENET_STAGE2_BATCH_SIZE stage2=class_conditional"
echo "vctk codec_capacity=$VCTK_CODEC_CAPACITY dilation_cycle=$VCTK_DILATION_CYCLE audio_num_samples=$VCTK_AUDIO_NUM_SAMPLES duration_filter=[$VCTK_AUDIO_MIN_DURATION_SECONDS,$VCTK_AUDIO_MAX_DURATION_SECONDS] require_text=$VCTK_REQUIRE_TEXT atoms=$VCTK_NUM_EMBEDDINGS q=$VCTK_COEFF_BINS coeff_quant=$VCTK_COEFF_QUANTIZATION mu=$VCTK_COEFF_MU cache_coef_max=$VCTK_CACHE_COEF_MAX support_order=$VCTK_SUPPORT_ORDER nodes/job=$AUDIO_NODES gpus/node=$AUDIO_GPUS; stage2=text_conditional"

IFS=',' read -r -a DOWNSAMPLE_ARRAY <<< "$DOWNSAMPLE_LAYERS"
IFS=',' read -r -a SPARSITY_ARRAY <<< "$SPARSITY_LEVELS"

for raw_downsample in "${DOWNSAMPLE_ARRAY[@]}"; do
  downsample="${raw_downsample// /}"
  [[ -n "$downsample" ]] || continue
  for raw_sparsity in "${SPARSITY_ARRAY[@]}"; do
    sparsity="${raw_sparsity// /}"
    [[ -n "$sparsity" ]] || continue
    if case_enabled celebahq; then submit_image_dataset celebahq "$downsample" "$sparsity"; fi
    if case_enabled ffhq; then submit_image_dataset ffhq "$downsample" "$sparsity"; fi
    if case_enabled imagenet; then submit_image_dataset imagenet "$downsample" "$sparsity"; fi
    if case_enabled vctk; then submit_vctk "$downsample" "$sparsity"; fi
  done
done

echo "=== submissions complete ==="
