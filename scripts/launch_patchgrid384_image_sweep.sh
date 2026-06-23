#!/usr/bin/env bash
# Launch d4/p2s2/k3, d3/p4s4/k3, and d2/p8s8/k3 LASER image runs.
#
# All three variants keep a 256px image at an 8x8 sparse patch grid:
#   d4 latent 16x16 with p2s2 -> 8x8 patches
#   d3 latent 32x32 with p4s4 -> 8x8 patches
#   d2 latent 64x64 with p8s8 -> 8x8 patches
# With k=3 and interleaved atom/coeff tokens, stage 2 sees 8*8*3*2 = 384 tokens.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

SUBMIT="${SUBMIT:-$HERE/submit_multimodal_sweep.py}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-0}"

PARTITION="${PARTITION:-gpu-redhat}"
NODES="${NODES:-1}"
GPUS="${GPUS:-2}"
CPUS="${CPUS:-12}"
MEM_MB="${MEM_MB:-240000}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
RUN_TAG="${RUN_TAG:-patchgrid384-k3-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_patchgrid384_image_sweep}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
# gpu031 produced uncorrectable ECC errors on 2026-06-20 and caused a cascade
# of early NCCL failures. Keep it out of relaunches unless explicitly overridden.
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu031,gpu018,gpuk[005-018]}"
DRY_RUN="${DRY_RUN:-0}"
STAGE1_ONLY="${STAGE1_ONLY:-0}"

CASES="${CASES:-celebahq,ffhq,imagenet}"
VARIANTS="${VARIANTS:-d4p2,d3p4,d2p8}"

CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/Projects/data/celeba_hq}"
FFHQ_DIR="${FFHQ_DIR:-/cache/home/$USER/datasets/ffhq/images1024x1024_webp}"
IMAGENET_DIR="${IMAGENET_DIR:-/scratch/$USER/Projects/data/imagenet}"

# Step-bounded so each chained job can reach adversarial continuation, cache
# extraction, and stage 2 within the allocation on large image folders. ImageNet
# needs a longer stage-1 path than the face datasets; otherwise the stage-2 cache
# is built from a sharp but still undertrained adversarial continuation.
STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE1_ADV_EPOCHS="${STAGE1_ADV_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
USER_STAGE1_STEPS="${STAGE1_STEPS:-}"
USER_STAGE1_ADV_STEPS="${STAGE1_ADV_STEPS:-}"
STAGE1_STEPS="${STAGE1_STEPS:-60000}"
STAGE1_ADV_STEPS="${STAGE1_ADV_STEPS:-20000}"
IMAGENET_STAGE1_STEPS="${IMAGENET_STAGE1_STEPS:-${USER_STAGE1_STEPS:-160000}}"
IMAGENET_STAGE1_ADV_STEPS="${IMAGENET_STAGE1_ADV_STEPS:-${USER_STAGE1_ADV_STEPS:-80000}}"
STAGE2_STEPS="${STAGE2_STEPS:-120000}"

STAGE1_LR="${STAGE1_LR:-7.5e-5}"
STAGE1_ADV_LR="${STAGE1_ADV_LR:-5.0e-5}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-5000}"
STAGE2_LR="${STAGE2_LR:-2.5e-4}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-1500}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"

COEFF_BINS="${COEFF_BINS:-128}"
COEF_MAX="${COEF_MAX:-auto_p99.9}"
CACHE_COEF_MAX="${CACHE_COEF_MAX:-$COEF_MAX}"
CACHE_COEFF_MAX_PADDING="${CACHE_COEFF_MAX_PADDING:-1.05}"
STAGE1_COEF_MAX="${STAGE1_COEF_MAX:-null}"
SUPPORT_ORDER="${SUPPORT_ORDER:-atom_id}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
DICT_LR="${DICT_LR:-2.5e-5}"
BOUNDED_OMP_REFINE_STEPS="${BOUNDED_OMP_REFINE_STEPS:-16}"
SPARSITY_REG_WEIGHT="${SPARSITY_REG_WEIGHT:-0.0}"
STAGE1_INIT_CKPT="${STAGE1_INIT_CKPT:-}"

RECON_MSE_WEIGHT="${RECON_MSE_WEIGHT:-0.25}"
RECON_L1_WEIGHT="${RECON_L1_WEIGHT:-1.0}"
RECON_EDGE_WEIGHT="${RECON_EDGE_WEIGHT:-0.5}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.2}"
PERCEPTUAL_START_STEP="${PERCEPTUAL_START_STEP:-2000}"
PERCEPTUAL_WARMUP_STEPS="${PERCEPTUAL_WARMUP_STEPS:-4000}"

ADVERSARIAL_WEIGHT="${ADVERSARIAL_WEIGHT:-0.05}"
ADVERSARIAL_WARMUP_STEPS="${ADVERSARIAL_WARMUP_STEPS:-2000}"
DISCRIMINATOR_LR="${DISCRIMINATOR_LR:-5.0e-5}"
DISCRIMINATOR_CHANNELS="${DISCRIMINATOR_CHANNELS:-64}"
DISCRIMINATOR_LAYERS="${DISCRIMINATOR_LAYERS:-3}"
USE_ADAPTIVE_DISC_WEIGHT="${USE_ADAPTIVE_DISC_WEIGHT:-false}"

VIS_LOG_EVERY_N_STEPS="${VIS_LOG_EVERY_N_STEPS:-1000}"
DIAG_LOG_INTERVAL="${DIAG_LOG_INTERVAL:-100}"
DICTIONARY_VIS_MAX_VECTORS="${DICTIONARY_VIS_MAX_VECTORS:-1024}"

STAGE2_D_MODEL="${STAGE2_D_MODEL:-512}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-8}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-12}"
STAGE2_D_FF="${STAGE2_D_FF:-2048}"
STAGE2_GLOBAL_TOKENS="${STAGE2_GLOBAL_TOKENS:-16}"
SAMPLE_EVERY_N_STEPS="${SAMPLE_EVERY_N_STEPS:-5000}"
SAMPLE_NUM_IMAGES="${SAMPLE_NUM_IMAGES:-16}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.8}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: PYTHON_SUBMIT must be Python >= 3.8; got $PYTHON_SUBMIT" >&2
  exit 2
fi

case_enabled() {
  local needle="$1"
  case ",$CASES," in
    *,"$needle",*) return 0 ;;
    *) return 1 ;;
  esac
}

variant_enabled() {
  local needle="$1"
  case ",$VARIANTS," in
    *,"$needle",*) return 0 ;;
    *) return 1 ;;
  esac
}

append_overrides() {
  local flag="$1"
  shift
  for item in "$@"; do
    CMD+=("$flag" "$item")
  done
}

variant_settings() {
  local variant="$1"
  case "$variant" in
    d4p2)
      NUM_DOWNSAMPLES=4
      CHANNEL_MULTIPLIERS="[1,1,2,2,4]"
      ATTN_RESOLUTIONS="[16,32]"
      PATCH_SIZE=2
      PATCH_STRIDE=2
      NUM_EMBEDDINGS="${D4_NUM_EMBEDDINGS:-8192}"
      EMBEDDING_DIM="${D4_EMBEDDING_DIM:-128}"
      PATCH_DIM=$((PATCH_SIZE * PATCH_SIZE * EMBEDDING_DIM))
      VARIANT_LABEL="d4-p2s2-k3-a${NUM_EMBEDDINGS}-e${EMBEDDING_DIM}-pdim${PATCH_DIM}"
      NUM_HIDDENS=192
      NUM_RESIDUAL_HIDDENS=128
      NUM_RESIDUAL_BLOCKS=4
      DECODER_EXTRA_RESIDUAL_LAYERS=3
      BACKBONE_LATENT_CHANNELS=512
      BOTTLENECK_LOSS_WEIGHT="${D4_BOTTLENECK_LOSS_WEIGHT:-0.75}"
      FACE_STAGE1_BATCH_SIZE="${D4_FACE_STAGE1_BATCH_SIZE:-3}"
      IMAGENET_STAGE1_BATCH_SIZE="${D4_IMAGENET_STAGE1_BATCH_SIZE:-2}"
      FACE_STAGE2_BATCH_SIZE="${D4_FACE_STAGE2_BATCH_SIZE:-4}"
      IMAGENET_STAGE2_BATCH_SIZE="${D4_IMAGENET_STAGE2_BATCH_SIZE:-2}"
      CACHE_BATCH_SIZE="${D4_CACHE_BATCH_SIZE:-8}"
      ;;
    d3p4)
      VARIANT_LABEL="d3-p4s4-k3-a16384-e64-pdim1024"
      NUM_DOWNSAMPLES=3
      CHANNEL_MULTIPLIERS="[1,1,2,2]"
      ATTN_RESOLUTIONS="[32]"
      PATCH_SIZE=4
      PATCH_STRIDE=4
      NUM_EMBEDDINGS=16384
      EMBEDDING_DIM=64
      PATCH_DIM=1024
      NUM_HIDDENS=192
      NUM_RESIDUAL_HIDDENS=112
      NUM_RESIDUAL_BLOCKS=4
      DECODER_EXTRA_RESIDUAL_LAYERS=3
      BACKBONE_LATENT_CHANNELS=384
      BOTTLENECK_LOSS_WEIGHT="${D3_BOTTLENECK_LOSS_WEIGHT:-0.5}"
      FACE_STAGE1_BATCH_SIZE="${D3_FACE_STAGE1_BATCH_SIZE:-2}"
      IMAGENET_STAGE1_BATCH_SIZE="${D3_IMAGENET_STAGE1_BATCH_SIZE:-2}"
      FACE_STAGE2_BATCH_SIZE="${D3_FACE_STAGE2_BATCH_SIZE:-4}"
      IMAGENET_STAGE2_BATCH_SIZE="${D3_IMAGENET_STAGE2_BATCH_SIZE:-2}"
      CACHE_BATCH_SIZE="${D3_CACHE_BATCH_SIZE:-6}"
      ;;
    d2p8)
      VARIANT_LABEL="d2-p8s8-k3-a32768-e32-pdim2048"
      NUM_DOWNSAMPLES=2
      CHANNEL_MULTIPLIERS="[1,1,2]"
      ATTN_RESOLUTIONS="[]"
      PATCH_SIZE=8
      PATCH_STRIDE=8
      NUM_EMBEDDINGS=32768
      EMBEDDING_DIM=32
      PATCH_DIM=2048
      NUM_HIDDENS=160
      NUM_RESIDUAL_HIDDENS=96
      NUM_RESIDUAL_BLOCKS=3
      DECODER_EXTRA_RESIDUAL_LAYERS=2
      BACKBONE_LATENT_CHANNELS=256
      BOTTLENECK_LOSS_WEIGHT="${D2_BOTTLENECK_LOSS_WEIGHT:-0.35}"
      FACE_STAGE1_BATCH_SIZE="${D2_FACE_STAGE1_BATCH_SIZE:-1}"
      IMAGENET_STAGE1_BATCH_SIZE="${D2_IMAGENET_STAGE1_BATCH_SIZE:-1}"
      FACE_STAGE2_BATCH_SIZE="${D2_FACE_STAGE2_BATCH_SIZE:-2}"
      IMAGENET_STAGE2_BATCH_SIZE="${D2_IMAGENET_STAGE2_BATCH_SIZE:-2}"
      CACHE_BATCH_SIZE="${D2_CACHE_BATCH_SIZE:-4}"
      ;;
    *)
      echo "ERROR: unknown variant '$variant' (expected d4p2,d3p4,d2p8)" >&2
      exit 2
      ;;
  esac
}

dataset_settings() {
  local dataset="$1"
  DATASET_STAGE1_STEPS="$STAGE1_STEPS"
  DATASET_STAGE1_ADV_STEPS="$STAGE1_ADV_STEPS"
  case "$dataset" in
    celebahq)
      DATA_DIR="$CELEBAHQ_DIR"
      DATA_AUGMENT=false
      STAGE1_BATCH_SIZE="$FACE_STAGE1_BATCH_SIZE"
      STAGE2_BATCH_SIZE="$FACE_STAGE2_BATCH_SIZE"
      CLASS_CONDITIONAL=false
      NUM_CLASSES=0
      SAMPLE_CLASS_LABELS="[]"
      ;;
    ffhq)
      DATA_DIR="$FFHQ_DIR"
      DATA_AUGMENT=true
      STAGE1_BATCH_SIZE="$FACE_STAGE1_BATCH_SIZE"
      STAGE2_BATCH_SIZE="$FACE_STAGE2_BATCH_SIZE"
      CLASS_CONDITIONAL=false
      NUM_CLASSES=0
      SAMPLE_CLASS_LABELS="[]"
      ;;
    imagenet)
      DATA_DIR="$IMAGENET_DIR"
      DATA_AUGMENT=true
      STAGE1_BATCH_SIZE="$IMAGENET_STAGE1_BATCH_SIZE"
      STAGE2_BATCH_SIZE="$IMAGENET_STAGE2_BATCH_SIZE"
      CLASS_CONDITIONAL=true
      NUM_CLASSES=1000
      SAMPLE_CLASS_LABELS="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"
      DATASET_STAGE1_STEPS="$IMAGENET_STAGE1_STEPS"
      DATASET_STAGE1_ADV_STEPS="$IMAGENET_STAGE1_ADV_STEPS"
      ;;
    *)
      echo "ERROR: unknown dataset '$dataset' (expected celebahq,ffhq,imagenet)" >&2
      exit 2
      ;;
  esac
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: required dataset directory not found for $dataset: $DATA_DIR" >&2
    exit 1
  fi
  if [[ "$dataset" == "imagenet" && "$DATASET_STAGE1_ADV_STEPS" =~ ^[0-9]+$ && "$ADVERSARIAL_WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
    if (( DATASET_STAGE1_ADV_STEPS <= ADVERSARIAL_WARMUP_STEPS )); then
      echo "ERROR: imagenet DATASET_STAGE1_ADV_STEPS=$DATASET_STAGE1_ADV_STEPS must exceed ADVERSARIAL_WARMUP_STEPS=$ADVERSARIAL_WARMUP_STEPS before stage2 cache extraction." >&2
      exit 2
    fi
  fi
}

submit_dataset_variant() {
  local dataset="$1"
  local variant="$2"
  variant_settings "$variant"
  dataset_settings "$dataset"

  local label="${RUN_TAG}-${dataset}-${VARIANT_LABEL}-seq384-s1${DATASET_STAGE1_STEPS}-adv${DATASET_STAGE1_ADV_STEPS}-s2${STAGE2_STEPS}"

  CMD=(
    "$PYTHON_SUBMIT" "$SUBMIT"
    --cases "$dataset"
    --full-training
    --model-family laser
    --project "$PROJECT"
    --partition "$PARTITION"
    --nodes "$NODES"
    --gpus "$GPUS"
    --cpus-per-task "$CPUS"
    --mem-mb "$MEM_MB"
    --time-limit "$TIME_LIMIT"
    --run-root-base "$RUN_ROOT_BASE"
    --snapshot-root "$SNAPSHOT_ROOT"
    --celebahq-dir "$CELEBAHQ_DIR"
    --ffhq-dir "$FFHQ_DIR"
    --imagenet-dir "$IMAGENET_DIR"
    --run-label "$label"
    --stage1-epochs "$STAGE1_EPOCHS"
    --stage1-adv-epochs "$STAGE1_ADV_EPOCHS"
    --stage2-epochs "$STAGE2_EPOCHS"
    --cache-arg=--coeff-bins
    --cache-arg "$COEFF_BINS"
    --cache-arg=--coeff-max
    --cache-arg "$CACHE_COEF_MAX"
    --cache-arg=--coeff-max-padding
    --cache-arg "$CACHE_COEFF_MAX_PADDING"
    --cache-arg=--coeff-quantization
    --cache-arg mu_law
    --cache-arg=--coeff-mu
    --cache-arg 255.0
    --cache-arg=--support-order
    --cache-arg "$SUPPORT_ORDER"
    --cache-arg=--batch-size
    --cache-arg "$CACHE_BATCH_SIZE"
  )
  if [[ -n "${EXCLUDE_NODES// }" ]]; then
    CMD+=(--exclude-nodes "$EXCLUDE_NODES")
  fi
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    CMD+=(--dry-run)
  fi
  if [[ "$STAGE1_ONLY" == "1" || "$STAGE1_ONLY" == "true" ]]; then
    CMD+=(--stage1-only)
  fi

  append_overrides --stage1-override \
    "model=laser" \
    "data.image_size=256" \
    "data.batch_size=$STAGE1_BATCH_SIZE" \
    "data.num_workers=8" \
    "data.augment=$DATA_AUGMENT" \
    "train.max_epochs=$STAGE1_EPOCHS" \
    "train.max_steps=$DATASET_STAGE1_STEPS" \
    "train.learning_rate=$STAGE1_LR" \
    "train.warmup_steps=$STAGE1_WARMUP_STEPS" \
    "train.min_lr_ratio=$MIN_LR_RATIO" \
    "train.val_check_interval=15000" \
    "train.limit_val_batches=256" \
    "train.limit_test_batches=0" \
    "train.log_every_n_steps=50" \
    "train.gradient_clip_val=1.0" \
    "train.run_test_after_fit=false" \
    "checkpoint.save_top_k=3" \
    "checkpoint.save_last=true" \
    "model.backbone=ddpm" \
    "model.num_downsamples=$NUM_DOWNSAMPLES" \
    "model.channel_multipliers=$CHANNEL_MULTIPLIERS" \
    "model.attn_resolutions=$ATTN_RESOLUTIONS" \
    "model.use_mid_attention=true" \
    "model.num_hiddens=$NUM_HIDDENS" \
    "model.num_residual_hiddens=$NUM_RESIDUAL_HIDDENS" \
    "model.num_residual_blocks=$NUM_RESIDUAL_BLOCKS" \
    "model.decoder_extra_residual_layers=$DECODER_EXTRA_RESIDUAL_LAYERS" \
    "model.backbone_latent_channels=$BACKBONE_LATENT_CHANNELS" \
    "model.patch_based=true" \
    "model.patch_size=$PATCH_SIZE" \
    "model.patch_stride=$PATCH_STRIDE" \
    "model.patch_reconstruction=tile" \
    "model.sparsity_level=3" \
    "model.num_embeddings=$NUM_EMBEDDINGS" \
    "model.embedding_dim=$EMBEDDING_DIM" \
    "model.commitment_cost=$COMMITMENT_COST" \
    "model.bottleneck_loss_weight=$BOTTLENECK_LOSS_WEIGHT" \
    "model.dict_learning_rate=$DICT_LR" \
    "model.coef_max=$STAGE1_COEF_MAX" \
    "model.sparsity_reg_weight=$SPARSITY_REG_WEIGHT" \
    "model.data_init_from_first_batch=true" \
    "model.recon_mse_weight=$RECON_MSE_WEIGHT" \
    "model.recon_l1_weight=$RECON_L1_WEIGHT" \
    "model.recon_edge_weight=$RECON_EDGE_WEIGHT" \
    "model.perceptual_weight=$PERCEPTUAL_WEIGHT" \
    "model.perceptual_start_step=$PERCEPTUAL_START_STEP" \
    "model.perceptual_warmup_steps=$PERCEPTUAL_WARMUP_STEPS" \
    "model.adversarial_weight=0.0" \
    "model.adversarial_start_step=1000000000" \
    "model.adversarial_warmup_steps=0" \
    "model.disc_start_step=1000000000" \
    "model.disc_learning_rate=$DISCRIMINATOR_LR" \
    "model.disc_channels=$DISCRIMINATOR_CHANNELS" \
    "model.disc_num_layers=$DISCRIMINATOR_LAYERS" \
    "model.disc_norm=group" \
    "model.disc_loss=hinge" \
    "model.use_adaptive_disc_weight=$USE_ADAPTIVE_DISC_WEIGHT" \
    "model.compute_fid=true" \
    "model.log_images_every_n_steps=$VIS_LOG_EVERY_N_STEPS" \
    "model.diag_log_interval=$DIAG_LOG_INTERVAL" \
    "model.enable_val_latent_visuals=true" \
    "model.codebook_visual_max_vectors=$DICTIONARY_VIS_MAX_VECTORS" \
    "wandb.name=$dataset-$VARIANT_LABEL-stage1-noadv" \
    "wandb.tags=[train,laser,$dataset,stage1,patchgrid384,$variant]"
  if [[ -n "${STAGE1_INIT_CKPT// }" ]]; then
    append_overrides --stage1-override "init_ckpt_path=$STAGE1_INIT_CKPT"
  fi

  append_overrides --stage1-adv-override \
    "train.max_epochs=$STAGE1_ADV_EPOCHS" \
    "train.max_steps=$DATASET_STAGE1_ADV_STEPS" \
    "train.learning_rate=$STAGE1_ADV_LR" \
    "model.adversarial_weight=$ADVERSARIAL_WEIGHT" \
    "model.adversarial_start_step=0" \
    "model.adversarial_warmup_steps=$ADVERSARIAL_WARMUP_STEPS" \
    "model.disc_start_step=0" \
    "model.use_adaptive_disc_weight=$USE_ADAPTIVE_DISC_WEIGHT" \
    "wandb.name=$dataset-$VARIANT_LABEL-stage1-adv" \
    "wandb.tags=[train,laser,$dataset,stage1,adversarial,patchgrid384,$variant]"

  append_overrides --stage2-override \
    "ar.max_steps=$STAGE2_STEPS" \
    "ar.d_model=$STAGE2_D_MODEL" \
    "ar.n_heads=$STAGE2_N_HEADS" \
    "ar.n_layers=$STAGE2_N_LAYERS" \
    "ar.d_ff=$STAGE2_D_FF" \
    "ar.n_global_spatial_tokens=$STAGE2_GLOBAL_TOKENS" \
    "ar.dropout=0.1" \
    "ar.learning_rate=$STAGE2_LR" \
    "ar.warmup_steps=$STAGE2_WARMUP_STEPS" \
    "ar.min_lr_ratio=$MIN_LR_RATIO" \
    "ar.autoregressive_coeffs=true" \
    "ar.class_conditional=$CLASS_CONDITIONAL" \
    "ar.num_classes=$NUM_CLASSES" \
    "ar.coeff_loss_type=auto" \
    "ar.coeff_loss_weight=1.0" \
    "ar.coeff_huber_delta=0.25" \
    "train_ar.batch_size=$STAGE2_BATCH_SIZE" \
    "train_ar.gradient_clip_val=1.0" \
    "train_ar.val_check_interval=1.0" \
    "train_ar.log_every_n_steps=20" \
    "train_ar.checkpoint_save_top_k=3" \
    "train_ar.checkpoint_save_last=true" \
    "train_ar.checkpoint_keep_recent=3" \
    "train_ar.checkpoint_every_n_epochs=1" \
    "train_ar.sample_every_n_epochs=0" \
    "train_ar.sample_every_n_steps=$SAMPLE_EVERY_N_STEPS" \
    "train_ar.sample_log_to_wandb=true" \
    "train_ar.sample_num_images=$SAMPLE_NUM_IMAGES" \
    "train_ar.sample_class_labels=$SAMPLE_CLASS_LABELS" \
    "train_ar.sample_temperature=$SAMPLE_TEMPERATURE" \
    "train_ar.sample_top_k=0" \
    "train_ar.sample_coeff_mode=gaussian" \
    "train_ar.compute_generation_fid=false" \
    "train_ar.generation_metric_num_samples=0" \
    "train_ar.run_test_after_fit=false" \
    "train_ar.save_final_samples_after_fit=true" \
    "wandb.name=$dataset-$VARIANT_LABEL-stage2" \
    "wandb.tags=[train,laser,$dataset,stage2,transformer,generation,patchgrid384,$variant]"

  echo "=========================================================="
  echo "Submitting $dataset $VARIANT_LABEL: seq=384 patch_dim=$PATCH_DIM atoms=$NUM_EMBEDDINGS embedding_dim=$EMBEDDING_DIM stage1_steps=$DATASET_STAGE1_STEPS stage1_adv_steps=$DATASET_STAGE1_ADV_STEPS"
  "${CMD[@]}"
}

for variant in d4p2 d3p4 d2p8; do
  variant_enabled "$variant" || continue
  for dataset in celebahq ffhq imagenet; do
    case_enabled "$dataset" || continue
    submit_dataset_variant "$dataset" "$variant"
  done
done

echo "=========================================================="
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "(dry run) previewed patch-grid 384 image sweep."
else
  echo "Submitted patch-grid 384 image sweep."
fi
