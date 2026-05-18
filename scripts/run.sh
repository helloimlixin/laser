#!/bin/bash
# Canonical maintained manual runner.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PROFILE_SH="$PROJECT_DIR/scripts/profile.sh"
if [[ ! -f "$PROFILE_SH" ]]; then
  echo "ERROR: missing profile file: $PROFILE_SH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$PROFILE_SH"

usage() {
  cat <<'EOF'
Usage: bash scripts/run.sh <stage1|extract|stage2|all>

Important environment variables:
  CASE_NAME
  RUN_DIR
  DATA_DIR
  NUM_EMBEDDINGS
  SPARSITY_LEVEL
  EMBEDDING_DIM
  PATCH_BASED
  PATCH_SIZE
  PATCH_STRIDE
  PATCH_RECONSTRUCTION
EOF
}

MODE="${1:-}"
case "$MODE" in
  stage1|extract|stage2|all) ;;
  *)
    usage >&2
    exit 1
    ;;
esac

CASE_NAME="${CASE_NAME:-debug_case}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/$CASE_NAME}"
STAGE1_DIR="${STAGE1_DIR:-$RUN_DIR/stage1}"
STAGE2_DIR="${STAGE2_DIR:-$RUN_DIR/stage2}"
TOKEN_CACHE="${TOKEN_CACHE:-$RUN_DIR/token_cache.pt}"
STAGE1_CKPT="${STAGE1_CKPT:-}"

PYTHONUSERBASE="${PYTHONUSERBASE:-/scratch/$USER/.pydeps/laser_src_py311}"
export PYTHONUSERBASE
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python3/python not found" >&2
  exit 127
fi

python_version() {
  "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || printf 'unknown\n'
}

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: $PYTHON_BIN is Python $(python_version "$PYTHON_BIN"); LASER requires Python >= 3.10." >&2
  echo "Set PYTHON_BIN to a supported environment before running this script." >&2
  exit 2
fi

INSTALL_DEPS="${INSTALL_DEPS:-false}"
if [[ "$INSTALL_DEPS" == "true" ]]; then
  "$PYTHON_BIN" -m pip install --user --quiet \
    scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \
    torch-fidelity matplotlib lpips soundfile 2>/dev/null || true
fi

debug_profile_load
if [[ -z "${MODEL_CONFIG:-}" || "$MODEL_CONFIG" == "auto" ]]; then
  if [[ "$DATASET" == "vctk" ]]; then
    MODEL_CONFIG="laser_audio"
  else
    MODEL_CONFIG="laser"
  fi
fi

CASE_LABEL="${CASE_LABEL:-$(debug_profile_case_label "$CASE_NAME")}"
DICT_LABEL="${DICT_LABEL:-$(debug_profile_dict_label "$NUM_EMBEDDINGS")}"
RUN_FAMILY="${RUN_FAMILY:-$(debug_profile_run_family)}"
EXP_SLUG="${EXP_SLUG:-$(debug_profile_slug "$NUM_EMBEDDINGS" "$CASE_NAME")}"
WANDB_GROUP="${WANDB_GROUP:-$EXP_SLUG}"
WANDB_STAGE1_NAME="${WANDB_STAGE1_NAME:-${EXP_SLUG}-s1}"
WANDB_STAGE2_NAME="${WANDB_STAGE2_NAME:-${EXP_SLUG}-s2}"

mkdir -p "$RUN_DIR" "$STAGE1_DIR" "$STAGE2_DIR"

print_summary() {
  echo ""
  echo "========================================"
  echo "RUN"
  echo "========================================"
  echo "mode:               $MODE"
  echo "case:               $CASE_NAME"
  echo "case_label:         $CASE_LABEL"
  echo "dict_label:         $DICT_LABEL"
  echo "exp_slug:           $EXP_SLUG"
  echo "run_dir:            $RUN_DIR"
  echo "stage1_dir:         $STAGE1_DIR"
  echo "stage2_dir:         $STAGE2_DIR"
  echo "token_cache:        $TOKEN_CACHE"
  echo "dataset:            $DATASET"
  echo "model_config:       $MODEL_CONFIG"
  echo "data_dir:           $DATA_DIR"
  echo "image_size:         $IMAGE_SIZE"
  echo "dictionary_size:    $NUM_EMBEDDINGS"
  echo "sparsity:           $SPARSITY_LEVEL"
  echo "embedding_dim:      $EMBEDDING_DIM"
  echo "patch_based:        $PATCH_BASED"
  if [[ "$PATCH_BASED" == "true" ]]; then
    echo "patch:              size=$PATCH_SIZE stride=$PATCH_STRIDE recon=$PATCH_RECONSTRUCTION"
  fi
  echo "devices:            $DEVICES"
  echo "strategy:           $TRAIN_STRATEGY"
  echo "stage1_epochs:      $STAGE1_EPOCHS"
  echo "stage2_epochs:      $STAGE2_EPOCHS"
  echo "python:             $PYTHON_BIN ($(python_version "$PYTHON_BIN"))"
  echo "wandb_group:        $WANDB_GROUP"
  echo "wandb_stage1:       $WANDB_STAGE1_NAME"
  echo "wandb_stage2:       $WANDB_STAGE2_NAME"
  echo "========================================"
}

find_stage1_ckpt() {
  if [[ -n "$STAGE1_CKPT" ]]; then
    if [[ ! -f "$STAGE1_CKPT" ]]; then
      echo "ERROR: STAGE1_CKPT does not exist: $STAGE1_CKPT" >&2
      exit 1
    fi
    printf '%s\n' "$STAGE1_CKPT"
    return
  fi

  local ckpt
  ckpt="$(find "$STAGE1_DIR" -path '*/checkpoints/*/final.ckpt' -type f | sort | tail -1)"
  if [[ -z "$ckpt" ]]; then
    ckpt="$(find "$STAGE1_DIR" -path '*/checkpoints/*/last.ckpt' -type f | sort | tail -1)"
  fi
  if [[ -z "$ckpt" ]]; then
    echo "ERROR: no stage-1 checkpoint found under $STAGE1_DIR" >&2
    exit 1
  fi
  printf '%s\n' "$ckpt"
}

run_stage1() {
  local -a args=(
    train_stage1_autoencoder.py
    "seed=$SEED"
    "output_dir=$STAGE1_DIR"
    "model=$MODEL_CONFIG"
    "data=$DATASET"
    "model.backbone=vqgan"
    "model.num_downsamples=$NUM_DOWNSAMPLES"
    "model.attn_resolutions=$ATTN_RESOLUTIONS"
    "model.channel_multipliers=$CHANNEL_MULTIPLIERS"
    "model.backbone_latent_channels=$BACKBONE_LATENT_CHANNELS"
    "model.max_ch_mult=$MAX_CH_MULT"
    "model.decoder_extra_residual_layers=$DECODER_EXTRA_RESIDUAL_LAYERS"
    "model.use_mid_attention=$USE_MID_ATTENTION"
    "model.num_hiddens=$NUM_HIDDENS"
    "model.num_residual_blocks=$NUM_RESIDUAL_BLOCKS"
    "model.num_residual_hiddens=$NUM_RESIDUAL_HIDDENS"
    "model.num_embeddings=$NUM_EMBEDDINGS"
    "model.embedding_dim=$EMBEDDING_DIM"
    "model.sparsity_level=$SPARSITY_LEVEL"
    "model.patch_based=$PATCH_BASED"
    "model.coef_max=$COEFF_MAX"
    "model.dict_learning_rate=$DICT_LR"
    "model.variational_coeffs=$VARIATIONAL_COEFFS"
    "model.variational_coeff_kl_weight=$VARIATIONAL_COEFF_KL_WEIGHT"
    "model.variational_coeff_prior_std=$VARIATIONAL_COEFF_PRIOR_STD"
    "model.variational_coeff_min_std=$VARIATIONAL_COEFF_MIN_STD"
    "model.perceptual_weight=$PERCEPTUAL_WEIGHT"
    "model.log_images_every_n_steps=$S1_LOG_IMAGES_EVERY"
    "model.diag_log_interval=$S1_DIAG_LOG_INTERVAL"
    "model.enable_val_latent_visuals=$S1_ENABLE_VAL_LATENT_VISUALS"
    "model.compute_fid=$S1_COMPUTE_FID"
    "model.fid_feature=2048"
    "data.dataset=$DATASET"
    "data.data_dir=$DATA_DIR"
    "data.image_size=$IMAGE_SIZE"
    "data.batch_size=$BATCH_SIZE"
    "data.num_workers=$NUM_WORKERS"
    "train.learning_rate=$STAGE1_LR"
    "train.warmup_steps=$WARMUP_STEPS"
    "train.min_lr_ratio=$MIN_LR_RATIO"
    "train.max_epochs=$STAGE1_EPOCHS"
    "train.accelerator=$ACCELERATOR"
    "train.num_nodes=$NUM_NODES"
    "train.devices=$DEVICES"
    "train.strategy=$TRAIN_STRATEGY"
    "train.precision=$PRECISION"
    "train.gradient_clip_val=$GRAD_CLIP_VAL"
    "train.log_every_n_steps=25"
    "train.val_check_interval=$S1_VAL_CHECK_INTERVAL"
    "wandb.project=$WANDB_PROJECT"
    "wandb.name=$WANDB_STAGE1_NAME"
    "wandb.group=$WANDB_GROUP"
    "wandb.tags=[$DATASET,$RUN_FAMILY,$DICT_LABEL,$CASE_LABEL,s1]"
    "wandb.append_timestamp=false"
  )

  if [[ "$PATCH_BASED" == "true" ]]; then
    args+=(
      "model.patch_size=$PATCH_SIZE"
      "model.patch_stride=$PATCH_STRIDE"
      "model.patch_reconstruction=$PATCH_RECONSTRUCTION"
    )
  fi

  if [[ "$DATASET" == "vctk" ]]; then
    args+=(
      "data.sample_rate=$AUDIO_SAMPLE_RATE"
      "data.audio_num_samples=$AUDIO_NUM_SAMPLES"
      "data.stft_n_fft=$AUDIO_STFT_N_FFT"
      "data.stft_hop_length=$AUDIO_STFT_HOP_LENGTH"
      "data.stft_win_length=$AUDIO_STFT_WIN_LENGTH"
      "data.stft_power=$AUDIO_STFT_POWER"
      "data.stft_log_offset=$AUDIO_STFT_LOG_OFFSET"
    )
  fi

  "$PYTHON_BIN" "${args[@]}"
}

run_extract() {
  local ckpt
  ckpt="$(find_stage1_ckpt)"
  local -a args=(
    cache.py
    --stage1-checkpoint "$ckpt" \
    --output-path "$TOKEN_CACHE" \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --coeff-max "$COEFF_MAX" \
    --coeff-bins "$COEFF_BINS" \
    --coeff-quantization "$COEFF_QUANTIZATION" \
    --coeff-mu "$COEFF_MU"
  )
  if [[ "$DATASET" == "vctk" ]]; then
    args+=(
      --sample-rate "$AUDIO_SAMPLE_RATE"
      --audio-num-samples "$AUDIO_NUM_SAMPLES"
      --stft-n-fft "$AUDIO_STFT_N_FFT"
      --stft-hop-length "$AUDIO_STFT_HOP_LENGTH"
      --stft-win-length "$AUDIO_STFT_WIN_LENGTH"
      --stft-power "$AUDIO_STFT_POWER"
      --stft-log-offset "$AUDIO_STFT_LOG_OFFSET"
    )
  fi
  "$PYTHON_BIN" "${args[@]}"
}

run_stage2() {
  if [[ ! -f "$TOKEN_CACHE" ]]; then
    echo "ERROR: token cache not found: $TOKEN_CACHE" >&2
    exit 1
  fi

  local -a args=(
    train_stage2_prior.py
    "token_cache_path=$TOKEN_CACHE"
    "output_dir=$STAGE2_DIR"
    "seed=$SEED"
    "ar.type=sparse_spatial_depth"
    "ar.d_model=$AR_D_MODEL"
    "ar.n_heads=$AR_N_HEADS"
    "ar.n_layers=$AR_N_LAYERS"
    "ar.d_ff=$AR_D_FF"
    "ar.dropout=0.1"
    "ar.learning_rate=$STAGE2_LR"
    "ar.warmup_steps=$STAGE2_WARMUP_STEPS"
    "ar.min_lr_ratio=$STAGE2_MIN_LR_RATIO"
    "ar.autoregressive_coeffs=$AR_AUTOREGRESSIVE_COEFFS"
    "ar.coeff_loss_type=$AR_COEFF_LOSS_TYPE"
    "ar.coeff_loss_weight=$AR_COEFF_LOSS_WEIGHT"
    "ar.sample_coeff_temperature=$AR_SAMPLE_COEFF_TEMPERATURE"
    "ar.sample_coeff_mode=$AR_SAMPLE_COEFF_MODE"
    "train_ar.batch_size=$STAGE2_BATCH_SIZE"
    "train_ar.max_epochs=$STAGE2_EPOCHS"
    "train_ar.accelerator=$ACCELERATOR"
    "train_ar.num_nodes=$NUM_NODES"
    "train_ar.devices=$DEVICES"
    "train_ar.strategy=$TRAIN_AR_STRATEGY"
    "train_ar.precision=$PRECISION"
    "train_ar.gradient_clip_val=$GRAD_CLIP_VAL"
    "train_ar.log_every_n_steps=$S2_LOG_EVERY_N_STEPS"
    "train_ar.sample_every_n_epochs=$S2_SAMPLE_EVERY_N_EPOCHS"
    "train_ar.sample_log_to_wandb=true"
    "train_ar.sample_num_images=$S2_SAMPLE_NUM_IMAGES"
    "train_ar.sample_temperature=$SAMPLE_TEMP"
    "train_ar.sample_coeff_temperature=$AR_SAMPLE_COEFF_TEMPERATURE"
    "train_ar.sample_coeff_mode=$AR_SAMPLE_COEFF_MODE"
    "train_ar.compute_generation_fid=$S2_COMPUTE_GENERATION_FID"
    "train_ar.compute_audio_generation_metrics=$S2_COMPUTE_AUDIO_GENERATION_METRICS"
    "train_ar.generation_metric_num_samples=$S2_GENERATION_METRIC_NUM_SAMPLES"
    "data.dataset=$DATASET"
    "data.data_dir=$DATA_DIR"
    "data.image_size=$IMAGE_SIZE"
    "data.num_workers=$NUM_WORKERS"
    "wandb.project=$WANDB_PROJECT"
    "wandb.name=$WANDB_STAGE2_NAME"
    "wandb.group=$WANDB_GROUP"
    "wandb.tags=[$DATASET,$RUN_FAMILY,$DICT_LABEL,$CASE_LABEL,s2]"
    "wandb.append_timestamp=false"
  )

  "$PYTHON_BIN" "${args[@]}"
}

print_summary

case "$MODE" in
  stage1) run_stage1 ;;
  extract) run_extract ;;
  stage2) run_stage2 ;;
  all)
    run_stage1
    run_extract
    run_stage2
    ;;
esac

echo "DONE: $EXP_SLUG ($MODE)"
