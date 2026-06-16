#!/usr/bin/env bash
# Stage-1-only reconstruction quality sweep with real validation metrics.

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
GPUS="${GPUS:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-240000}"
PROJECT="${PROJECT:-laser-debugging}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/stage1_quality_metrics}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
COCO_DIR="${COCO_DIR:-/scratch/$USER/data/coco}"
FFHQ_DIR="${FFHQ_DIR:-/scratch/$USER/datasets/ffhq}"
FFHQ_TRAIN_CROP_SIZE="${FFHQ_TRAIN_CROP_SIZE:-192}"
MAESTRO_DIR="${MAESTRO_DIR:-/scratch/$USER/datasets/maestro/maestro-v3.0.0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-coco,celebahq}"
DEPENDENCY="${DEPENDENCY:-}"

LASER_BACKBONE_LATENT_CHANNELS="${LASER_BACKBONE_LATENT_CHANNELS:-512}"
LASER_EMBEDDING_DIM="${LASER_EMBEDDING_DIM:-512}"
LASER_SPARSITY_LEVEL="${LASER_SPARSITY_LEVEL:-16}"
LASER_COEF_MAX="${LASER_COEF_MAX:-16.0}"
LASER_COMMITMENT_COST="${LASER_COMMITMENT_COST:-0.05}"
LASER_RECON_MSE_WEIGHT="${LASER_RECON_MSE_WEIGHT:-0.25}"
LASER_RECON_L1_WEIGHT="${LASER_RECON_L1_WEIGHT:-1.0}"
LASER_RECON_EDGE_WEIGHT="${LASER_RECON_EDGE_WEIGHT:-0.25}"
LASER_PERCEPTUAL_WEIGHT="${LASER_PERCEPTUAL_WEIGHT:-0.0}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi

COMMON_ARGS=(
  --full-training
  --stage1-only
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage2-epochs 1
  --partition "$PARTITION"
  --time-limit "$TIME_LIMIT"
  --gpus "$GPUS"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem-mb "$MEM_MB"
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
  --coco-dir "$COCO_DIR"
  --ffhq-dir "$FFHQ_DIR"
  --maestro-dir "$MAESTRO_DIR"
)
if [[ -n "$DEPENDENCY" ]]; then
  COMMON_ARGS+=(--dependency "$DEPENDENCY")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

COMMON_STAGE1=(
  --stage1-override train.limit_train_batches=1.0
  --stage1-override train.limit_val_batches=256
  --stage1-override train.limit_test_batches=256
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override train.val_check_interval=0.25
  --stage1-override model.compute_fid=true
  --stage1-override model.out_tanh=true
)

case_train_workers() {
  local dataset="$1"
  if [[ "$dataset" == "celebahq" ]]; then
    printf "0"
  else
    printf "8"
  fi
}

case_attn_f32() {
  local dataset="$1"
  case "$dataset" in
    coco) printf "[16]" ;;
    ffhq|celebahq) printf "[8]" ;;
    celeba) printf "[4]" ;;
    *) printf "[]" ;;
  esac
}

laser_global_batch_for() {
  local dataset="$1"
  case "$dataset" in
    coco) printf "$GPUS" ;;
    celebahq) printf "$((GPUS * 4))" ;;
    ffhq) printf "$((GPUS * 8))" ;;
    celeba) printf "$((GPUS * 8))" ;;
    *) printf "$((GPUS * 4))" ;;
  esac
}

laser_eval_batch_for() {
  local dataset="$1"
  case "$dataset" in
    ffhq) printf "$((GPUS * 4))" ;;
    *) laser_global_batch_for "$dataset" ;;
  esac
}

vqvae_global_batch_for() {
  local dataset="$1"
  case "$dataset" in
    coco) printf "$((GPUS * 2))" ;;
    celebahq) printf "$((GPUS * 8))" ;;
    ffhq) printf "$((GPUS * 16))" ;;
    celeba) printf "$((GPUS * 16))" ;;
    *) printf "$((GPUS * 8))" ;;
  esac
}

vqvae_eval_batch_for() {
  local dataset="$1"
  case "$dataset" in
    ffhq) printf "$((GPUS * 8))" ;;
    *) vqvae_global_batch_for "$dataset" ;;
  esac
}

train_crop_size_for() {
  local dataset="$1"
  case "$dataset" in
    ffhq) printf "%s" "$FFHQ_TRAIN_CROP_SIZE" ;;
    *) printf "0" ;;
  esac
}

submit_laser_f32() {
  local dataset="$1"
  local workers batch eval_batch train_crop attn
  workers="$(case_train_workers "$dataset")"
  batch="$(laser_global_batch_for "$dataset")"
  eval_batch="$(laser_eval_batch_for "$dataset")"
  train_crop="$(train_crop_size_for "$dataset")"
  attn="$(case_attn_f32 "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "${dataset}-stage1-rfid-rqpaper-laser-f32-d4-k16384-z${LASER_EMBEDDING_DIM}-recon-s1-${STAGE1_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override data.eval_batch_size="$eval_batch" \
    --stage1-override data.train_crop_size="$train_crop" \
    --stage1-override model.backbone=vqgan \
    --stage1-override model.num_downsamples=5 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4,4] \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_residual_blocks=2 \
    --stage1-override model.num_residual_hiddens=64 \
    --stage1-override model.backbone_latent_channels="$LASER_BACKBONE_LATENT_CHANNELS" \
    --stage1-override model.embedding_dim="$LASER_EMBEDDING_DIM" \
    --stage1-override model.num_embeddings=16384 \
    --stage1-override model.sparsity_level="$LASER_SPARSITY_LEVEL" \
    --stage1-override "model.attn_resolutions=${attn}" \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.decoder_extra_residual_layers=1 \
    --stage1-override model.bottleneck_loss_weight=0.25 \
    --stage1-override model.commitment_cost="$LASER_COMMITMENT_COST" \
    --stage1-override model.coef_max="$LASER_COEF_MAX" \
    --stage1-override model.recon_mse_weight="$LASER_RECON_MSE_WEIGHT" \
    --stage1-override model.recon_l1_weight="$LASER_RECON_L1_WEIGHT" \
    --stage1-override model.recon_edge_weight="$LASER_RECON_EDGE_WEIGHT" \
    --stage1-override model.perceptual_weight="$LASER_PERCEPTUAL_WEIGHT" \
    --stage1-override model.perceptual_start_step=0 \
    --stage1-override model.perceptual_warmup_steps=0 \
    --stage1-override train.learning_rate=4.0e-5 \
    --stage1-override train.precision=bf16-mixed
}

submit_laser_f16() {
  local dataset="$1"
  local workers batch eval_batch train_crop
  workers="$(case_train_workers "$dataset")"
  batch="$(laser_global_batch_for "$dataset")"
  eval_batch="$(laser_eval_batch_for "$dataset")"
  train_crop="$(train_crop_size_for "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "${dataset}-stage1-rfid-control-laser-f16-d4-k16384-z${LASER_EMBEDDING_DIM}-recon-s1-${STAGE1_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override data.eval_batch_size="$eval_batch" \
    --stage1-override data.train_crop_size="$train_crop" \
    --stage1-override model.backbone=vqgan \
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4] \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_residual_blocks=2 \
    --stage1-override model.num_residual_hiddens=64 \
    --stage1-override model.backbone_latent_channels="$LASER_BACKBONE_LATENT_CHANNELS" \
    --stage1-override model.embedding_dim="$LASER_EMBEDDING_DIM" \
    --stage1-override model.num_embeddings=16384 \
    --stage1-override model.sparsity_level="$LASER_SPARSITY_LEVEL" \
    --stage1-override model.attn_resolutions=[] \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.decoder_extra_residual_layers=1 \
    --stage1-override model.bottleneck_loss_weight=0.25 \
    --stage1-override model.commitment_cost="$LASER_COMMITMENT_COST" \
    --stage1-override model.coef_max="$LASER_COEF_MAX" \
    --stage1-override model.recon_mse_weight="$LASER_RECON_MSE_WEIGHT" \
    --stage1-override model.recon_l1_weight="$LASER_RECON_L1_WEIGHT" \
    --stage1-override model.recon_edge_weight="$LASER_RECON_EDGE_WEIGHT" \
    --stage1-override model.perceptual_weight="$LASER_PERCEPTUAL_WEIGHT" \
    --stage1-override model.perceptual_start_step=0 \
    --stage1-override model.perceptual_warmup_steps=0 \
    --stage1-override train.learning_rate=4.0e-5 \
    --stage1-override train.precision=bf16-mixed
}

submit_vqvae_f16() {
  local dataset="$1"
  local workers batch eval_batch train_crop
  workers="$(case_train_workers "$dataset")"
  batch="$(vqvae_global_batch_for "$dataset")"
  eval_batch="$(vqvae_eval_batch_for "$dataset")"
  train_crop="$(train_crop_size_for "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases "$dataset" \
    --model-family vqvae \
    --run-label "${dataset}-stage1-rfid-control-vqvae-f16-k8192-z256-s1-${STAGE1_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override data.eval_batch_size="$eval_batch" \
    --stage1-override data.train_crop_size="$train_crop" \
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.num_hiddens=192 \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.num_embeddings=8192 \
    --stage1-override model.embedding_dim=256 \
    --stage1-override model.commitment_cost=0.25 \
    --stage1-override model.decay=0.99 \
    --stage1-override model.codebook_init=true \
    --stage1-override model.dead_code_threshold=1.0 \
    --stage1-override model.perceptual_weight=0.0 \
    --stage1-override train.learning_rate=1.0e-4 \
    --stage1-override train.precision=32
}

IFS=',' read -ra selected <<< "$DATASETS"
for raw_dataset in "${selected[@]}"; do
  dataset="$(echo "$raw_dataset" | xargs)"
  [[ -z "$dataset" ]] && continue
  submit_laser_f32 "$dataset"
  submit_laser_f16 "$dataset"
  submit_vqvae_f16 "$dataset"
done
