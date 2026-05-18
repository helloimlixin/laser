#!/usr/bin/env bash
# Launch image debugging sweeps on COCO512, CelebA128, and CelebA-HQ256.
#
# Models:
#   1. RQ-paper-style LASER f=32: K=16384, reconstruction-focused objective.
#   2. LASER f=16 control: K=16384, reconstruction-focused objective.
#   3. VQ-VAE f=16 control: K=8192, z=256.

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-240000}"
PROJECT="${PROJECT:-laser-debugging}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/image_debugging_sweeps}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
COCO_DIR="${COCO_DIR:-/scratch/$USER/data/coco}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
SANITY_STAGE1_TRAIN_BATCHES="${SANITY_STAGE1_TRAIN_BATCHES:-200}"
SANITY_STAGE1_VAL_BATCHES="${SANITY_STAGE1_VAL_BATCHES:-20}"
SANITY_CACHE_MAX_ITEMS="${SANITY_CACHE_MAX_ITEMS:-4096}"
DRY_RUN="${DRY_RUN:-0}"

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

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

if [[ ! -d "$COCO_DIR/train2017" || ! -d "$COCO_DIR/val2017" ]]; then
  echo "COCO directory must contain train2017/ and val2017/: $COCO_DIR" >&2
  exit 1
fi
if [[ ! -d "/scratch/$USER/datasets/celeba_packed_128" ]]; then
  echo "CelebA directory not found: /scratch/$USER/datasets/celeba_packed_128" >&2
  exit 1
fi
if [[ ! -d "/scratch/$USER/datasets/celebahq_packed_256" ]]; then
  echo "CelebA-HQ directory not found: /scratch/$USER/datasets/celebahq_packed_256" >&2
  exit 1
fi

COMMON_ARGS=(
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
  --coco-dir "$COCO_DIR"
)
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

COMMON_STAGE1=(
  --stage1-override train.limit_train_batches="$SANITY_STAGE1_TRAIN_BATCHES"
  --stage1-override train.limit_val_batches="$SANITY_STAGE1_VAL_BATCHES"
  --stage1-override train.limit_test_batches="$SANITY_STAGE1_VAL_BATCHES"
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override model.compute_fid=false
  --stage1-override model.out_tanh=true
)

COMMON_STAGE2=(
  --stage2-override train_ar.max_items="$SANITY_CACHE_MAX_ITEMS"
  --stage2-override train_ar.sample_every_n_epochs=2
  --stage2-override train_ar.sample_num_images=8
  --stage2-override train_ar.generation_metric_num_samples=0
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

COMMON_CACHE=(
  --cache-arg=--max-items
  --cache-arg="$SANITY_CACHE_MAX_ITEMS"
)

case_train_workers() {
  local dataset="$1"
  if [[ "$dataset" == "celebahq" ]]; then
    printf "0"
  else
    printf "4"
  fi
}

case_stage2_workers() {
  local dataset="$1"
  if [[ "$dataset" == "celebahq" ]]; then
    printf "0"
  else
    printf "2"
  fi
}

case_attn_f32() {
  local dataset="$1"
  case "$dataset" in
    coco) printf "[16]" ;;
    celebahq) printf "[8]" ;;
    celeba) printf "[4]" ;;
    *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
  esac
}

laser_batch_for() {
  local dataset="$1"
  case "$dataset" in
    coco) printf "1" ;;
    celebahq) printf "4" ;;
    celeba) printf "8" ;;
    *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
  esac
}

vqvae_batch_for() {
  local dataset="$1"
  case "$dataset" in
    coco) printf "2" ;;
    celebahq) printf "8" ;;
    celeba) printf "16" ;;
    *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
  esac
}

stage2_batch_for() {
  local dataset="$1"
  local variant="$2"
  case "$dataset:$variant" in
    coco:f32) printf "4" ;;
    coco:f16) printf "2" ;;
    coco:vqvae) printf "16" ;;
    celebahq:f32) printf "8" ;;
    celebahq:f16) printf "4" ;;
    celebahq:vqvae) printf "32" ;;
    celeba:f32) printf "16" ;;
    celeba:f16) printf "8" ;;
    celeba:vqvae) printf "64" ;;
    *) echo "unknown dataset/variant: $dataset/$variant" >&2; exit 2 ;;
  esac
}

submit_laser_f32() {
  local dataset="$1"
  local workers stage2_workers batch stage2_batch attn
  workers="$(case_train_workers "$dataset")"
  stage2_workers="$(case_stage2_workers "$dataset")"
  batch="$(laser_batch_for "$dataset")"
  stage2_batch="$(stage2_batch_for "$dataset" f32)"
  attn="$(case_attn_f32 "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "${dataset}-rqpaper-laser-f32-d4-k16384-z${LASER_EMBEDDING_DIM}-recon-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    "${COMMON_CACHE[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
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
    --stage1-override train.precision=bf16-mixed \
    --stage2-override data.num_workers="$stage2_workers" \
    --stage2-override train_ar.batch_size="$stage2_batch" \
    --stage2-override ar.coeff_loss_type=huber \
    --stage2-override ar.coeff_loss_weight=1.0 \
    --stage2-override ar.coeff_huber_delta=0.5 \
    --cache-arg=--num-workers \
    --cache-arg="$workers" \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg="$LASER_COEF_MAX"
}

submit_laser_f16() {
  local dataset="$1"
  local workers stage2_workers batch stage2_batch
  workers="$(case_train_workers "$dataset")"
  stage2_workers="$(case_stage2_workers "$dataset")"
  batch="$(laser_batch_for "$dataset")"
  stage2_batch="$(stage2_batch_for "$dataset" f16)"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "${dataset}-control-laser-f16-d4-k16384-z${LASER_EMBEDDING_DIM}-recon-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    "${COMMON_CACHE[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
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
    --stage1-override train.precision=bf16-mixed \
    --stage2-override data.num_workers="$stage2_workers" \
    --stage2-override train_ar.batch_size="$stage2_batch" \
    --stage2-override ar.coeff_loss_type=huber \
    --stage2-override ar.coeff_loss_weight=1.0 \
    --stage2-override ar.coeff_huber_delta=0.5 \
    --cache-arg=--num-workers \
    --cache-arg="$workers" \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg="$LASER_COEF_MAX"
}

submit_vqvae_f16() {
  local dataset="$1"
  local workers stage2_workers batch stage2_batch
  workers="$(case_train_workers "$dataset")"
  stage2_workers="$(case_stage2_workers "$dataset")"
  batch="$(vqvae_batch_for "$dataset")"
  stage2_batch="$(stage2_batch_for "$dataset" vqvae)"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases "$dataset" \
    --model-family vqvae \
    --run-label "${dataset}-control-vqvae-f16-k8192-z256-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    "${COMMON_CACHE[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
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
    --stage1-override train.precision=32 \
    --stage2-override data.num_workers="$stage2_workers" \
    --stage2-override train_ar.batch_size="$stage2_batch"
}

for dataset in coco celeba celebahq; do
  submit_laser_f32 "$dataset"
  submit_laser_f16 "$dataset"
  submit_vqvae_f16 "$dataset"
done
