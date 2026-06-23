#!/usr/bin/env bash
# Full-data 20/20 vision capacity sweep for sharper reconstructions.
#
# Datasets: COCO512, CelebA128, CelebA-HQ256, FFHQ256.
# Models: LASER and VQ-VAE.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
PROJECT="${PROJECT:-laser-debugging}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/vision_capacity_20ep}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
COCO_DIR="${COCO_DIR:-/scratch/$USER/data/coco}"
FFHQ_DIR="${FFHQ_DIR:-/scratch/$USER/datasets/ffhq}"
FFHQ_TRAIN_CROP_SIZE="${FFHQ_TRAIN_CROP_SIZE:-192}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/datasets/VCTK-Corpus-0.92}"
MAESTRO_DIR="${MAESTRO_DIR:-/scratch/$USER/datasets/maestro/maestro-v3.0.0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

COMMON_ARGS=(
  --full-training
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
  --partition "$PARTITION"
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
  --coco-dir "$COCO_DIR"
  --ffhq-dir "$FFHQ_DIR"
  --vctk-dir "$VCTK_DIR"
  --maestro-dir "$MAESTRO_DIR"
)
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  COMMON_ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

COMMON_STAGE1=(
  --stage1-override train.limit_train_batches=1.0
  --stage1-override train.limit_val_batches=1.0
  --stage1-override train.limit_test_batches=1.0
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override train.val_check_interval=1.0
  --stage1-override train.warmup_steps=1500
  --stage1-override train.min_lr_ratio=0.05
  --stage1-override model.compute_fid=true
)

COMMON_STAGE2=(
  --stage2-override train_ar.max_items=0
  --stage2-override train_ar.limit_train_batches=1.0
  --stage2-override train_ar.limit_val_batches=1.0
  --stage2-override train_ar.limit_test_batches=1.0
  --stage2-override train_ar.sample_every_n_epochs=5
  --stage2-override train_ar.sample_log_to_wandb=true
  --stage2-override train_ar.sample_temperature=0.8
  --stage2-override train_ar.sample_top_k=0
  --stage2-override ar.type=sparse_spatial_depth
  --stage2-override ar.warmup_steps=1500
  --stage2-override ar.min_lr_ratio=0.05
)

stage1_warmup_for() {
  case "$1" in
    ffhq) printf "750" ;;
    *) printf "1500" ;;
  esac
}

stage2_warmup_for() {
  case "$1" in
    ffhq) printf "1000" ;;
    *) printf "1500" ;;
  esac
}

train_crop_size_for() {
  case "$1" in
    ffhq) printf "%s" "$FFHQ_TRAIN_CROP_SIZE" ;;
    *) printf "0" ;;
  esac
}

image_size_for() {
  case "$1" in
    coco) printf "512" ;;
    celeba) printf "128" ;;
    celebahq|ffhq) printf "256" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

vision_gpus_for() {
  case "$1" in
    celeba) printf "2" ;;
    coco|celebahq|ffhq) printf "3" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

vision_mem_for() {
  case "$1" in
    celeba) printf "180000" ;;
    coco|celebahq|ffhq) printf "360000" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

vision_time_for() {
  case "$1" in
    coco) printf "3-00:00:00" ;;
    *) printf "2-00:00:00" ;;
  esac
}

vision_workers_for() {
  case "$1" in
    celebahq) printf "0" ;;
    *) printf "8" ;;
  esac
}

vision_stage2_workers_for() {
  case "$1" in
    celebahq) printf "0" ;;
    *) printf "4" ;;
  esac
}

laser_downsamples_for() {
  case "$1" in
    coco) printf "5" ;;
    *) printf "4" ;;
  esac
}

laser_channel_mult_for() {
  case "$1" in
    coco) printf "[1,1,2,2,4,4]" ;;
    *) printf "[1,1,2,2,4]" ;;
  esac
}

laser_attn_for() {
  case "$1" in
    coco) printf "[16]" ;;
    celeba) printf "[8]" ;;
    celebahq|ffhq) printf "[16]" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

laser_hidden_for() {
  case "$1" in
    coco) printf "224" ;;
    celeba) printf "224" ;;
    celebahq|ffhq) printf "256" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

laser_embed_dim_for() {
  case "$1" in
    coco) printf "128" ;;
    *) printf "192" ;;
  esac
}

laser_atoms_for() {
  case "$1" in
    coco) printf "32768" ;;
    *) printf "32768" ;;
  esac
}

laser_backbone_latent_for() {
  case "$1" in
    coco) printf "896" ;;
    *) printf "768" ;;
  esac
}

laser_batch_for() {
  case "$1" in
    coco) printf "1" ;;
    celeba) printf "12" ;;
    celebahq) printf "4" ;;
    ffhq) printf "12" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

laser_eval_batch_for() {
  case "$1" in
    ffhq) printf "6" ;;
    *) laser_batch_for "$1" ;;
  esac
}

laser_stage2_batch_for() {
  case "$1" in
    coco) printf "2" ;;
    celeba) printf "16" ;;
    celebahq) printf "4" ;;
    ffhq) printf "6" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

laser_stage1_lr_for() {
  case "$1" in
    ffhq) printf "5.0e-5" ;;
    *) printf "3.0e-5" ;;
  esac
}

laser_dict_lr_for() {
  case "$1" in
    ffhq) printf "2.5e-4" ;;
    *) printf "1.5e-4" ;;
  esac
}

laser_stage2_lr_for() {
  case "$1" in
    ffhq) printf "4.0e-4" ;;
    *) printf "3.5e-4" ;;
  esac
}

vq_downsamples_for() {
  case "$1" in
    coco) printf "5" ;;
    celeba) printf "3" ;;
    celebahq|ffhq) printf "4" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

vq_atoms_for() {
  case "$1" in
    coco) printf "32768" ;;
    celeba) printf "8192" ;;
    celebahq|ffhq) printf "16384" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

vq_embed_dim_for() {
  case "$1" in
    celeba) printf "192" ;;
    *) printf "384" ;;
  esac
}

vq_hidden_for() {
  case "$1" in
    celeba) printf "224" ;;
    *) printf "256" ;;
  esac
}

vq_res_hidden_for() {
  case "$1" in
    celeba) printf "112" ;;
    *) printf "128" ;;
  esac
}

vq_batch_for() {
  case "$1" in
    coco) printf "2" ;;
    celeba) printf "24" ;;
    celebahq) printf "8" ;;
    ffhq) printf "24" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

vq_eval_batch_for() {
  case "$1" in
    ffhq) printf "12" ;;
    *) vq_batch_for "$1" ;;
  esac
}

vq_stage2_batch_for() {
  case "$1" in
    coco) printf "4" ;;
    celeba) printf "32" ;;
    celebahq) printf "8" ;;
    ffhq) printf "12" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

vq_stage1_lr_for() {
  case "$1" in
    ffhq) printf "2.5e-4" ;;
    *) printf "1.5e-4" ;;
  esac
}

vq_stage2_lr_for() {
  case "$1" in
    ffhq) printf "4.5e-4" ;;
    *) printf "4.0e-4" ;;
  esac
}

submit_vision_laser() {
  local dataset="$1"
  local gpus mem time workers s2_workers image_size downsamples latent_grid ch_mult attn hidden zdim atoms backbone_latent batch s2_batch
  local eval_batch train_crop stage1_lr dict_lr stage2_lr stage1_warmup stage2_warmup
  gpus="$(vision_gpus_for "$dataset")"
  mem="$(vision_mem_for "$dataset")"
  time="$(vision_time_for "$dataset")"
  workers="$(vision_workers_for "$dataset")"
  s2_workers="$(vision_stage2_workers_for "$dataset")"
  image_size="$(image_size_for "$dataset")"
  downsamples="$(laser_downsamples_for "$dataset")"
  latent_grid=$(( image_size / (1 << downsamples) ))
  ch_mult="$(laser_channel_mult_for "$dataset")"
  attn="$(laser_attn_for "$dataset")"
  hidden="$(laser_hidden_for "$dataset")"
  zdim="$(laser_embed_dim_for "$dataset")"
  atoms="$(laser_atoms_for "$dataset")"
  backbone_latent="$(laser_backbone_latent_for "$dataset")"
  batch="$(laser_batch_for "$dataset")"
  eval_batch="$(laser_eval_batch_for "$dataset")"
  train_crop="$(train_crop_size_for "$dataset")"
  s2_batch="$(laser_stage2_batch_for "$dataset")"
  stage1_lr="$(laser_stage1_lr_for "$dataset")"
  dict_lr="$(laser_dict_lr_for "$dataset")"
  stage2_lr="$(laser_stage2_lr_for "$dataset")"
  stage1_warmup="$(stage1_warmup_for "$dataset")"
  stage2_warmup="$(stage2_warmup_for "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "cap20-${dataset}-vision-laser-f${latent_grid}-h${hidden}-z${zdim}-k${atoms}-s8-b$((batch * gpus))" \
    --gpus "$gpus" \
    --cpus-per-task 16 \
    --mem-mb "$mem" \
    --time-limit "$time" \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    --stage1-override data.image_size="$image_size" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override data.eval_batch_size="$eval_batch" \
    --stage1-override data.train_crop_size="$train_crop" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override train.warmup_steps="$stage1_warmup" \
    --stage1-override model.backbone=ddpm \
    --stage1-override model.num_downsamples="$downsamples" \
    --stage1-override "model.channel_multipliers=${ch_mult}" \
    --stage1-override model.num_hiddens="$hidden" \
    --stage1-override model.num_residual_blocks=4 \
    --stage1-override model.num_residual_hiddens=128 \
    --stage1-override model.backbone_latent_channels="$backbone_latent" \
    --stage1-override model.embedding_dim="$zdim" \
    --stage1-override model.num_embeddings="$atoms" \
    --stage1-override model.sparsity_level=8 \
    --stage1-override "model.attn_resolutions=${attn}" \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.decoder_extra_residual_layers=3 \
    --stage1-override model.bottleneck_loss_weight=0.75 \
    --stage1-override model.commitment_cost=1.0 \
    --stage1-override model.dict_learning_rate="$dict_lr" \
    --stage1-override model.coef_max=16.0 \
    --stage1-override model.sparsity_reg_weight=0.0 \
    --stage1-override model.recon_mse_weight=1.0 \
    --stage1-override model.recon_l1_weight=0.0 \
    --stage1-override model.recon_edge_weight=0.0 \
    --stage1-override model.perceptual_weight=0.2 \
    --stage1-override model.perceptual_start_step=0 \
    --stage1-override model.perceptual_warmup_steps=1000 \
    --stage1-override train.learning_rate="$stage1_lr" \
    --stage1-override train.precision=bf16-mixed \
    --stage2-override data.num_workers="$s2_workers" \
    --stage2-override train_ar.batch_size="$s2_batch" \
    --stage2-override ar.warmup_steps="$stage2_warmup" \
    --stage2-override train_ar.sample_num_images=8 \
    --stage2-override train_ar.generation_metric_num_samples=32 \
    --stage2-override train_ar.compute_generation_fid=true \
    --stage2-override ar.learning_rate="$stage2_lr" \
    --stage2-override ar.d_model=512 \
    --stage2-override ar.n_heads=8 \
    --stage2-override ar.n_layers=8 \
    --stage2-override ar.d_ff=2048 \
    --stage2-override ar.coeff_loss_type=huber \
    --stage2-override ar.coeff_loss_weight=1.0 \
    --stage2-override ar.coeff_huber_delta=0.5 \
    --stage2-override train_ar.sample_coeff_mode=mean \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg=16.0
}

submit_vision_vqvae() {
  local dataset="$1"
  local gpus mem time workers s2_workers image_size downsamples latent_grid atoms zdim hidden res_hidden batch s2_batch
  local eval_batch train_crop stage1_lr stage2_lr stage1_warmup stage2_warmup
  gpus="$(vision_gpus_for "$dataset")"
  mem="$(vision_mem_for "$dataset")"
  time="$(vision_time_for "$dataset")"
  workers="$(vision_workers_for "$dataset")"
  s2_workers="$(vision_stage2_workers_for "$dataset")"
  image_size="$(image_size_for "$dataset")"
  downsamples="$(vq_downsamples_for "$dataset")"
  latent_grid=$(( image_size / (1 << downsamples) ))
  atoms="$(vq_atoms_for "$dataset")"
  zdim="$(vq_embed_dim_for "$dataset")"
  hidden="$(vq_hidden_for "$dataset")"
  res_hidden="$(vq_res_hidden_for "$dataset")"
  batch="$(vq_batch_for "$dataset")"
  eval_batch="$(vq_eval_batch_for "$dataset")"
  train_crop="$(train_crop_size_for "$dataset")"
  s2_batch="$(vq_stage2_batch_for "$dataset")"
  stage1_lr="$(vq_stage1_lr_for "$dataset")"
  stage2_lr="$(vq_stage2_lr_for "$dataset")"
  stage1_warmup="$(stage1_warmup_for "$dataset")"
  stage2_warmup="$(stage2_warmup_for "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --cases "$dataset" \
    --model-family vqvae \
    --run-label "cap20-${dataset}-vision-vqvae-f${latent_grid}-h${hidden}-z${zdim}-k${atoms}-b$((batch * gpus))" \
    --gpus "$gpus" \
    --cpus-per-task 16 \
    --mem-mb "$mem" \
    --time-limit "$time" \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    --stage1-override data.image_size="$image_size" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override data.eval_batch_size="$eval_batch" \
    --stage1-override data.train_crop_size="$train_crop" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override train.warmup_steps="$stage1_warmup" \
    --stage1-override model.num_downsamples="$downsamples" \
    --stage1-override model.num_hiddens="$hidden" \
    --stage1-override model.num_residual_blocks=4 \
    --stage1-override model.num_residual_hiddens="$res_hidden" \
    --stage1-override model.num_embeddings="$atoms" \
    --stage1-override model.embedding_dim="$zdim" \
    --stage1-override model.commitment_cost=0.25 \
    --stage1-override model.decay=0.99 \
    --stage1-override model.codebook_init=true \
    --stage1-override model.dead_code_threshold=1.0 \
    --stage1-override model.perceptual_weight=0.05 \
    --stage1-override train.learning_rate="$stage1_lr" \
    --stage1-override train.precision=bf16-mixed \
    --stage2-override data.num_workers="$s2_workers" \
    --stage2-override train_ar.batch_size="$s2_batch" \
    --stage2-override ar.warmup_steps="$stage2_warmup" \
    --stage2-override train_ar.sample_num_images=8 \
    --stage2-override train_ar.generation_metric_num_samples=32 \
    --stage2-override train_ar.compute_generation_fid=true \
    --stage2-override ar.learning_rate="$stage2_lr" \
    --stage2-override ar.d_model=512 \
    --stage2-override ar.n_heads=8 \
    --stage2-override ar.n_layers=8 \
    --stage2-override ar.d_ff=2048
}

for dataset in coco celeba celebahq ffhq; do
  submit_vision_laser "$dataset"
  submit_vision_vqvae "$dataset"
done
