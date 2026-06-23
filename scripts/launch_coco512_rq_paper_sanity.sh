#!/usr/bin/env bash
# COCO512 sanity checks after reviewing RQ-VAE/RQ-Transformer.
#
# The paper keeps an aggressive f=32 spatial compression, but does not use a
# single token per site. Its ImageNet-256 tokenizer uses 8x8x4 residual codes,
# a 16,384-entry codebook, 256-d latent channels, bottleneck attention, LPIPS,
# and GAN loss. This launcher tests the closest supported variants in our
# maintained pipeline plus a no-extra-downsample control.

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-384000}"
PROJECT="${PROJECT:-laser}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/coco512_rq_paper_sanity}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
COCO_DIR="${COCO_DIR:-/scratch/$USER/data/coco}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
SANITY_STAGE1_TRAIN_BATCHES="${SANITY_STAGE1_TRAIN_BATCHES:-200}"
SANITY_STAGE1_VAL_BATCHES="${SANITY_STAGE1_VAL_BATCHES:-20}"
SANITY_CACHE_MAX_ITEMS="${SANITY_CACHE_MAX_ITEMS:-4096}"
DRY_RUN="${DRY_RUN:-0}"

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

COMMON_ARGS=(
  --cases coco
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
  --stage1-override data.num_workers=4
  --stage1-override train.limit_train_batches="$SANITY_STAGE1_TRAIN_BATCHES"
  --stage1-override train.limit_val_batches="$SANITY_STAGE1_VAL_BATCHES"
  --stage1-override train.limit_test_batches="$SANITY_STAGE1_VAL_BATCHES"
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override model.compute_fid=false
)

COMMON_STAGE2=(
  --stage2-override data.num_workers=2
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

submit_laser_rqstyle_f32() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --model-family laser \
    --run-label "coco512-rqpaper-laser-f32-d4-k16384-z256-lpips-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    "${COMMON_CACHE[@]}" \
    --stage1-override data.batch_size=1 \
    --stage1-override model.backbone=ddpm \
    --stage1-override model.num_downsamples=5 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4,4] \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_residual_blocks=2 \
    --stage1-override model.num_residual_hiddens=64 \
    --stage1-override model.backbone_latent_channels=256 \
    --stage1-override model.embedding_dim=256 \
    --stage1-override model.num_embeddings=16384 \
    --stage1-override model.sparsity_level=4 \
    --stage1-override model.attn_resolutions=[16] \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.decoder_extra_residual_layers=1 \
    --stage1-override model.bottleneck_loss_weight=0.25 \
    --stage1-override model.commitment_cost=0.25 \
    --stage1-override model.coef_max=8.0 \
    --stage1-override model.recon_mse_weight=1.0 \
    --stage1-override model.recon_l1_weight=0.0 \
    --stage1-override model.recon_edge_weight=0.0 \
    --stage1-override model.perceptual_weight=1.0 \
    --stage1-override model.perceptual_start_step=0 \
    --stage1-override model.perceptual_warmup_steps=0 \
    --stage1-override train.learning_rate=4.0e-5 \
    --stage1-override train.precision=bf16-mixed \
    --stage2-override train_ar.batch_size=4 \
    --stage2-override ar.coeff_loss_type=huber \
    --stage2-override ar.coeff_loss_weight=1.0 \
    --stage2-override ar.coeff_huber_delta=0.5 \
    --cache-arg=--num-workers \
    --cache-arg=4 \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg=8.0
}

submit_laser_f16_control() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --model-family laser \
    --run-label "coco512-control-laser-f16-d4-k16384-z256-lpips-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    "${COMMON_CACHE[@]}" \
    --stage1-override data.batch_size=1 \
    --stage1-override model.backbone=ddpm \
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4] \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_residual_blocks=2 \
    --stage1-override model.num_residual_hiddens=64 \
    --stage1-override model.backbone_latent_channels=256 \
    --stage1-override model.embedding_dim=256 \
    --stage1-override model.num_embeddings=16384 \
    --stage1-override model.sparsity_level=4 \
    --stage1-override model.attn_resolutions=[] \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.decoder_extra_residual_layers=1 \
    --stage1-override model.bottleneck_loss_weight=0.25 \
    --stage1-override model.commitment_cost=0.25 \
    --stage1-override model.coef_max=8.0 \
    --stage1-override model.recon_mse_weight=1.0 \
    --stage1-override model.recon_l1_weight=0.0 \
    --stage1-override model.recon_edge_weight=0.0 \
    --stage1-override model.perceptual_weight=1.0 \
    --stage1-override model.perceptual_start_step=0 \
    --stage1-override model.perceptual_warmup_steps=0 \
    --stage1-override train.learning_rate=4.0e-5 \
    --stage1-override train.precision=bf16-mixed \
    --stage2-override train_ar.batch_size=2 \
    --stage2-override ar.coeff_loss_type=huber \
    --stage2-override ar.coeff_loss_weight=1.0 \
    --stage2-override ar.coeff_huber_delta=0.5 \
    --cache-arg=--num-workers \
    --cache-arg=4 \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg=8.0
}

submit_vqvae_f16_control() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --model-family vqvae \
    --run-label "coco512-control-vqvae-f16-k8192-z256-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    "${COMMON_CACHE[@]}" \
    --stage1-override data.batch_size=2 \
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
    --stage2-override train_ar.batch_size=16
}

submit_laser_rqstyle_f32
submit_laser_f16_control
submit_vqvae_f16_control
