#!/usr/bin/env bash
# Submit COCO 512x512 VQ-VAE and LASER two-stage jobs from frozen snapshots.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-384000}"
PROJECT="${PROJECT:-laser}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/coco512_src_sweeps}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
COCO_DIR="${COCO_DIR:-/scratch/$USER/data/coco}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
MODEL_FAMILY="${MODEL_FAMILY:-both}"
COCO_STAGE1_NUM_WORKERS="${COCO_STAGE1_NUM_WORKERS:-4}"
COCO_STAGE2_NUM_WORKERS="${COCO_STAGE2_NUM_WORKERS:-2}"
COCO_VQVAE_STAGE1_BATCH="${COCO_VQVAE_STAGE1_BATCH:-8}"
COCO_LASER_STAGE1_BATCH="${COCO_LASER_STAGE1_BATCH:-4}"
COCO_VQVAE_STAGE2_BATCH="${COCO_VQVAE_STAGE2_BATCH:-32}"
COCO_LASER_STAGE2_BATCH="${COCO_LASER_STAGE2_BATCH:-8}"
COCO_VQVAE_DOWNSAMPLES="${COCO_VQVAE_DOWNSAMPLES:-5}"
COCO_LASER_DOWNSAMPLES="${COCO_LASER_DOWNSAMPLES:-5}"
COCO_LASER_SPARSITY="${COCO_LASER_SPARSITY:-16}"
COCO_LASER_NUM_EMBEDDINGS="${COCO_LASER_NUM_EMBEDDINGS:-8192}"
COCO_LASER_EMBEDDING_DIM="${COCO_LASER_EMBEDDING_DIM:-64}"

case "$COCO_LASER_DOWNSAMPLES" in
  4)
    COCO_LASER_CHANNEL_MULTIPLIERS="${COCO_LASER_CHANNEL_MULTIPLIERS:-[1,1,2,2,4]}"
    ;;
  5)
    COCO_LASER_CHANNEL_MULTIPLIERS="${COCO_LASER_CHANNEL_MULTIPLIERS:-[1,1,2,2,4,4]}"
    ;;
  *)
    echo "Set COCO_LASER_CHANNEL_MULTIPLIERS explicitly for COCO_LASER_DOWNSAMPLES=$COCO_LASER_DOWNSAMPLES" >&2
    exit 2
    ;;
esac

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
  --stage1-override data.num_workers="$COCO_STAGE1_NUM_WORKERS"
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override train.precision=bf16-mixed
  --stage1-override model.compute_fid=false
  --stage1-override model.perceptual_weight=0.0
  --stage2-override data.num_workers="$COCO_STAGE2_NUM_WORKERS"
  --stage2-override train_ar.sample_every_n_epochs=5
  --stage2-override train_ar.sample_num_images=8
  --stage2-override train_ar.generation_metric_num_samples=0
  --stage2-override train_ar.compute_generation_fid=false
  --stage2-override train_ar.sample_temperature=0.8
  --stage2-override train_ar.sample_top_k=0
  --stage2-override ar.d_model=512
  --stage2-override ar.n_heads=8
  --stage2-override ar.n_layers=8
  --stage2-override ar.d_ff=2048
  --stage2-override ar.learning_rate=3.0e-4
  --stage2-override ar.warmup_steps=1000
  --stage2-override ar.min_lr_ratio=0.01
)

submit_vqvae() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --model-family vqvae \
    --run-label "coco512-vqvae-ds${COCO_VQVAE_DOWNSAMPLES}-k8192-d128-b${COCO_VQVAE_STAGE1_BATCH}-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    --stage1-override data.batch_size="$COCO_VQVAE_STAGE1_BATCH" \
    --stage1-override model.num_downsamples="$COCO_VQVAE_DOWNSAMPLES" \
    --stage1-override model.num_hiddens=192 \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.num_embeddings=8192 \
    --stage1-override model.embedding_dim=128 \
    --stage1-override model.decay=0.99 \
    --stage1-override model.commitment_cost=0.25 \
    --stage1-override model.codebook_init=true \
    --stage1-override model.dead_code_threshold=1.0 \
    --stage1-override train.learning_rate=1.0e-4 \
    --stage1-override train.precision=32 \
    --stage2-override train_ar.batch_size="$COCO_VQVAE_STAGE2_BATCH"
}

submit_laser() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --model-family laser \
    --run-label "coco512-laser-ds${COCO_LASER_DOWNSAMPLES}-a${COCO_LASER_NUM_EMBEDDINGS}-d${COCO_LASER_EMBEDDING_DIM}-s${COCO_LASER_SPARSITY}-b${COCO_LASER_STAGE1_BATCH}-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    --stage1-override data.batch_size="$COCO_LASER_STAGE1_BATCH" \
    --stage1-override model.backbone=ddpm \
    --stage1-override model.num_downsamples="$COCO_LASER_DOWNSAMPLES" \
    --stage1-override "model.channel_multipliers=$COCO_LASER_CHANNEL_MULTIPLIERS" \
    --stage1-override model.backbone_latent_channels=512 \
    --stage1-override model.decoder_extra_residual_layers=1 \
    --stage1-override model.use_mid_attention=false \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=64 \
    --stage1-override model.num_embeddings="$COCO_LASER_NUM_EMBEDDINGS" \
    --stage1-override model.embedding_dim="$COCO_LASER_EMBEDDING_DIM" \
    --stage1-override model.sparsity_level="$COCO_LASER_SPARSITY" \
    --stage1-override model.patch_based=false \
    --stage1-override model.bottleneck_loss_weight=0.25 \
    --stage1-override model.recon_mse_weight=0.25 \
    --stage1-override model.recon_l1_weight=1.0 \
    --stage1-override model.recon_edge_weight=0.25 \
    --stage1-override model.commitment_cost=0.05 \
    --stage1-override model.coef_max=8.0 \
    --stage1-override train.learning_rate=2.0e-4 \
    --stage2-override train_ar.batch_size="$COCO_LASER_STAGE2_BATCH" \
    --stage2-override ar.coeff_loss_type=huber \
    --stage2-override ar.coeff_loss_weight=0.1 \
    --stage2-override ar.coeff_huber_delta=1.0 \
    --stage2-override train_ar.sample_coeff_mode=gaussian \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg=8.0
}

case "$MODEL_FAMILY" in
  both|all)
    submit_vqvae
    submit_laser
    ;;
  vqvae)
    submit_vqvae
    ;;
  laser)
    submit_laser
    ;;
  *)
    echo "MODEL_FAMILY must be one of: both, all, vqvae, laser" >&2
    exit 2
    ;;
esac
