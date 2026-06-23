#!/usr/bin/env bash
# FFHQ 512x512 two-downsample capacity sweep for VQ-VAE and patch-based LASER.
#
# Both model families keep a 128x128 latent grid. The VQ prior trains on
# 32x32 latent crops so stage 2 remains tractable; LASER keeps full-image
# generation tractable by grouping the 128x128 grid into sparse patch codes
# with larger dictionaries.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser-debugging}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/ffhq512_down2_capacity_sweep}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
FFHQ_DIR="${FFHQ_DIR:-/scratch/$USER/datasets/ffhq}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018}"
RUN_VQVAE="${RUN_VQVAE:-1}"
RUN_LASER="${RUN_LASER:-1}"
FFHQ512_STAGE1_WARMUP_STEPS="${FFHQ512_STAGE1_WARMUP_STEPS:-500}"
FFHQ512_STAGE2_WARMUP_STEPS="${FFHQ512_STAGE2_WARMUP_STEPS:-500}"
FFHQ512_MIN_LR_RATIO="${FFHQ512_MIN_LR_RATIO:-0.05}"
FFHQ512_TRAIN_CROP_SIZE="${FFHQ512_TRAIN_CROP_SIZE:-384}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

if [[ ! -d "$FFHQ_DIR" ]]; then
  echo "FFHQ directory not found: $FFHQ_DIR" >&2
  exit 1
fi

COMMON_ARGS=(
  --cases ffhq
  --full-training
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
  --partition "$PARTITION"
  --time-limit "$TIME_LIMIT"
  --cpus-per-task 16
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
  --ffhq-dir "$FFHQ_DIR"
)
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  COMMON_ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

COMMON_STAGE1=(
  --stage1-override data=ffhq
  --stage1-override data.data_dir="$FFHQ_DIR"
  --stage1-override data.image_size=512
  --stage1-override data.num_workers=8
  --stage1-override train.limit_train_batches=1.0
  --stage1-override train.limit_val_batches=1.0
  --stage1-override train.limit_test_batches=1.0
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override train.val_check_interval=1.0
  --stage1-override train.warmup_steps="$FFHQ512_STAGE1_WARMUP_STEPS"
  --stage1-override train.min_lr_ratio="$FFHQ512_MIN_LR_RATIO"
  --stage1-override model.compute_fid=false
)

COMMON_STAGE2=(
  --stage2-override data.dataset=ffhq
  --stage2-override data.data_dir="$FFHQ_DIR"
  --stage2-override data.image_size=512
  --stage2-override data.num_workers=4
  --stage2-override train_ar.max_items=0
  --stage2-override train_ar.limit_train_batches=1.0
  --stage2-override train_ar.limit_val_batches=1.0
  --stage2-override train_ar.limit_test_batches=1.0
  --stage2-override train_ar.sample_every_n_epochs=5
  --stage2-override train_ar.sample_log_to_wandb=true
  --stage2-override train_ar.sample_num_images=8
  --stage2-override train_ar.generation_metric_num_samples=16
  --stage2-override train_ar.compute_generation_fid=true
  --stage2-override train_ar.compute_audio_generation_metrics=false
  --stage2-override train_ar.run_test_after_fit=false
  --stage2-override train_ar.save_final_samples_after_fit=false
  --stage2-override train_ar.sample_temperature=0.8
  --stage2-override train_ar.sample_top_k=0
  --stage2-override ar.type=sparse_spatial_depth
  --stage2-override ar.warmup_steps="$FFHQ512_STAGE2_WARMUP_STEPS"
  --stage2-override ar.min_lr_ratio="$FFHQ512_MIN_LR_RATIO"
  --cache-arg=--image-size
  --cache-arg=512
)

submit_vqvae() {
  local case_name="$1"
  local atoms="$2"
  local zdim="$3"
  local hidden="$4"
  local res_hidden="$5"
  local per_gpu_batch="$6"
  local per_gpu_s2_batch="$7"
  local per_gpu_eval_batch="$8"
  local lr="$9"
  local s2_lr="${10}"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --model-family vqvae \
    --run-label "ffhq512-vqvae-down2-${case_name}-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    --gpus 4 \
    --mem-mb 320000 \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    --stage1-override data.batch_size="$per_gpu_batch" \
    --stage1-override data.eval_batch_size="$per_gpu_eval_batch" \
    --stage1-override data.train_crop_size="$FFHQ512_TRAIN_CROP_SIZE" \
    --stage1-override model.num_downsamples=2 \
    --stage1-override model.num_hiddens="$hidden" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens="$res_hidden" \
    --stage1-override model.num_embeddings="$atoms" \
    --stage1-override model.embedding_dim="$zdim" \
    --stage1-override model.commitment_cost=0.25 \
    --stage1-override model.decay=0.99 \
    --stage1-override model.codebook_init=true \
    --stage1-override model.dead_code_threshold=1.0 \
    --stage1-override model.perceptual_weight=0.0 \
    --stage1-override train.learning_rate="$lr" \
    --stage1-override train.precision=32 \
    --stage2-override train_ar.batch_size="$per_gpu_s2_batch" \
    --stage2-override train_ar.crop_h_sites=32 \
    --stage2-override train_ar.crop_w_sites=32 \
    --stage2-override train_ar.sample_every_n_epochs=0 \
    --stage2-override train_ar.sample_num_images=0 \
    --stage2-override train_ar.generation_metric_num_samples=0 \
    --stage2-override train_ar.compute_generation_fid=false \
    --stage2-override train_ar.save_final_samples_after_fit=false \
    --stage2-override ar.d_model=512 \
    --stage2-override ar.n_heads=8 \
    --stage2-override ar.n_layers=8 \
    --stage2-override ar.d_ff=2048 \
    --stage2-override ar.learning_rate="$s2_lr"
}

submit_laser() {
  local case_name="$1"
  local patch="$2"
  local sparsity="$3"
  local atoms="$4"
  local zdim="$5"
  local hidden="$6"
  local res_hidden="$7"
  local per_gpu_batch="$8"
  local per_gpu_s2_batch="$9"
  local per_gpu_eval_batch="${10}"
  local lr="${11}"
  local dict_lr="${12}"
  local s2_lr="${13}"
  local d_model="${14}"
  local n_heads="${15}"
  local n_layers="${16}"
  local d_ff="${17}"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --model-family laser \
    --run-label "ffhq512-laser-down2-${case_name}-p${patch}-k${sparsity}-a${atoms}-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    --gpus 4 \
    --mem-mb 320000 \
    "${COMMON_STAGE1[@]}" \
    "${COMMON_STAGE2[@]}" \
    --stage1-override data.batch_size="$per_gpu_batch" \
    --stage1-override data.eval_batch_size="$per_gpu_eval_batch" \
    --stage1-override data.train_crop_size="$FFHQ512_TRAIN_CROP_SIZE" \
    --stage1-override model.backbone=ddpm \
    --stage1-override model.num_downsamples=2 \
    --stage1-override model.channel_multipliers=[1,1,2] \
    --stage1-override model.num_hiddens="$hidden" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens="$res_hidden" \
    --stage1-override model.backbone_latent_channels="$hidden" \
    --stage1-override model.embedding_dim="$zdim" \
    --stage1-override model.patch_based=true \
    --stage1-override model.patch_size="$patch" \
    --stage1-override model.patch_stride="$patch" \
    --stage1-override model.patch_reconstruction=tile \
    --stage1-override model.num_embeddings="$atoms" \
    --stage1-override model.sparsity_level="$sparsity" \
    --stage1-override model.attn_resolutions=[] \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.decoder_extra_residual_layers=2 \
    --stage1-override model.bottleneck_loss_weight=0.75 \
    --stage1-override model.commitment_cost=1.0 \
    --stage1-override model.dict_learning_rate="$dict_lr" \
    --stage1-override model.coef_max=16.0 \
    --stage1-override model.sparsity_reg_weight=0.0 \
    --stage1-override model.recon_mse_weight=0.5 \
    --stage1-override model.recon_l1_weight=0.5 \
    --stage1-override model.recon_edge_weight=0.0 \
    --stage1-override model.perceptual_weight=0.10 \
    --stage1-override model.perceptual_start_step=0 \
    --stage1-override model.perceptual_warmup_steps=1000 \
    --stage1-override train.learning_rate="$lr" \
    --stage1-override train.precision=bf16-mixed \
    --stage2-override train_ar.batch_size="$per_gpu_s2_batch" \
    --stage2-override ar.d_model="$d_model" \
    --stage2-override ar.n_heads="$n_heads" \
    --stage2-override ar.n_layers="$n_layers" \
    --stage2-override ar.d_ff="$d_ff" \
    --stage2-override ar.learning_rate="$s2_lr" \
    --stage2-override ar.coeff_loss_type=huber \
    --stage2-override ar.coeff_loss_weight=1.0 \
    --stage2-override ar.coeff_huber_delta=0.5 \
    --stage2-override train_ar.sample_coeff_mode=mean \
    --cache-arg=--coeff-bins \
    --cache-arg=0 \
    --cache-arg=--coeff-max \
    --cache-arg=16.0
}

if [[ "$RUN_VQVAE" == "1" || "$RUN_VQVAE" == "true" ]]; then
  submit_vqvae f128-k16384-z128 16384 128 192 96 6 8 3 1.75e-4 3.0e-4
  submit_vqvae f128-k32768-z256 32768 256 224 112 4 6 2 1.4e-4 2.75e-4
fi

if [[ "$RUN_LASER" == "1" || "$RUN_LASER" == "true" ]]; then
  # Full VQ would be 128x128 = 16384 prior sites.
  # LASER p16: 128 latent / 16 patch = 8x8 sites, 16 atom+coeff pairs => 2048 prior tokens.
  submit_laser f128-z16 16 16 131072 16 160 80 4 8 2 7.0e-5 1.4e-4 4.0e-4 384 6 8 1536
  # LASER p32: 128 latent / 32 patch = 4x4 sites, 16 atom+coeff pairs => 512 prior tokens.
  submit_laser f128-z8 32 16 65536 8 160 80 4 8 2 5.5e-5 1.0e-4 4.0e-4 256 4 6 1024
fi
