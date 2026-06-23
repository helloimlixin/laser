#!/usr/bin/env bash
# COCO512 stage-1 recovery sweep for a drifting fixed-depth sparse bottleneck.
#
# The goal is to keep the sparse token depth fixed at 4 while making the
# bottleneck easier to fit and giving the encoder/decoder more capacity:
# - lower sparse embedding dim (64/128 instead of 256+) so k=4 OMP is viable;
# - keep a wide DDPM-style backbone around the bottleneck via 1x1 projections;
# - use per-GPU batch size 1 to avoid OMP and activation OOMs on 512x512 images.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-2-12:00:00}"
GPUS="${GPUS:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-240000}"
PROJECT="${PROJECT:-laser-debugging}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/coco_bottleneck_power}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
COCO_DIR="${COCO_DIR:-/scratch/$USER/data/coco}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"

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
  --cases coco
  --model-family laser
)
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
  --stage1-override train.warmup_steps=1000
  --stage1-override train.min_lr_ratio=0.05
  --stage1-override model.compute_fid=true
  --stage1-override model.backbone=ddpm
  --stage1-override model.sparsity_level=4
  --stage1-override model.sparsity_reg_weight=0.0
  --stage1-override model.coef_max=16.0
  --stage1-override model.dict_learning_rate=1.0e-4
  --stage1-override model.recon_mse_weight=1.0
  --stage1-override model.recon_l1_weight=0.0
  --stage1-override model.recon_edge_weight=0.0
  --stage1-override model.perceptual_start_step=0
  --stage1-override model.perceptual_warmup_steps=1000
  --stage1-override model.use_mid_attention=true
  --stage1-override model.log_images_every_n_steps=500
  --stage1-override data.batch_size=1
  --stage1-override data.num_workers=8
  --stage1-override train.precision=bf16-mixed
)

submit_variant() {
  local label="$1"
  local downsamples="$2"
  local ch_mult="$3"
  local embedding_dim="$4"
  local num_embeddings="$5"
  local bottleneck_weight="$6"
  local commitment="$7"
  local latent_channels="$8"
  local attn_resolutions="$9"
  local decoder_extra="${10}"
  local perceptual_weight="${11}"
  local learning_rate="${12}"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --run-label "$label" \
    "${COMMON_STAGE1[@]}" \
    --stage1-override "model.num_downsamples=${downsamples}" \
    --stage1-override "model.channel_multipliers=${ch_mult}" \
    --stage1-override model.num_hiddens=160 \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=80 \
    --stage1-override "model.backbone_latent_channels=${latent_channels}" \
    --stage1-override "model.embedding_dim=${embedding_dim}" \
    --stage1-override "model.num_embeddings=${num_embeddings}" \
    --stage1-override "model.attn_resolutions=${attn_resolutions}" \
    --stage1-override "model.decoder_extra_residual_layers=${decoder_extra}" \
    --stage1-override "model.bottleneck_loss_weight=${bottleneck_weight}" \
    --stage1-override "model.commitment_cost=${commitment}" \
    --stage1-override "model.perceptual_weight=${perceptual_weight}" \
    --stage1-override "train.learning_rate=${learning_rate}"
}

# Quality-oriented control: no extra COCO downsample, but fixed sparse depth.
submit_variant \
  "coco-stage1-power-f16-z64-k16384-s4-bw05-c1-s1-${STAGE1_EPOCHS}" \
  4 "[1,1,2,2,4]" 64 16384 0.5 1.0 512 "[]" 2 0.25 2.0e-5

# Stage-2-friendly grid: one extra downsample, stronger decoder, compact sparse dim.
submit_variant \
  "coco-stage1-power-f32-z64-k32768-s4-bw1-c15-s1-${STAGE1_EPOCHS}" \
  5 "[1,1,2,2,4,4]" 64 32768 1.0 1.5 640 "[16]" 3 0.15 2.0e-5

# Same spatial grid, wider sparse latent to test whether z64 is under-capacity.
submit_variant \
  "coco-stage1-power-f32-z128-k32768-s4-bw075-c1-s1-${STAGE1_EPOCHS}" \
  5 "[1,1,2,2,4,4]" 128 32768 0.75 1.0 640 "[16]" 3 0.15 2.0e-5
