#!/usr/bin/env bash
set -euo pipefail

# CelebA-HQ 256 recovery sweep for broken line-like generation.
# All jobs run through maintained src code and are submitted from frozen
# snapshots via scripts/submit_multimodal_sweep.py.

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-384000}"
PROJECT="${PROJECT:-laser}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/xl598/runs/celebahq_src_sweeps}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/xl598/submission_snapshots}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-200}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

COMMON_ARGS=(
  --cases celebahq
  --model-family laser
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
  --stage1-override data.num_workers=0
  --stage1-override model.backbone=scratch_vqvae
  --stage1-override model.num_downsamples=4
  --stage1-override model.num_hiddens=128
  --stage1-override model.num_residual_blocks=2
  --stage1-override model.num_residual_hiddens=32
  --stage1-override model.embedding_dim=16
  --stage1-override model.commitment_cost=0.25
  --stage1-override model.bottleneck_loss_weight=1.0
  --stage1-override model.recon_mse_weight=1.0
  --stage1-override model.recon_l1_weight=0.0
  --stage1-override model.recon_edge_weight=0.0
  --stage1-override model.perceptual_weight=0.0
  --stage1-override model.compute_fid=false
  --stage1-override model.coef_max=3.0
  --stage1-override train.learning_rate=2.0e-4
  --stage1-override train.gradient_clip_val=1.0
  --stage2-override data.num_workers=0
  --stage2-override ar.learning_rate=1.0e-3
  --stage2-override ar.warmup_steps=1000
  --stage2-override ar.min_lr_ratio=0.01
  --stage2-override ar.coeff_loss_type=gt_atom_recon_mse
  --stage2-override ar.coeff_loss_weight=0.1
  --stage2-override ar.coeff_huber_delta=1.0
  --stage2-override train_ar.batch_size=4
  --stage2-override train_ar.sample_every_n_epochs=5
  --stage2-override train_ar.sample_num_images=16
  --stage2-override train_ar.generation_metric_num_samples=0
  --stage2-override train_ar.compute_generation_fid=false
  --stage2-override train_ar.sample_temperature=0.5
  --stage2-override train_ar.sample_top_k=0
  --cache-arg=--num-workers
  --cache-arg=0
  --cache-arg=--coeff-max
  --cache-arg=3.0
  --cache-arg=--coeff-bins
  --cache-arg=256
  --cache-arg=--coeff-quantization
  --cache-arg=uniform
)

launch_patch() {
  local patch_size="$1"
  local label="celebahq-linefix-p${patch_size}s${patch_size}-src-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}"
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --run-label "$label" \
    --stage1-override model.patch_based=true \
    --stage1-override model.patch_size="$patch_size" \
    --stage1-override model.patch_stride="$patch_size" \
    --stage1-override model.patch_reconstruction=tile \
    --stage1-override model.num_embeddings=4096 \
    --stage1-override model.sparsity_level=24 \
    --stage2-override ar.type=sparse_spatial_depth \
    --stage2-override ar.d_model=512 \
    --stage2-override ar.n_heads=8 \
    --stage2-override ar.n_layers=12 \
    --stage2-override ar.d_ff=1024
}

launch_capacity() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --run-label "celebahq-linefix-full-cap768-src-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    --stage1-override model.patch_based=false \
    --stage1-override model.num_embeddings=1024 \
    --stage1-override model.sparsity_level=8 \
    --stage2-override ar.type=sparse_spatial_depth \
    --stage2-override ar.d_model=768 \
    --stage2-override ar.n_heads=12 \
    --stage2-override ar.n_layers=18 \
    --stage2-override ar.d_ff=3072 \
    --stage2-override train_ar.batch_size=2
}

launch_sliding_window() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --run-label "celebahq-linefix-fullgrid-win64-gpt-src-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    --stage1-override model.patch_based=false \
    --stage1-override model.num_embeddings=1024 \
    --stage1-override model.sparsity_level=8 \
    --stage2-override ar.type=gpt \
    --stage2-override ar.window_sites=64 \
    --stage2-override ar.n_global_spatial_tokens=8 \
    --stage2-override ar.d_model=512 \
    --stage2-override ar.n_heads=8 \
    --stage2-override ar.n_layers=12 \
    --stage2-override ar.d_ff=1024 \
    --stage2-override train_ar.crop_h_sites=0 \
    --stage2-override train_ar.crop_w_sites=0
}

launch_patch 2
launch_patch 4
launch_patch 8
launch_capacity
launch_sliding_window
