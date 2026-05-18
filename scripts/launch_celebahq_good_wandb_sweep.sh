#!/usr/bin/env bash
set -euo pipefail

# Relaunch CelebA-HQ from maintained src code using the two CelebA-128 W&B
# runs that had promising recon/prior behavior:
#   - boppnb0r: compact non-quantized sparse profile
#   - 351lokx0: quantized sparse-coeff profile with a 12-layer prior
#
# The previous src CelebA-HQ launch died from CPU/shm pressure after manually
# overriding per-rank batch and worker counts. Keep this launcher one-GPU and
# zero-worker end to end unless explicitly overridden by the caller.

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-384000}"
PROJECT="${PROJECT:-laser}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/xl598/runs/celebahq_src_sweeps}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/xl598/submission_snapshots}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
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
  --stage2-override train_ar.batch_size=4
  --stage2-override train_ar.sample_num_images=16
  --stage2-override train_ar.generation_metric_num_samples=16
  --cache-arg=--num-workers
  --cache-arg=0
  --cache-arg=--coeff-max
  --cache-arg=3.0
  --cache-arg=--coeff-quantization
  --cache-arg=uniform
)

launch_boppnb0r() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --run-label celebahq-boppnb0r-src-s1-20-s2-100-memfix \
    --stage1-override model.num_downsamples=3 \
    --stage1-override model.num_embeddings=128 \
    --stage1-override model.sparsity_level=3 \
    --stage2-override ar.d_model=256 \
    --stage2-override ar.n_heads=8 \
    --stage2-override ar.n_layers=6 \
    --stage2-override ar.d_ff=1024 \
    --stage2-override ar.learning_rate=4.0e-4 \
    --stage2-override ar.coeff_loss_weight=1.0 \
    --stage2-override train_ar.sample_temperature=0.6 \
    --stage2-override train_ar.sample_top_k=0 \
    --cache-arg=--coeff-bins \
    --cache-arg=0
}

launch_351lokx0() {
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --run-label celebahq-351lokx0-src-s1-20-s2-100-memfix \
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.num_embeddings=1024 \
    --stage1-override model.sparsity_level=8 \
    --stage2-override ar.d_model=512 \
    --stage2-override ar.n_heads=8 \
    --stage2-override ar.n_layers=12 \
    --stage2-override ar.d_ff=1024 \
    --stage2-override ar.learning_rate=1.0e-3 \
    --stage2-override ar.coeff_loss_weight=0.1 \
    --stage2-override ar.coeff_huber_delta=1.0 \
    --stage2-override train_ar.sample_temperature=0.5 \
    --stage2-override train_ar.sample_top_k=0 \
    --cache-arg=--coeff-bins \
    --cache-arg=256
}

launch_boppnb0r
launch_351lokx0
