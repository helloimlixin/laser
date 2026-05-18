#!/usr/bin/env bash
# COCO stage-1 replacements for runs where the sparse bottleneck drifts upward.

set -euo pipefail

PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
GPUS="${GPUS:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-240000}"
PROJECT="${PROJECT:-laser-debugging}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/coco_bottleneck_recovery}"
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
  --stage1-override model.compute_fid=true
  --stage1-override model.out_tanh=true
  --stage1-override data.num_workers=8
)

"$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
  "${COMMON_ARGS[@]}" \
  --run-label "coco-stage1-bneck-cap-laser-f32-k32768-z256-s4-c1-bw05-coef16-s1-${STAGE1_EPOCHS}" \
  "${COMMON_STAGE1[@]}" \
  --stage1-override data.batch_size=1 \
  --stage1-override model.backbone=vqgan \
  --stage1-override model.num_downsamples=5 \
  --stage1-override model.channel_multipliers=[1,1,2,2,4,4] \
  --stage1-override model.num_hiddens=128 \
  --stage1-override model.num_residual_blocks=3 \
  --stage1-override model.num_residual_hiddens=64 \
  --stage1-override model.backbone_latent_channels=512 \
  --stage1-override model.embedding_dim=256 \
  --stage1-override model.num_embeddings=32768 \
  --stage1-override model.sparsity_level=4 \
  --stage1-override model.attn_resolutions=[16] \
  --stage1-override model.use_mid_attention=true \
  --stage1-override model.decoder_extra_residual_layers=1 \
  --stage1-override model.bottleneck_loss_weight=0.5 \
  --stage1-override model.commitment_cost=1.0 \
  --stage1-override model.dict_learning_rate=1.0e-4 \
  --stage1-override model.coef_max=16.0 \
  --stage1-override model.bounded_omp_refine_steps=12 \
  --stage1-override model.recon_mse_weight=1.0 \
  --stage1-override model.recon_l1_weight=0.0 \
  --stage1-override model.recon_edge_weight=0.0 \
  --stage1-override model.perceptual_weight=0.5 \
  --stage1-override model.perceptual_start_step=0 \
  --stage1-override model.perceptual_warmup_steps=0 \
  --stage1-override train.learning_rate=2.0e-5 \
  --stage1-override train.precision=bf16-mixed

"$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
  "${COMMON_ARGS[@]}" \
  --run-label "coco-stage1-bneck-control-laser-f16-k16384-z256-s4-c025-bw025-s1-${STAGE1_EPOCHS}" \
  "${COMMON_STAGE1[@]}" \
  --stage1-override data.batch_size=1 \
  --stage1-override model.backbone=vqgan \
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
  --stage1-override train.precision=bf16-mixed
