#!/usr/bin/env bash
# Faithful rerun of the rq3ivx3d "power" pipeline with a 3x longer stage-2.
#
# rq3ivx3d = CelebA-HQ stage-2 sparse_spatial_depth prior (d768, 18 spatial /
# 9 depth layers, 16 global tokens, QUANTIZED 256-bin coeffs, coef_max 16,
# support_order=atom_id, p2s2/k4 8x8 grid): generation FID 43.8 — the best
# CelebA-HQ number so far. It stopped at exactly ar.max_steps=250000 (epoch
# 124/150) with the cosine LR fully decayed, i.e. cut off by the step limit.
#
# Lineage being reproduced (Lambda runs, commit 3cc46dd):
#   stage1 no-adv : qinvce8o -> vegdxqp9 -> ylc69j0u (3x10ep continue chain
#                   = 30 effective clean epochs, b16, lr 2.3e-4)
#   stage1 adv    : gwmkhbd8 (20ep, FIXED adversarial_weight=0.03, warmup 375,
#                   hinge PatchGAN 64ch/3L, disc lr 5e-5, NO adaptive weight)
#   cache         : token_cache_q256.pt (--coeff-bins 256 --coeff-max 16
#                   --support-order atom_id)
#   stage2        : rq3ivx3d (lr 2.5e-4, warmup 1500, min_lr_ratio 0.03,
#                   b12, max_steps 250000)
#
# Changes vs the original:
#   - stage2 ar.max_steps 250k -> 750k (the ask: train past the step limit;
#     the cosine horizon stretches with max_steps so LR stays useful longer)
#   - stage1 clean epochs 30 -> 40 in ONE cosine cycle (the original's 3
#     restarted 10-ep schedules are not worth reproducing)
#   - 2 GPUs DDP: stage1 per-device batch 8 (eff 16 = original), stage2
#     per-device batch 12 (eff 24 = 2x original throughput per step)
#   - disc_norm=group (DDP-safe; original used batch norm on 1 GPU)
#   - June-2026 flag renames: discriminator_* -> disc_*; the retired
#     dictionary_through_decoder / dead_atom_revival_steps are now the
#     model's default behavior / no-ops and are not passed.
#
# Stage-2 checkpoints save_top_k=3 + last, so a wall-clock kill near 750k is
# recoverable with scripts/submit_stage2_resume.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
# W&B group = "laser-train-<run-label>-<stamp>" must stay under 128 chars.
RUN_TAG="${RUN_TAG:-powerlong-$(date +%m%d)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_power_s2_long}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/Projects/data/celeba_hq}"
# Image dataset selector (celebahq|ffhq|imagenet); DATA_DIR defaults per dataset.
DATASET="${DATASET:-celebahq}"
case "$DATASET" in
  celebahq) DATA_DIR="${DATA_DIR:-$CELEBAHQ_DIR}" ;;
  ffhq)     DATA_DIR="${DATA_DIR:-/scratch/$USER/Projects/data/ffhq}" ;;
  imagenet) DATA_DIR="${DATA_DIR:-/scratch/$USER/Projects/data/imagenet}" ;;
  *) echo "ERROR: unsupported DATASET=$DATASET (celebahq|ffhq|imagenet)" >&2; exit 1 ;;
esac

GPUS="${GPUS:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEM_MB="${MEM_MB:-240000}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018,gpuk[005-018]}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-40}"
STAGE1_ADV_EPOCHS="${STAGE1_ADV_EPOCHS:-20}"
# 667 epochs at eff batch 24 (27k cached items / 24 = 1125 steps/epoch);
# the epoch cap just needs to clear the step budget.
STAGE2_EPOCHS="${STAGE2_EPOCHS:-800}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-750000}"

# Sweepable recipe knobs (defaults reproduce rq3ivx3d exactly).
# SPARSITY_LEVEL changes the codec (new stage-1); COEFF_BINS is cache-time only
# (same codec re-quantized) — see run-label k${SPARSITY_LEVEL} q${COEFF_BINS}.
SPARSITY_LEVEL="${SPARSITY_LEVEL:-4}"
COEFF_BINS="${COEFF_BINS:-256}"

STAGE1_BATCH_PER_GPU="${STAGE1_BATCH_PER_GPU:-8}"
STAGE2_BATCH_PER_GPU="${STAGE2_BATCH_PER_GPU:-12}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: dataset directory not found: $DATA_DIR" >&2
  exit 1
fi

COMMON_ARGS=(
  --full-training
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage1-adv-epochs "$STAGE1_ADV_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
  --partition "$PARTITION"
  --time-limit "$TIME_LIMIT"
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
  "--${DATASET}-dir" "$DATA_DIR"
)
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  COMMON_ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

echo "=== rq3ivx3d power-recipe long rerun ==="
echo "RUN_TAG=$RUN_TAG PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT DRY_RUN=$DRY_RUN"
echo "stage1=$STAGE1_EPOCHS ep (eff b16) + adv=$STAGE1_ADV_EPOCHS ep; stage2 max_steps=$STAGE2_MAX_STEPS (orig 250000), eff b$((STAGE2_BATCH_PER_GPU * GPUS))"

"$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
  "${COMMON_ARGS[@]}" \
  --gpus "$GPUS" \
  --cpus-per-task "$CPUS_PER_TASK" \
  --mem-mb "$MEM_MB" \
  --cases "$DATASET" \
  --model-family laser \
  --run-label "${RUN_TAG}-${DATASET}-p2s2k${SPARSITY_LEVEL}-q${COEFF_BINS}" \
  --cache-arg=--coeff-bins \
  --cache-arg="$COEFF_BINS" \
  --cache-arg=--coeff-max \
  --cache-arg=16.0 \
  --cache-arg=--support-order \
  --cache-arg=atom_id \
  --cache-arg=--batch-size \
  --cache-arg="$CACHE_BATCH_SIZE" \
  --stage2-override ar.type=sparse_spatial_depth \
  --stage2-override ar.autoregressive_coeffs=true \
  --stage2-override ar.max_steps="$STAGE2_MAX_STEPS" \
  --stage2-override ar.d_model=768 \
  --stage2-override ar.n_heads=12 \
  --stage2-override ar.n_layers=18 \
  --stage2-override ar.d_ff=3072 \
  --stage2-override ar.n_global_spatial_tokens=16 \
  --stage2-override ar.dropout=0.1 \
  --stage2-override ar.learning_rate=2.5e-4 \
  --stage2-override ar.warmup_steps=1500 \
  --stage2-override ar.min_lr_ratio=0.03 \
  --stage2-override ar.coeff_loss_type=auto \
  --stage2-override ar.coeff_loss_weight=1.0 \
  --stage2-override ar.coeff_huber_delta=0.25 \
  --stage2-override train_ar.batch_size="$STAGE2_BATCH_PER_GPU" \
  --stage2-override train_ar.gradient_clip_val=1.0 \
  --stage2-override train_ar.val_check_interval=1.0 \
  --stage2-override train_ar.log_every_n_steps=20 \
  --stage2-override train_ar.checkpoint_save_top_k=1 \
  --stage2-override train_ar.checkpoint_save_last=false \
  --stage2-override train_ar.checkpoint_keep_recent=1 \
  --stage2-override train_ar.checkpoint_every_n_epochs=25 \
  --stage2-override train_ar.sample_every_n_epochs=12 \
  --stage2-override train_ar.sample_num_images=8 \
  --stage2-override train_ar.sample_temperature=0.7 \
  --stage2-override train_ar.sample_top_k=0 \
  --stage2-override train_ar.compute_generation_fid=true \
  --stage2-override train_ar.generation_metric_num_samples=256 \
  --stage2-override train_ar.run_test_after_fit=false \
  --stage2-override train_ar.save_final_samples_after_fit=true \
  --stage1-override data.image_size=256 \
  --stage1-override data.batch_size="$STAGE1_BATCH_PER_GPU" \
  --stage1-override data.num_workers=4 \
  --stage1-override data.augment=true \
  --stage1-override train.learning_rate=2.3e-4 \
  --stage1-override train.warmup_steps=94 \
  --stage1-override train.min_lr_ratio=0.05 \
  --stage1-override train.gradient_clip_val=1.0 \
  --stage1-override train.val_check_interval=0.25 \
  --stage1-override train.limit_val_batches=256 \
  --stage1-override train.limit_test_batches=256 \
  --stage1-override train.log_every_n_steps=20 \
  --stage1-override train.run_test_after_fit=false \
  --stage1-override checkpoint.save_top_k=0 \
  --stage1-override checkpoint.save_last=false \
  --stage1-override model=laser \
  --stage1-override model.backbone=vqgan \
  --stage1-override model.num_hiddens=128 \
  --stage1-override model.num_downsamples=4 \
  --stage1-override model.channel_multipliers=[1,1,2,2,4] \
  --stage1-override model.backbone_latent_channels=512 \
  --stage1-override model.max_ch_mult=4 \
  --stage1-override model.embedding_dim=128 \
  --stage1-override model.num_embeddings=4096 \
  --stage1-override model.sparsity_level="$SPARSITY_LEVEL" \
  --stage1-override model.patch_based=true \
  --stage1-override model.patch_size=2 \
  --stage1-override model.patch_stride=2 \
  --stage1-override model.patch_reconstruction=tile \
  --stage1-override model.bottleneck_loss_weight=0.25 \
  --stage1-override model.commitment_cost=0.05 \
  --stage1-override model.coef_max=16.0 \
  --stage1-override model.dict_learning_rate=5.8e-4 \
  --stage1-override model.num_residual_blocks=3 \
  --stage1-override model.num_residual_hiddens=96 \
  --stage1-override model.decoder_extra_residual_layers=2 \
  --stage1-override model.use_mid_attention=true \
  --stage1-override model.attn_resolutions=[16,32] \
  --stage1-override model.data_init_from_first_batch=true \
  --stage1-override model.out_tanh=true \
  --stage1-override model.recon_mse_weight=0.25 \
  --stage1-override model.recon_l1_weight=1.0 \
  --stage1-override model.recon_edge_weight=0.50 \
  --stage1-override model.perceptual_weight=0.20 \
  --stage1-override model.perceptual_start_step=188 \
  --stage1-override model.perceptual_warmup_steps=375 \
  --stage1-override model.adversarial_weight=0.0 \
  --stage1-override model.adversarial_start_step=1000000000 \
  --stage1-override model.adversarial_warmup_steps=0 \
  --stage1-override model.disc_start_step=1000000000 \
  --stage1-override model.compute_fid=true \
  --stage1-override model.log_images_every_n_steps=200 \
  --stage1-adv-override model.adversarial_weight=0.03 \
  --stage1-adv-override model.adversarial_start_step=0 \
  --stage1-adv-override model.adversarial_warmup_steps=375 \
  --stage1-adv-override model.disc_start_step=0 \
  --stage1-adv-override model.disc_learning_rate=5.0e-5 \
  --stage1-adv-override model.disc_channels=64 \
  --stage1-adv-override model.disc_num_layers=3 \
  --stage1-adv-override model.disc_norm=group \
  --stage1-adv-override model.disc_loss=hinge \
  --stage1-adv-override model.use_adaptive_disc_weight=false

echo "=== submission complete ==="
