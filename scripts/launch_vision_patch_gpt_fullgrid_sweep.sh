#!/usr/bin/env bash
# Full-grid local-window GPT stage-2 sweep for the patch VQGAN-style LASER setup.
#
# This is the corrected variant of the FFHQ p8 GPT run that trained on 4x4
# token crops. Here stage 2 trains on the full token grid, while ar.window_sites
# keeps the GPT attention local.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
GPUS="${GPUS:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-320000}"
PROJECT="${PROJECT:-laser-debugging}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_patch_gpt_fullgrid_sweep}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
FFHQ_DIR="${FFHQ_DIR:-/scratch/$USER/datasets/ffhq}"
CASES="${CASES:-ffhq,celebahq}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

ARGS=(
  --full-training
  --cases "$CASES"
  --model-family laser
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
  --ffhq-dir "$FFHQ_DIR"
  --run-label "vision-patch-p8-fullgrid-gptwin16-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}"
  --stage1-override data.batch_size=4
  --stage1-override data.eval_batch_size=4
  --stage1-override data.num_workers=4
  --stage1-override train.limit_train_batches=1.0
  --stage1-override train.limit_val_batches=1.0
  --stage1-override train.limit_test_batches=1.0
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override train.val_check_interval=1.0
  --stage1-override train.warmup_steps=1000
  --stage1-override train.min_lr_ratio=0.05
  --stage1-override train.learning_rate=5.0e-5
  --stage1-override train.precision=bf16-mixed
  --stage1-override model.compute_fid=false
  --stage1-override model.out_tanh=true
  --stage1-override model.log_images_every_n_steps=0
  --stage1-override model.enable_val_latent_visuals=true
  --stage1-override model.backbone=vqgan
  --stage1-override model.num_downsamples=2
  --stage1-override 'model.channel_multipliers=[1,1,2]'
  --stage1-override model.num_hiddens=160
  --stage1-override model.num_residual_blocks=3
  --stage1-override model.num_residual_hiddens=80
  --stage1-override model.backbone_latent_channels=160
  --stage1-override model.embedding_dim=32
  --stage1-override model.patch_based=true
  --stage1-override model.patch_size=8
  --stage1-override model.patch_stride=8
  --stage1-override model.patch_reconstruction=tile
  --stage1-override model.num_embeddings=65536
  --stage1-override model.sparsity_level=16
  --stage1-override 'model.attn_resolutions=[]'
  --stage1-override model.use_mid_attention=true
  --stage1-override model.decoder_extra_residual_layers=2
  --stage1-override model.bottleneck_loss_weight=0.75
  --stage1-override model.commitment_cost=1.0
  --stage1-override model.dict_learning_rate=1.0e-4
  --stage1-override model.coef_max=16.0
  --stage1-override model.sparsity_reg_weight=0.0
  --stage1-override model.recon_mse_weight=0.5
  --stage1-override model.recon_l1_weight=0.5
  --stage1-override model.recon_edge_weight=0.0
  --stage1-override model.perceptual_weight=0.10
  --stage1-override model.perceptual_start_step=0
  --stage1-override model.perceptual_warmup_steps=1000
  --stage2-override data.num_workers=4
  --stage2-override train_ar.batch_size=2
  --stage2-override train_ar.limit_train_batches=1.0
  --stage2-override train_ar.limit_val_batches=1.0
  --stage2-override train_ar.limit_test_batches=1.0
  --stage2-override train_ar.crop_h_sites=0
  --stage2-override train_ar.crop_w_sites=0
  --stage2-override train_ar.sample_every_n_epochs=10
  --stage2-override train_ar.sample_log_to_wandb=true
  --stage2-override train_ar.sample_num_images=4
  --stage2-override train_ar.generation_metric_num_samples=0
  --stage2-override train_ar.compute_generation_fid=false
  --stage2-override train_ar.compute_audio_generation_metrics=false
  --stage2-override train_ar.run_test_after_fit=false
  --stage2-override train_ar.save_final_samples_after_fit=false
  --stage2-override train_ar.sample_temperature=0.9
  --stage2-override train_ar.sample_top_k=128
  --stage2-override ar.type=gpt
  --stage2-override ar.window_sites=16
  --stage2-override ar.n_global_spatial_tokens=8
  --stage2-override ar.d_model=512
  --stage2-override ar.n_heads=8
  --stage2-override ar.n_layers=10
  --stage2-override ar.d_ff=2048
  --stage2-override ar.learning_rate=3.0e-4
  --stage2-override ar.warmup_steps=1000
  --stage2-override ar.min_lr_ratio=0.05
  --stage2-override ar.coeff_loss_type=auto
  --stage2-override ar.coeff_loss_weight=1.0
  --cache-arg=--num-workers
  --cache-arg=4
  --cache-arg=--coeff-bins
  --cache-arg=512
  --cache-arg=--coeff-max
  --cache-arg=16.0
)

if [[ -n "${EXCLUDE_NODES// }" ]]; then
  ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  ARGS+=(--dry-run)
fi

"$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py "${ARGS[@]}"
