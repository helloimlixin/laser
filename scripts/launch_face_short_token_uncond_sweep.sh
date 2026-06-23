#!/bin/bash
# Full CelebA-HQ/FFHQ pipeline sweep with ImageNet-short-token style settings.
# Stage 2 is unconditional generation: no class labels and no class embedding.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$HERE/submit_multimodal_sweep.py"
PYTHON_BIN="${PYTHON_BIN:-/projects/community/miniconda/2023.11/bd387/base/bin/python}"

PARTITION="${PARTITION:-gpu-redhat}"
GPUS="${GPUS:-2}"
CPUS="${CPUS:-12}"
MEM_MB="${MEM_MB:-220000}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
PROJECT="${PROJECT:-laser}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_face_short_token_uncond}"
CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/Projects/data/celeba_hq}"
FFHQ_DIR="${FFHQ_DIR:-/cache/home/$USER/datasets/ffhq/images1024x1024_webp}"
CASES="${CASES:-celebahq,ffhq}"
DRY_RUN="${DRY_RUN:-0}"

# Step-bounded so each chained job reaches stage1_adv, cache extraction, and
# stage2 within the SLURM time limit even on large image folders.
STAGE1_STEPS="${STAGE1_STEPS:-60000}"
STAGE1_ADV_STEPS="${STAGE1_ADV_STEPS:-20000}"
STAGE2_STEPS="${STAGE2_STEPS:-120000}"

COMMON_STAGE1=(
  "model=laser_imagenet_site"
  "data.batch_size=4"
  "data.num_workers=8"
  "train.max_epochs=50"
  "train.max_steps=$STAGE1_STEPS"
  "train.learning_rate=1.0e-4"
  "train.val_check_interval=15000"
  "train.limit_val_batches=256"
  "train.limit_test_batches=0"
  "train.log_every_n_steps=50"
  "train.gradient_clip_val=1.0"
  "checkpoint.save_top_k=1"
  "checkpoint.save_last=false"
  "model.compute_fid=true"
  "model.backbone=ddpm"
  "model.use_mid_attention=true"
  "model.recon_l1_weight=1.0"
  "model.recon_mse_weight=0.25"
  "model.recon_edge_weight=0.5"
  "model.perceptual_weight=0.2"
  "model.perceptual_start_step=2000"
  "model.perceptual_warmup_steps=4000"
  "model.adversarial_weight=0.0"
  "model.data_init_from_first_batch=true"
  "model.dict_learning_rate=2.5e-4"
  "model.commitment_cost=0.25"
  "model.bottleneck_loss_weight=1.0"
  "model.coef_max=16.0"
  "model.log_images_every_n_steps=1000"
  "model.enable_val_latent_visuals=true"
  "model.patch_based=false"
  "model.patch_size=1"
  "model.patch_stride=1"
  "model.patch_reconstruction=tile"
  "model.num_embeddings=8192"
  "model.embedding_dim=128"
  "model.num_hiddens=160"
  "model.num_residual_hiddens=112"
  "model.num_residual_blocks=3"
  "model.decoder_extra_residual_layers=3"
  "model.backbone_latent_channels=512"
)

COMMON_STAGE1_ADV=(
  "train.max_epochs=20"
  "train.max_steps=$STAGE1_ADV_STEPS"
  "train.learning_rate=7.5e-5"
  "model.adversarial_weight=0.02"
  "model.adversarial_start_step=0"
  "model.adversarial_warmup_steps=2000"
  "model.disc_start_step=0"
  "model.disc_learning_rate=5.0e-5"
  "model.disc_channels=64"
  "model.disc_num_layers=3"
  "model.disc_norm=group"
  "model.disc_loss=hinge"
  "model.use_adaptive_disc_weight=false"
)

COMMON_STAGE2=(
  "ar.max_steps=$STAGE2_STEPS"
  "ar.d_model=512"
  "ar.n_heads=8"
  "ar.n_layers=12"
  "ar.d_ff=2048"
  "ar.n_global_spatial_tokens=16"
  "ar.dropout=0.1"
  "ar.learning_rate=2.5e-4"
  "ar.warmup_steps=1500"
  "ar.min_lr_ratio=0.03"
  "ar.autoregressive_coeffs=true"
  "ar.class_conditional=false"
  "ar.num_classes=0"
  "ar.coeff_loss_type=auto"
  "ar.coeff_loss_weight=1.0"
  "ar.coeff_huber_delta=0.25"
  "train_ar.batch_size=4"
  "train_ar.gradient_clip_val=1.0"
  "train_ar.val_check_interval=1.0"
  "train_ar.log_every_n_steps=20"
  "train_ar.checkpoint_save_top_k=1"
  "train_ar.checkpoint_save_last=false"
  "train_ar.checkpoint_keep_recent=1"
  "train_ar.checkpoint_every_n_epochs=1"
  "train_ar.sample_every_n_epochs=0"
  "train_ar.sample_every_n_steps=5000"
  "train_ar.sample_num_images=16"
  "train_ar.sample_temperature=0.8"
  "train_ar.sample_top_k=0"
  "train_ar.compute_generation_fid=false"
  "train_ar.generation_metric_num_samples=0"
  "train_ar.run_test_after_fit=false"
  "train_ar.save_final_samples_after_fit=true"
)

append_overrides() {
  local flag="$1"
  shift
  for item in "$@"; do
    CMD+=("$flag" "$item")
  done
}

submit_variant() {
  local label="$1"
  local sparsity="$2"
  local downsamples="$3"
  local ch_mults="$4"
  local attn_res="$5"

  CMD=(
    "$PYTHON_BIN" "$SUBMIT"
    --cases "$CASES"
    --full-training
    --model-family laser
    --project "$PROJECT"
    --partition "$PARTITION"
    --gpus "$GPUS"
    --cpus-per-task "$CPUS"
    --mem-mb "$MEM_MB"
    --time-limit "$TIME_LIMIT"
    --run-root-base "$RUN_ROOT_BASE"
    --celebahq-dir "$CELEBAHQ_DIR"
    --ffhq-dir "$FFHQ_DIR"
    --run-label "$label"
    --stage1-epochs 50
    --stage1-adv-epochs 20
    --stage2-epochs 50
    --cache-arg=--coeff-bins
    --cache-arg 256
    --cache-arg=--coeff-max
    --cache-arg 16.0
    --cache-arg=--support-order
    --cache-arg atom_id
  )
  [[ "$DRY_RUN" == "1" ]] && CMD+=(--dry-run)

  append_overrides --stage1-override "${COMMON_STAGE1[@]}"
  append_overrides --stage1-override "model.sparsity_level=$sparsity"
  append_overrides --stage1-override "model.num_downsamples=$downsamples"
  append_overrides --stage1-override "model.channel_multipliers=$ch_mults"
  append_overrides --stage1-override "model.attn_resolutions=$attn_res"

  append_overrides --stage1-adv-override "${COMMON_STAGE1_ADV[@]}"

  append_overrides --stage2-override "${COMMON_STAGE2[@]}"
  append_overrides --stage2-override "wandb.tags=[train,laser,stage2,transformer,generation,unconditional,short_tokens,$label]"

  echo "=========================================================="
  echo "Submitting $label for $CASES: d=$downsamples k=$sparsity atoms=8192 e=128"
  "${CMD[@]}"
}

submit_variant "face-d5-k2-a8192-e128-w160-uncond" 2 5 "[1,1,2,2,4,4]" "[8,16,32]"
submit_variant "face-d6-k4-a8192-e128-w160-uncond" 4 6 "[1,1,2,2,4,4,4]" "[4,8,16]"
submit_variant "face-d6-k2-a8192-e128-w160-uncond" 2 6 "[1,1,2,2,4,4,4]" "[4,8,16]"

echo "=========================================================="
if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run) previewed CelebA-HQ/FFHQ short-token unconditional sweep."
else
  echo "Submitted CelebA-HQ/FFHQ short-token unconditional sweep."
fi
