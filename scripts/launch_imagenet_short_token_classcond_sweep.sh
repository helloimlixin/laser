#!/bin/bash
# Full ImageNet-256 pipeline sweep based on W&B run ethltg23
# (imgcap-site-k8-a8192-e128-w160-60k), varying token length pressure for
# class-conditional stage-2 generation.

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
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_imagenet_short_token_classcond}"
IMAGENET_DIR="${IMAGENET_DIR:-/scratch/$USER/Projects/data/imagenet}"
DRY_RUN="${DRY_RUN:-0}"
export LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-0}"
VIS_LOG_EVERY_N_STEPS="${VIS_LOG_EVERY_N_STEPS:-1000}"
DIAG_LOG_INTERVAL="${DIAG_LOG_INTERVAL:-100}"
DICTIONARY_VIS_MAX_VECTORS="${DICTIONARY_VIS_MAX_VECTORS:-1024}"

# Stage lengths are step-bounded because one literal ImageNet epoch is too large
# for a chained stage1 -> stage1_adv -> cache -> stage2 sweep job.
STAGE1_STEPS="${STAGE1_STEPS:-60000}"
STAGE1_ADV_STEPS="${STAGE1_ADV_STEPS:-20000}"
STAGE2_STEPS="${STAGE2_STEPS:-120000}"

COMMON_STAGE1=(
  "model=laser_imagenet_site"
  "data.batch_size=2"
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
  "model.backbone=vqgan"
  "model.max_ch_mult=4"
  "model.use_mid_attention=true"
  "model.recon_l1_weight=1.0"
  "model.recon_mse_weight=0.25"
  "model.recon_edge_weight=0.5"
  "model.perceptual_weight=0.2"
  "model.perceptual_start_step=2000"
  "model.perceptual_warmup_steps=4000"
  "model.adversarial_weight=0.0"
  "model.out_tanh=true"
  "model.data_init_from_first_batch=true"
  "model.dict_learning_rate=2.5e-4"
  "model.commitment_cost=0.25"
  "model.bottleneck_loss_weight=1.0"
  "model.coef_max=16.0"
  "model.log_images_every_n_steps=$VIS_LOG_EVERY_N_STEPS"
  "model.diag_log_interval=$DIAG_LOG_INTERVAL"
  "model.enable_val_latent_visuals=true"
  "model.codebook_visual_max_vectors=$DICTIONARY_VIS_MAX_VECTORS"
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
  "ar.class_conditional=true"
  "ar.num_classes=1000"
  "ar.coeff_loss_type=auto"
  "ar.coeff_loss_weight=1.0"
  "ar.coeff_huber_delta=0.25"
  "train_ar.batch_size=2"
  "train_ar.gradient_clip_val=1.0"
  "train_ar.val_check_interval=1.0"
  "train_ar.log_every_n_steps=20"
  "train_ar.checkpoint_save_top_k=1"
  "train_ar.checkpoint_save_last=false"
  "train_ar.checkpoint_keep_recent=1"
  "train_ar.checkpoint_every_n_epochs=1"
  "train_ar.sample_every_n_epochs=0"
  "train_ar.sample_every_n_steps=5000"
  "train_ar.sample_log_to_wandb=false"
  "train_ar.sample_num_images=16"
  "train_ar.sample_class_labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"
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
    --cases imagenet
    --full-training
    --model-family laser
    --project "$PROJECT"
    --partition "$PARTITION"
    --gpus "$GPUS"
    --cpus-per-task "$CPUS"
    --mem-mb "$MEM_MB"
    --time-limit "$TIME_LIMIT"
    --run-root-base "$RUN_ROOT_BASE"
    --imagenet-dir "$IMAGENET_DIR"
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
  append_overrides --stage1-override "wandb.name=imagenet-$label-stage1-noadv"

  append_overrides --stage1-adv-override "${COMMON_STAGE1_ADV[@]}"
  append_overrides --stage1-adv-override "wandb.name=imagenet-$label-stage1-adv"

  append_overrides --stage2-override "${COMMON_STAGE2[@]}"
  append_overrides --stage2-override "wandb.name=imagenet-$label-stage2-classcond"
  append_overrides --stage2-override "wandb.tags=[train,laser,imagenet,stage2,transformer,generation,class_conditional,short_tokens,$label]"

  echo "=========================================================="
  echo "Submitting $label: d=$downsamples k=$sparsity atoms=8192 e=128"
  "${CMD[@]}"
}

submit_variant "imgcap-site-d4-k4-a8192-e128-w160-classcond" 4 4 "[1,1,2,2,4]" "[16,32]"
submit_variant "imgcap-site-d4-k2-a8192-e128-w160-classcond" 2 4 "[1,1,2,2,4]" "[16,32]"
submit_variant "imgcap-site-d5-k8-a8192-e128-w160-classcond" 8 5 "[1,1,2,2,4,4]" "[8,16,32]"
submit_variant "imgcap-site-d5-k4-a8192-e128-w160-classcond" 4 5 "[1,1,2,2,4,4]" "[8,16,32]"

echo "=========================================================="
if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run) previewed ImageNet short-token class-conditioned sweep."
else
  echo "Submitted ImageNet short-token class-conditioned sweep."
fi
