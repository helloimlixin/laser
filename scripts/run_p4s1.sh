#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/p4g.sh"

p4g_mkdirs

python3 train.py \
  model=laser \
  data=celeba \
  seed=42 \
  output_dir="$(p4g_s1_out)" \
  data.data_dir="$DATA_DIR" \
  data.image_size="$IMG" \
  data.batch_size="$S1_BSZ" \
  data.num_workers="$S1_WORKERS" \
  train.accelerator=gpu \
  train.devices="$S1_GPUS" \
  train.strategy="$(if [ "${S1_GPUS}" -gt 1 ]; then echo ddp; else echo auto; fi)" \
  train.precision=16-mixed \
  train.max_epochs="$S1_EPOCHS" \
  train.learning_rate="$S1_LR" \
  train.warmup_steps="$S1_WARMUP_STEPS" \
  train.min_lr_ratio="$S1_MIN_LR_RATIO" \
  train.log_every_n_steps="$S1_LOG_EVERY" \
  train.val_check_interval="$S1_VAL_INTERVAL" \
  wandb.project=laser-s1 \
  wandb.name=c256_p4_s1 \
  wandb.save_dir="$(p4g_wandb_dir)" \
  model.num_embeddings="$ATOMS" \
  model.sparsity_level="$K" \
  model.num_hiddens="$NHID" \
  model.embedding_dim="$EMB" \
  model.num_residual_blocks="$RBLK" \
  model.num_residual_hiddens="$RHID" \
  model.patch_based=true \
  model.patch_size="$PATCH" \
  model.patch_stride="$STRIDE" \
  model.patch_reconstruction=tile \
  model.coef_max="$CMAX" \
  model.dict_learning_rate="$S1_DICT_LR" \
  model.bottleneck_loss_weight="$S1_BOTTLENECK_W" \
  model.perceptual_weight="$S1_PERCEPTUAL_W" \
  model.sparsity_reg_weight="$S1_SPARSITY_REG_W" \
  model.coherence_weight="$S1_COHERENCE_W" \
  model.bounded_omp_refine_steps="$S1_BOUNDED_OMP" \
  model.compute_fid=true \
  model.fid_feature=2048 \
  model.log_images_every_n_steps="$S1_IMG_EVERY" \
  model.diag_log_interval="$S1_DIAG_EVERY" \
  model.enable_val_latent_visuals="$S1_LATENT_VIS"

S1_CKPT="$(find "$(p4g_s1_out)/checkpoints" -path '*/laser/last.ckpt' | sort | tail -n 1)"
if [[ -z "$S1_CKPT" ]]; then
  echo "stage1_checkpoint_not_found" >&2
  exit 1
fi
p4g_write_ref "$(p4g_s1_ckpt_ref)" "$S1_CKPT"

python3 build_token_cache.py \
  --stage1_checkpoint "$S1_CKPT" \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --split train \
  --image_size "$IMG" \
  --batch_size 16 \
  --num_workers 8 \
  --device cuda \
  --coeff_vocab_size "$BINS" \
  --coeff_max "$CMAX" \
  --output "$(p4g_cache_pt)"

p4g_write_ref "$(p4g_cache_ref)" "$(p4g_cache_pt)"
echo "stage1_checkpoint=$S1_CKPT"
echo "token_cache=$(p4g_cache_pt)"
