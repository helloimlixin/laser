#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/pg.sh"

pg_mkdirs

python3 train.py \
  model=laser \
  data=celeba \
  seed=42 \
  output_dir="$(pg_s1_out)" \
  data.data_dir="$DATA_DIR" \
  data.image_size="$IMG" \
  data.batch_size="$S1_BSZ" \
  data.num_workers="$S1_WORKERS" \
  train.accelerator=gpu \
  train.devices=1 \
  train.strategy=auto \
  train.precision=16-mixed \
  train.max_epochs="$S1_EPOCHS" \
  train.learning_rate="$S1_LR" \
  train.log_every_n_steps=25 \
  train.val_check_interval=1.0 \
  wandb.project="$WANDB_PROJECT" \
  wandb.name="$(pg_tag)_s1" \
  wandb.save_dir="$(pg_wandb_dir)" \
  model.num_embeddings="$ATOMS" \
  model.sparsity_level="$K" \
  model.num_hiddens="$NHID" \
  model.embedding_dim="$EMB" \
  model.num_residual_blocks="$RBLK" \
  model.num_residual_hiddens="$RHID" \
  model.patch_based=true \
  model.patch_size="$(pg_patch)" \
  model.patch_stride="$(pg_stride)" \
  model.patch_reconstruction=tile \
  model.coef_max="$CMAX" \
  model.compute_fid=true \
  model.fid_feature=2048 \
  model.log_images_every_n_steps=500

S1_CKPT="$(find "$(pg_s1_out)/checkpoints" -path '*/laser/last.ckpt' | sort | tail -n 1)"
if [[ -z "$S1_CKPT" ]]; then
  echo "stage1_checkpoint_not_found" >&2
  exit 1
fi

pg_write_ref "$(pg_s1_ckpt_ref)" "$S1_CKPT"
echo "tag=$(pg_tag)"
echo "stage1_checkpoint=$S1_CKPT"
