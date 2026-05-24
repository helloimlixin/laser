#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/pg.sh"

BIN="${BIN:?BIN is required}"
S1_CKPT="$(pg_read_ref "$(pg_s1_ckpt_ref)")"

mkdir -p "$(pg_s2_out "$BIN")" "$RUN_ROOT/cache"

python3 cache.py \
  --stage1-checkpoint "$S1_CKPT" \
  --dataset celeba \
  --data-dir "$DATA_DIR" \
  --split train \
  --image-size "$IMG" \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --coeff-bins "$BIN" \
  --coeff-max "$CMAX" \
  --output-path "$(pg_cache_pt "$BIN")"

python3 train_stage2_prior.py \
  seed=42 \
  output_dir="$(pg_s2_out "$BIN")" \
  token_cache_path="$(pg_cache_pt "$BIN")" \
  data.num_workers="$S2_WORKERS" \
  ar.type=gpt \
  ar.window_sites="$WIN" \
  ar.d_model="$D_MODEL" \
  ar.n_heads="$HEADS" \
  ar.n_layers="$LAYERS" \
  ar.d_ff="$D_FF" \
  ar.dropout=0.1 \
  ar.learning_rate="$S2_LR" \
  ar.weight_decay=0.01 \
  ar.warmup_steps=1000 \
  wandb.project="$WANDB_PROJECT" \
  wandb.name="$(pg_tag)_b${BIN}" \
  wandb.save_dir="$(pg_wandb_dir)" \
  train_ar.batch_size="$S2_BSZ" \
  train_ar.max_epochs="$S2_EPOCHS" \
  train_ar.accelerator=gpu \
  train_ar.devices=1 \
  train_ar.strategy=auto \
  train_ar.precision=16-mixed \
  train_ar.val_check_interval=0.25 \
  train_ar.sample_every_n_epochs=5 \
  train_ar.sample_num_images=16 \
  train_ar.sample_log_to_wandb=true

echo "tag=$(pg_tag)"
echo "stage1_checkpoint=$S1_CKPT"
echo "token_cache=$(pg_cache_pt "$BIN")"
echo "bins=$BIN"
