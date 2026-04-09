#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/p4g.sh"

WIN="${WIN:?WIN is required}"
GST="${GST:-0}"
S1_CKPT="$(p4g_read_ref "$(p4g_s1_ckpt_ref)")"
CACHE_PT="$(p4g_read_ref "$(p4g_cache_ref)")"

mkdir -p "$(p4g_s2_out "$WIN" "$GST")"

python3 train_s2.py \
  seed=42 \
  output_dir="$(p4g_s2_out "$WIN" "$GST")" \
  token_cache_path="$CACHE_PT" \
  data.num_workers="$S2_WORKERS" \
  ar.type=gpt \
  ar.window_sites="$WIN" \
  ar.n_global_spatial_tokens="$GST" \
  ar.d_model="$D_MODEL" \
  ar.n_heads="$HEADS" \
  ar.n_layers="$LAYERS" \
  ar.d_ff="$D_FF" \
  ar.dropout=0.1 \
  ar.learning_rate="$S2_LR" \
  ar.weight_decay=0.01 \
  ar.warmup_steps=1000 \
  wandb.project=laser-s2 \
  wandb.name="c256_p4_w${WIN}_g${GST}" \
  wandb.save_dir="$(p4g_wandb_dir)" \
  train_ar.batch_size="$S2_BSZ" \
  train_ar.max_epochs="$S2_EPOCHS" \
  train_ar.accelerator=gpu \
  train_ar.devices="$S2_GPUS" \
  train_ar.strategy="$(if [ "${S2_GPUS}" -gt 1 ]; then echo ddp; else echo auto; fi)" \
  train_ar.precision=16-mixed \
  train_ar.val_check_interval="$S2_VAL_INTERVAL" \
  train_ar.sample_every_n_steps="$S2_SAMPLE_STEP_EVERY" \
  train_ar.sample_every_n_epochs="$S2_SAMPLE_EPOCH_EVERY" \
  train_ar.sample_num_images="$S2_SAMPLE_IMAGES" \
  train_ar.sample_log_to_wandb=true

echo "stage1_checkpoint=$S1_CKPT"
echo "token_cache=$CACHE_PT"
echo "window_sites=$WIN"
echo "global_spatial_tokens=$GST"
