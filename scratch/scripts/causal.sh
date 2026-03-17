#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/xl598/anaconda3/etc/profile.d/conda.sh
conda activate laser

cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
if [[ -n "${WANDB_ANONYMOUS:-}" ]]; then
  export WANDB_ANONYMOUS
fi

args=(
  --dataset "${DATASET:-celeba}"
  --data_dir "${DATA_DIR:-/home/xl598/Projects/data/celeba}"
  --stage1_source_ckpt "${STAGE1_CKPT:-$ROOT_DIR/checkpoints/quantized/stage1/ae_best.pt}"
  --out_dir "${OUT_DIR:-$ROOT_DIR/runs/laser_scale_var_causal_train_gpu2}"
  --epochs "${EPOCHS:-100}"
  --batch_size "${BATCH_SIZE:-8}"
  --num_workers "${NUM_WORKERS:-4}"
  --token_num_workers "${TOKEN_NUM_WORKERS:-2}"
  --cpu_threads "${CPU_THREADS:-4}"
  --dist_timeout_minutes "${DIST_TIMEOUT_MINUTES:-120}"
  --lr "${LR:-2e-4}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --var_d_model "${VAR_D_MODEL:-256}"
  --var_heads "${VAR_HEADS:-8}"
  --var_layers "${VAR_LAYERS:-8}"
  --var_ff "${VAR_FF:-1024}"
  --sample_every_steps "${SAMPLE_EVERY_STEPS:-1000}"
  --sample_batch_size "${SAMPLE_BATCH_SIZE:-8}"
  --sample_temperature "${SAMPLE_TEMPERATURE:-1.0}"
  --sample_top_k "${SAMPLE_TOP_K:-0}"
  --teacher_forced_recon_every_steps "${TEACHER_FORCED_RECON_EVERY_STEPS:-1000}"
  --teacher_forced_recon_num_samples "${TEACHER_FORCED_RECON_NUM_SAMPLES:-8}"
  --wandb_mode "${WANDB_MODE:-disabled}"
  --amp "${AMP:-true}"
)

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  args+=(--wandb_project "${WANDB_PROJECT}")
fi

if [[ -n "${WANDB_NAME:-}" ]]; then
  args+=(--wandb_name "${WANDB_NAME}")
fi

exec /home/xl598/anaconda3/envs/laser/bin/torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  "$ROOT_DIR/scale_var.py" \
  "${args[@]}"
