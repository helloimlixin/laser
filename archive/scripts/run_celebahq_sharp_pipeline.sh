#!/usr/bin/env bash
# CelebA-HQ sharpness pipeline: stage 1 autoencoder, token cache, stage 2 prior.
#
# Follow-up to W&B run helloimlixin-rutgers/laser/vc17pae5. The stage-1
# defaults below keep the sharper d4/LPIPS/attention settings from
# run_celebahq_stage1_sharp.sh, then train a real-valued sparse-token prior.
set -euo pipefail

cd /home/xl598/Projects/laser

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/xl598/Projects/laser/runs/celebahq_sharp_pipeline_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba_hq}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_GROUP="${WANDB_GROUP:-celebahq_sharp_pipeline_${STAMP}}"

STAGE1_DIR="${STAGE1_DIR:-${RUN_ROOT}/stage1}"
STAGE2_DIR="${STAGE2_DIR:-${RUN_ROOT}/stage2}"
CACHE="${CACHE:-${RUN_ROOT}/token_cache_train_real.pt}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-3}"
STAGE1_LR="${STAGE1_LR:-1.0e-4}"
STAGE1_DICT_LR="${STAGE1_DICT_LR:-2.5e-4}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-500}"
STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-0.05}"

NUM_HIDDENS="${NUM_HIDDENS:-128}"
NUM_RESIDUAL_BLOCKS="${NUM_RESIDUAL_BLOCKS:-3}"
NUM_RESIDUAL_HIDDENS="${NUM_RESIDUAL_HIDDENS:-96}"
DECODER_EXTRA_RESIDUAL_LAYERS="${DECODER_EXTRA_RESIDUAL_LAYERS:-2}"
BACKBONE_LATENT_CHANNELS="${BACKBONE_LATENT_CHANNELS:-512}"

NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-8192}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-24}"
COEF_MAX="${COEF_MAX:-16.0}"

RECON_MSE_WEIGHT="${RECON_MSE_WEIGHT:-0.25}"
RECON_L1_WEIGHT="${RECON_L1_WEIGHT:-1.0}"
RECON_EDGE_WEIGHT="${RECON_EDGE_WEIGHT:-0.50}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.20}"
PERCEPTUAL_START_STEP="${PERCEPTUAL_START_STEP:-1000}"
PERCEPTUAL_WARMUP_STEPS="${PERCEPTUAL_WARMUP_STEPS:-2000}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
CACHE_MAX_ITEMS="${CACHE_MAX_ITEMS:-0}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE2_LR="${STAGE2_LR:-3.0e-4}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-1000}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.05}"
STAGE2_D_MODEL="${STAGE2_D_MODEL:-512}"
STAGE2_N_HEADS="${STAGE2_N_HEADS:-8}"
STAGE2_N_LAYERS="${STAGE2_N_LAYERS:-8}"
STAGE2_D_FF="${STAGE2_D_FF:-2048}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"
STAGE2_SAMPLE_EVERY_N_EPOCHS="${STAGE2_SAMPLE_EVERY_N_EPOCHS:-5}"
STAGE2_SAMPLE_NUM_IMAGES="${STAGE2_SAMPLE_NUM_IMAGES:-8}"
STAGE2_SAMPLE_TEMPERATURE="${STAGE2_SAMPLE_TEMPERATURE:-0.7}"
STAGE2_SAMPLE_TOP_K="${STAGE2_SAMPLE_TOP_K:-0}"
STAGE2_SAMPLE_COEFF_MODE="${STAGE2_SAMPLE_COEFF_MODE:-mean}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/home/xl598/Projects/laser
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}/logs" "${STAGE1_DIR}" "${STAGE2_DIR}"

echo "[$(date --iso-8601=seconds)] pipeline start"
echo "RUN_ROOT=${RUN_ROOT}"
echo "STAGE1_DIR=${STAGE1_DIR}"
echo "STAGE2_DIR=${STAGE2_DIR}"
echo "CACHE=${CACHE}"
echo "WANDB_GROUP=${WANDB_GROUP}"

"${PYTHON_BIN}" train.py stage1 \
  output_dir="${STAGE1_DIR}" \
  hydra.run.dir="${STAGE1_DIR}/hydra" \
  model=laser \
  data=celebahq \
  data.data_dir="${DATA_DIR}" \
  data.image_size=256 \
  data.batch_size="${STAGE1_BATCH_SIZE}" \
  data.num_workers=4 \
  data.augment=true \
  train.accelerator=gpu \
  train.devices=2 \
  train.strategy=ddp \
  train.precision=bf16-mixed \
  train.max_epochs="${STAGE1_EPOCHS}" \
  train.learning_rate="${STAGE1_LR}" \
  train.warmup_steps="${STAGE1_WARMUP_STEPS}" \
  train.min_lr_ratio="${STAGE1_MIN_LR_RATIO}" \
  train.gradient_clip_val=1.0 \
  train.log_every_n_steps=20 \
  checkpoint.save_top_k=1 \
  model.backbone=ddpm \
  model.num_hiddens="${NUM_HIDDENS}" \
  model.num_downsamples=4 \
  "model.channel_multipliers=[1,1,2,2,4]" \
  model.backbone_latent_channels="${BACKBONE_LATENT_CHANNELS}" \
  model.embedding_dim="${EMBEDDING_DIM}" \
  model.num_embeddings="${NUM_EMBEDDINGS}" \
  model.sparsity_level="${SPARSITY_LEVEL}" \
  model.patch_based=false \
  model.patch_size=2 \
  model.patch_stride=2 \
  model.patch_reconstruction=tile \
  model.bottleneck_loss_weight=0.25 \
  model.commitment_cost=0.05 \
  model.coef_max="${COEF_MAX}" \
  model.dict_learning_rate="${STAGE1_DICT_LR}" \
  model.num_residual_blocks="${NUM_RESIDUAL_BLOCKS}" \
  model.num_residual_hiddens="${NUM_RESIDUAL_HIDDENS}" \
  model.decoder_extra_residual_layers="${DECODER_EXTRA_RESIDUAL_LAYERS}" \
  model.use_mid_attention=true \
  "model.attn_resolutions=[16,32]" \
  +model.dead_atom_revival_steps=100 \
  +model.data_init_from_first_batch=true \
  model.recon_mse_weight="${RECON_MSE_WEIGHT}" \
  model.recon_l1_weight="${RECON_L1_WEIGHT}" \
  model.recon_edge_weight="${RECON_EDGE_WEIGHT}" \
  model.perceptual_weight="${PERCEPTUAL_WEIGHT}" \
  model.perceptual_start_step="${PERCEPTUAL_START_STEP}" \
  model.perceptual_warmup_steps="${PERCEPTUAL_WARMUP_STEPS}" \
  model.compute_fid=false \
  model.log_images_every_n_steps=200 \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="celebahq_s1_sharp_d4_lpips_${STAMP}"

CKPT="$(
  find "${STAGE1_DIR}/checkpoints" -type f -name "final.ckpt" -printf "%T@ %p\n" \
    | sort -nr | head -1 | cut -d" " -f2-
)"
if [[ -z "${CKPT}" ]]; then
  CKPT="$(
    find "${STAGE1_DIR}/checkpoints" -type f -name "*.ckpt" ! -name "last.ckpt" -printf "%T@ %p\n" \
      | sort -nr | head -1 | cut -d" " -f2-
  )"
fi
if [[ -z "${CKPT}" ]]; then
  CKPT="$(find "${STAGE1_DIR}/checkpoints" -type f -name "last.ckpt" | head -1)"
fi
[[ -n "${CKPT}" ]] || { echo "No stage-1 checkpoint found under ${STAGE1_DIR}" >&2; exit 1; }
echo "[$(date --iso-8601=seconds)] stage-1 checkpoint: ${CKPT}"

"${PYTHON_BIN}" cache.py \
  --stage1-checkpoint "${CKPT}" \
  --model-type laser \
  --output-path "${CACHE}" \
  --dataset celebahq \
  --data-dir "${DATA_DIR}" \
  --image-size 256 \
  --batch-size "${CACHE_BATCH_SIZE}" \
  --num-workers "${CACHE_NUM_WORKERS}" \
  --seed 42 \
  --max-items "${CACHE_MAX_ITEMS}" \
  --coeff-bins 0 \
  --coeff-max "${COEF_MAX}" \
  --device auto

echo "[$(date --iso-8601=seconds)] token cache: ${CACHE}"

"${PYTHON_BIN}" train.py stage2 \
  output_dir="${STAGE2_DIR}" \
  hydra.run.dir="${STAGE2_DIR}/hydra" \
  token_cache_path="${CACHE}" \
  seed=42 \
  ar.type=sparse_spatial_depth \
  ar.d_model="${STAGE2_D_MODEL}" \
  ar.n_heads="${STAGE2_N_HEADS}" \
  ar.n_layers="${STAGE2_N_LAYERS}" \
  ar.d_ff="${STAGE2_D_FF}" \
  ar.learning_rate="${STAGE2_LR}" \
  ar.warmup_steps="${STAGE2_WARMUP_STEPS}" \
  ar.min_lr_ratio="${STAGE2_MIN_LR_RATIO}" \
  ar.coeff_loss_type=huber \
  ar.coeff_loss_weight=1.0 \
  ar.coeff_huber_delta=0.5 \
  ar.sample_coeff_mode="${STAGE2_SAMPLE_COEFF_MODE}" \
  train_ar.accelerator=gpu \
  train_ar.devices=2 \
  train_ar.strategy=ddp \
  train_ar.precision=bf16-mixed \
  train_ar.max_epochs="${STAGE2_EPOCHS}" \
  train_ar.batch_size="${STAGE2_BATCH_SIZE}" \
  train_ar.gradient_clip_val=1.0 \
  train_ar.log_every_n_steps=20 \
  train_ar.val_check_interval=1.0 \
  train_ar.sample_every_n_epochs="${STAGE2_SAMPLE_EVERY_N_EPOCHS}" \
  train_ar.sample_num_images="${STAGE2_SAMPLE_NUM_IMAGES}" \
  train_ar.sample_temperature="${STAGE2_SAMPLE_TEMPERATURE}" \
  train_ar.sample_top_k="${STAGE2_SAMPLE_TOP_K}" \
  train_ar.sample_coeff_mode="${STAGE2_SAMPLE_COEFF_MODE}" \
  train_ar.sample_log_to_wandb=true \
  train_ar.compute_generation_fid=false \
  train_ar.generation_metric_num_samples=0 \
  train_ar.run_test_after_fit=false \
  train_ar.save_final_samples_after_fit=false \
  data.num_workers="${STAGE2_NUM_WORKERS}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="celebahq_s2_sharp_d4_lpips_${STAMP}" \
  wandb.save_dir="${STAGE2_DIR}/wandb"

echo "[$(date --iso-8601=seconds)] pipeline done: ${RUN_ROOT}"
