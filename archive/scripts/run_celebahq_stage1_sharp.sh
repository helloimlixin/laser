#!/usr/bin/env bash
# CelebA-HQ stage-1 sharpness probe.
#
# This is a targeted follow-up to W&B run helloimlixin-rutgers/laser/vc17pae5.
# The prior run used an 8x8 latent grid for 256px images, no LPIPS, and no
# attention, which produced smooth reconstructions despite good PSNR.
set -euo pipefail

cd /home/xl598/Projects/laser

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/xl598/Projects/laser/runs/celebahq_stage1_sharp_d4_lpips_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba_hq}"

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/home/xl598/Projects/laser
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_ROOT}"

"${PYTHON_BIN}" train.py stage1 \
  output_dir="${RUN_ROOT}" \
  hydra.run.dir="${RUN_ROOT}/hydra" \
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
  wandb.project=laser \
  wandb.name="celebahq_s1_sharp_d4_lpips_${STAMP}"
