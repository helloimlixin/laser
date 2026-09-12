#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
IMAGENET_DIR="${IMAGENET_DIR:-/workspace/Projects/data/imagenet}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/imagenet-a512k2/stage1_adv}"
RUN_NAME="${RUN_NAME:-imagenet-stage1-a512k2}"

NUM_GPUS="${NUM_GPUS:-1}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-${NUM_GPUS}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
NUM_WORKERS="${NUM_WORKERS:-24}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
MAX_STEPS="${MAX_STEPS:-"-1"}"
PRECISION="${PRECISION:-bf16-mixed}"
DRY_RUN="${DRY_RUN:-0}"
VIS_LOG_EVERY_N_STEPS="${VIS_LOG_EVERY_N_STEPS:-5000}"
ENABLE_VAL_LATENT_VISUALS="${ENABLE_VAL_LATENT_VISUALS:-true}"
CODEBOOK_VISUAL_MAX_VECTORS="${CODEBOOK_VISUAL_MAX_VECTORS:-512}"

ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE:-512}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"

TMP_ROOT="${TMP_ROOT:-/workspace/tmp/laser}"
TMPDIR="${TMPDIR:-${TMP_ROOT}/${RUN_NAME}}"
TEMP="${TEMP:-${TMPDIR}}"
TMP="${TMP:-${TMPDIR}}"
export TMPDIR TEMP TMP

WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${RUN_ROOT}/wandb/cache}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${RUN_ROOT}/wandb/artifacts}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/.cache}"
export WANDB_CACHE_DIR WANDB_ARTIFACT_DIR XDG_CACHE_HOME

mkdir -p "${RUN_ROOT}" "${TMPDIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}" "${XDG_CACHE_HOME}"
cd "${REPO_ROOT}"

if [[ ! -d "${IMAGENET_DIR}" ]]; then
  echo "ImageNet directory not found: ${IMAGENET_DIR}" >&2
  echo "Set IMAGENET_DIR=/path/to/imagenet." >&2
  exit 1
fi

EXTRA_INIT=()
if [[ -n "${INIT_CKPT:-}" ]]; then
  EXTRA_INIT=("init_ckpt_path=${INIT_CKPT}")
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  DRY_RUN_ARGS=("--dry-run")
fi

cat <<EOF
Output: ${RUN_ROOT}
Temp: ${TMPDIR}
ImageNet: ${IMAGENET_DIR}
Recipe: atoms=${ATOM_VOCAB_SIZE} k=${SPARSITY_LEVEL} embedding_dim=${EMBEDDING_DIM}
Schedule: epochs=${MAX_EPOCHS} max_steps=${MAX_STEPS} batch=${BATCH_SIZE} accumulate=${ACCUMULATE_GRAD_BATCHES}
Visuals: recon_every_steps=${VIS_LOG_EVERY_N_STEPS} val_latents=${ENABLE_VAL_LATENT_VISUALS} codebook_vectors=${CODEBOOK_VISUAL_MAX_VECTORS}
EOF

export WANDB_MODE="${WANDB_MODE:-online}"
export LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-0}"
export HYDRA_FULL_ERROR=1

python train.py stage1 \
  --dataset imagenet \
  --modality image \
  --conditioning none \
  --adversarial true \
  --num-gpus "${NUM_GPUS}" \
  --devices-per-node "${DEVICES_PER_NODE}" \
  --downsample-layers 5 \
  --sparsity-level "${SPARSITY_LEVEL}" \
  --num-embeddings "${ATOM_VOCAB_SIZE}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --image-size 256 \
  --data-dir "${IMAGENET_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --precision "${PRECISION}" \
  --learning-rate 4.0e-5 \
  --dict-learning-rate 4.0e-5 \
  --epochs "${MAX_EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --output-dir "${RUN_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project laser \
  "${DRY_RUN_ARGS[@]}" \
  "${EXTRA_INIT[@]}" \
  "train.beta=0.5" \
  "train.beta2=0.9" \
  "train.warmup_steps=5005" \
  "train.min_lr_ratio=1.0" \
  "train.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}" \
  "train.limit_train_batches=1.0" \
  "train.limit_val_batches=512" \
  "train.limit_test_batches=0" \
  "train.val_check_interval=5000" \
  "train.compute_rfid_after_fit=true" \
  "train.rfid_split=val" \
  "train.rfid_batch_size=64" \
  "train.rfid_num_workers=8" \
  "train.rfid_max_samples=0" \
  "train.rfid_device=auto" \
  "train.rfid_feature=2048" \
  "data.batch_size=${BATCH_SIZE}" \
  "data.num_workers=${NUM_WORKERS}" \
  "data.train_crop_size=null" \
  "model.backbone=ddpm" \
  "model.num_downsamples=5" \
  "model.channel_multipliers=[1,1,2,2,4,4]" \
  "model.backbone_latent_channels=256" \
  "model.attn_resolutions=[8]" \
  "model.decoder_extra_residual_layers=0" \
  "model.use_mid_attention=true" \
  "model.dropout=0.0" \
  "model.num_hiddens=128" \
  "model.num_residual_blocks=2" \
  "model.num_residual_hiddens=96" \
  "model.num_embeddings=${ATOM_VOCAB_SIZE}" \
  "model.embedding_dim=${EMBEDDING_DIM}" \
  "model.sparsity_level=${SPARSITY_LEVEL}" \
  "model.patch_based=false" \
  "model.patch_size=1" \
  "model.patch_stride=1" \
  "model.dict_learning_rate=4.0e-5" \
  "model.data_init_from_first_batch=true" \
  "model.recon_mse_weight=1.0" \
  "model.recon_l1_weight=0.1" \
  "model.recon_edge_weight=0.02" \
  "model.bottleneck_loss_weight=0.75" \
  "model.dictionary_loss_weight=0.75" \
  "model.sparsity_reg_weight=0.0" \
  "model.perceptual_weight=1.0" \
  "model.perceptual_start_step=0" \
  "model.perceptual_warmup_steps=0" \
  "model.adversarial_weight=0.75" \
  "model.adversarial_start_step=0" \
  "model.adversarial_warmup_steps=0" \
  "model.disc_start_step=0" \
  "model.disc_learning_rate=4.0e-5" \
  "model.discriminator_beta1=0.5" \
  "model.discriminator_beta2=0.9" \
  "model.disc_channels=64" \
  "model.disc_num_layers=3" \
  "model.disc_norm=group" \
  "model.disc_loss=hinge" \
  "model.use_adaptive_disc_weight=true" \
  "model.compute_fid=true" \
  "model.log_images_every_n_steps=${VIS_LOG_EVERY_N_STEPS}" \
  "model.enable_val_latent_visuals=${ENABLE_VAL_LATENT_VISUALS}" \
  "model.codebook_visual_max_vectors=${CODEBOOK_VISUAL_MAX_VECTORS}" \
  "checkpoint.monitor=val/rfid" \
  "checkpoint.mode=min" \
  "checkpoint.upload_to_wandb=true" \
  "checkpoint.upload_every_n_epochs=1" \
  "wandb.group=imagenet-a512k2-stage1" \
  "wandb.tags=[stage1,imagenet,laser,tokenizer,a512,k2,adversarial]" \
  "wandb.save_dir=${RUN_ROOT}/wandb" \
  "$@"
