#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
IMAGENET_DIR="${IMAGENET_DIR:-/workspace/Projects/data/imagenet}"
ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE:-256}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-2}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/imagenet-rqvae-no-perceptual-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}-${STAMP}}"
RUN_NAME="${RUN_NAME:-imagenet-rqvae-no-perceptual-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}-${STAMP}}"
NUM_GPUS="${NUM_GPUS:-1}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-${NUM_GPUS}}"
# The paper used four A100s for a global batch of 128. Accumulating four
# per-device batches of 32 reproduces that global batch on this single H100.
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
NUM_WORKERS="${NUM_WORKERS:-24}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON="$(command -v python)"
  fi
fi
DRY_RUN="${DRY_RUN:-0}"

TRAIN_DIR="${IMAGENET_DIR}/train"
VAL_DIR="${IMAGENET_DIR}/val"

if [[ ! -d "${TRAIN_DIR}" || ! -d "${VAL_DIR}" ]]; then
  echo "Expected ImageNet-1k directories at ${TRAIN_DIR} and ${VAL_DIR}." >&2
  exit 1
fi

NUM_CLASSES="$(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "${NUM_CLASSES}" -ne 1000 ]]; then
  echo "Expected 1000 ImageNet training classes, found ${NUM_CLASSES}." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found at ${PYTHON}. Set PYTHON=/path/to/python." >&2
  exit 1
fi

GLOBAL_BATCH_SIZE=$((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES * NUM_GPUS))
if [[ "${ALLOW_NON128_BATCH:-0}" != "1" && "${GLOBAL_BATCH_SIZE}" -ne 128 ]]; then
  echo "Expected an effective global batch of 128, got ${GLOBAL_BATCH_SIZE}." >&2
  exit 1
fi

TMPDIR="${TMPDIR:-/workspace/tmp/laser/${RUN_NAME}}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${REPO_ROOT}/.cache_root/wandb}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${REPO_ROOT}/.cache_root/wandb-artifacts}"
export TMPDIR TEMP="${TMPDIR}" TMP="${TMPDIR}" WANDB_CACHE_DIR WANDB_ARTIFACT_DIR
export WANDB_MODE="${WANDB_MODE:-online}"
export LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-0}"
LOG_IMAGES_EVERY_N_STEPS="${LOG_IMAGES_EVERY_N_STEPS:-500}"
mkdir -p "${TMPDIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}"

cleanup_nonselected_checkpoints() {
  # train.py writes final.ckpt after Lightning's selected-checkpoint callback.
  # It is redundant with last.ckpt and is removed to preserve the four-file cap.
  if [[ -d "${RUN_ROOT}/checkpoints" ]]; then
    find "${RUN_ROOT}/checkpoints" -type f -name final.ckpt -delete
  fi
  if [[ -d "${RUN_ROOT}/wandb_checkpoints" ]]; then
    find "${RUN_ROOT}/wandb_checkpoints" -type f -name '*.tmp' -delete
  fi
}
trap cleanup_nonselected_checkpoints EXIT

DRY_RUN_ARGS=()
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

# ImageNet-1k has 1,281,167 training examples. At batch size 128 this is
# ceil(1,281,167 / 128) = 10,010 optimizer steps per epoch, so 5,005 steps is
# exactly half an epoch of linear warmup. min_lr_ratio=1 keeps the LR constant
# after warmup (the scheduler's cosine term becomes a no-op).
"${PYTHON}" train.py stage1 \
  --dataset imagenet \
  --modality image \
  --conditioning none \
  --adversarial false \
  --num-gpus "${NUM_GPUS}" \
  --devices-per-node "${DEVICES_PER_NODE}" \
  --downsample-layers 5 \
  --sparsity-level "${SPARSITY_LEVEL}" \
  --num-embeddings "${ATOM_VOCAB_SIZE}" \
  --embedding-dim 256 \
  --image-size 256 \
  --data-dir "${IMAGENET_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --precision bf16-mixed \
  --learning-rate 4.0e-5 \
  --dict-learning-rate 4.0e-5 \
  --epochs 10 \
  --max-steps -1 \
  --output-dir "${RUN_ROOT}" \
  --run-name "${RUN_NAME}" \
  --project laser \
  "${DRY_RUN_ARGS[@]}" \
  train.beta=0.5 \
  train.beta2=0.9 \
  train.warmup_steps=5005 \
  train.min_lr_ratio=1.0 \
  train.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
  train.limit_train_batches=1.0 \
  train.limit_val_batches=512 \
  train.limit_test_batches=0 \
  train.val_check_interval=1.0 \
  train.run_test_after_fit=false \
  train.compute_rfid_after_fit=false \
  data.batch_size="${BATCH_SIZE}" \
  data.eval_batch_size=64 \
  data.num_workers="${NUM_WORKERS}" \
  data.pin_memory=true \
  data.prefetch_factor=4 \
  data.train_crop_size=null \
  model.backbone=ddpm \
  model.num_hiddens=128 \
  model.num_residual_blocks=2 \
  model.num_residual_hiddens=96 \
  model.num_downsamples=5 \
  model.channel_multipliers='[1,1,2,2,4,4]' \
  model.backbone_latent_channels=256 \
  model.attn_resolutions='[8]' \
  model.decoder_extra_residual_layers=0 \
  model.use_mid_attention=true \
  model.dropout=0.0 \
  model.num_embeddings="${ATOM_VOCAB_SIZE}" \
  model.embedding_dim=256 \
  model.sparsity_level="${SPARSITY_LEVEL}" \
  model.patch_based=false \
  model.patch_size=1 \
  model.patch_stride=1 \
  model.dict_learning_rate=4.0e-5 \
  model.perceptual_weight=0.0 \
  model.perceptual_start_step=0 \
  model.perceptual_warmup_steps=0 \
  model.adversarial_weight=0.0 \
  model.adversarial_start_step=1000000000 \
  model.disc_start_step=1000000000 \
  model.compute_fid=false \
  model.log_images_every_n_steps="${LOG_IMAGES_EVERY_N_STEPS}" \
  checkpoint.monitor=val/loss \
  checkpoint.mode=min \
  checkpoint.save_top_k=3 \
  checkpoint.save_last=true \
  checkpoint.every_n_epochs=1 \
  checkpoint.upload_to_wandb=true \
  checkpoint.upload_every_n_epochs=1 \
  +checkpoint.upload_mode=files \
  wandb.group=imagenet-laser-rqvae10 \
  wandb.tags="[stage1,imagenet,laser,a${ATOM_VOCAB_SIZE},k${SPARSITY_LEVEL},rqvae10,no-perceptual,no-adversarial,effective-batch128,fixed-checkpoint-slots]" \
  wandb.save_dir="${RUN_ROOT}/wandb" \
  "$@"
