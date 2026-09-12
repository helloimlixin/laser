#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${REPO_ROOT}/outputs/imagenet-rqvae-a512k2-gan-rqvae-lpips-20260714_002101}"
FIDELITY_CKPT="${SOURCE_RUN_ROOT}/imagenet_stage1/checkpoints/recovery_sources/laser-epoch=007-fidelity.ckpt"
BEST_RFID_CKPT="${SOURCE_RUN_ROOT}/imagenet_stage1/checkpoints/run_20260714_002106/laser/laser-epoch=009.ckpt"
if [[ -z "${SOURCE_CKPT:-}" ]]; then
  if [[ -f "${FIDELITY_CKPT}" ]]; then
    SOURCE_CKPT="${FIDELITY_CKPT}"
  else
    SOURCE_CKPT="${BEST_RFID_CKPT}"
  fi
fi
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/imagenet-a512k2-reconstruction-recovery-${STAMP}}"
RUN_NAME="${RUN_NAME:-imagenet-a512k2-reconstruction-recovery-${STAMP}}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "xfxxy13y fidelity checkpoint not found: ${SOURCE_CKPT}" >&2
  exit 1
fi

# Initialize instead of resuming: the recovery deliberately removes the
# discriminator and starts a fresh optimizer under a fidelity-first objective.
INIT_CKPT="${SOURCE_CKPT}" \
RUN_ROOT="${RUN_ROOT}" \
RUN_NAME="${RUN_NAME}" \
MAX_EPOCHS="${MAX_EPOCHS}" \
scripts/run_imagenet_stage1_a512k2.sh \
  "train.learning_rate=1.0e-5" \
  "train.warmup_steps=0" \
  "model.dict_learning_rate=1.0e-5" \
  "model.data_init_from_first_batch=false" \
  "model.recon_mse_weight=0.25" \
  "model.recon_l1_weight=1.0" \
  "model.recon_edge_weight=0.5" \
  "model.perceptual_weight=0.2" \
  "model.perceptual_start_step=0" \
  "model.perceptual_warmup_steps=0" \
  "model.adversarial_weight=0.0" \
  "model.adversarial_start_step=1000000000" \
  "model.adversarial_warmup_steps=0" \
  "model.disc_start_step=1000000000" \
  "model.use_adaptive_disc_weight=false" \
  "checkpoint.monitor=val/recon_mse_loss" \
  "checkpoint.mode=min" \
  "wandb.group=imagenet-a512k2-reconstruction-recovery" \
  "wandb.tags=[stage1,imagenet,laser,tokenizer,a512,k2,reconstruction_recovery,no_gan,from_xfxxy13y]" \
  "$@"
