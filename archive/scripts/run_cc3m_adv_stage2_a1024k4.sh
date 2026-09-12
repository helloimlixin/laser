#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
IMAGENET_DIR="${IMAGENET_DIR:-/workspace/Projects/data/imagenet}"
CC3M_DIR="${CC3M_DIR:-/workspace/Projects/data/cc3m}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

# RQ-VAE/RQ-Transformer-style recipe used by the referenced ImageNet run
# helloimlixin-rutgers/laser/6mcsatyq.
ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE:-1024}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
COEFF_BINS="${COEFF_BINS:-256}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-32}"
TEXT_TOKENIZER="${TEXT_TOKENIZER:-rq_bpe16k}"

RECIPE_NAME="${RECIPE_NAME:-cc3m-adv-stage2-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}}"
PIPE_ROOT="${PIPE_ROOT:-${REPO_ROOT}/outputs/${RECIPE_NAME}-${STAMP}}"
STAGE1_ADV_ROOT="${STAGE1_ADV_ROOT:-${PIPE_ROOT}/imagenet_stage1_adv}"
S2_ROOT="${S2_ROOT:-${PIPE_ROOT}/cc3m_stage2}"
LOG_DIR="${LOG_DIR:-${PIPE_ROOT}/logs}"
STAGE1_WANDB_ROOT="${STAGE1_WANDB_ROOT:-${STAGE1_ADV_ROOT}/wandb}"
STAGE1_WANDB_CACHE_DIR="${STAGE1_WANDB_CACHE_DIR:-${STAGE1_WANDB_ROOT}/cache}"
STAGE1_WANDB_DATA_DIR="${STAGE1_WANDB_DATA_DIR:-${STAGE1_WANDB_ROOT}/data}"
STAGE1_WANDB_ARTIFACT_DIR="${STAGE1_WANDB_ARTIFACT_DIR:-${STAGE1_WANDB_ROOT}/artifacts}"

# Defaults to the local continuation of W&B run 6mcsatyq. Override
# STAGE1_RECON_CKPT to initialize adversarial training from another checkpoint,
# or STAGE1_ADV_CKPT to skip adversarial training and run only CC3M stage 2.
STAGE1_RECON_ROOT="${STAGE1_RECON_ROOT:-${REPO_ROOT}/outputs/imagenet-fixed-a1024k4-20260710_190712/imagenet_stage1}"
STAGE1_RECON_CKPT="${STAGE1_RECON_CKPT:-}"
STAGE1_ADV_CKPT="${STAGE1_ADV_CKPT:-}"
STAGE1_ADV_RESUME_CKPT="${STAGE1_ADV_RESUME_CKPT:-}"
SKIP_STAGE1_ADV="${SKIP_STAGE1_ADV:-0}"
CHECK_STAGE1_RECIPE="${CHECK_STAGE1_RECIPE:-1}"
DRY_RUN="${DRY_RUN:-0}"

ADV_MAX_EPOCHS="${ADV_MAX_EPOCHS:-3}"
ADV_MAX_STEPS="${ADV_MAX_STEPS:--1}"
ADV_BATCH_SIZE="${ADV_BATCH_SIZE:-32}"
ADV_ACCUMULATE_GRAD_BATCHES="${ADV_ACCUMULATE_GRAD_BATCHES:-1}"
ADV_NUM_WORKERS="${ADV_NUM_WORKERS:-24}"
ADV_LR="${ADV_LR:-4.0e-5}"
ADV_DICT_LR="${ADV_DICT_LR:-4.0e-5}"
ADV_WARMUP_STEPS="${ADV_WARMUP_STEPS:-5005}"
ADV_LIMIT_VAL_BATCHES="${ADV_LIMIT_VAL_BATCHES:-512}"
ADV_VAL_CHECK_INTERVAL="${ADV_VAL_CHECK_INTERVAL:-5000}"
ADV_RFID_BATCH_SIZE="${ADV_RFID_BATCH_SIZE:-64}"
ADV_RFID_NUM_WORKERS="${ADV_RFID_NUM_WORKERS:-8}"
ADV_ADVERSARIAL_WEIGHT="${ADV_ADVERSARIAL_WEIGHT:-0.75}"
ADV_DISC_LR="${ADV_DISC_LR:-4.0e-5}"
ADV_LOG_IMAGES_EVERY_N_STEPS="${ADV_LOG_IMAGES_EVERY_N_STEPS:-1000}"
ADV_ENABLE_VAL_LATENT_VISUALS="${ADV_ENABLE_VAL_LATENT_VISUALS:-true}"
ADV_CODEBOOK_VISUAL_MAX_VECTORS="${ADV_CODEBOOK_VISUAL_MAX_VECTORS:-512}"
ADV_WANDB_NAME="${ADV_WANDB_NAME:-imagenet-stage1-adv-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}-${STAMP}}"
ADV_WANDB_GROUP="${ADV_WANDB_GROUP:-imagenet-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}-stage1-adv}"

# CC-3M prior defaults follow Section 4.2 and the paper supplement: 32 BPE
# text tokens prepended as spatial-transformer context, 0.1/0.9 text/image
# loss weights, AdamW lr=5e-4/weight_decay=1e-4, and effective batch 2048.
# The model width/layer count mirror the 654M CC-3M RQ-Transformer as closely
# as this LASER sparse-token prior allows.
S2_MAX_EPOCHS="${S2_MAX_EPOCHS:-100}"
S2_BATCH_SIZE="${S2_BATCH_SIZE:-32}"
S2_ACCUMULATE_GRAD_BATCHES="${S2_ACCUMULATE_GRAD_BATCHES:-64}"
S2_D_MODEL="${S2_D_MODEL:-1280}"
S2_LAYERS="${S2_LAYERS:-24}"
S2_HEADS="${S2_HEADS:-20}"
S2_FF="${S2_FF:-5120}"
S2_LR="${S2_LR:-5e-4}"
S2_WEIGHT_DECAY="${S2_WEIGHT_DECAY:-1e-4}"
S2_SAMPLE_TOP_K="${S2_SAMPLE_TOP_K:-${ATOM_VOCAB_SIZE}}"
S2_SAMPLE_TEMPERATURE="${S2_SAMPLE_TEMPERATURE:-0.9}"

TMP_ROOT="${TMP_ROOT:-/workspace/tmp/laser}"
TMPDIR="${TMPDIR:-${TMP_ROOT}/${RECIPE_NAME}-${STAMP}}"
TEMP="${TEMP:-${TMPDIR}}"
TMP="${TMP:-${TMPDIR}}"
export TMPDIR TEMP TMP

mkdir -p \
  "${PIPE_ROOT}" \
  "${STAGE1_ADV_ROOT}" \
  "${S2_ROOT}" \
  "${LOG_DIR}" \
  "${TMPDIR}" \
  "${STAGE1_WANDB_CACHE_DIR}" \
  "${STAGE1_WANDB_DATA_DIR}" \
  "${STAGE1_WANDB_ARTIFACT_DIR}"
cd "${REPO_ROOT}"

find_latest_ckpt() {
  local root="$1"
  local ckpt=""
  if [[ -d "${root}" ]]; then
    ckpt="$(find "${root}" -path '*/laser/final.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
    if [[ -z "${ckpt}" ]]; then
      ckpt="$(find "${root}" -path '*/laser/last.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
    fi
    if [[ -z "${ckpt}" ]]; then
      ckpt="$(find "${root}" -path '*/laser/*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
    fi
  fi
  printf '%s\n' "${ckpt}"
}

resolve_recon_ckpt() {
  if [[ -n "${STAGE1_RECON_CKPT}" ]]; then
    printf '%s\n' "${STAGE1_RECON_CKPT}"
    return
  fi

  local ckpt=""
  ckpt="$(find_latest_ckpt "${STAGE1_RECON_ROOT}")"
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(find_latest_ckpt "${REPO_ROOT}/outputs/imagenet-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}/stage1")"
  fi
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(find_latest_ckpt "${REPO_ROOT}/outputs/imagenet-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}/stage1_recon")"
  fi
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(
      find "${REPO_ROOT}/outputs" -path "*/imagenet*stage1*/checkpoints/*/laser/final.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | cut -d' ' -f2- \
        | grep "a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}" \
        | tail -1 \
        || true
    )"
  fi

  if [[ -z "${ckpt}" ]]; then
    cat >&2 <<EOF
No ImageNet reconstruction checkpoint found for a${ATOM_VOCAB_SIZE}/k${SPARSITY_LEVEL}.

Set STAGE1_RECON_CKPT=/path/to/recon.ckpt, or point STAGE1_RECON_ROOT at the
6mcsatyq output directory. Default root:
  ${STAGE1_RECON_ROOT}
EOF
    exit 1
  fi
  printf '%s\n' "${ckpt}"
}

check_recipe() {
  local ckpt="$1"
  if [[ "${CHECK_STAGE1_RECIPE}" == "0" || "${CHECK_STAGE1_RECIPE}" == "false" ]]; then
    return
  fi
  python - "${ckpt}" "${ATOM_VOCAB_SIZE}" "${SPARSITY_LEVEL}" "${EMBEDDING_DIM}" <<'PY'
import sys
from pathlib import Path

import torch

ckpt_path = Path(sys.argv[1])
expected = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(ckpt_path, map_location="cpu")

hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
actual = (
    hparams.get("num_embeddings"),
    hparams.get("sparsity_level"),
    hparams.get("embedding_dim"),
)
if any(value is None for value in actual):
    missing = [
        name
        for name, value in zip(("num_embeddings", "sparsity_level", "embedding_dim"), actual)
        if value is None
    ]
    print(
        "Warning: checkpoint metadata missing "
        + ", ".join(missing)
        + "; continuing without full recipe verification.",
        file=sys.stderr,
    )
    raise SystemExit(0)

actual_int = tuple(int(value) for value in actual)
if actual_int != expected:
    raise SystemExit(
        "Checkpoint recipe mismatch: "
        f"expected num_embeddings/sparsity_level/embedding_dim={expected}, got {actual_int}. "
        "Override ATOM_VOCAB_SIZE/SPARSITY_LEVEL/EMBEDDING_DIM intentionally or pass the right checkpoint."
    )
PY
}

dry_run_enabled() {
  [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_stage1_adv() {
  if [[ ! -d "${IMAGENET_DIR}" ]]; then
    echo "ImageNet directory not found: ${IMAGENET_DIR}" >&2
    echo "Set IMAGENET_DIR=/path/to/imagenet, or pass STAGE1_ADV_CKPT to skip adversarial continuation." >&2
    exit 1
  fi

  local source_args=()
  if [[ -n "${STAGE1_ADV_RESUME_CKPT}" ]]; then
    if [[ ! -f "${STAGE1_ADV_RESUME_CKPT}" ]]; then
      echo "Stage-1 adversarial resume checkpoint not found: ${STAGE1_ADV_RESUME_CKPT}" >&2
      exit 1
    fi
    source_args=("ckpt_path=${STAGE1_ADV_RESUME_CKPT}")
  else
    source_args=("init_ckpt_path=${STAGE1_RECON_CKPT}")
  fi

  local dry_args=()
  if dry_run_enabled; then
    dry_args=("--dry-run")
  fi

  local cmd=(
    python train.py stage1
    --dataset imagenet
    --modality image
    --conditioning none
    --adversarial true
    --num-gpus 1
    --devices-per-node 1
    --downsample-layers 5
    --sparsity-level "${SPARSITY_LEVEL}"
    --num-embeddings "${ATOM_VOCAB_SIZE}"
    --embedding-dim "${EMBEDDING_DIM}"
    --image-size 256
    --data-dir "${IMAGENET_DIR}"
    --batch-size "${ADV_BATCH_SIZE}"
    --num-workers "${ADV_NUM_WORKERS}"
    --precision bf16-mixed
    --learning-rate "${ADV_LR}"
    --dict-learning-rate "${ADV_DICT_LR}"
    --epochs "${ADV_MAX_EPOCHS}"
    --max-steps "${ADV_MAX_STEPS}"
    --output-dir "${STAGE1_ADV_ROOT}"
    --run-name "${ADV_WANDB_NAME}"
    --project laser
    "${dry_args[@]}"
    "${source_args[@]}"
    "hydra.run.dir=${STAGE1_ADV_ROOT}/hydra"
    "train.beta=0.5"
    "train.beta2=0.9"
    "train.warmup_steps=${ADV_WARMUP_STEPS}"
    "train.min_lr_ratio=1.0"
    "train.accumulate_grad_batches=${ADV_ACCUMULATE_GRAD_BATCHES}"
    "train.limit_train_batches=1.0"
    "train.limit_val_batches=${ADV_LIMIT_VAL_BATCHES}"
    "train.limit_test_batches=0"
    "train.val_check_interval=${ADV_VAL_CHECK_INTERVAL}"
    "train.run_test_after_fit=false"
    "train.compute_rfid_after_fit=true"
    "train.rfid_split=val"
    "train.rfid_batch_size=${ADV_RFID_BATCH_SIZE}"
    "train.rfid_num_workers=${ADV_RFID_NUM_WORKERS}"
    "train.rfid_max_samples=0"
    "train.rfid_device=auto"
    "train.rfid_feature=2048"
    "data.batch_size=${ADV_BATCH_SIZE}"
    "data.eval_batch_size=${ADV_BATCH_SIZE}"
    "data.num_workers=${ADV_NUM_WORKERS}"
    "data.train_crop_size=null"
    "model.backbone=ddpm"
    "model.num_downsamples=5"
    "model.channel_multipliers=[1,1,2,2,4,4]"
    "model.backbone_latent_channels=256"
    "model.attn_resolutions=[8]"
    "model.decoder_extra_residual_layers=0"
    "model.use_mid_attention=true"
    "model.dropout=0.0"
    "model.num_hiddens=128"
    "model.num_residual_blocks=2"
    "model.num_residual_hiddens=96"
    "model.num_embeddings=${ATOM_VOCAB_SIZE}"
    "model.embedding_dim=${EMBEDDING_DIM}"
    "model.sparsity_level=${SPARSITY_LEVEL}"
    "model.patch_based=false"
    "model.patch_size=1"
    "model.patch_stride=1"
    "model.dict_learning_rate=${ADV_DICT_LR}"
    "model.data_init_from_first_batch=true"
    "model.recon_mse_weight=1.0"
    "model.recon_l1_weight=0.1"
    "model.recon_edge_weight=0.02"
    "model.bottleneck_loss_weight=0.75"
    "model.dictionary_loss_weight=0.75"
    "model.sparsity_reg_weight=0.0"
    "model.perceptual_weight=1.0"
    "model.perceptual_start_step=0"
    "model.perceptual_warmup_steps=0"
    "model.adversarial_weight=${ADV_ADVERSARIAL_WEIGHT}"
    "model.adversarial_start_step=0"
    "model.adversarial_warmup_steps=0"
    "model.disc_start_step=0"
    "model.disc_learning_rate=${ADV_DISC_LR}"
    "model.discriminator_beta1=0.5"
    "model.discriminator_beta2=0.9"
    "model.disc_channels=64"
    "model.disc_num_layers=3"
    "model.disc_norm=group"
    "model.disc_loss=hinge"
    "model.use_adaptive_disc_weight=true"
    "model.compute_fid=true"
    "model.log_images_every_n_steps=${ADV_LOG_IMAGES_EVERY_N_STEPS}"
    "model.enable_val_latent_visuals=${ADV_ENABLE_VAL_LATENT_VISUALS}"
    "model.codebook_visual_max_vectors=${ADV_CODEBOOK_VISUAL_MAX_VECTORS}"
    "checkpoint.monitor=val/rfid"
    "checkpoint.mode=min"
    "checkpoint.save_top_k=3"
    "checkpoint.save_last=true"
    "checkpoint.upload_to_wandb=true"
    "checkpoint.upload_every_n_epochs=1"
    "wandb.name=${ADV_WANDB_NAME}"
    "wandb.group=${ADV_WANDB_GROUP}"
    "wandb.tags=[stage1_adv,imagenet,laser,tokenizer,a${ATOM_VOCAB_SIZE},k${SPARSITY_LEVEL},adversarial,rqvae_style,cc3m_text_to_image]"
    "wandb.save_dir=${STAGE1_ADV_ROOT}/wandb"
  )

  if dry_run_enabled; then
    echo "Stage-1 adversarial command:"
    echo "Stage-1 W&B workspace dirs:"
    echo "  WANDB_DIR=${STAGE1_WANDB_ROOT}"
    echo "  WANDB_CACHE_DIR=${STAGE1_WANDB_CACHE_DIR}"
    echo "  WANDB_DATA_DIR=${STAGE1_WANDB_DATA_DIR}"
    echo "  WANDB_ARTIFACT_DIR=${STAGE1_WANDB_ARTIFACT_DIR}"
    print_command "${cmd[@]}"
    return
  fi

  env \
    "WANDB_DIR=${STAGE1_WANDB_ROOT}" \
    "WANDB_CACHE_DIR=${STAGE1_WANDB_CACHE_DIR}" \
    "WANDB_DATA_DIR=${STAGE1_WANDB_DATA_DIR}" \
    "WANDB_ARTIFACT_DIR=${STAGE1_WANDB_ARTIFACT_DIR}" \
    "${cmd[@]}" 2>&1 | tee "${LOG_DIR}/imagenet_stage1_adv.log"
}

run_stage2() {
  local env_cmd=(
    env
    "REPO_ROOT=${REPO_ROOT}"
    "DATA_DIR=${CC3M_DIR}"
    "STAMP=${STAMP}"
    "PIPE_ROOT=${PIPE_ROOT}"
    "S2_ROOT=${S2_ROOT}"
    "STAGE1_CKPT=${STAGE1_ADV_CKPT}"
    "ATOM_VOCAB_SIZE=${ATOM_VOCAB_SIZE}"
    "SPARSITY_LEVEL=${SPARSITY_LEVEL}"
    "EMBEDDING_DIM=${EMBEDDING_DIM}"
    "COEFF_BINS=${COEFF_BINS}"
    "TEXT_MAX_LENGTH=${TEXT_MAX_LENGTH}"
    "TEXT_TOKENIZER=${TEXT_TOKENIZER}"
    "S2_MAX_EPOCHS=${S2_MAX_EPOCHS}"
    "S2_BATCH_SIZE=${S2_BATCH_SIZE}"
    "S2_ACCUMULATE_GRAD_BATCHES=${S2_ACCUMULATE_GRAD_BATCHES}"
    "S2_D_MODEL=${S2_D_MODEL}"
    "S2_LAYERS=${S2_LAYERS}"
    "S2_HEADS=${S2_HEADS}"
    "S2_FF=${S2_FF}"
    "S2_LR=${S2_LR}"
    "S2_WEIGHT_DECAY=${S2_WEIGHT_DECAY}"
    "S2_SAMPLE_TOP_K=${S2_SAMPLE_TOP_K}"
    "S2_SAMPLE_TEMPERATURE=${S2_SAMPLE_TEMPERATURE}"
    "RECIPE_NAME=${RECIPE_NAME}"
    "CHECK_STAGE1_RECIPE=${CHECK_STAGE1_RECIPE}"
    scripts/run_cc3m_stage2_imagenet_a512k2.sh
    "$@"
  )

  if dry_run_enabled; then
    echo "CC3M stage-2 command:"
    print_command "${env_cmd[@]}"
    return
  fi

  "${env_cmd[@]}" 2>&1 | tee "${LOG_DIR}/cc3m_stage2.log"
}

cat <<EOF | tee "${PIPE_ROOT}/run.info"
Recipe: ${RECIPE_NAME}
Pipeline root: ${PIPE_ROOT}
ImageNet reconstruction root: ${STAGE1_RECON_ROOT}
ImageNet adversarial root: ${STAGE1_ADV_ROOT}
CC3M stage-2 root: ${S2_ROOT}
Stage-1 W&B root: ${STAGE1_WANDB_ROOT}
Stage-1 W&B artifact dir: ${STAGE1_WANDB_ARTIFACT_DIR}
Expected tokenizer recipe: atoms=${ATOM_VOCAB_SIZE} k=${SPARSITY_LEVEL} embedding_dim=${EMBEDDING_DIM}
CC3M text recipe: tokenizer=${TEXT_TOKENIZER} length=${TEXT_MAX_LENGTH}
CC3M prior recipe: d_model=${S2_D_MODEL} layers=${S2_LAYERS} heads=${S2_HEADS} batch=${S2_BATCH_SIZE} accumulate=${S2_ACCUMULATE_GRAD_BATCHES}
EOF

if [[ -z "${STAGE1_ADV_CKPT}" ]]; then
  STAGE1_RECON_CKPT="$(resolve_recon_ckpt)"
  if [[ ! -f "${STAGE1_RECON_CKPT}" ]]; then
    echo "Stage-1 reconstruction checkpoint not found: ${STAGE1_RECON_CKPT}" >&2
    exit 1
  fi
  check_recipe "${STAGE1_RECON_CKPT}"
  echo "Stage-1 reconstruction checkpoint: ${STAGE1_RECON_CKPT}" | tee "${PIPE_ROOT}/stage1_recon_checkpoint.txt"

  if [[ "${SKIP_STAGE1_ADV}" == "1" || "${SKIP_STAGE1_ADV}" == "true" ]]; then
    STAGE1_ADV_CKPT="$(find_latest_ckpt "${STAGE1_ADV_ROOT}")"
    if [[ -z "${STAGE1_ADV_CKPT}" ]]; then
      echo "SKIP_STAGE1_ADV was set, but no adversarial checkpoint was found under ${STAGE1_ADV_ROOT}" >&2
      exit 1
    fi
  else
    echo "Starting ImageNet adversarial continuation from: ${STAGE1_RECON_CKPT}"
    run_stage1_adv
    STAGE1_ADV_CKPT="$(find_latest_ckpt "${STAGE1_ADV_ROOT}")"
  fi
fi

if [[ -z "${STAGE1_ADV_CKPT}" || ! -f "${STAGE1_ADV_CKPT}" ]]; then
  if dry_run_enabled; then
    STAGE1_ADV_CKPT="${STAGE1_ADV_ROOT}/checkpoints/<run>/laser/final.ckpt"
  else
    echo "Could not resolve stage-1 adversarial checkpoint under ${STAGE1_ADV_ROOT}" >&2
    exit 1
  fi
fi

if ! dry_run_enabled; then
  check_recipe "${STAGE1_ADV_CKPT}"
fi
echo "Stage-1 adversarial checkpoint: ${STAGE1_ADV_CKPT}" | tee "${PIPE_ROOT}/stage1_adv_checkpoint.txt"

echo "Starting CC3M text-to-image stage 2."
run_stage2 "$@"
