#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
DATA_DIR="${DATA_DIR:-/workspace/Projects/data/cc3m}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

# Section 4.2-style CC-3M stage 2: reuse an ImageNet-trained tokenizer rather
# than finetuning stage 1 on CC-3M. The LASER tokenizer differs from RQ-VAE's
# K=16384 quantizer, but the stage-2 defaults follow the released 650M CC3M
# topology (1280 wide, 26 spatial + 4 depth layers, 20 heads, 32 BPE tokens).
ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE:-512}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
COEFF_BINS="${COEFF_BINS:-128}"
RECIPE_NAME="${RECIPE_NAME:-cc3m-s2-imagenet-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}}"
CHECK_STAGE1_RECIPE="${CHECK_STAGE1_RECIPE:-1}"

STAGE1_ROOT="${STAGE1_ROOT:-${REPO_ROOT}/outputs/imagenet-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}/stage1_adv}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
PIPE_ROOT="${PIPE_ROOT:-${REPO_ROOT}/outputs/${RECIPE_NAME}-${STAMP}}"
S2_ROOT="${S2_ROOT:-${PIPE_ROOT}/stage2}"

TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-32}"
TEXT_TOKENIZER="${TEXT_TOKENIZER:-rq_bpe16k}"
CACHE_MAX_ITEMS="${CACHE_MAX_ITEMS:-0}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-8}"

S2_MAX_STEPS="${S2_MAX_STEPS:-"-1"}"
S2_MAX_EPOCHS="${S2_MAX_EPOCHS:-100}"
S2_BATCH_SIZE="${S2_BATCH_SIZE:-8}"
S2_ACCUMULATE_GRAD_BATCHES="${S2_ACCUMULATE_GRAD_BATCHES:-256}"
S2_NUM_WORKERS="${S2_NUM_WORKERS:-8}"
S2_D_MODEL="${S2_D_MODEL:-1280}"
S2_LAYERS="${S2_LAYERS:-26}"
S2_DEPTH_LAYERS="${S2_DEPTH_LAYERS:-4}"
S2_HEADS="${S2_HEADS:-20}"
S2_FF="${S2_FF:-5120}"
S2_LR="${S2_LR:-5e-4}"
S2_WEIGHT_DECAY="${S2_WEIGHT_DECAY:-1e-4}"
S2_WARMUP_STEPS="${S2_WARMUP_STEPS:-1000}"
S2_VAL_BATCHES="${S2_VAL_BATCHES:-512}"
S2_GENERATION_FID_EVERY_N_EPOCHS="${S2_GENERATION_FID_EVERY_N_EPOCHS:-1}"
S2_GENERATION_FID_NUM_SAMPLES="${S2_GENERATION_FID_NUM_SAMPLES:-5000}"
S2_SAMPLE_EVERY_N_STEPS="${S2_SAMPLE_EVERY_N_STEPS:-5000}"
S2_SAMPLE_NUM_IMAGES="${S2_SAMPLE_NUM_IMAGES:-8}"
S2_SAMPLE_TEMPERATURE="${S2_SAMPLE_TEMPERATURE:-0.9}"
S2_SAMPLE_TOP_K="${S2_SAMPLE_TOP_K:-${ATOM_VOCAB_SIZE}}"

TMP_ROOT="${TMP_ROOT:-/workspace/tmp/laser}"
TMPDIR="${TMPDIR:-${TMP_ROOT}/${RECIPE_NAME}-${STAMP}-s2}"
TEMP="${TEMP:-${TMPDIR}}"
TMP="${TMP:-${TMPDIR}}"
export TMPDIR TEMP TMP

mkdir -p "${PIPE_ROOT}" "${S2_ROOT}/logs" "${TMPDIR}"
cd "${REPO_ROOT}"

find_latest_ckpt() {
  local root="$1"
  local ckpt=""
  if [[ -d "${root}" ]]; then
    ckpt="$(find "${root}" -path '*/laser/final.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
    if [[ -z "${ckpt}" ]]; then
      ckpt="$(find "${root}" -path '*/laser/*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
    fi
  fi
  printf '%s\n' "${ckpt}"
}

resolve_stage1_ckpt() {
  if [[ -n "${STAGE1_CKPT}" ]]; then
    printf '%s\n' "${STAGE1_CKPT}"
    return
  fi

  local ckpt=""
  ckpt="$(find_latest_ckpt "${STAGE1_ROOT}")"
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(find_latest_ckpt "${REPO_ROOT}/outputs/imagenet-nonpatch-d5-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}/stage1_adv")"
  fi
  if [[ -z "${ckpt}" ]]; then
    cat >&2 <<EOF
No ImageNet a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL} stage-1 checkpoint found.

Set STAGE1_CKPT=/path/to/final.ckpt, or run:
  scripts/run_imagenet_stage1_a512k2.sh

Looked under:
  ${STAGE1_ROOT}
  ${REPO_ROOT}/outputs/imagenet-nonpatch-d5-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}/stage1_adv
EOF
    exit 1
  fi
  printf '%s\n' "${ckpt}"
}

STAGE1_CKPT="$(resolve_stage1_ckpt)"
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "Stage-1 checkpoint not found: ${STAGE1_CKPT}" >&2
  exit 1
fi

check_stage1_recipe() {
  if [[ "${CHECK_STAGE1_RECIPE}" == "0" || "${CHECK_STAGE1_RECIPE}" == "false" ]]; then
    return
  fi
  python - "${STAGE1_CKPT}" "${ATOM_VOCAB_SIZE}" "${SPARSITY_LEVEL}" "${EMBEDDING_DIM}" <<'PY'
import sys
from pathlib import Path

import torch

ckpt_path = Path(sys.argv[1])
expected_atoms = int(sys.argv[2])
expected_k = int(sys.argv[3])
expected_dim = int(sys.argv[4])

try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(ckpt_path, map_location="cpu")

hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
actual_atoms = hparams.get("num_embeddings")
actual_k = hparams.get("sparsity_level")
actual_dim = hparams.get("embedding_dim")

missing = [
    name
    for name, value in (
        ("num_embeddings", actual_atoms),
        ("sparsity_level", actual_k),
        ("embedding_dim", actual_dim),
    )
    if value is None
]
if missing:
    print(
        "Warning: checkpoint metadata missing "
        + ", ".join(missing)
        + "; continuing without full recipe verification.",
        file=sys.stderr,
    )
    raise SystemExit(0)

actual = (int(actual_atoms), int(actual_k), int(actual_dim))
expected = (expected_atoms, expected_k, expected_dim)
if actual != expected:
    raise SystemExit(
        "Stage-1 checkpoint recipe mismatch: "
        f"expected num_embeddings/sparsity_level/embedding_dim={expected}, got {actual}. "
        "Set STAGE1_CKPT to an ImageNet a512/k2 checkpoint, or override "
        "ATOM_VOCAB_SIZE/SPARSITY_LEVEL/EMBEDDING_DIM intentionally."
    )
PY
}

check_stage1_recipe

TOKEN_CACHE="${S2_ROOT}/token_cache/cc3m_train_imagenet_a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}_q${COEFF_BINS}_${TEXT_TOKENIZER}_text${TEXT_MAX_LENGTH}.pt"

cat <<EOF
Pipeline root: ${PIPE_ROOT}
Temp: ${TMPDIR}
Stage-1 ImageNet tokenizer: ${STAGE1_CKPT}
Expected tokenizer recipe: atoms=${ATOM_VOCAB_SIZE} k=${SPARSITY_LEVEL} embedding_dim=${EMBEDDING_DIM}
CC-3M cache max items: ${CACHE_MAX_ITEMS} (0 means full split)
Stage-2 topology: d_model=${S2_D_MODEL} spatial_layers=${S2_LAYERS} depth_layers=${S2_DEPTH_LAYERS} heads=${S2_HEADS} d_ff=${S2_FF}
Stage-2 schedule: epochs=${S2_MAX_EPOCHS} max_steps=${S2_MAX_STEPS} batch=${S2_BATCH_SIZE} accumulate=${S2_ACCUMULATE_GRAD_BATCHES}
Stage-2 effective batch: $((S2_BATCH_SIZE * S2_ACCUMULATE_GRAD_BATCHES))
Stage-2 checkpoint metric: s2/generation_fid every ${S2_GENERATION_FID_EVERY_N_EPOCHS} epoch(s), ${S2_GENERATION_FID_NUM_SAMPLES} samples
Stage-2 sampling: temperature=${S2_SAMPLE_TEMPERATURE} top_k=${S2_SAMPLE_TOP_K}
EOF

export LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1

run_stage2() {
  local run_name="${RECIPE_NAME}-${STAMP}"
  local tmpdir="${TMPDIR}"
  mkdir -p "${tmpdir}" "$(dirname "${TOKEN_CACHE}")" "${S2_ROOT}/wandb/cache" "${S2_ROOT}/wandb/data" "${S2_ROOT}/wandb/artifacts"
  export TMPDIR="${tmpdir}"
  export MPLCONFIGDIR="${S2_ROOT}/mplconfig"
  export WANDB_DIR="${S2_ROOT}/wandb"
  export WANDB_CACHE_DIR="${S2_ROOT}/wandb/cache"
  export WANDB_DATA_DIR="${S2_ROOT}/wandb/data"
  export WANDB_ARTIFACT_DIR="${S2_ROOT}/wandb/artifacts"

  python train.py stage2 \
    "token_cache_path=${TOKEN_CACHE}" \
    "output_dir=${S2_ROOT}" \
    "seed=42" \
    "token_cache.build=true" \
    "token_cache.force=false" \
    "token_cache.stage1_checkpoint=${STAGE1_CKPT}" \
    "token_cache.output=${TOKEN_CACHE}" \
    "token_cache.split=train" \
    "token_cache.cache_mode=quantized" \
    "token_cache.coeff_vocab_size=${COEFF_BINS}" \
    "token_cache.coeff_max=auto" \
    "token_cache.coeff_quantization=quantile" \
    "token_cache.coeff_calibration_percentile=99.5" \
    "token_cache.text_max_length=${TEXT_MAX_LENGTH}" \
    "token_cache.text_tokenizer=${TEXT_TOKENIZER}" \
    "token_cache.batch_size=${CACHE_BATCH_SIZE}" \
    "token_cache.num_workers=${CACHE_NUM_WORKERS}" \
    "token_cache.max_items=${CACHE_MAX_ITEMS}" \
    "token_cache.device=auto" \
    "data.dataset=cc3m" \
    "data.data_dir=${DATA_DIR}" \
    "data.image_size=256" \
    "data.num_workers=${S2_NUM_WORKERS}" \
    "ar.type=sparse_spatial_depth" \
    "ar.autoregressive_coeffs=true" \
    "ar.text_conditional=true" \
    "ar.text_conditioning_mode=rq_prefix" \
    "ar.text_prefix_length=${TEXT_MAX_LENGTH}" \
    "ar.text_loss_weight=0.1" \
    "ar.image_loss_weight=0.9" \
    "ar.n_global_spatial_tokens=0" \
    "ar.d_model=${S2_D_MODEL}" \
    "ar.n_heads=${S2_HEADS}" \
    "ar.n_layers=${S2_LAYERS}" \
    "ar.n_depth_layers=${S2_DEPTH_LAYERS}" \
    "ar.d_ff=${S2_FF}" \
    "ar.dropout=0.1" \
    "ar.learning_rate=${S2_LR}" \
    "ar.weight_decay=${S2_WEIGHT_DECAY}" \
    "ar.optimizer_beta1=0.9" \
    "ar.optimizer_beta2=0.95" \
    "ar.warmup_steps=${S2_WARMUP_STEPS}" \
    "ar.max_steps=${S2_MAX_STEPS}" \
    "ar.min_lr_ratio=0.01" \
    "ar.atom_loss_weight=1.0" \
    "ar.coeff_loss_weight=1.0" \
    "ar.atom_label_smoothing=0.02" \
    "ar.atom_coverage_weight=0.02" \
    "ar.coeff_loss_type=huber" \
    "ar.coeff_huber_delta=0.25" \
    "ar.coeff_head_hidden_mult=2.0" \
    "ar.coeff_head_depth=2" \
    "ar.coeff_head_dropout=0.05" \
    "ar.sample_coeff_mode=mean" \
    "train_ar.max_epochs=${S2_MAX_EPOCHS}" \
    "train_ar.batch_size=${S2_BATCH_SIZE}" \
    "train_ar.accumulate_grad_batches=${S2_ACCUMULATE_GRAD_BATCHES}" \
    "train_ar.max_items=0" \
    "train_ar.limit_train_batches=1.0" \
    "train_ar.limit_val_batches=${S2_VAL_BATCHES}" \
    "train_ar.limit_test_batches=0" \
    "train_ar.val_check_interval=1.0" \
    "train_ar.validation_split=0.05" \
    "train_ar.test_split=0.00" \
    "train_ar.log_every_n_steps=20" \
    "train_ar.devices=1" \
    "train_ar.num_nodes=1" \
    "train_ar.strategy=auto" \
    "train_ar.precision=bf16-mixed" \
    "train_ar.deterministic=false" \
    "train_ar.gradient_clip_val=1.0" \
    "train_ar.checkpoint_save_top_k=3" \
    "train_ar.checkpoint_save_last=true" \
    "train_ar.checkpoint_every_n_epochs=1" \
    "train_ar.checkpoint_monitor=s2/generation_fid" \
    "train_ar.checkpoint_mode=min" \
    "train_ar.generation_fid_every_n_epochs=${S2_GENERATION_FID_EVERY_N_EPOCHS}" \
    "train_ar.generation_fid_num_samples=${S2_GENERATION_FID_NUM_SAMPLES}" \
    "train_ar.sample_every_n_steps=${S2_SAMPLE_EVERY_N_STEPS}" \
    "train_ar.sample_every_n_epochs=0" \
    "train_ar.sample_num_images=${S2_SAMPLE_NUM_IMAGES}" \
    "train_ar.sample_temperature=${S2_SAMPLE_TEMPERATURE}" \
    "train_ar.sample_top_k=${S2_SAMPLE_TOP_K}" \
    "train_ar.sample_coeff_mode=mean" \
    "train_ar.sample_log_to_wandb=true" \
    "train_ar.sample_text_prompts=[\"a red sports car parked on a city street\",\"a small dog running through green grass\",\"a plate of fresh fruit on a wooden table\",\"a bedroom with a large window and white sheets\",\"eiffel tower on a desert\",\"a painting by vincent van gogh\"]" \
    "train_ar.compute_generation_fid=true" \
    "train_ar.generation_metric_num_samples=${S2_GENERATION_FID_NUM_SAMPLES}" \
    "train_ar.run_test_after_fit=false" \
    "train_ar.save_final_samples_after_fit=true" \
    "+train_ar.checkpoint_upload_to_wandb=true" \
    "+train_ar.checkpoint_upload_every_n_epochs=1" \
    "wandb.project=laser" \
    "wandb.name=${run_name}" \
    "wandb.group=${RECIPE_NAME}" \
    "wandb.tags=[stage2,cc3m,laser,rq_prefix,text_conditional,bpe32,imagenet_tokenizer,a${ATOM_VOCAB_SIZE},k${SPARSITY_LEVEL},q${COEFF_BINS}]" \
    "wandb.append_timestamp=false" \
    "wandb.save_dir=${S2_ROOT}/wandb" \
    "$@" \
    2>&1 | tee "${S2_ROOT}/logs/train.log"
}

echo "Stage-1 checkpoint: ${STAGE1_CKPT}" | tee "${PIPE_ROOT}/stage1_checkpoint.txt"
run_stage2 "$@"
