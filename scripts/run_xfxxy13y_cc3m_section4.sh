#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
DATA_DIR="${DATA_DIR:-/workspace/Projects/data/cc3m}"
STAGE1_CKPT="${STAGE1_CKPT:-${REPO_ROOT}/outputs/imagenet-rqvae-a512k2-gan-rqvae-lpips-20260714_002101/imagenet_stage1/checkpoints/run_20260714_002106/laser/final.ckpt}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
PIPE_ROOT="${PIPE_ROOT:-${REPO_ROOT}/outputs/cc3m-section4-xfxxy13y-${STAMP}}"
S2_ROOT="${S2_ROOT:-${PIPE_ROOT}/stage2}"
RECON_ITEMS="${RECON_ITEMS:-16}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-8}"
CACHE_MAX_ITEMS="${CACHE_MAX_ITEMS:-0}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

RECON_CACHE="${PIPE_ROOT}/stage1_recon/cc3m_val${RECON_ITEMS}_imagenet_a512k2_q128_bpe32.pt"
TOKEN_CACHE="${S2_ROOT}/token_cache/cc3m_train_imagenet_a512k2_q128_rq_bpe16k_text32.pt"

mkdir -p "${PIPE_ROOT}/logs" "${PIPE_ROOT}/stage1_recon" "${S2_ROOT}/token_cache"
cd "${REPO_ROOT}"

printf '%s\n' "$$" > "${PIPE_ROOT}/driver.pid"
printf '%s\n' "${STAGE1_CKPT}" > "${PIPE_ROOT}/stage1_checkpoint.txt"
printf '%s\n' "${PIPE_ROOT}" > "${REPO_ROOT}/outputs/cc3m-section4-xfxxy13y-latest.txt"

finish() {
  local rc=$?
  printf '%s\n' "${rc}" > "${PIPE_ROOT}/exit.status"
}
trap finish EXIT

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "ImageNet checkpoint not found: ${STAGE1_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${RECON_CACHE}" ]]; then
  python scripts/tools/build_token_cache.py \
    --stage1_checkpoint "${STAGE1_CKPT}" \
    --dataset cc3m \
    --data_dir "${DATA_DIR}" \
    --split val \
    --cache_mode quantized \
    --image_size 256 \
    --batch_size 4 \
    --num_workers 2 \
    --coeff_vocab_size 128 \
    --coeff_max auto \
    --coeff_quantization quantile \
    --coeff_calibration_percentile 99.5 \
    --text_max_length 32 \
    --text_tokenizer rq_bpe16k \
    --max_items "${RECON_ITEMS}" \
    --device auto \
    --output "${RECON_CACHE}" \
    2>&1 | tee "${PIPE_ROOT}/logs/stage1_recon_cache.log"
fi

if [[ ! -f "${PIPE_ROOT}/stage1_recon/diagnostics/cc3m_laser_recon_metrics.json" ]]; then
  python scripts/tools/diagnose_cc3m_stage1_recon.py \
    --cache "${RECON_CACHE}" \
    --checkpoint "${STAGE1_CKPT}" \
    --out-dir "${PIPE_ROOT}/stage1_recon/diagnostics" \
    --count "${RECON_ITEMS}" \
    --device auto \
    2>&1 | tee "${PIPE_ROOT}/logs/stage1_recon_diagnostics.log"
fi

if [[ ! -f "${TOKEN_CACHE}" ]]; then
  python scripts/tools/build_token_cache.py \
    --stage1_checkpoint "${STAGE1_CKPT}" \
    --dataset cc3m \
    --data_dir "${DATA_DIR}" \
    --split train \
    --cache_mode quantized \
    --image_size 256 \
    --batch_size "${CACHE_BATCH_SIZE}" \
    --num_workers "${CACHE_NUM_WORKERS}" \
    --coeff_vocab_size 128 \
    --coeff_max auto \
    --coeff_quantization quantile \
    --coeff_calibration_percentile 99.5 \
    --text_max_length 32 \
    --text_tokenizer rq_bpe16k \
    --max_items "${CACHE_MAX_ITEMS}" \
    --device auto \
    --output "${TOKEN_CACHE}" \
    2>&1 | tee "${PIPE_ROOT}/logs/token_cache_train.log"
fi

python - "${TOKEN_CACHE}" "${STAGE1_CKPT}" <<'PY'
import sys
from pathlib import Path

import torch

cache_path = Path(sys.argv[1]).resolve()
checkpoint = str(Path(sys.argv[2]).resolve())
cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
meta = dict(cache.get("meta", {}) or {})
tokens = cache.get("tokens_flat")
text = cache.get("text_tokens")
assert torch.is_tensor(tokens) and tokens.ndim == 2 and int(tokens.size(0)) > 0
assert torch.is_tensor(text) and tuple(text.shape) == (int(tokens.size(0)), 32)
assert tuple(cache.get("shape", ())) == (8, 8, 4)
assert int(meta.get("num_atoms", 0)) == 512
assert int(meta.get("sparsity_level", 0)) == 2
assert meta.get("text_tokenizer") == "rq_bpe16k"
assert str(Path(meta.get("stage1_checkpoint", "")).resolve()) == checkpoint
print(f"Validated CC3M cache: {tokens.size(0)} rows, tokens={tuple(tokens.shape)}, text={tuple(text.shape)}")
PY

if [[ -n "${WAIT_FOR_PID}" ]]; then
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    printf '%s waiting for PID %s before 650M stage-2 training\n' "$(date -u +%FT%TZ)" "${WAIT_FOR_PID}" \
      | tee -a "${PIPE_ROOT}/logs/wait_for_gpu.log"
    sleep 60
  done
fi

export STAGE1_CKPT PIPE_ROOT S2_ROOT DATA_DIR CACHE_BATCH_SIZE CACHE_NUM_WORKERS CACHE_MAX_ITEMS
export RECIPE_NAME="cc3m-s2-section4-650m-imagenet-a512k2"
export STAMP
bash scripts/run_cc3m_stage2_imagenet_a512k2.sh "$@"
