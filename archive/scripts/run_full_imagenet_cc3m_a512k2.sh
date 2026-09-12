#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE:-512}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"

PIPE_ROOT="${PIPE_ROOT:-${REPO_ROOT}/outputs/full-imagenet-cc3m-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}-${STAMP}}"
IMAGENET_STAGE1_ROOT="${IMAGENET_STAGE1_ROOT:-${PIPE_ROOT}/imagenet_stage1}"
CC3M_STAGE2_ROOT="${CC3M_STAGE2_ROOT:-${PIPE_ROOT}/cc3m_stage2}"
LOG_DIR="${LOG_DIR:-${PIPE_ROOT}/logs}"

TMP_ROOT="${TMP_ROOT:-/workspace/tmp/laser}"
TMPDIR="${TMPDIR:-${TMP_ROOT}/full-imagenet-cc3m-${STAMP}}"
TEMP="${TEMP:-${TMPDIR}}"
TMP="${TMP:-${TMPDIR}}"
export TMPDIR TEMP TMP

mkdir -p "${LOG_DIR}" "${TMPDIR}"
cd "${REPO_ROOT}"

find_latest_stage1_ckpt() {
  local root="$1"
  local ckpt=""
  ckpt="$(find "${root}" -path '*/laser/final.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(find "${root}" -path '*/laser/*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
  fi
  printf '%s\n' "${ckpt}"
}

cat <<EOF
Full pipeline root: ${PIPE_ROOT}
Temp: ${TMPDIR}
ImageNet stage-1 root: ${IMAGENET_STAGE1_ROOT}
CC-3M stage-2 root: ${CC3M_STAGE2_ROOT}
Recipe: atoms=${ATOM_VOCAB_SIZE} k=${SPARSITY_LEVEL} embedding_dim=${EMBEDDING_DIM}
EOF

if [[ -z "${STAGE1_CKPT:-}" ]]; then
  echo "Starting ImageNet stage-1 tokenizer..."
  RUN_ROOT="${IMAGENET_STAGE1_ROOT}" \
  RUN_NAME="imagenet-stage1-a${ATOM_VOCAB_SIZE}k${SPARSITY_LEVEL}-${STAMP}" \
  ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE}" \
  SPARSITY_LEVEL="${SPARSITY_LEVEL}" \
  EMBEDDING_DIM="${EMBEDDING_DIM}" \
  scripts/run_imagenet_stage1_a512k2.sh 2>&1 | tee "${LOG_DIR}/imagenet_stage1.log"

  STAGE1_CKPT="$(find_latest_stage1_ckpt "${IMAGENET_STAGE1_ROOT}")"
fi

if [[ -z "${STAGE1_CKPT}" || ! -f "${STAGE1_CKPT}" ]]; then
  echo "Could not resolve completed ImageNet stage-1 checkpoint under ${IMAGENET_STAGE1_ROOT}" >&2
  exit 1
fi

echo "Resolved ImageNet stage-1 checkpoint: ${STAGE1_CKPT}" | tee "${PIPE_ROOT}/stage1_checkpoint.txt"
echo "Starting CC-3M stage-2 prior..."

STAGE1_CKPT="${STAGE1_CKPT}" \
PIPE_ROOT="${PIPE_ROOT}" \
S2_ROOT="${CC3M_STAGE2_ROOT}" \
ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE}" \
SPARSITY_LEVEL="${SPARSITY_LEVEL}" \
EMBEDDING_DIM="${EMBEDDING_DIM}" \
scripts/run_cc3m_stage2_imagenet_a512k2.sh 2>&1 | tee "${LOG_DIR}/cc3m_stage2.log"
