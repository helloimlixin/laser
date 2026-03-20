#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ENTRYPOINT="${ENTRYPOINT:-proto.py}"

export DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
export OUT_DIR="${OUT_DIR:-/scratch/$USER/runs/celebahq256_patch_qbest}"
export WANDB_NAME="${WANDB_NAME:-celebahq256_patch_qbest}"
export LOG_PREFIX="${LOG_PREFIX:-celebahq256_patch_qbest}"
export JOB_NAME="${JOB_NAME:-chq256-patch-qbest}"

export PARTITION="${PARTITION:-gpu-redhat}"
export NODES="${NODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export MEM_MB="${MEM_MB:-128000}"
export TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"

export IMAGE_SIZE="${IMAGE_SIZE:-256}"
export STAGE2_SAMPLE_IMAGE_SIZE="${STAGE2_SAMPLE_IMAGE_SIZE:-$IMAGE_SIZE}"

export AE_NUM_DOWNSAMPLES="${AE_NUM_DOWNSAMPLES:-4}"
export NUM_HIDDENS="${NUM_HIDDENS:-128}"
export NUM_RES_HIDDENS="${NUM_RES_HIDDENS:-64}"
export NUM_RES_LAYERS="${NUM_RES_LAYERS:-2}"

export BATCH_SIZE="${BATCH_SIZE:-6}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-6}"

# Balanced winner from the 2026-03-19 coeff comparison:
# a4096/k24 had the lowest stage-2 loss; a6144/k24 was best on stage-1 recon.
export NUM_ATOMS="${NUM_ATOMS:-4096}"
export SPARSITY_LEVEL="${SPARSITY_LEVEL:-24}"
export QUANTIZE_SPARSE_COEFFS="${QUANTIZE_SPARSE_COEFFS:-true}"
export N_BINS="${N_BINS:-512}"
export COEF_MAX="${COEF_MAX:-4.0}"
export COEF_QUANTIZATION="${COEF_QUANTIZATION:-uniform}"
export COEF_MU="${COEF_MU:-0.0}"

export PATCH_BASED="${PATCH_BASED:-true}"
export PATCH_SIZE="${PATCH_SIZE:-4}"
export PATCH_STRIDE="${PATCH_STRIDE:-2}"
export PATCH_RECONSTRUCTION="${PATCH_RECONSTRUCTION:-hann}"

exec "$ROOT_DIR/scripts/patch100.sh" "$@"
