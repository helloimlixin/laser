#!/usr/bin/env bash
# Linear-scaled duplicate of imga16384k4s1fair-20260814213455.
# Defaults to 4x H100 (local batch 64), and also supports 8x A100/L40S
# (set NPROC=8 for local batch 32).

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"

export STAMP
export PYTHON_BIN="${PYTHON_BIN:-/workspace/tmp/laser-h100-venv/bin/python}"
export NPROC="${NPROC:-4}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
export LEARNING_RATE="${LEARNING_RATE:-8.0e-5}"
export DICTIONARY_LEARNING_RATE="${DICTIONARY_LEARNING_RATE:-$LEARNING_RATE}"
export SOURCE_RUN="${SOURCE_RUN:-helloimlixin-rutgers/laser/imga16384k4s1fair-20260814213455}"
export RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/imagenet-a16384-k4-fair-stage1-b256-$STAMP}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-imga16384k4s1fair-b256-${STAMP//-/}}"
export WANDB_NAME="${WANDB_NAME:-imagenet-a16384-k4-fair-stage1-b256-$STAMP}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet-a16384-k4-fair-scaled-b256-$STAMP}"
# The current H100 host/driver exposes NVLink but rejects NCCL's NVLS setup
# with CUDA error 401. Ring/tree collectives retain full DDP correctness.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

exec "$ROOT/scripts/run_imagenet_a16384_k4_fair_stage1.sh" "$@"
