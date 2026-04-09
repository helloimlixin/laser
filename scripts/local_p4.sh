#!/bin/bash
set -euo pipefail

# Simple local launcher for the current p4s4 experiments on a 2-GPU machine.
#
# Fresh run:
#   DATA_DIR=/data/celebahq_packed_256 RUN_ROOT=$PWD/runs/local_p4 bash scripts/local_p4.sh
#
# Stage-2 only from an existing run root:
#   MODE=s2 RUN_ROOT=$PWD/runs/local_p4 WIN_LIST=32 bash scripts/local_p4.sh
#
# Stage-2 only with explicit refs:
#   MODE=s2 RUN_ROOT=$PWD/runs/local_p4 \
#   S1_CKPT=/path/to/last.ckpt CACHE_PT=/path/to/tok_q256.pt \
#   WIN_LIST=32 bash scripts/local_p4.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODE:=all}"               # all | s1 | s2
: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${RUN_ROOT:=$ROOT_DIR/runs/local_p4}"
: "${DATA_DIR:=}"

# Current experiment shape.
: "${IMG:=256}"
: "${PATCH:=4}"
: "${STRIDE:=4}"
: "${ATOMS:=4096}"
: "${K:=16}"
: "${BINS:=256}"
: "${CMAX:=8.0}"

# Stage 1: conservative defaults for a 2x RTX 4000 box.
: "${S1_EPOCHS:=5}"
: "${S1_GPUS:=2}"
: "${S1_BSZ:=1}"
: "${S1_WORKERS:=2}"
: "${S1_LR:=2e-4}"
: "${S1_DICT_LR:=2.5e-4}"
: "${S1_WARMUP_STEPS:=500}"
: "${S1_MIN_LR_RATIO:=0.01}"
: "${S1_LOG_EVERY:=25}"
: "${S1_VAL_INTERVAL:=1.0}"
: "${S1_IMG_EVERY:=500}"
: "${S1_DIAG_EVERY:=100}"
: "${S1_LATENT_VIS:=true}"
: "${S1_BOTTLENECK_W:=1.0}"
: "${S1_PERCEPTUAL_W:=0.0}"
: "${S1_SPARSITY_REG_W:=0.01}"
: "${S1_COHERENCE_W:=0.0}"
: "${S1_BOUNDED_OMP:=8}"
: "${EMB:=4}"
: "${NHID:=128}"
: "${RBLK:=2}"
: "${RHID:=32}"

# Stage 2: simple GPT + sliding window.
: "${S2_EPOCHS:=5}"
: "${S2_GPUS:=2}"
: "${S2_BSZ:=1}"
: "${S2_WORKERS:=2}"
: "${S2_LR:=1e-4}"
: "${S2_VAL_INTERVAL:=0.25}"
: "${S2_SAMPLE_STEP_EVERY:=500}"
: "${S2_SAMPLE_EPOCH_EVERY:=5}"
: "${S2_SAMPLE_IMAGES:=16}"
: "${D_MODEL:=256}"
: "${HEADS:=8}"
: "${LAYERS:=8}"
: "${D_FF:=768}"
: "${WIN_LIST:=32}"

source "$ROOT_DIR/scripts/p4g.sh"

if [[ -z "$DATA_DIR" && "$MODE" != "s2" ]]; then
  echo "DATA_DIR is required for stage 1 / cache build." >&2
  exit 1
fi

p4g_mkdirs

if [[ -n "${S1_CKPT:-}" ]]; then
  p4g_write_ref "$(p4g_s1_ckpt_ref)" "$S1_CKPT"
fi
if [[ -n "${CACHE_PT:-}" ]]; then
  p4g_write_ref "$(p4g_cache_ref)" "$CACHE_PT"
fi

echo "ROOT_DIR=$ROOT_DIR"
echo "MODE=$MODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RUN_ROOT=$RUN_ROOT"
echo "WIN_LIST=$WIN_LIST"

case "$MODE" in
  all|s1)
    bash "$ROOT_DIR/scripts/run_p4s1.sh"
    ;;
  s2)
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    exit 1
    ;;
esac

if [[ "$MODE" == "all" || "$MODE" == "s2" ]]; then
  if [[ ! -f "$(p4g_s1_ckpt_ref)" ]]; then
    echo "Missing stage-1 checkpoint ref: $(p4g_s1_ckpt_ref)" >&2
    exit 1
  fi
  if [[ ! -f "$(p4g_cache_ref)" ]]; then
    echo "Missing token cache ref: $(p4g_cache_ref)" >&2
    exit 1
  fi

  IFS=',' read -r -a wins <<< "$WIN_LIST"
  for win in "${wins[@]}"; do
    win="$(echo "$win" | xargs)"
    [[ -n "$win" ]] || continue
    echo
    echo "=== Stage 2, window=$win ==="
    WIN="$win" bash "$ROOT_DIR/scripts/run_p4s2.sh"
  done
fi
