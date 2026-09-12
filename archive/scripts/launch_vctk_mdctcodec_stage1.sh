#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export VISQOL_BINARY="${VISQOL_BINARY:-${PWD}/outputs/visqol/bin/visqol}"
if [[ ! -x "$VISQOL_BINARY" ]]; then
  echo "Build official ViSQOL with scripts/setup_mdctcodec_audio.sh first." >&2
  exit 1
fi
if [[ ! -f outputs/mdctcodec_recovery/s2er91dm/checkpoints/last.ckpt ]]; then
  python scripts/recover_mdctcodec_wandb.py
fi
exec python -u train.py stage1 --config-name "${MDCT_CONFIG:-vctk_mdctcodec_stage1_6kbps}" "$@"
