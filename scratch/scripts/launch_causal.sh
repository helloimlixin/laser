#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

out_root="${OUT_ROOT:-$ROOT_DIR/runs/laser_scale_var_causal_train_gpu2}"
ts="$(date +%Y%m%d_%H%M%S)"
session="${SESSION_NAME:-scale_var_causal_train_${ts}}"
log="${LOG_PATH:-${out_root}/tmux_${ts}.log}"

mkdir -p "${out_root}"

cmd_env=("OUT_DIR='${out_root}'")
for key in WANDB_MODE WANDB_ANONYMOUS WANDB_PROJECT WANDB_NAME NPROC_PER_NODE DATASET DATA_DIR STAGE1_CKPT EPOCHS BATCH_SIZE NUM_WORKERS TOKEN_NUM_WORKERS CPU_THREADS DIST_TIMEOUT_MINUTES LR WEIGHT_DECAY VAR_D_MODEL VAR_HEADS VAR_LAYERS VAR_FF SAMPLE_EVERY_STEPS SAMPLE_BATCH_SIZE SAMPLE_TEMPERATURE SAMPLE_TOP_K TEACHER_FORCED_RECON_EVERY_STEPS TEACHER_FORCED_RECON_NUM_SAMPLES AMP; do
  if [[ -n "${!key:-}" ]]; then
    cmd_env+=("${key}='${!key}'")
  fi
done
env_prefix="$(printf "%s " "${cmd_env[@]}")"

tmux new-session -d -s "${session}" \
  /bin/bash -lc \
  "${env_prefix}'${ROOT_DIR}/scripts/causal.sh' > '${log}' 2>&1; status=\$?; echo [launcher] exit_status=\$status >> '${log}'; exec bash"

printf 'session=%s\n' "${session}"
printf 'log=%s\n' "${log}"
