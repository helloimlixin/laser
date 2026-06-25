#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
WAIT_PID="${WAIT_PID:-194347}"
SRC_DATA_DIR="${SRC_DATA_DIR:-/workspace/Projects/data/ffhq}"
SPLIT_DATA_DIR="${SPLIT_DATA_DIR:-outputs/ffhq-splits/ffhq_60k10k_seed42}"
QUEUE_ROOT="${QUEUE_ROOT:-outputs/ffhq-experiments/queue_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${QUEUE_ROOT}"
LOG="${LOG:-${QUEUE_ROOT}/queue.log}"

RUN_CONTINUE_200K="${RUN_CONTINUE_200K:-1}"
RUN_BYPASS="${RUN_BYPASS:-1}"
RUN_LPIPS_NO_EDGE="${RUN_LPIPS_NO_EDGE:-1}"
RUN_RQ_BASELINE="${RUN_RQ_BASELINE:-1}"
RUN_LATE_GAN="${RUN_LATE_GAN:-0}"

CONT_MAX_STEPS="${CONT_MAX_STEPS:-200000}"
CONT_MAX_EPOCHS="${CONT_MAX_EPOCHS:-400}"
ABLATION_MAX_STEPS="${ABLATION_MAX_STEPS:-120000}"
ABLATION_MAX_EPOCHS="${ABLATION_MAX_EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
RFID_BATCH_SIZE="${RFID_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-32}"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "${LOG}"
}

latest_ckpt() {
  local root="$1"
  find "${root}" -type f \( -name 'final.ckpt' -o -name 'last.ckpt' \) -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

latest_current_ckpt() {
  find outputs/ffhq-full/stage1/checkpoints -type f -path '*/laser/last.ckpt' -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

common_stage1_args() {
  local out_dir="$1"
  local data_dir="$2"
  local max_steps="$3"
  local max_epochs="$4"
  shift 4
  printf '%s\0' \
    train.py stage1 \
    model=laser_image_nonpatch_d5 \
    data=ffhq \
    "output_dir=${out_dir}" \
    wandb.project=laser \
    wandb.group=ffhq-debug-rfid \
    wandb.append_timestamp=false \
    "wandb.save_dir=${out_dir}/wandb" \
    train.accelerator=gpu \
    train.devices=1 \
    train.strategy=auto \
    train.precision=bf16-mixed \
    train.deterministic=false \
    train.accumulate_grad_batches=1 \
    "train.max_epochs=${max_epochs}" \
    "train.max_steps=${max_steps}" \
    train.limit_train_batches=1.0 \
    train.limit_val_batches=1.0 \
    train.limit_test_batches=0 \
    train.val_check_interval=1.0 \
    train.run_test_after_fit=false \
    train.compute_rfid_after_fit=false \
    train.learning_rate=0.0002 \
    train.warmup_steps=2000 \
    train.min_lr_ratio=0.05 \
    train.gradient_clip_val=1.0 \
    train.log_every_n_steps=20 \
    "data.data_dir=${data_dir}" \
    "data.batch_size=${BATCH_SIZE}" \
    "data.eval_batch_size=${EVAL_BATCH_SIZE}" \
    "data.num_workers=${NUM_WORKERS}" \
    data.pin_memory=true \
    data.prefetch_factor=6 \
    data.image_size=256 \
    data.train_crop_size=null \
    data.augment=true \
    model.backbone=simple \
    model.num_embeddings=4096 \
    model.embedding_dim=128 \
    model.sparsity_level=4 \
    model.commitment_cost=0.25 \
    model.bottleneck_loss_weight=0.75 \
    model.dictionary_loss_weight=null \
    model.dict_learning_rate=0.0005 \
    model.coef_max=null \
    model.patch_based=false \
    model.patch_size=1 \
    model.patch_stride=1 \
    model.patch_reconstruction=tile \
    model.data_init_from_first_batch=true \
    model.recon_mse_weight=0.25 \
    model.recon_l1_weight=1.0 \
    model.recon_edge_weight=0.5 \
    model.perceptual_weight=0.2 \
    model.perceptual_start_step=1000 \
    model.perceptual_warmup_steps=2000 \
    model.adversarial_weight=0.0 \
    model.adversarial_start_step=1000000000 \
    model.adversarial_warmup_steps=0 \
    model.disc_start_step=1000000000 \
    model.compute_fid=true \
    model.log_images_every_n_steps=250 \
    model.diag_log_interval=250 \
    model.enable_val_latent_visuals=true \
    model.codebook_visual_max_vectors=4096 \
    "$@"
}

run_stage1() {
  local label="$1"
  local out_dir="$2"
  local data_dir="$3"
  local max_steps="$4"
  local max_epochs="$5"
  local ckpt_path="$6"
  shift 6
  mkdir -p "${out_dir}"
  local run_log="${out_dir}/train.log"
  log "starting ${label}: out=${out_dir} data=${data_dir} max_steps=${max_steps} ckpt=${ckpt_path:-<scratch>}"
  local -a cmd
  mapfile -d '' -t cmd < <(common_stage1_args "${out_dir}" "${data_dir}" "${max_steps}" "${max_epochs}" "$@")
  cmd+=("wandb.name=${label}")
  cmd+=("wandb.tags=[stage1,ffhq,debug,${label}]")
  if [[ -n "${ckpt_path}" ]]; then
    cmd+=("ckpt_path=${ckpt_path}")
  fi
  printf '[%s] command:' "$(date --iso-8601=seconds)" | tee -a "${LOG}" "${run_log}"
  printf ' %q' "${PYTHON_BIN}" "${cmd[@]}" | tee -a "${LOG}" "${run_log}"
  printf '\n' | tee -a "${LOG}" "${run_log}"
  "${PYTHON_BIN}" "${cmd[@]}" 2>&1 | tee -a "${run_log}"
  log "finished ${label}"
}

eval_rfid() {
  local label="$1"
  local ckpt="$2"
  local data_dir="$3"
  local split="$4"
  local max_samples="$5"
  local eval_log="${QUEUE_ROOT}/${label}_rfid_${split}_${max_samples}.log"
  log "rFID ${label}: split=${split} max_samples=${max_samples} ckpt=${ckpt} data=${data_dir}"
  "${PYTHON_BIN}" compute_rfid.py \
    --ckpt "${ckpt}" \
    --dataset ffhq \
    --data-dir "${data_dir}" \
    --image-size 256 \
    --mean 0.5 0.5 0.5 \
    --std 0.5 0.5 0.5 \
    --split "${split}" \
    --batch-size "${RFID_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-samples "${max_samples}" \
    --device auto \
    --fid-debug-mode recon \
    2>&1 | tee -a "${eval_log}"
}

if [[ -n "${WAIT_PID}" ]]; then
  if kill -0 "${WAIT_PID}" 2>/dev/null; then
    log "waiting for active FFHQ process pid=${WAIT_PID}"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
      sleep 300
    done
    log "wait pid exited: ${WAIT_PID}"
  else
    log "wait pid is not running: ${WAIT_PID}"
  fi
fi

if [[ "${RUN_CONTINUE_200K}" == "1" ]]; then
  ckpt="$(latest_current_ckpt)"
  [[ -n "${ckpt}" ]] || { log "ERROR: no current FFHQ checkpoint found"; exit 1; }
  cont_dir="outputs/ffhq-full/stage1_200k"
  run_stage1 ffhq-current-continue-200k "${cont_dir}" "${SRC_DATA_DIR}" "${CONT_MAX_STEPS}" "${CONT_MAX_EPOCHS}" "${ckpt}"
  cont_ckpt="$(latest_ckpt "${cont_dir}/checkpoints")"
  [[ -n "${cont_ckpt}" ]] || { log "ERROR: no continuation checkpoint found"; exit 1; }
  eval_rfid current_200k_20k "${cont_ckpt}" "${SRC_DATA_DIR}" train 20000
  eval_rfid current_200k_fulltrain "${cont_ckpt}" "${SRC_DATA_DIR}" train 0
fi

if [[ "${RUN_BYPASS}" == "1" ]]; then
  out_dir="outputs/ffhq-experiments/bypass_bottleneck_ceiling"
  run_stage1 ffhq-bypass-bottleneck-ceiling "${out_dir}" "${SPLIT_DATA_DIR}" "${ABLATION_MAX_STEPS}" "${ABLATION_MAX_EPOCHS}" "" \
    model.bypass_bottleneck=true \
    model.bottleneck_loss_weight=0.0 \
    model.dictionary_loss_weight=0.0
  ckpt="$(latest_ckpt "${out_dir}/checkpoints")"
  [[ -n "${ckpt}" ]] || { log "ERROR: no bypass checkpoint found"; exit 1; }
  eval_rfid bypass_20k "${ckpt}" "${SPLIT_DATA_DIR}" train 20000
  eval_rfid bypass_fulltrain "${ckpt}" "${SPLIT_DATA_DIR}" train 0
fi

if [[ "${RUN_LPIPS_NO_EDGE}" == "1" ]]; then
  out_dir="outputs/ffhq-experiments/lpips1_no_edge"
  run_stage1 ffhq-lpips1-no-edge "${out_dir}" "${SPLIT_DATA_DIR}" "${ABLATION_MAX_STEPS}" "${ABLATION_MAX_EPOCHS}" "" \
    model.recon_edge_weight=0.0 \
    model.perceptual_weight=1.0 \
    model.perceptual_start_step=0 \
    model.perceptual_warmup_steps=2000
  ckpt="$(latest_ckpt "${out_dir}/checkpoints")"
  [[ -n "${ckpt}" ]] || { log "ERROR: no LPIPS/no-edge checkpoint found"; exit 1; }
  eval_rfid lpips1_no_edge_20k "${ckpt}" "${SPLIT_DATA_DIR}" train 20000
  eval_rfid lpips1_no_edge_fulltrain "${ckpt}" "${SPLIT_DATA_DIR}" train 0
fi

if [[ "${RUN_LATE_GAN}" == "1" ]]; then
  log "RUN_LATE_GAN=1 was requested, but this queue intentionally does not auto-launch GAN. Inspect LPIPS/no-edge first."
else
  log "late GAN not launched; inspect LPIPS/no-edge result before enabling GAN."
fi

if [[ "${RUN_RQ_BASELINE}" == "1" ]]; then
  out_dir="outputs/ffhq-experiments/rqvae_baseline"
  run_stage1 ffhq-rqvae-baseline "${out_dir}" "${SPLIT_DATA_DIR}" "${ABLATION_MAX_STEPS}" "${ABLATION_MAX_EPOCHS}" "" \
    +model.bottleneck_type=rq \
    +model.rq_code_depth=4 \
    +model.rq_shared_codebook=true \
    +model.rq_decay=0.99 \
    +model.rq_restart_unused_codes=true
  ckpt="$(latest_ckpt "${out_dir}/checkpoints")"
  [[ -n "${ckpt}" ]] || { log "ERROR: no RQ baseline checkpoint found"; exit 1; }
  eval_rfid rqvae_20k "${ckpt}" "${SPLIT_DATA_DIR}" train 20000
  eval_rfid rqvae_fulltrain "${ckpt}" "${SPLIT_DATA_DIR}" train 0
fi

log "queue complete"
