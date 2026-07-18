#!/usr/bin/env bash
# FFHQ-256 two-phase dictionary tokenizer based on KakaoBrain's
# ffhq256-rqvae-8x8x4 stage-1 configuration.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
DATA_DIR="${DATA_DIR:-/workspace/Projects/data/ffhq}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/ffhq-rqvae-config-a1024k2-recon100-adv150-${STAMP}}"
RECON_DIR="${RUN_ROOT}/recon100"
ADV_DIR="${RUN_ROOT}/perceptual_adv150"
LOG_DIR="${RUN_ROOT}/logs"
GROUP="${WANDB_GROUP:-ffhq-rqvae-config-a1024k2-${STAMP}}"
RECON_WANDB_ID="${RECON_WANDB_ID:-ffhqa1024k2r100${STAMP}}"
ADV_WANDB_ID="${ADV_WANDB_ID:-ffhqa1024k2pa150${STAMP}}"

# The reference was trained on four A100s with batch_size=32 per rank. A
# microbatch of 32 plus four-way accumulation retains its effective batch of
# 128 and its 4e-5 learning rate on one H100. The lighter reconstruction phase
# can safely share the currently free H100 memory; the heavier GAN phase waits
# for essentially the full GPU if another job is still present.
MICROBATCH="${MICROBATCH:-32}"
ACCUMULATE="${ACCUMULATE:-4}"
EVAL_BATCH="${EVAL_BATCH:-32}"
LEARNING_RATE="${LEARNING_RATE:-4.0e-5}"
WARMUP_UPDATES="${WARMUP_UPDATES:-2465}"
NUM_WORKERS="${NUM_WORKERS:-12}"
IMAGE_LOG_EVERY_N_STEPS="${IMAGE_LOG_EVERY_N_STEPS:-1000}"
RECON_MIN_FREE_MIB="${RECON_MIN_FREE_MIB:-50000}"
ADV_MIN_FREE_MIB="${ADV_MIN_FREE_MIB:-70000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
PRINT_ONLY="${PRINT_ONLY:-0}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME="allow"
export LASER_DISABLE_WANDB_MEDIA="${LASER_DISABLE_WANDB_MEDIA:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TMPDIR="${TMPDIR:-/tmp/laser_ffhq_a1024k2_${STAMP}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
unset WANDB_RUN_ID WANDB_ID

mkdir -p \
  "$RECON_DIR/wandb" \
  "$ADV_DIR/wandb" \
  "$LOG_DIR" \
  "$TMPDIR"
printf '%s\n' "$$" > "$RUN_ROOT/driver.pid"
if [[ ! -s "$RUN_ROOT/status.tsv" ]]; then
  printf 'time_utc\tphase\tstate\tdetail\n' > "$RUN_ROOT/status.tsv"
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "FFHQ data directory not found: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$REPO_ROOT"

status() {
  local phase="$1"
  local state="$2"
  local detail="${3:-}"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$phase" "$state" "$detail" \
    >> "$RUN_ROOT/status.tsv"
}

active_phase="driver"
on_exit() {
  local exit_code="$?"
  if (( exit_code == 0 )); then
    status "$active_phase" complete "pipeline exit=0"
  else
    status "$active_phase" failed "pipeline exit=${exit_code}"
  fi
}
trap on_exit EXIT

latest_checkpoint() {
  local root="$1"
  local name="$2"
  find "$root" -type f -name "$name" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print; }'
}

wait_for_gpu_memory() {
  local required_mib="$1"
  if [[ "$PRINT_ONLY" == "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      | head -n 1 | tr -d '[:space:]')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= required_mib )); then
      echo "GPU memory gate passed: ${free_mib} MiB free (required ${required_mib} MiB)."
      return 0
    fi
    echo "Waiting for GPU memory: ${free_mib:-unknown} MiB free; need ${required_mib} MiB."
    sleep "$GPU_WAIT_SECONDS"
  done
}

run_training() {
  local log_file="$1"
  shift
  local command=("$PYTHON_BIN" train.py stage1 "$@")
  if [[ "$PRINT_ONLY" == "1" ]]; then
    printf 'COMMAND '
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi
  "${command[@]}" 2>&1 | tee -a "$log_file"
}

COMMON_ARGS=(
  model=laser_image_nonpatch_d5
  data=ffhq
  seed=42
  "data.data_dir=${DATA_DIR}"
  data.image_size=256
  "data.batch_size=${MICROBATCH}"
  "data.eval_batch_size=${EVAL_BATCH}"
  "data.num_workers=${NUM_WORKERS}"
  data.train_crop_size=null
  data.augment=true
  train.accelerator=gpu
  train.num_nodes=1
  train.devices=1
  train.strategy=auto
  train.precision=bf16-mixed
  train.max_steps=-1
  "train.learning_rate=${LEARNING_RATE}"
  train.beta=0.5
  train.beta2=0.9
  "train.warmup_steps=${WARMUP_UPDATES}"
  train.min_lr_ratio=1.0
  "train.accumulate_grad_batches=${ACCUMULATE}"
  train.gradient_clip_val=0.0
  train.deterministic=false
  train.log_every_n_steps=25
  train.limit_train_batches=1.0
  train.limit_val_batches=1.0
  train.limit_test_batches=0
  train.val_check_interval=1.0
  train.run_test_after_fit=false
  train.compute_rfid_after_fit=false
  model.backbone=ddpm
  model.num_downsamples=5
  'model.channel_multipliers=[1,1,2,2,4,4]'
  model.backbone_latent_channels=256
  'model.attn_resolutions=[16]'
  model.decoder_extra_residual_layers=1
  model.use_mid_attention=true
  model.dropout=0.0
  model.num_hiddens=128
  model.num_residual_blocks=2
  model.num_residual_hiddens=96
  model.bottleneck_type=dictionary
  model.num_embeddings=1024
  model.embedding_dim=256
  model.sparsity_level=2
  model.patch_based=false
  model.patch_size=1
  model.patch_stride=1
  model.commitment_cost=0.25
  "model.dict_learning_rate=${LEARNING_RATE}"
  model.dead_atom_revival=true
  model.dead_atom_revival_interval=500
  model.dead_atom_revival_max_fraction=0.05
  model.dead_atom_revival_noise=0.05
  model.dead_atom_revival_patience=5
  model.recon_mse_weight=1.0
  model.recon_l1_weight=0.0
  model.recon_edge_weight=0.0
  model.bottleneck_loss_weight=0.25
  model.dictionary_loss_weight=0.25
  model.sparsity_reg_weight=0.0
  model.compute_fid=false
  "model.log_images_every_n_steps=${IMAGE_LOG_EVERY_N_STEPS}"
  model.enable_val_latent_visuals=false
  checkpoint.monitor=val/loss
  checkpoint.mode=min
  checkpoint.save_top_k=3
  checkpoint.save_last=true
  checkpoint.every_n_epochs=5
  checkpoint.upload_to_wandb=true
  checkpoint.upload_every_n_epochs=5
  +checkpoint.upload_mode=files
  wandb.project=laser
  wandb.append_timestamp=false
)

run_reconstruction_phase() {
  active_phase="recon100"
  local marker="$RECON_DIR/.phase_complete"
  local last_ckpt
  last_ckpt="$(latest_checkpoint "$RECON_DIR" last.ckpt)"
  if [[ -f "$marker" && -n "$last_ckpt" && "$PRINT_ONLY" != "1" ]]; then
    echo "Reusing completed reconstruction phase: $last_ckpt"
    return 0
  fi

  local resume_arg=()
  if [[ -n "$last_ckpt" ]]; then
    echo "Resuming reconstruction phase from: $last_ckpt"
    resume_arg=("ckpt_path=${last_ckpt}")
  fi

  wait_for_gpu_memory "$RECON_MIN_FREE_MIB"
  status "$active_phase" starting "microbatch=${MICROBATCH} accum=${ACCUMULATE} effective_batch=$((MICROBATCH * ACCUMULATE)) lr=${LEARNING_RATE}"
  run_training "$LOG_DIR/recon100.log" \
    "${COMMON_ARGS[@]}" \
    "output_dir=${RECON_DIR}" \
    "hydra.run.dir=${RECON_DIR}/hydra" \
    train.max_epochs=100 \
    model.data_init_from_first_batch=true \
    model.perceptual_weight=0.0 \
    model.perceptual_start_step=1000000000 \
    model.perceptual_warmup_steps=0 \
    model.adversarial_weight=0.0 \
    model.adversarial_start_step=1000000000 \
    model.adversarial_warmup_steps=0 \
    model.disc_start_step=1000000000 \
    "wandb.id=${RECON_WANDB_ID}" \
    wandb.resume=allow \
    "wandb.name=ffhq-rqvae-config-recon100-a1024k2-${STAMP}" \
    "wandb.group=${GROUP}" \
    'wandb.tags=[stage1,ffhq,rqvae_config,dictionary,a1024,k2,dim256,recon100,no_lpips,no_gan,effective_batch128,h100]' \
    "wandb.save_dir=${RECON_DIR}/wandb" \
    "${resume_arg[@]}"

  if [[ "$PRINT_ONLY" == "1" ]]; then
    return 0
  fi
  last_ckpt="$(latest_checkpoint "$RECON_DIR" last.ckpt)"
  if [[ -z "$last_ckpt" ]]; then
    echo "Reconstruction phase finished without last.ckpt" >&2
    return 1
  fi
  touch "$marker"
  find "$RECON_DIR" -type f -name final.ckpt -delete
  status "$active_phase" complete "checkpoint=${last_ckpt}"
}

run_adversarial_phase() {
  active_phase="perceptual_adv150"
  local marker="$ADV_DIR/.phase_complete"
  local recon_ckpt adv_last
  recon_ckpt="$(latest_checkpoint "$RECON_DIR" last.ckpt)"
  if [[ "$PRINT_ONLY" == "1" ]]; then
    recon_ckpt="${recon_ckpt:-${RECON_DIR}/checkpoints/last.ckpt}"
  elif [[ -z "$recon_ckpt" ]]; then
    echo "Missing reconstruction checkpoint under $RECON_DIR" >&2
    return 1
  fi

  adv_last="$(latest_checkpoint "$ADV_DIR" last.ckpt)"
  if [[ -f "$marker" && -n "$adv_last" && "$PRINT_ONLY" != "1" ]]; then
    echo "Reusing completed perceptual/adversarial phase: $adv_last"
    return 0
  fi

  local start_arg
  if [[ -n "$adv_last" ]]; then
    echo "Resuming perceptual/adversarial phase from: $adv_last"
    start_arg="ckpt_path=${adv_last}"
  else
    echo "Initializing perceptual/adversarial phase from: $recon_ckpt"
    start_arg="init_ckpt_path=${recon_ckpt}"
  fi

  wait_for_gpu_memory "$ADV_MIN_FREE_MIB"
  status "$active_phase" starting "microbatch=${MICROBATCH} accum=${ACCUMULATE} effective_batch=$((MICROBATCH * ACCUMULATE)) lr=${LEARNING_RATE}"
  run_training "$LOG_DIR/perceptual_adv150.log" \
    "${COMMON_ARGS[@]}" \
    "output_dir=${ADV_DIR}" \
    "hydra.run.dir=${ADV_DIR}/hydra" \
    train.max_epochs=150 \
    model.data_init_from_first_batch=false \
    model.perceptual_weight=1.0 \
    model.perceptual_start_step=0 \
    model.perceptual_warmup_steps=0 \
    model.adversarial_weight=0.75 \
    model.adversarial_start_step=0 \
    model.adversarial_warmup_steps=0 \
    model.disc_start_step=0 \
    "model.disc_learning_rate=${LEARNING_RATE}" \
    model.discriminator_beta1=0.5 \
    model.discriminator_beta2=0.9 \
    model.disc_channels=64 \
    model.disc_num_layers=2 \
    model.disc_norm=batch \
    model.disc_spectral=false \
    model.disc_loss=hinge \
    model.use_adaptive_disc_weight=true \
    model.disc_factor=1.0 \
    model.disc_weight_max=10000.0 \
    "wandb.id=${ADV_WANDB_ID}" \
    wandb.resume=allow \
    "wandb.name=ffhq-rqvae-config-lpips-gan150-a1024k2-${STAMP}" \
    "wandb.group=${GROUP}" \
    'wandb.tags=[stage1,ffhq,rqvae_config,dictionary,a1024,k2,dim256,lpips1,gan150,patchgan2,batchnorm,hinge,effective_batch128,h100]' \
    "wandb.save_dir=${ADV_DIR}/wandb" \
    "$start_arg"

  if [[ "$PRINT_ONLY" == "1" ]]; then
    return 0
  fi
  adv_last="$(latest_checkpoint "$ADV_DIR" last.ckpt)"
  if [[ -z "$adv_last" ]]; then
    echo "Perceptual/adversarial phase finished without last.ckpt" >&2
    return 1
  fi
  touch "$marker"
  find "$ADV_DIR" -type f -name final.ckpt -delete
  status "$active_phase" complete "checkpoint=${adv_last}"
}

run_reconstruction_phase
run_adversarial_phase
active_phase="pipeline"
status "$active_phase" complete "recon100 plus perceptual_adv150"
