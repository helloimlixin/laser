#!/usr/bin/env bash
# Train an identity-preserving FFHQ tokenizer before a short GAN continuation,
# then fit the unconditional sparse spatial-depth prior.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
DATA_DIR="${DATA_DIR:-/workspace/Projects/data/ffhq}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/ffhq-a256k2-recon150-adv50-stage2-1000-${STAMP}}"
RECON_DIR="${RUN_ROOT}/stage1_recon150"
ADV_DIR="${RUN_ROOT}/stage1_adv50"
STAGE2_DIR="${RUN_ROOT}/stage2"
TOKEN_CACHE="${STAGE2_DIR}/token_cache/ffhq_train_img256_laser_a256k2_q256.pt"
GROUP="${WANDB_GROUP:-ffhq-a256k2-recon150-adv50-stage2-1000-${STAMP}}"
RECON_WANDB_ID="${RECON_WANDB_ID:-ffhqa256k2r150${STAMP}}"
ADV_WANDB_ID="${ADV_WANDB_ID:-ffhqa256k2adv50${STAMP}}"
STAGE2_WANDB_ID="${STAGE2_WANDB_ID:-ffhqa256k2s2e1000${STAMP}}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-35000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME="allow"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TMPDIR="${TMPDIR:-/tmp/laser_ffhq_a256k2_${STAMP}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
unset WANDB_RUN_ID WANDB_ID

mkdir -p \
  "$RECON_DIR/wandb" \
  "$ADV_DIR/wandb" \
  "$STAGE2_DIR/wandb" \
  "$(dirname "$TOKEN_CACHE")" \
  "$RUN_ROOT/logs" \
  "$TMPDIR"
printf '%s\n' "$$" > "$RUN_ROOT/driver.pid"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "FFHQ data directory not found: $DATA_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"

latest_checkpoint() {
  local root="$1"
  local name="$2"
  find "$root" -type f -name "$name" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print; }'
}

wait_for_gpu_memory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_GPU_MIB )); then
      echo "GPU memory gate passed: ${free_mib} MiB free (required ${MIN_FREE_GPU_MIB} MiB)."
      return 0
    fi
    echo "Waiting for GPU memory: ${free_mib:-unknown} MiB free; need ${MIN_FREE_GPU_MIB} MiB."
    sleep "$GPU_WAIT_SECONDS"
  done
}

COMMON_STAGE1_ARGS=(
  model=laser_image_nonpatch_d5
  data=ffhq
  seed=42
  "data.data_dir=${DATA_DIR}"
  data.image_size=256
  data.batch_size=16
  data.eval_batch_size=16
  data.num_workers=8
  data.train_crop_size=null
  train.accelerator=gpu
  train.num_nodes=1
  train.devices=1
  train.strategy=auto
  train.precision=bf16-mixed
  train.max_steps=-1
  train.learning_rate=4.0e-5
  train.beta=0.5
  train.beta2=0.9
  train.min_lr_ratio=1.0
  train.accumulate_grad_batches=1
  train.gradient_clip_val=1.0
  train.deterministic=false
  train.limit_train_batches=1.0
  train.limit_val_batches=512
  train.limit_test_batches=0
  train.val_check_interval=5000
  train.run_test_after_fit=false
  train.compute_rfid_after_fit=true
  train.rfid_split=val
  train.rfid_batch_size=32
  train.rfid_num_workers=4
  train.rfid_max_samples=0
  train.rfid_device=auto
  train.rfid_feature=2048
  model.backbone=ddpm
  model.num_downsamples=5
  'model.channel_multipliers=[1,1,2,2,4,4]'
  model.backbone_latent_channels=256
  'model.attn_resolutions=[16]'
  model.decoder_extra_residual_layers=0
  model.use_mid_attention=true
  model.dropout=0.0
  model.num_hiddens=128
  model.num_residual_blocks=2
  model.num_residual_hiddens=96
  model.num_embeddings=256
  model.embedding_dim=128
  model.sparsity_level=2
  model.patch_based=false
  model.patch_size=1
  model.patch_stride=1
  model.dict_learning_rate=4.0e-5
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
  model.perceptual_weight=1.0
  model.perceptual_start_step=0
  model.perceptual_warmup_steps=0
  model.compute_fid=true
  model.log_images_every_n_steps=1000
  model.enable_val_latent_visuals=true
  model.codebook_visual_max_vectors=256
  checkpoint.monitor=val/rfid
  checkpoint.mode=min
  checkpoint.save_top_k=3
  checkpoint.save_last=true
  checkpoint.every_n_epochs=1
  checkpoint.upload_to_wandb=true
  checkpoint.upload_every_n_epochs=25
  wandb.project=laser
  wandb.append_timestamp=false
)

run_reconstruction_stage() {
  local final_ckpt last_ckpt
  final_ckpt="$(latest_checkpoint "$RECON_DIR" final.ckpt)"
  if [[ -n "$final_ckpt" ]]; then
    echo "Reusing completed 150-epoch reconstruction checkpoint: $final_ckpt"
    return 0
  fi

  last_ckpt="$(latest_checkpoint "$RECON_DIR" last.ckpt)"
  local resume_arg=()
  if [[ -n "$last_ckpt" ]]; then
    echo "Resuming reconstruction training from: $last_ckpt"
    resume_arg=("ckpt_path=${last_ckpt}")
  fi

  wait_for_gpu_memory
  echo "Starting stage 1a: 256 atoms, k=2, 150 epochs, no adversarial loss."
  python train.py stage1 \
    "${COMMON_STAGE1_ARGS[@]}" \
    "output_dir=${RECON_DIR}" \
    "hydra.run.dir=${RECON_DIR}/hydra" \
    train.max_epochs=150 \
    train.warmup_steps=19690 \
    model.data_init_from_first_batch=true \
    model.adversarial_weight=0.0 \
    model.adversarial_start_step=1000000000 \
    model.adversarial_warmup_steps=0 \
    model.disc_start_step=1000000000 \
    "wandb.id=${RECON_WANDB_ID}" \
    wandb.resume=allow \
    "wandb.name=ffhq-stage1-recon150-a256k2-${STAMP}" \
    "wandb.group=${GROUP}" \
    'wandb.tags=[stage1,ffhq,laser,tokenizer,a256,k2,reconstruction,nonadversarial,recon150,lpips1,dead_atom_revival,batch16]' \
    "wandb.save_dir=${RECON_DIR}/wandb" \
    "${resume_arg[@]}" \
    2>&1 | tee -a "$RUN_ROOT/logs/stage1_recon150.log"
}

run_adversarial_stage() {
  local recon_ckpt final_ckpt last_ckpt
  recon_ckpt="$(latest_checkpoint "$RECON_DIR" final.ckpt)"
  if [[ -z "$recon_ckpt" ]]; then
    echo "Missing reconstruction checkpoint under $RECON_DIR" >&2
    exit 1
  fi

  final_ckpt="$(latest_checkpoint "$ADV_DIR" final.ckpt)"
  if [[ -n "$final_ckpt" ]]; then
    echo "Reusing completed 50-epoch adversarial checkpoint: $final_ckpt"
    return 0
  fi

  last_ckpt="$(latest_checkpoint "$ADV_DIR" last.ckpt)"
  local start_arg
  if [[ -n "$last_ckpt" ]]; then
    echo "Resuming adversarial training from: $last_ckpt"
    start_arg="ckpt_path=${last_ckpt}"
  else
    echo "Initializing adversarial training from: $recon_ckpt"
    start_arg="init_ckpt_path=${recon_ckpt}"
  fi

  wait_for_gpu_memory
  echo "Starting stage 1b: 50-epoch adversarial continuation."
  python train.py stage1 \
    "${COMMON_STAGE1_ARGS[@]}" \
    "output_dir=${ADV_DIR}" \
    "hydra.run.dir=${ADV_DIR}/hydra" \
    train.max_epochs=50 \
    train.warmup_steps=0 \
    model.data_init_from_first_batch=false \
    model.adversarial_weight=0.75 \
    model.adversarial_start_step=0 \
    model.adversarial_warmup_steps=0 \
    model.disc_start_step=0 \
    model.disc_learning_rate=4.0e-5 \
    model.discriminator_beta1=0.5 \
    model.discriminator_beta2=0.9 \
    model.disc_channels=64 \
    model.disc_num_layers=2 \
    model.disc_norm=none \
    model.disc_loss=hinge \
    model.use_adaptive_disc_weight=true \
    "wandb.id=${ADV_WANDB_ID}" \
    wandb.resume=allow \
    "wandb.name=ffhq-stage1-adv50-a256k2-from-recon150-${STAMP}" \
    "wandb.group=${GROUP}" \
    'wandb.tags=[stage1,ffhq,laser,tokenizer,a256,k2,adversarial,adv50,from_recon150,lpips1,dead_atom_revival,batch16]' \
    "wandb.save_dir=${ADV_DIR}/wandb" \
    "$start_arg" \
    2>&1 | tee -a "$RUN_ROOT/logs/stage1_adv50.log"
}

run_stage2() {
  local stage1_ckpt stage2_last
  stage1_ckpt="$(latest_checkpoint "$ADV_DIR" final.ckpt)"
  if [[ -z "$stage1_ckpt" ]]; then
    echo "Missing adversarial checkpoint under $ADV_DIR" >&2
    exit 1
  fi

  stage2_last="$(latest_checkpoint "$STAGE2_DIR" last.ckpt)"
  local resume_arg=()
  if [[ -n "$stage2_last" ]]; then
    echo "Resuming stage 2 from: $stage2_last"
    resume_arg=("ckpt_path=${stage2_last}")
  fi

  wait_for_gpu_memory
  echo "Starting stage 2: quantized sparse spatial-depth prior for 1000 epochs."
  python train.py stage2 \
    "token_cache_path=${TOKEN_CACHE}" \
    "output_dir=${STAGE2_DIR}" \
    seed=42 \
    token_cache.build=true \
    token_cache.force=false \
    "token_cache.stage1_checkpoint=${stage1_ckpt}" \
    "token_cache.output=${TOKEN_CACHE}" \
    token_cache.split=train \
    token_cache.cache_mode=quantized \
    token_cache.coeff_vocab_size=256 \
    token_cache.coeff_max=auto \
    token_cache.coeff_quantization=quantile \
    token_cache.coeff_calibration_percentile=99.5 \
    token_cache.coeff_mu=0.0 \
    token_cache.batch_size=64 \
    token_cache.num_workers=24 \
    token_cache.max_items=0 \
    token_cache.device=auto \
    data.dataset=ffhq \
    "data.data_dir=${DATA_DIR}" \
    data.image_size=256 \
    data.num_workers=8 \
    ar.type=sparse_spatial_depth \
    ar.autoregressive_coeffs=true \
    ar.class_conditional=false \
    ar.vocab_size=null \
    ar.atom_vocab_size=null \
    ar.coeff_vocab_size=null \
    ar.window_sites=0 \
    ar.n_global_spatial_tokens=16 \
    ar.d_model=768 \
    ar.n_heads=12 \
    ar.n_layers=18 \
    ar.d_ff=3072 \
    ar.dropout=0.1 \
    ar.learning_rate=1.25e-4 \
    ar.weight_decay=0.01 \
    ar.warmup_steps=5000 \
    ar.max_steps=-1 \
    ar.min_lr_ratio=0.08 \
    ar.atom_loss_weight=1.0 \
    ar.coeff_loss_weight=1.0 \
    ar.coeff_depth_weighting=none \
    ar.coeff_focal_gamma=0.0 \
    ar.atom_label_smoothing=0.0 \
    ar.atom_coverage_weight=0.0 \
    ar.coeff_loss_type=auto \
    ar.coeff_huber_delta=0.25 \
    train_ar.max_epochs=1000 \
    train_ar.batch_size=32 \
    train_ar.max_items=0 \
    train_ar.limit_train_batches=1.0 \
    train_ar.limit_val_batches=1.0 \
    train_ar.limit_test_batches=0 \
    train_ar.val_check_interval=1.0 \
    train_ar.validation_split=0.05 \
    train_ar.test_split=0.05 \
    train_ar.log_every_n_steps=20 \
    train_ar.devices=1 \
    train_ar.num_nodes=1 \
    train_ar.strategy=auto \
    train_ar.precision=bf16-mixed \
    train_ar.accelerator=gpu \
    train_ar.deterministic=false \
    train_ar.accumulate_grad_batches=1 \
    train_ar.gradient_clip_val=1.0 \
    train_ar.checkpoint_save_top_k=1 \
    train_ar.checkpoint_save_last=true \
    train_ar.checkpoint_keep_recent=3 \
    train_ar.checkpoint_every_n_epochs=5 \
    +train_ar.checkpoint_upload_to_wandb=true \
    +train_ar.checkpoint_upload_every_n_epochs=25 \
    train_ar.sample_every_n_epochs=20 \
    train_ar.sample_every_n_steps=0 \
    train_ar.sample_log_to_wandb=true \
    train_ar.sample_num_images=8 \
    train_ar.sample_temperature=0.7 \
    train_ar.sample_top_k=0 \
    train_ar.compute_generation_fid=false \
    train_ar.compute_audio_generation_metrics=false \
    train_ar.generation_metric_num_samples=32 \
    train_ar.run_test_after_fit=false \
    train_ar.save_final_samples_after_fit=true \
    wandb.project=laser \
    "wandb.name=ffhq-stage2-1000-a256k2-from-adv50-${STAMP}" \
    "wandb.id=${STAGE2_WANDB_ID}" \
    wandb.resume=allow \
    "wandb.group=${GROUP}" \
    'wandb.tags=[stage2,ffhq,laser,sparse_spatial_depth,unconditional,a256,k2,from_adv50,stage2_1000,quantized_coeffs,h100]' \
    wandb.append_timestamp=false \
    "wandb.save_dir=${STAGE2_DIR}/wandb" \
    "${resume_arg[@]}" \
    2>&1 | tee -a "$RUN_ROOT/logs/stage2_1000.log"
}

cat > "$RUN_ROOT/run.info" <<EOF
run_root=$RUN_ROOT
stamp=$STAMP
group=$GROUP
recon_wandb_id=$RECON_WANDB_ID
adv_wandb_id=$ADV_WANDB_ID
stage2_wandb_id=$STAGE2_WANDB_ID
dictionary_atoms=256
sparsity_level=2
stage1_reconstruction_epochs=150
stage1_adversarial_epochs=50
stage2_epochs=1000
EOF

echo "Pipeline root: $RUN_ROOT"
echo "W&B group: $GROUP"
run_reconstruction_stage
run_adversarial_stage
run_stage2
echo "Pipeline complete: $RUN_ROOT"
