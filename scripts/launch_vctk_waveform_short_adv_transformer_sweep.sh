#!/usr/bin/env bash
# Submit short-sequence VCTK waveform LASER runs:
#   stage 1 reconstruction -> stage 1 adversarial continuation -> transformer prior.
#
# The [8,8] waveform downsampling schedule gives a 64x latent stride. For the
# default 32,768-sample VCTK crop this keeps the stage-2 spatial sequence at 512
# sites before sparse depth expansion.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
GPUS="${GPUS:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEM_MB="${MEM_MB:-240000}"
PROJECT="${PROJECT:-laser-vctk-short-adv}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/vctk_waveform_short_adv_transformer}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/Projects/data/VCTK-Corpus}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE1_ADV_EPOCHS="${STAGE1_ADV_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-60}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018,gpuk[005-018]}"

LASER_COEFF_BINS="${LASER_COEFF_BINS:-512}"
LASER_COEFF_MAX="${LASER_COEFF_MAX:-16.0}"
LASER_COEFF_QUANTIZATION="${LASER_COEFF_QUANTIZATION:-uniform}"
# Within-site support ordering for the token cache. "magnitude" sorts each site's
# atoms by descending |coeff| (matching-pursuit order) -> far more learnable stage-2
# prior than the legacy "atom_id" order. The prior auto-detects this from cache meta.
LASER_SUPPORT_ORDER="${LASER_SUPPORT_ORDER:-magnitude}"
VCTK_STAGE1_WARMUP_STEPS="${VCTK_STAGE1_WARMUP_STEPS:-750}"
VCTK_STAGE2_WARMUP_STEPS="${VCTK_STAGE2_WARMUP_STEPS:-750}"
VCTK_MIN_LR_RATIO="${VCTK_MIN_LR_RATIO:-0.05}"
VCTK_LASER_LR="${VCTK_LASER_LR:-1.5e-4}"
VCTK_LASER_DICT_LR="${VCTK_LASER_DICT_LR:-1.5e-4}"
VCTK_LASER_STAGE2_LR="${VCTK_LASER_STAGE2_LR:-4.0e-4}"
VCTK_ADV_WEIGHT="${VCTK_ADV_WEIGHT:-0.03}"
VCTK_DISC_LR="${VCTK_DISC_LR:-5.0e-5}"
VCTK_DISC_CHANNELS="${VCTK_DISC_CHANNELS:-32}"
VCTK_DISC_LAYERS="${VCTK_DISC_LAYERS:-3}"
VCTK_AUDIO_DISC_NUM_SCALES="${VCTK_AUDIO_DISC_NUM_SCALES:-3}"
VCTK_AUDIO_DISC_MAX_CHANNELS="${VCTK_AUDIO_DISC_MAX_CHANNELS:-512}"

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi

if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

if [[ ! -d "$VCTK_DIR" ]]; then
  echo "VCTK directory not found: $VCTK_DIR" >&2
  exit 1
fi

COMMON_ARGS=(
  --cases vctk
  --model-family laser
  --full-training
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage1-adv-epochs "$STAGE1_ADV_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
  --stage2-kind transformer
  --partition "$PARTITION"
  --time-limit "$TIME_LIMIT"
  --gpus "$GPUS"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem-mb "$MEM_MB"
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
  --vctk-dir "$VCTK_DIR"
  --stage1-override model=laser_audio_waveform
  --stage1-override data=vctk_waveform
  --stage1-override data.data_dir="$VCTK_DIR"
  --stage1-override data.batch_size=4
  --stage1-override data.num_workers=4
  --stage1-override model.compute_fid=false
  --stage1-override model.audio_downsample_rates=[8,8]
  --stage1-override model.audio_waveform_l1_weight=1.0
  --stage1-override model.audio_multires_stft_loss_weight=1.0
  --stage1-override model.audio_multires_stft_fft_sizes=[512,1024,2048]
  --stage1-override model.out_tanh=true
  --stage1-override data.audio_dc_remove=true
  --stage1-override data.audio_peak_normalize=true
  --stage1-override data.audio_target_peak=0.95
  --stage1-override data.audio_rms_normalize=true
  --stage1-override data.audio_target_rms=0.12
  --stage1-override data.audio_max_gain=8.0
  --stage1-override data.audio_min_crop_rms=0.03
  --stage1-override data.audio_crop_attempts=64
  --stage1-override data.audio_fade_samples=1024
  --stage1-override train.limit_train_batches=1.0
  --stage1-override train.limit_val_batches=1.0
  --stage1-override train.limit_test_batches=1.0
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override train.val_check_interval=1.0
  --stage1-override train.warmup_steps="$VCTK_STAGE1_WARMUP_STEPS"
  --stage1-override train.min_lr_ratio="$VCTK_MIN_LR_RATIO"
  --stage1-override model.adversarial_weight=0.0
  --stage1-adv-override model.adversarial_weight="$VCTK_ADV_WEIGHT"
  --stage1-adv-override model.adversarial_start_step=0
  --stage1-adv-override model.adversarial_warmup_steps=0
  --stage1-adv-override model.disc_start_step=0
  --stage1-adv-override model.audio_adversarial_type=hifigan
  --stage1-adv-override model.disc_channels="$VCTK_DISC_CHANNELS"
  --stage1-adv-override model.disc_num_layers="$VCTK_DISC_LAYERS"
  --stage1-adv-override model.audio_disc_num_scales="$VCTK_AUDIO_DISC_NUM_SCALES"
  --stage1-adv-override model.audio_disc_max_channels="$VCTK_AUDIO_DISC_MAX_CHANNELS"
  --stage1-adv-override model.disc_learning_rate="$VCTK_DISC_LR"
  --stage1-adv-override model.disc_factor=1.0
  --stage1-adv-override model.use_adaptive_disc_weight=true
  --stage2-override data.dataset=vctk
  --stage2-override data.data_dir="$VCTK_DIR"
  --stage2-override data.num_workers=2
  --stage2-override train_ar.max_items=0
  --stage2-override train_ar.limit_train_batches=1.0
  --stage2-override train_ar.limit_val_batches=1.0
  --stage2-override train_ar.limit_test_batches=1.0
  --stage2-override train_ar.sample_every_n_epochs=1
  --stage2-override train_ar.sample_log_to_wandb=true
  --stage2-override train_ar.sample_num_images=8
  --stage2-override train_ar.generation_metric_num_samples=8
  --stage2-override train_ar.compute_generation_fid=false
  --stage2-override train_ar.compute_audio_generation_metrics=true
  --stage2-override train_ar.sample_temperature=0.8
  --stage2-override train_ar.sample_top_k=0
  --stage2-override train_ar.sample_coeff_mode=mean
  --stage2-override ar.type=sparse_spatial_depth
  --stage2-override ar.warmup_steps="$VCTK_STAGE2_WARMUP_STEPS"
  --stage2-override ar.min_lr_ratio="$VCTK_MIN_LR_RATIO"
  --stage2-override ar.learning_rate="$VCTK_LASER_STAGE2_LR"
  --stage2-override ar.coeff_loss_type=huber
  --stage2-override ar.coeff_loss_weight=1.0
  --stage2-override ar.coeff_huber_delta=1.0
  --cache-arg=--audio-representation
  --cache-arg=waveform
  --cache-arg=--audio-dc-remove
  --cache-arg=--audio-peak-normalize
  --cache-arg=--audio-target-peak
  --cache-arg=0.95
  --cache-arg=--audio-rms-normalize
  --cache-arg=--audio-target-rms
  --cache-arg=0.12
  --cache-arg=--audio-max-gain
  --cache-arg=8.0
  --cache-arg=--audio-min-crop-rms
  --cache-arg=0.03
  --cache-arg=--audio-crop-attempts
  --cache-arg=64
  --cache-arg=--audio-fade-samples
  --cache-arg=1024
  --cache-arg=--coeff-bins
  --cache-arg="$LASER_COEFF_BINS"
  --cache-arg=--coeff-max
  --cache-arg="$LASER_COEFF_MAX"
  --cache-arg=--coeff-quantization
  --cache-arg="$LASER_COEFF_QUANTIZATION"
  --cache-arg=--support-order
  --cache-arg="$LASER_SUPPORT_ORDER"
)
if [[ -n "${EXCLUDE_NODES// }" ]]; then
  COMMON_ARGS+=(--exclude-nodes "$EXCLUDE_NODES")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

submit_laser() {
  local case_name="$1"
  local atoms="$2"
  local zdim="$3"
  local hidden="$4"
  local res_hidden="$5"
  local sparsity="$6"
  local batch="$7"
  local s2_batch="$8"
  local d_model="$9"
  local n_heads="${10}"
  local n_layers="${11}"
  local d_ff="${12}"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --run-label "vctk-shortseq-adv-transformer-${case_name}" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override model.num_embeddings="$atoms" \
    --stage1-override model.embedding_dim="$zdim" \
    --stage1-override model.num_hiddens="$hidden" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens="$res_hidden" \
    --stage1-override model.sparsity_level="$sparsity" \
    --stage1-override model.commitment_cost=1.0 \
    --stage1-override model.bottleneck_loss_weight=0.75 \
    --stage1-override model.dict_learning_rate="$VCTK_LASER_DICT_LR" \
    --stage1-override model.coef_max="$LASER_COEFF_MAX" \
    --stage1-override model.sparsity_reg_weight=0.0 \
    --stage1-override model.recon_mse_weight=0.5 \
    --stage1-override model.recon_l1_weight=0.5 \
    --stage1-override train.learning_rate="$VCTK_LASER_LR" \
    --stage1-override train.precision=bf16-mixed \
    --stage2-override train_ar.batch_size="$s2_batch" \
    --stage2-override ar.d_model="$d_model" \
    --stage2-override ar.n_heads="$n_heads" \
    --stage2-override ar.n_layers="$n_layers" \
    --stage2-override ar.d_ff="$d_ff"
}

submit_laser ds64-a8192-d96-s8 8192 96 192 96 8 8 4 384 6 8 1536
submit_laser ds64-a16384-d128-s12 16384 128 224 112 12 6 4 512 8 10 2048
