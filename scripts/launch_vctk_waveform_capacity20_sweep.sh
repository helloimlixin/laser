#!/usr/bin/env bash
# Submit VCTK waveform 20/20 capacity sweeps for LASER and VQ-VAE.
#
# The sweep uses two waveform downsampling stages. A total 64x schedule keeps
# the stage-2 sequence manageable while avoiding the over-compressed 4-stage
# 128x bottleneck that was hurting audio quality.
#
# LASER stage 2 uses quantized sparse coefficients by default. The previous
# raw-coefficient setup trained a deterministic coefficient regressor
# (variational_coeffs=false), so generation collapsed toward near-zero
# coefficient means and produced almost silent audio.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
GPUS="${GPUS:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEM_MB="${MEM_MB:-240000}"
PROJECT="${PROJECT:-laser-debugging}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/vctk_waveform_down2_capacity20}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/datasets/VCTK-Corpus-0.92}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu018}"
RUN_LASER="${RUN_LASER:-1}"
RUN_VQVAE="${RUN_VQVAE:-1}"
LASER_COEFF_BINS="${LASER_COEFF_BINS:-512}"
LASER_COEFF_MAX="${LASER_COEFF_MAX:-16.0}"
LASER_COEFF_QUANTIZATION="${LASER_COEFF_QUANTIZATION:-uniform}"
VCTK_STAGE1_WARMUP_STEPS="${VCTK_STAGE1_WARMUP_STEPS:-750}"
VCTK_STAGE2_WARMUP_STEPS="${VCTK_STAGE2_WARMUP_STEPS:-750}"
VCTK_MIN_LR_RATIO="${VCTK_MIN_LR_RATIO:-0.05}"
VCTK_LASER_LR="${VCTK_LASER_LR:-1.5e-4}"
VCTK_LASER_DICT_LR="${VCTK_LASER_DICT_LR:-1.5e-4}"
VCTK_LASER_STAGE2_LR="${VCTK_LASER_STAGE2_LR:-4.0e-4}"
VCTK_VQVAE_LR="${VCTK_VQVAE_LR:-3.0e-4}"
VCTK_VQVAE_STAGE2_LR="${VCTK_VQVAE_STAGE2_LR:-4.0e-4}"

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
  --full-training
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
  --partition "$PARTITION"
  --time-limit "$TIME_LIMIT"
  --gpus "$GPUS"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem-mb "$MEM_MB"
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
  --vctk-dir "$VCTK_DIR"
  --stage1-override data=vctk_waveform
  --stage1-override data.data_dir="$VCTK_DIR"
  --stage1-override data.batch_size=4
  --stage1-override data.num_workers=4
  --stage1-override model.compute_fid=false
  --stage1-override model.audio_downsample_rates=[8,8]
  --stage1-override model.audio_waveform_l1_weight=1.0
  --stage1-override model.audio_multires_stft_loss_weight=1.0
  --stage1-override model.audio_multires_stft_fft_sizes=[512,1024,2048]
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
  --stage2-override ar.type=sparse_spatial_depth
  --stage2-override ar.d_model=384
  --stage2-override ar.n_heads=6
  --stage2-override ar.n_layers=8
  --stage2-override ar.d_ff=1536
  --stage2-override ar.warmup_steps="$VCTK_STAGE2_WARMUP_STEPS"
  --stage2-override ar.min_lr_ratio="$VCTK_MIN_LR_RATIO"
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
    --model-family laser \
    --run-label "vctk-down2-cap20-laser-${case_name}" \
    --stage1-override model=laser_audio_waveform \
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
    --stage1-override model.coef_max=16.0 \
    --stage1-override model.sparsity_reg_weight=0.0 \
    --stage1-override model.recon_mse_weight=0.5 \
    --stage1-override model.recon_l1_weight=0.5 \
    --stage1-override train.learning_rate="$VCTK_LASER_LR" \
    --stage1-override train.precision=bf16-mixed \
    --stage2-override train_ar.batch_size="$s2_batch" \
    --stage2-override ar.d_model="$d_model" \
    --stage2-override ar.n_heads="$n_heads" \
    --stage2-override ar.n_layers="$n_layers" \
    --stage2-override ar.d_ff="$d_ff" \
    --stage2-override ar.learning_rate="$VCTK_LASER_STAGE2_LR" \
    --stage2-override ar.coeff_loss_type=huber \
    --stage2-override ar.coeff_loss_weight=1.0 \
    --stage2-override ar.coeff_huber_delta=1.0 \
    --stage2-override train_ar.sample_coeff_mode=mean \
    --cache-arg=--coeff-bins \
    --cache-arg="$LASER_COEFF_BINS" \
    --cache-arg=--coeff-max \
    --cache-arg="$LASER_COEFF_MAX" \
    --cache-arg=--coeff-quantization \
    --cache-arg="$LASER_COEFF_QUANTIZATION"
}

submit_vqvae() {
  local case_name="$1"
  local atoms="$2"
  local zdim="$3"
  local hidden="$4"
  local res_hidden="$5"
  local batch="$6"
  local s2_batch="$7"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --model-family vqvae \
    --run-label "vctk-down2-cap20-vqvae-${case_name}" \
    --stage1-override model=vqvae_audio_waveform \
    --stage1-override data.batch_size="$batch" \
    --stage1-override model.num_embeddings="$atoms" \
    --stage1-override model.embedding_dim="$zdim" \
    --stage1-override model.num_hiddens="$hidden" \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens="$res_hidden" \
    --stage1-override model.commitment_cost=0.25 \
    --stage1-override model.decay=0.95 \
    --stage1-override model.codebook_init=true \
    --stage1-override model.dead_code_threshold=1.0 \
    --stage1-override train.learning_rate="$VCTK_VQVAE_LR" \
    --stage1-override train.precision=32 \
    --stage2-override train_ar.batch_size="$s2_batch" \
    --stage2-override ar.learning_rate="$VCTK_VQVAE_STAGE2_LR"
}

if [[ "$RUN_LASER" == "1" || "$RUN_LASER" == "true" ]]; then
  submit_laser ds64-a8192-d96-s8 8192 96 192 96 8 8 4 384 6 8 1536
  submit_laser ds64-a16384-d128-s12 16384 128 224 112 12 6 4 512 8 10 2048
fi
if [[ "$RUN_VQVAE" == "1" || "$RUN_VQVAE" == "true" ]]; then
  submit_vqvae ds64-a8192-d96 8192 96 192 96 12 8
  submit_vqvae ds64-a16384-d128 16384 128 224 112 10 8
fi
