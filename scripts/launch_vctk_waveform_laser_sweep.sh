#!/usr/bin/env bash
# Submit raw-waveform VCTK LASER two-stage runs from maintained src.
#
# Use quantized sparse coefficients by default. Raw real-valued coefficient
# caches require a variational coefficient head to sample well; without it,
# stage-2 generation predicts near-zero coefficient means and can become silent.

set -euo pipefail

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
PROJECT="${PROJECT:-laser}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/vctk_src_sweeps}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/datasets/VCTK-Corpus-0.92}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
VCTK_LASER_STAGE1_BATCH="${VCTK_LASER_STAGE1_BATCH:-12}"
VCTK_LASER_STAGE2_BATCH="${VCTK_LASER_STAGE2_BATCH:-8}"
VCTK_LASER_STAGE1_LR="${VCTK_LASER_STAGE1_LR:-4.0e-4}"
VCTK_LASER_STAGE2_LR="${VCTK_LASER_STAGE2_LR:-4.0e-4}"
VCTK_LASER_STAGE1_WARMUP_STEPS="${VCTK_LASER_STAGE1_WARMUP_STEPS:-750}"
VCTK_LASER_STAGE2_WARMUP_STEPS="${VCTK_LASER_STAGE2_WARMUP_STEPS:-750}"
VCTK_LASER_MIN_LR_RATIO="${VCTK_LASER_MIN_LR_RATIO:-0.05}"
VCTK_LASER_AR_D_MODEL="${VCTK_LASER_AR_D_MODEL:-512}"
VCTK_LASER_AR_N_HEADS="${VCTK_LASER_AR_N_HEADS:-8}"
VCTK_LASER_AR_N_LAYERS="${VCTK_LASER_AR_N_LAYERS:-8}"
VCTK_LASER_AR_D_FF="${VCTK_LASER_AR_D_FF:-2048}"
VCTK_LASER_COEFF_LOSS_WEIGHT="${VCTK_LASER_COEFF_LOSS_WEIGHT:-1.0}"
VCTK_LASER_COEFF_BINS="${VCTK_LASER_COEFF_BINS:-512}"
VCTK_LASER_COEFF_MAX="${VCTK_LASER_COEFF_MAX:-8.0}"
VCTK_LASER_COEFF_QUANTIZATION="${VCTK_LASER_COEFF_QUANTIZATION:-uniform}"

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
  --stage1-override model=laser_audio_waveform
  --stage1-override data=vctk_waveform
  --stage1-override data.num_workers=4
  --stage1-override data.batch_size="$VCTK_LASER_STAGE1_BATCH"
  --stage1-override train.learning_rate="$VCTK_LASER_STAGE1_LR"
  --stage1-override train.warmup_steps="$VCTK_LASER_STAGE1_WARMUP_STEPS"
  --stage1-override train.min_lr_ratio="$VCTK_LASER_MIN_LR_RATIO"
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override train.precision=bf16-mixed
  --stage2-override data.num_workers=2
  --stage2-override train_ar.batch_size="$VCTK_LASER_STAGE2_BATCH"
  --stage2-override train_ar.sample_num_images=8
  --stage2-override train_ar.generation_metric_num_samples=8
  --stage2-override ar.d_model="$VCTK_LASER_AR_D_MODEL"
  --stage2-override ar.n_heads="$VCTK_LASER_AR_N_HEADS"
  --stage2-override ar.n_layers="$VCTK_LASER_AR_N_LAYERS"
  --stage2-override ar.d_ff="$VCTK_LASER_AR_D_FF"
  --stage2-override ar.learning_rate="$VCTK_LASER_STAGE2_LR"
  --stage2-override ar.warmup_steps="$VCTK_LASER_STAGE2_WARMUP_STEPS"
  --stage2-override ar.min_lr_ratio="$VCTK_LASER_MIN_LR_RATIO"
  --stage2-override ar.coeff_loss_type=huber
  --stage2-override ar.coeff_loss_weight="$VCTK_LASER_COEFF_LOSS_WEIGHT"
  --stage2-override ar.coeff_huber_delta=1.0
  --stage2-override train_ar.sample_temperature=0.8
  --stage2-override train_ar.sample_coeff_mode=mean
  --cache-arg=--audio-representation
  --cache-arg=waveform
  --cache-arg=--num-workers
  --cache-arg=4
  --cache-arg=--coeff-bins
  --cache-arg="$VCTK_LASER_COEFF_BINS"
  --cache-arg=--coeff-max
  --cache-arg="$VCTK_LASER_COEFF_MAX"
  --cache-arg=--coeff-quantization
  --cache-arg="$VCTK_LASER_COEFF_QUANTIZATION"
)

CASES=(
  "ds64_a1024_d64_s8|[4,4,4]|1024|64|128|3|64|8"
  "ds64_a2048_d64_s8|[4,4,4]|2048|64|128|3|64|8"
  "ds64_a1024_d96_s8|[4,4,4]|1024|96|160|3|80|8"
  "ds32_a1024_d64_s8|[4,4,2]|1024|64|128|3|64|8"
)

for case_def in "${CASES[@]}"; do
  IFS="|" read -r case_name rates num_embeddings embedding_dim num_hiddens residual_blocks residual_hiddens sparsity <<<"$case_def"
  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --run-label "vctk-waveform-laser-${case_name}-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}" \
    --stage1-override "model.audio_downsample_rates=${rates}" \
    --stage1-override "model.num_embeddings=${num_embeddings}" \
    --stage1-override "model.embedding_dim=${embedding_dim}" \
    --stage1-override "model.num_hiddens=${num_hiddens}" \
    --stage1-override "model.num_residual_blocks=${residual_blocks}" \
    --stage1-override "model.num_residual_hiddens=${residual_hiddens}" \
    --stage1-override "model.sparsity_level=${sparsity}"
done
