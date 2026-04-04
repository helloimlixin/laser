#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_patch_nooverlap_largepatch_quick_sweep}"
PARTITION="${PARTITION:-gpu-redhat}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"

ENTRYPOINT="${ENTRYPOINT:-laser.py}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE1_LR="${STAGE1_LR:-2e-4}"
STAGE1_LR_SCHEDULE="${STAGE1_LR_SCHEDULE:-cosine}"
STAGE1_WARMUP_EPOCHS="${STAGE1_WARMUP_EPOCHS:-1}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-500}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.1}"
STAGE2_WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-4}"
TOKEN_SUBSET="${TOKEN_SUBSET:-98304}"
AE_NUM_DOWNSAMPLES="${AE_NUM_DOWNSAMPLES:-2}"
NUM_HIDDENS="${NUM_HIDDENS:-64}"
NUM_RES_HIDDENS="${NUM_RES_HIDDENS:-32}"
NUM_RES_LAYERS="${NUM_RES_LAYERS:-1}"
PATCH_BASED="${PATCH_BASED:-true}"
PATCH_RECONSTRUCTION="${PATCH_RECONSTRUCTION:-tile}"
STAGE2_SAMPLE_EVERY_STEPS="${STAGE2_SAMPLE_EVERY_STEPS:-500}"
STAGE2_SAMPLE_START_STEP="${STAGE2_SAMPLE_START_STEP:-0}"
STAGE2_SAMPLE_BATCH_SIZE="${STAGE2_SAMPLE_BATCH_SIZE:-4}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
RUN_PREFIX="${RUN_PREFIX:-celebahq256_patch_noov_largepatch_5ep}"
JOB_PREFIX="${JOB_PREFIX:-chq256-pnolp}"
CASE_FILTER="${CASE_FILTER:-}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# name | patch_size | patch_stride | quantize | num_atoms | sparsity | n_bins | coef_max | batch_size | stage2_batch | stage2_lr
cases=(
  "p8r|8|8|false|4096|8|256|8.0|24|24|6e-4"
  "p8q|8|8|true|4096|8|512|8.0|24|24|6e-4"
  "p16r|16|16|false|6144|12|256|8.0|8|32|8e-4"
  "p16q|16|16|true|6144|12|512|8.0|8|32|8e-4"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name patch_size patch_stride quantize_sparse_coeffs num_atoms sparsity_level n_bins coef_max batch_size stage2_batch_size stage2_lr <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}_p${patch_size}s${patch_stride}_a${num_atoms}_k${sparsity_level}"
  job_name="${JOB_PREFIX}-${case_name}"
  out_dir="$OUT_ROOT/$run_name"

  echo "[Sweep] submitting $run_name"
  submit_output="$(
    DATA_DIR="$DATA_DIR" \
    OUT_DIR="$out_dir" \
    PARTITION="$PARTITION" \
    NODES="$NODES" \
    GPUS_PER_NODE="$GPUS_PER_NODE" \
    CPUS_PER_TASK="$CPUS_PER_TASK" \
    MEM_MB="$MEM_MB" \
    TIME_LIMIT="$TIME_LIMIT" \
    ENTRYPOINT="$ENTRYPOINT" \
    IMAGE_SIZE="$IMAGE_SIZE" \
    STAGE1_EPOCHS="$STAGE1_EPOCHS" \
    STAGE2_EPOCHS="$STAGE2_EPOCHS" \
    STAGE1_LR="$STAGE1_LR" \
    STAGE1_LR_SCHEDULE="$STAGE1_LR_SCHEDULE" \
    STAGE1_WARMUP_EPOCHS="$STAGE1_WARMUP_EPOCHS" \
    STAGE2_WARMUP_STEPS="$STAGE2_WARMUP_STEPS" \
    STAGE2_MIN_LR_RATIO="$STAGE2_MIN_LR_RATIO" \
    STAGE2_WEIGHT_DECAY="$STAGE2_WEIGHT_DECAY" \
    BATCH_SIZE="$batch_size" \
    STAGE2_BATCH_SIZE="$stage2_batch_size" \
    NUM_WORKERS="$NUM_WORKERS" \
    TOKEN_NUM_WORKERS="$TOKEN_NUM_WORKERS" \
    TOKEN_SUBSET="$TOKEN_SUBSET" \
    AE_NUM_DOWNSAMPLES="$AE_NUM_DOWNSAMPLES" \
    NUM_HIDDENS="$NUM_HIDDENS" \
    NUM_RES_HIDDENS="$NUM_RES_HIDDENS" \
    NUM_RES_LAYERS="$NUM_RES_LAYERS" \
    PATCH_BASED="$PATCH_BASED" \
    PATCH_SIZE="$patch_size" \
    PATCH_STRIDE="$patch_stride" \
    PATCH_RECONSTRUCTION="$PATCH_RECONSTRUCTION" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="$run_name" \
    LOG_PREFIX="$run_name" \
    JOB_NAME="$job_name" \
    NUM_ATOMS="$num_atoms" \
    SPARSITY_LEVEL="$sparsity_level" \
    QUANTIZE_SPARSE_COEFFS="$quantize_sparse_coeffs" \
    N_BINS="$n_bins" \
    COEF_MAX="$coef_max" \
    COEF_QUANTIZATION="uniform" \
    COEF_MU="0.0" \
    STAGE2_LR="$stage2_lr" \
    STAGE2_SCHED_SAMPLING_FINAL_PROB="0.0" \
    STAGE2_SAMPLE_EVERY_STEPS="$STAGE2_SAMPLE_EVERY_STEPS" \
    STAGE2_SAMPLE_START_STEP="$STAGE2_SAMPLE_START_STEP" \
    STAGE2_SAMPLE_BATCH_SIZE="$STAGE2_SAMPLE_BATCH_SIZE" \
    STAGE2_SAMPLE_IMAGE_SIZE="$IMAGE_SIZE" \
    RFID_NUM_SAMPLES="0" \
    STAGE1_AUTO_RESUME_FROM_LATEST="false" \
    "$ROOT_DIR/scripts/patch100.sh"
  )"
  printf '%s\n' "$submit_output"
  submitted=$((submitted + 1))
done

if ((submitted == 0)); then
  echo "No cases matched CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi
