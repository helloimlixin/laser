#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_patch100_arch_sweep}"
PARTITION="${PARTITION:-auto}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

ENTRYPOINT="${ENTRYPOINT:-laser.py}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-4}"

AE_NUM_DOWNSAMPLES="${AE_NUM_DOWNSAMPLES:-2}"
NUM_HIDDENS="${NUM_HIDDENS:-64}"
NUM_RES_HIDDENS="${NUM_RES_HIDDENS:-32}"
NUM_RES_LAYERS="${NUM_RES_LAYERS:-1}"
PATCH_BASED="${PATCH_BASED:-true}"
PATCH_SIZE="${PATCH_SIZE:-8}"
PATCH_STRIDE="${PATCH_STRIDE:-4}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
RUN_PREFIX="${RUN_PREFIX:-celebahq256_patch_arch100}"
JOB_PREFIX="${JOB_PREFIX:-chq256-pa100}"
CASE_FILTER="${CASE_FILTER:-}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# name | arch | patch_recon | stage2_batch | stage2_lr | warmup | min_lr_ratio | tf_d_model | tf_heads | tf_layers | tf_ff | tf_global_tokens
# Stage-2 batch sizes are sized conservatively for the 48 GB L40S nodes on gpu-redhat
# with the larger shared quantized vocab (4096 atoms + 2048 coeff bins).
cases=(
  "sd_hann|spatial_depth|hann|12|4e-4|600|0.05|512|8|12|1024|0"
  "sd_center|spatial_depth|center_crop|8|3e-4|800|0.05|512|8|12|1024|0"
  "gpt_hann|mingpt|hann|2|1.0e-4|1500|0.05|256|8|8|768|0"
  "gpt_center|mingpt|center_crop|1|7.5e-5|1800|0.05|256|8|8|768|0"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name stage2_arch patch_reconstruction stage2_batch_size stage2_lr stage2_warmup_steps stage2_min_lr_ratio tf_d_model tf_heads tf_layers tf_ff tf_global_tokens <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}_p${PATCH_SIZE}s${PATCH_STRIDE}_a4096_k16"
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
    BATCH_SIZE="$BATCH_SIZE" \
    STAGE2_BATCH_SIZE="$stage2_batch_size" \
    NUM_WORKERS="$NUM_WORKERS" \
    TOKEN_NUM_WORKERS="$TOKEN_NUM_WORKERS" \
    AE_NUM_DOWNSAMPLES="$AE_NUM_DOWNSAMPLES" \
    NUM_HIDDENS="$NUM_HIDDENS" \
    NUM_RES_HIDDENS="$NUM_RES_HIDDENS" \
    NUM_RES_LAYERS="$NUM_RES_LAYERS" \
    PATCH_BASED="$PATCH_BASED" \
    PATCH_SIZE="$PATCH_SIZE" \
    PATCH_STRIDE="$PATCH_STRIDE" \
    PATCH_RECONSTRUCTION="$patch_reconstruction" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="$run_name" \
    LOG_PREFIX="$run_name" \
    JOB_NAME="$job_name" \
    NUM_ATOMS="4096" \
    SPARSITY_LEVEL="16" \
    QUANTIZE_SPARSE_COEFFS="true" \
    N_BINS="2048" \
    COEF_MAX="20.0" \
    COEF_QUANTIZATION="uniform" \
    COEF_MU="0.0" \
    REBUILD_TOKEN_CACHE="true" \
    STAGE2_SOURCE_RUN="" \
    STAGE2_SOURCE_TOKEN_CACHE="" \
    STAGE2_SOURCE_CKPT="" \
    STAGE2_ARCH="$stage2_arch" \
    STAGE2_LR="$stage2_lr" \
    STAGE2_WARMUP_STEPS="$stage2_warmup_steps" \
    STAGE2_MIN_LR_RATIO="$stage2_min_lr_ratio" \
    STAGE2_WEIGHT_DECAY="0.01" \
    STAGE2_SCHED_SAMPLING_FINAL_PROB="0.0" \
    TF_D_MODEL="$tf_d_model" \
    TF_HEADS="$tf_heads" \
    TF_LAYERS="$tf_layers" \
    TF_FF="$tf_ff" \
    TF_GLOBAL_TOKENS="$tf_global_tokens" \
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
