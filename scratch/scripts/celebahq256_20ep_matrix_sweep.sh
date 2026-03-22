#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_20ep_matrix_sweep}"
PARTITION="${PARTITION:-auto}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

ENTRYPOINT="${ENTRYPOINT:-laser.py}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-4}"
TOKEN_SUBSET="${TOKEN_SUBSET:-98304}"

FAST_AE_NUM_DOWNSAMPLES="${FAST_AE_NUM_DOWNSAMPLES:-4}"
PATCH_AE_NUM_DOWNSAMPLES="${PATCH_AE_NUM_DOWNSAMPLES:-2}"
NUM_HIDDENS="${NUM_HIDDENS:-64}"
NUM_RES_HIDDENS="${NUM_RES_HIDDENS:-32}"
NUM_RES_LAYERS="${NUM_RES_LAYERS:-1}"
PATCH_SIZE="${PATCH_SIZE:-8}"
PATCH_STRIDE="${PATCH_STRIDE:-4}"
PATCH_RECONSTRUCTION="${PATCH_RECONSTRUCTION:-hann}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
RUN_PREFIX="${RUN_PREFIX:-celebahq256_m20}"
JOB_PREFIX="${JOB_PREFIX:-chq256-m20}"
CASE_FILTER="${CASE_FILTER:-}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

cases=(
  "fast|fast_q_e16_a1024_k8|true|16|1024|8|256|3.0"
  "fast|fast_q_e24_a1536_k12|true|24|1536|12|256|3.0"
  "fast|fast_q_e32_a2048_k16|true|32|2048|16|256|3.0"
  "fast|fast_r_e16_a1024_k8|false|16|1024|8|256|3.0"
  "fast|fast_r_e24_a1536_k12|false|24|1536|12|256|3.0"
  "fast|fast_r_e32_a2048_k16|false|32|2048|16|256|3.0"
  "patch|patch_q_p8s4_e16_a4096_k16|true|16|4096|16|512|8.0"
  "patch|patch_q_p8s4_e24_a4096_k24|true|24|4096|24|512|8.0"
  "patch|patch_q_p8s4_e32_a6144_k24|true|32|6144|24|512|8.0"
  "patch|patch_r_p8s4_e16_a4096_k16|false|16|4096|16|256|8.0"
  "patch|patch_r_p8s4_e24_a4096_k24|false|24|4096|24|256|8.0"
  "patch|patch_r_p8s4_e32_a6144_k24|false|32|6144|24|256|8.0"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r family case_name quantize_sparse_coeffs embedding_dim num_atoms sparsity_level n_bins coef_max <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  out_dir="$OUT_ROOT/$run_name"
  launcher="$ROOT_DIR/scripts/fast100.sh"
  patch_based="false"
  ae_num_downsamples="$FAST_AE_NUM_DOWNSAMPLES"
  if [[ "$family" == "patch" ]]; then
    launcher="$ROOT_DIR/scripts/patch100.sh"
    patch_based="true"
    ae_num_downsamples="$PATCH_AE_NUM_DOWNSAMPLES"
  fi

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
    STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    TOKEN_NUM_WORKERS="$TOKEN_NUM_WORKERS" \
    TOKEN_SUBSET="$TOKEN_SUBSET" \
    AE_NUM_DOWNSAMPLES="$ae_num_downsamples" \
    NUM_HIDDENS="$NUM_HIDDENS" \
    NUM_RES_HIDDENS="$NUM_RES_HIDDENS" \
    NUM_RES_LAYERS="$NUM_RES_LAYERS" \
    PATCH_BASED="$patch_based" \
    PATCH_SIZE="$PATCH_SIZE" \
    PATCH_STRIDE="$PATCH_STRIDE" \
    PATCH_RECONSTRUCTION="$PATCH_RECONSTRUCTION" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="$run_name" \
    LOG_PREFIX="$run_name" \
    JOB_NAME="$job_name" \
    EMBEDDING_DIM="$embedding_dim" \
    NUM_ATOMS="$num_atoms" \
    SPARSITY_LEVEL="$sparsity_level" \
    QUANTIZE_SPARSE_COEFFS="$quantize_sparse_coeffs" \
    N_BINS="$n_bins" \
    COEF_MAX="$coef_max" \
    COEF_QUANTIZATION="uniform" \
    COEF_MU="0.0" \
    STAGE2_SCHED_SAMPLING_FINAL_PROB="0.0" \
    STAGE2_SAMPLE_IMAGE_SIZE="$IMAGE_SIZE" \
    RFID_NUM_SAMPLES="0" \
    STAGE1_AUTO_RESUME_FROM_LATEST="false" \
    "$launcher"
  )"
  printf '%s\n' "$submit_output"
  submitted=$((submitted + 1))
done

if ((submitted == 0)); then
  echo "No cases matched CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi
