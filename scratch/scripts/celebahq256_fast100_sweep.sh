#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_fast100_sweep}"
PARTITION="${PARTITION:-gpu}"
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
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-12}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-4}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
RUN_PREFIX="${RUN_PREFIX:-celebahq256_fast100}"
JOB_PREFIX="${JOB_PREFIX:-chq256-f100}"
SWEEP_SET="${SWEEP_SET:-core}"
CASE_FILTER="${CASE_FILTER:-}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

core_cases=(
  "qbase|true|1024|8|256|3.0|uniform|0.0"
  "qmid|true|1536|12|256|3.0|uniform|0.0"
  "qwide|true|2048|12|256|3.0|uniform|0.0"
  "nqbase|false|1024|8|256|3.0|uniform|0.0"
)

expanded_cases=(
  "qwide16|true|2048|16|256|3.0|uniform|0.0"
  "qxl|true|3072|16|256|3.0|uniform|0.0"
  "nqmid|false|1536|12|256|3.0|uniform|0.0"
)

cases=()
case "$SWEEP_SET" in
  core)
    cases=("${core_cases[@]}")
    ;;
  expanded)
    cases=("${expanded_cases[@]}")
    ;;
  all)
    cases=("${core_cases[@]}" "${expanded_cases[@]}")
    ;;
  *)
    echo "Unsupported SWEEP_SET: $SWEEP_SET" >&2
    echo "Supported values: core, expanded, all" >&2
    exit 1
    ;;
esac

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name quantize_sparse_coeffs num_atoms sparsity_level n_bins coef_max coef_quantization coef_mu <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}_a${num_atoms}_k${sparsity_level}"
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
    STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    TOKEN_NUM_WORKERS="$TOKEN_NUM_WORKERS" \
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
    COEF_QUANTIZATION="$coef_quantization" \
    COEF_MU="$coef_mu" \
    STAGE2_SCHED_SAMPLING_FINAL_PROB="0.0" \
    STAGE2_SAMPLE_IMAGE_SIZE="$IMAGE_SIZE" \
    RFID_NUM_SAMPLES="0" \
    STAGE1_AUTO_RESUME_FROM_LATEST="false" \
    "$ROOT_DIR/scripts/fast100.sh"
  )"
  printf '%s\n' "$submit_output"
  submitted=$((submitted + 1))
done

if ((submitted == 0)); then
  echo "No cases matched SWEEP_SET=$SWEEP_SET CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi
