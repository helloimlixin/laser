#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_stage2_sampling_sweep}"
PARTITION="${PARTITION:-gpu-redhat}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"

ENTRYPOINT="${ENTRYPOINT:-proto.py}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-6}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-6}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
RUN_PREFIX="${RUN_PREFIX:-celebahq256_s2samp10}"
JOB_PREFIX="${JOB_PREFIX:-chq256-s2}"
SWEEP_SET="${SWEEP_SET:-core}"
CASE_FILTER="${CASE_FILTER:-}"

NUM_ATOMS="${NUM_ATOMS:-4096}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-24}"
PATCH_SIZE="${PATCH_SIZE:-4}"
PATCH_STRIDE="${PATCH_STRIDE:-2}"
N_BINS="${N_BINS:-512}"
COEF_MAX="${COEF_MAX:-4.0}"

STAGE1_AUTO_RESUME_FROM_LATEST="${STAGE1_AUTO_RESUME_FROM_LATEST:-true}"
STAGE1_CHECKPOINT_EVERY_STEPS="${STAGE1_CHECKPOINT_EVERY_STEPS:-500}"
RFID_NUM_SAMPLES="${RFID_NUM_SAMPLES:-0}"
STAGE2_FID_NUM_SAMPLES="${STAGE2_FID_NUM_SAMPLES:-128}"
STAGE2_FID_EVERY_EPOCHS="${STAGE2_FID_EVERY_EPOCHS:-2}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

core_cases=(
  "norerank|cf1|t050|k0|gauss|ctnan|q0|bw00|obw00|rdz00|rbz00|sort0"
  "baseline|cf4|t050|k0|gauss|ctnan|q1|bw10|obw10|rdz15|rbz15|sort1"
  "topk64|cf8|t020|k64|gauss|ct020|q1|bw00|obw10|rdz00|rbz10|sort1"
  "mean48|cf8|t016|k48|mean|ctnan|q1|bw20|obw20|rdz10|rbz10|sort1"
  "tight16|cf16|t010|k16|gauss|ct005|q1|bw30|obw30|rdz09|rbz09|sort1"
)

expanded_cases=(
  "mild128|cf4|t035|k128|gauss|ctnan|q1|bw05|obw10|rdz10|rbz10|sort1"
  "soft96|cf6|t025|k96|gauss|ct018|q1|bw05|obw10|rdz05|rbz10|sort1"
  "mid64|cf8|t022|k64|gauss|ct015|q1|bw10|obw15|rdz08|rbz10|sort1"
  "mid32|cf8|t018|k32|gauss|ct010|q1|bw10|obw15|rdz10|rbz10|sort1"
  "cool24|cf12|t012|k24|gauss|ct006|q1|bw15|obw20|rdz08|rbz10|sort1"
  "quality48|cf8|t020|k48|gauss|ct020|q1|bw00|obw00|rdz00|rbz00|sort1"
  "mean64|cf8|t022|k64|mean|ctnan|q1|bw10|obw10|rdz05|rbz10|sort1"
  "brightguard|cf8|t018|k48|gauss|ct010|q1|bw15|obw30|rdz05|rbz08|sort1"
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
  IFS='|' read -r case_name cf_tag t_tag k_tag coeff_tag ct_tag q_tag bw_tag obw_tag rdz_tag rbz_tag sort_tag <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  candidate_factor="${cf_tag#cf}"
  atom_temperature="0.${t_tag#t}"
  atom_top_k="${k_tag#k}"
  coeff_mode="${coeff_tag/gauss/gaussian}"
  coeff_temperature_tag="${ct_tag#ct}"
  selection_quality_weight="$(awk "BEGIN { print ${q_tag#q} / 1.0 }")"
  selection_brightness_weight="$(awk "BEGIN { print ${bw_tag#bw} / 10.0 }")"
  selection_overbright_weight="$(awk "BEGIN { print ${obw_tag#obw} / 10.0 }")"
  selection_reject_dark_z="$(awk "BEGIN { print ${rdz_tag#rdz} / 10.0 }")"
  selection_reject_bright_z="$(awk "BEGIN { print ${rbz_tag#rbz} / 10.0 }")"

  if [[ "$coeff_temperature_tag" == "nan" ]]; then
    coeff_temperature="nan"
  else
    coeff_temperature="0.${coeff_temperature_tag}"
  fi

  if [[ "${sort_tag#sort}" == "1" ]]; then
    sample_sort_by_quality="true"
  else
    sample_sort_by_quality="false"
  fi

  run_name="${RUN_PREFIX}_${case_name}_p${PATCH_SIZE}s${PATCH_STRIDE}_a${NUM_ATOMS}_k${SPARSITY_LEVEL}"
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
    STAGE2_SAMPLE_IMAGE_SIZE="$IMAGE_SIZE" \
    BATCH_SIZE="$BATCH_SIZE" \
    STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="$run_name" \
    LOG_PREFIX="$run_name" \
    JOB_NAME="$job_name" \
    NUM_ATOMS="$NUM_ATOMS" \
    SPARSITY_LEVEL="$SPARSITY_LEVEL" \
    N_BINS="$N_BINS" \
    COEF_MAX="$COEF_MAX" \
    STAGE1_AUTO_RESUME_FROM_LATEST="$STAGE1_AUTO_RESUME_FROM_LATEST" \
    STAGE1_CHECKPOINT_EVERY_STEPS="$STAGE1_CHECKPOINT_EVERY_STEPS" \
    RFID_NUM_SAMPLES="$RFID_NUM_SAMPLES" \
    STAGE2_FID_NUM_SAMPLES="$STAGE2_FID_NUM_SAMPLES" \
    STAGE2_FID_EVERY_EPOCHS="$STAGE2_FID_EVERY_EPOCHS" \
    STAGE2_SAMPLE_CANDIDATE_FACTOR="$candidate_factor" \
    STAGE2_SAMPLE_TEMPERATURE="$atom_temperature" \
    STAGE2_SAMPLE_TOP_K="$atom_top_k" \
    STAGE2_SAMPLE_COEFF_MODE="$coeff_mode" \
    STAGE2_SAMPLE_COEFF_TEMPERATURE="$coeff_temperature" \
    STAGE2_SAMPLE_QUALITY_WEIGHT="$selection_quality_weight" \
    STAGE2_SAMPLE_BRIGHTNESS_WEIGHT="$selection_brightness_weight" \
    STAGE2_SAMPLE_OVERBRIGHT_WEIGHT="$selection_overbright_weight" \
    STAGE2_SAMPLE_REJECT_DARK_Z="$selection_reject_dark_z" \
    STAGE2_SAMPLE_REJECT_BRIGHT_Z="$selection_reject_bright_z" \
    STAGE2_SAMPLE_SORT_BY_QUALITY="$sample_sort_by_quality" \
    "$ROOT_DIR/scripts/patch_celebahq256_best.sh"
  )"
  printf '%s\n' "$submit_output"
  submitted=$((submitted + 1))
done

if ((submitted == 0)); then
  echo "No cases matched SWEEP_SET=$SWEEP_SET CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi
