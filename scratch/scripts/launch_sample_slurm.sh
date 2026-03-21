#!/bin/bash

set -euo pipefail
unset LC_ALL || true
export LANG=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${JOB_NAME:-laser-sample}"
PARTITION="${PARTITION:-cgpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-32000}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER/.pydeps/proto_rqsd_py311}"
ENTRYPOINT="${ENTRYPOINT:-sample.py}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the training run directory}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the sample output directory}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-}"
TOKEN_CACHE="${TOKEN_CACHE:-}"
PROMPT_SPATIAL_STEPS="${PROMPT_SPATIAL_STEPS:-0}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-0.22}"
TOP_K="${TOP_K:-96}"
COEFF_TEMPERATURE="${COEFF_TEMPERATURE:-}"
COEFF_SAMPLE_MODE="${COEFF_SAMPLE_MODE:-gaussian}"
CANDIDATE_FACTOR="${CANDIDATE_FACTOR:-8}"
SELECTION_QUALITY_WEIGHT="${SELECTION_QUALITY_WEIGHT:-1.0}"
SELECTION_BRIGHTNESS_WEIGHT="${SELECTION_BRIGHTNESS_WEIGHT:-1.0}"
SELECTION_OVERBRIGHT_WEIGHT="${SELECTION_OVERBRIGHT_WEIGHT:-1.0}"
SELECTION_REJECT_DARK_Z="${SELECTION_REJECT_DARK_Z:-1.0}"
SELECTION_REJECT_BRIGHT_Z="${SELECTION_REJECT_BRIGHT_Z:-1.0}"
SELECTION_MODE="${SELECTION_MODE:-diverse}"
SELECTION_SORT_BY_QUALITY="${SELECTION_SORT_BY_QUALITY:-true}"
SELECTION_REFERENCE_MAX_ITEMS="${SELECTION_REFERENCE_MAX_ITEMS:-256}"
LOG_CANDIDATE_POOL="${LOG_CANDIDATE_POOL:-true}"
OUTPUT_IMAGE_SIZE="${OUTPUT_IMAGE_SIZE:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-samples}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-$(basename "$OUTPUT_DIR")}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
LOG_PREFIX="${LOG_PREFIX:-laser_sample}"

if [[ ! -f "$ROOT_DIR/$ENTRYPOINT" ]]; then
  echo "Missing $ENTRYPOINT under $ROOT_DIR" >&2
  exit 1
fi

SBATCH_EXTRA_ARGS=()
if [[ -n "$SBATCH_DEPENDENCY" ]]; then
  SBATCH_EXTRA_ARGS+=(--dependency="$SBATCH_DEPENDENCY")
fi

SBATCH_CMD=(sbatch)
if ((${#SBATCH_EXTRA_ARGS[@]} > 0)); then
  SBATCH_CMD+=("${SBATCH_EXTRA_ARGS[@]}")
fi

"${SBATCH_CMD[@]}" \
  --partition="$PARTITION" \
  --requeue \
  --job-name="$JOB_NAME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:1" \
  --mem="$MEM_MB" \
  --time="$TIME_LIMIT" \
  --chdir="$ROOT_DIR" \
  --output="$ROOT_DIR/${LOG_PREFIX}_%j.out" \
  --error="$ROOT_DIR/${LOG_PREFIX}_%j.err" \
  --wrap='set -euo pipefail
unset LC_ALL || true
export LANG=C
export PYTHONUNBUFFERED=1
USER_NAME=$(id -un)
PROJECT_DIR='"$ROOT_DIR"'
RUN_DIR='"$RUN_DIR"'
OUTPUT_DIR='"$OUTPUT_DIR"'
STAGE1_CHECKPOINT='"$STAGE1_CHECKPOINT"'
STAGE2_CHECKPOINT='"$STAGE2_CHECKPOINT"'
TOKEN_CACHE='"$TOKEN_CACHE"'
PROMPT_SPATIAL_STEPS='"$PROMPT_SPATIAL_STEPS"'
PROMPT_OFFSET='"$PROMPT_OFFSET"'
IMAGE='"$IMAGE"'
PYTHONUSERBASE_DIR='"$PYTHONUSERBASE_DIR"'
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"
ENTRYPOINT='"$ENTRYPOINT"'
NUM_SAMPLES='"$NUM_SAMPLES"'
SEED='"$SEED"'
TEMPERATURE='"$TEMPERATURE"'
TOP_K='"$TOP_K"'
COEFF_TEMPERATURE='"$COEFF_TEMPERATURE"'
COEFF_SAMPLE_MODE='"$COEFF_SAMPLE_MODE"'
CANDIDATE_FACTOR='"$CANDIDATE_FACTOR"'
SELECTION_QUALITY_WEIGHT='"$SELECTION_QUALITY_WEIGHT"'
SELECTION_BRIGHTNESS_WEIGHT='"$SELECTION_BRIGHTNESS_WEIGHT"'
SELECTION_OVERBRIGHT_WEIGHT='"$SELECTION_OVERBRIGHT_WEIGHT"'
SELECTION_REJECT_DARK_Z='"$SELECTION_REJECT_DARK_Z"'
SELECTION_REJECT_BRIGHT_Z='"$SELECTION_REJECT_BRIGHT_Z"'
SELECTION_MODE='"$SELECTION_MODE"'
SELECTION_SORT_BY_QUALITY='"$SELECTION_SORT_BY_QUALITY"'
SELECTION_REFERENCE_MAX_ITEMS='"$SELECTION_REFERENCE_MAX_ITEMS"'
LOG_CANDIDATE_POOL='"$LOG_CANDIDATE_POOL"'
OUTPUT_IMAGE_SIZE='"$OUTPUT_IMAGE_SIZE"'
WANDB_MODE='"$WANDB_MODE"'
WANDB_PROJECT='"$WANDB_PROJECT"'
WANDB_ENTITY='"$WANDB_ENTITY"'
WANDB_NAME='"$WANDB_NAME"'
WANDB_API_KEY_FILE='"$WANDB_API_KEY_FILE"'

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    set +u
    source /etc/profile.d/modules.sh
    set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u
    source /usr/share/Modules/init/bash
    set -u
  fi
fi
if ! command -v singularity >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  fi
fi
command -v singularity >/dev/null 2>&1 || { echo singularity_not_found >&2; exit 1; }

mkdir -p "$OUTPUT_DIR" "$PYTHONUSERBASE_DIR" "$(dirname "$WANDB_API_KEY_FILE")"
export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_API_KEY_FILE
unset WANDB_API_KEY || true
if [[ -f "$WANDB_API_KEY_FILE" ]]; then
  IFS= read -r WANDB_API_KEY < "$WANDB_API_KEY_FILE" || true
  export WANDB_API_KEY
fi

if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --nv --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" "$IMAGE" python3 - <<'"'"'PY_DEP'"'"' >/dev/null 2>&1
import importlib
import scipy
from PIL import Image
module = importlib.import_module("wandb")
raise SystemExit(0 if hasattr(module, "init") and hasattr(module, "Api") else 1)
PY_DEP
then
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --nv --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" "$IMAGE" python3 -m pip install --user scipy wandb pillow
fi

CMD=(
  python3 "$PROJECT_DIR/$ENTRYPOINT"
  --run_dir "$RUN_DIR"
  --output_dir "$OUTPUT_DIR"
  --device cuda
  --num_samples "$NUM_SAMPLES"
  --prompt_spatial_steps "$PROMPT_SPATIAL_STEPS"
  --prompt_offset "$PROMPT_OFFSET"
  --seed "$SEED"
  --temperature "$TEMPERATURE"
  --top_k "$TOP_K"
  --coeff_sample_mode "$COEFF_SAMPLE_MODE"
  --candidate_factor "$CANDIDATE_FACTOR"
  --selection_quality_weight "$SELECTION_QUALITY_WEIGHT"
  --selection_brightness_weight "$SELECTION_BRIGHTNESS_WEIGHT"
  --selection_overbright_weight "$SELECTION_OVERBRIGHT_WEIGHT"
  --selection_reject_dark_z "$SELECTION_REJECT_DARK_Z"
  --selection_reject_bright_z "$SELECTION_REJECT_BRIGHT_Z"
  --selection_mode "$SELECTION_MODE"
  --selection_reference_max_items "$SELECTION_REFERENCE_MAX_ITEMS"
  --wandb_project "$WANDB_PROJECT"
  --wandb_name "$WANDB_NAME"
  --wandb_mode "$WANDB_MODE"
  --wandb_dir "$OUTPUT_DIR/wandb"
)
if [[ -n "$STAGE1_CHECKPOINT" ]]; then
  CMD+=(--stage1_checkpoint "$STAGE1_CHECKPOINT")
fi
if [[ -n "$STAGE2_CHECKPOINT" ]]; then
  CMD+=(--stage2_checkpoint "$STAGE2_CHECKPOINT")
fi
if [[ -n "$TOKEN_CACHE" ]]; then
  CMD+=(--token_cache "$TOKEN_CACHE")
fi
if [[ -n "$COEFF_TEMPERATURE" ]]; then
  CMD+=(--coeff_temperature "$COEFF_TEMPERATURE")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ "$SELECTION_SORT_BY_QUALITY" == "1" || "$SELECTION_SORT_BY_QUALITY" == "true" || "$SELECTION_SORT_BY_QUALITY" == "TRUE" || "$SELECTION_SORT_BY_QUALITY" == "yes" || "$SELECTION_SORT_BY_QUALITY" == "YES" ]]; then
  CMD+=(--selection_sort_by_quality)
else
  CMD+=(--no_selection_sort_by_quality)
fi
if [[ "$LOG_CANDIDATE_POOL" == "1" || "$LOG_CANDIDATE_POOL" == "true" || "$LOG_CANDIDATE_POOL" == "TRUE" || "$LOG_CANDIDATE_POOL" == "yes" || "$LOG_CANDIDATE_POOL" == "YES" ]]; then
  CMD+=(--log_candidate_pool)
else
  CMD+=(--no_log_candidate_pool)
fi
if [[ "$WANDB_MODE" != "disabled" ]]; then
  CMD+=(--wandb)
fi
if [[ -n "$OUTPUT_IMAGE_SIZE" ]]; then
  CMD+=(--output_image_size "$OUTPUT_IMAGE_SIZE")
fi

echo "[Launch] RUN_DIR=$RUN_DIR"
echo "[Launch] OUTPUT_DIR=$OUTPUT_DIR"
echo "[Launch] STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-<auto>}"
echo "[Launch] STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-<auto>}"
echo "[Launch] TOKEN_CACHE=${TOKEN_CACHE:-<auto>}"
echo "[Launch] PROMPT_SPATIAL_STEPS=$PROMPT_SPATIAL_STEPS PROMPT_OFFSET=$PROMPT_OFFSET"
echo "[Launch] TEMPERATURE=$TEMPERATURE TOP_K=$TOP_K COEFF_SAMPLE_MODE=$COEFF_SAMPLE_MODE COEFF_TEMPERATURE=${COEFF_TEMPERATURE:-<auto>} CANDIDATE_FACTOR=$CANDIDATE_FACTOR"
echo "[Launch] selection_mode=$SELECTION_MODE selection_quality_weight=$SELECTION_QUALITY_WEIGHT selection_brightness_weight=$SELECTION_BRIGHTNESS_WEIGHT selection_overbright_weight=$SELECTION_OVERBRIGHT_WEIGHT selection_reject_dark_z=$SELECTION_REJECT_DARK_Z selection_reject_bright_z=$SELECTION_REJECT_BRIGHT_Z"
printf "[Launch] CMD:"
printf " %q" "${CMD[@]}"
printf "\n"

singularity exec --nv --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" "$IMAGE" "${CMD[@]}"
'
