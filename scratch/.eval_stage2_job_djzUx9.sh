#!/bin/bash
set -euo pipefail
unset LC_ALL || true
export LANG=C
export PYTHONUNBUFFERED=1
USER_NAME=$(id -un)
PROJECT_DIR="/scratch/xl598/Projects/laser/scratch"
RUN_DIR="/scratch/xl598/runs/celebahq256_bottleneck_coeff_sweep/celebahq256_bneck10_latent8q_baseline_a4096_k24/20260320_022449"
OUTPUT_DIR="/scratch/xl598/resamples/stage2_checkpoint_eval_20260320_r3/latent8q_baseline"
STAGE1_CHECKPOINT=""
STAGE2_CHECKPOINT=""
TOKEN_CACHE=""
DEVICE="cuda"
BATCH_SIZE="32"
MAX_ITEMS="128"
OFFSET="0"
PROMPT_STEPS="0,16,32,48,56,60,63"
GREEDY_TEMPERATURE="1.0"
GREEDY_TOP_K="1"
SEED="0"
TF_HEADS="8"
TF_DROPOUT="0.1"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
PYTHONUSERBASE_DIR="/scratch/xl598/.pydeps/proto_rqsd_py311"
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"
ENTRYPOINT="eval_stage2_checkpoint.py"

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

mkdir -p "$OUTPUT_DIR" "$PYTHONUSERBASE_DIR"
export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"

if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --nv --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" "$IMAGE" python3 -c "import scipy; from PIL import Image" >/dev/null 2>&1
then
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --nv --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" "$IMAGE" python3 -m pip install --user scipy pillow wandb
fi

CMD=(
  python3 "$PROJECT_DIR/$ENTRYPOINT"
  --run_dir "$RUN_DIR"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --max_items "$MAX_ITEMS"
  --offset "$OFFSET"
  --prompt_steps "$PROMPT_STEPS"
  --greedy_temperature "$GREEDY_TEMPERATURE"
  --greedy_top_k "$GREEDY_TOP_K"
  --seed "$SEED"
  --tf_heads "$TF_HEADS"
  --tf_dropout "$TF_DROPOUT"
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

echo "[Launch] RUN_DIR=$RUN_DIR"
echo "[Launch] OUTPUT_DIR=$OUTPUT_DIR"
echo "[Launch] STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-<auto>}"
echo "[Launch] STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-<auto>}"
echo "[Launch] TOKEN_CACHE=${TOKEN_CACHE:-<auto>}"
echo "[Launch] BATCH_SIZE=$BATCH_SIZE MAX_ITEMS=$MAX_ITEMS OFFSET=$OFFSET"
echo "[Launch] PROMPT_STEPS=$PROMPT_STEPS GREEDY_TEMPERATURE=$GREEDY_TEMPERATURE GREEDY_TOP_K=$GREEDY_TOP_K"
printf "[Launch] CMD:"
printf " %q" "${CMD[@]}"
printf "\n"

singularity exec --nv --bind "$PROJECT_DIR" --bind "/scratch/$USER_NAME" "$IMAGE" "${CMD[@]}"
