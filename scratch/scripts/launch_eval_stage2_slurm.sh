#!/bin/bash

set -euo pipefail
unset LC_ALL || true
export LANG=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${JOB_NAME:-laser-eval}"
PARTITION="${PARTITION:-cgpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-32000}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER/.pydeps/proto_rqsd_py311}"
ENTRYPOINT="${ENTRYPOINT:-eval_stage2_checkpoint.py}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the training run directory}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the evaluation output directory}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-}"
TOKEN_CACHE="${TOKEN_CACHE:-}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_ITEMS="${MAX_ITEMS:-512}"
OFFSET="${OFFSET:-0}"
PROMPT_STEPS="${PROMPT_STEPS:-0,16,32,48,56,60,63}"
GREEDY_TEMPERATURE="${GREEDY_TEMPERATURE:-1.0}"
GREEDY_TOP_K="${GREEDY_TOP_K:-1}"
SEED="${SEED:-0}"
TF_HEADS="${TF_HEADS:-8}"
TF_DROPOUT="${TF_DROPOUT:-0.1}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
LOG_PREFIX="${LOG_PREFIX:-laser_eval}"

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

JOB_SCRIPT="$(mktemp "$ROOT_DIR/.eval_stage2_job_XXXXXX.sh")"

cat >"$JOB_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
unset LC_ALL || true
export LANG=C
export PYTHONUNBUFFERED=1
USER_NAME=\$(id -un)
PROJECT_DIR="$ROOT_DIR"
RUN_DIR="$RUN_DIR"
OUTPUT_DIR="$OUTPUT_DIR"
STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"
STAGE2_CHECKPOINT="$STAGE2_CHECKPOINT"
TOKEN_CACHE="$TOKEN_CACHE"
DEVICE="$DEVICE"
BATCH_SIZE="$BATCH_SIZE"
MAX_ITEMS="$MAX_ITEMS"
OFFSET="$OFFSET"
PROMPT_STEPS="$PROMPT_STEPS"
GREEDY_TEMPERATURE="$GREEDY_TEMPERATURE"
GREEDY_TOP_K="$GREEDY_TOP_K"
SEED="$SEED"
TF_HEADS="$TF_HEADS"
TF_DROPOUT="$TF_DROPOUT"
IMAGE="$IMAGE"
PYTHONUSERBASE_DIR="$PYTHONUSERBASE_DIR"
PYTHON_SITE="\$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"
ENTRYPOINT="$ENTRYPOINT"

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

mkdir -p "\$OUTPUT_DIR" "\$PYTHONUSERBASE_DIR"
export PYTHONUSERBASE="\$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="\$PYTHON_SITE\${PYTHONPATH:+:\$PYTHONPATH}"

if ! PYTHONUSERBASE="\$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="\$PYTHON_SITE\${PYTHONPATH:+:\$PYTHONPATH}" singularity exec --nv --bind "\$PROJECT_DIR" --bind "/scratch/\$USER_NAME" "\$IMAGE" python3 -c "import scipy; from PIL import Image" >/dev/null 2>&1
then
  PYTHONUSERBASE="\$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="\$PYTHON_SITE\${PYTHONPATH:+:\$PYTHONPATH}" singularity exec --nv --bind "\$PROJECT_DIR" --bind "/scratch/\$USER_NAME" "\$IMAGE" python3 -m pip install --user scipy pillow wandb
fi

CMD=(
  python3 "\$PROJECT_DIR/\$ENTRYPOINT"
  --run_dir "\$RUN_DIR"
  --output_dir "\$OUTPUT_DIR"
  --device "\$DEVICE"
  --batch_size "\$BATCH_SIZE"
  --max_items "\$MAX_ITEMS"
  --offset "\$OFFSET"
  --prompt_steps "\$PROMPT_STEPS"
  --greedy_temperature "\$GREEDY_TEMPERATURE"
  --greedy_top_k "\$GREEDY_TOP_K"
  --seed "\$SEED"
  --tf_heads "\$TF_HEADS"
  --tf_dropout "\$TF_DROPOUT"
)
if [[ -n "\$STAGE1_CHECKPOINT" ]]; then
  CMD+=(--stage1_checkpoint "\$STAGE1_CHECKPOINT")
fi
if [[ -n "\$STAGE2_CHECKPOINT" ]]; then
  CMD+=(--stage2_checkpoint "\$STAGE2_CHECKPOINT")
fi
if [[ -n "\$TOKEN_CACHE" ]]; then
  CMD+=(--token_cache "\$TOKEN_CACHE")
fi

echo "[Launch] RUN_DIR=\$RUN_DIR"
echo "[Launch] OUTPUT_DIR=\$OUTPUT_DIR"
echo "[Launch] STAGE1_CHECKPOINT=\${STAGE1_CHECKPOINT:-<auto>}"
echo "[Launch] STAGE2_CHECKPOINT=\${STAGE2_CHECKPOINT:-<auto>}"
echo "[Launch] TOKEN_CACHE=\${TOKEN_CACHE:-<auto>}"
echo "[Launch] BATCH_SIZE=\$BATCH_SIZE MAX_ITEMS=\$MAX_ITEMS OFFSET=\$OFFSET"
echo "[Launch] PROMPT_STEPS=\$PROMPT_STEPS GREEDY_TEMPERATURE=\$GREEDY_TEMPERATURE GREEDY_TOP_K=\$GREEDY_TOP_K"
printf "[Launch] CMD:"
printf " %q" "\${CMD[@]}"
printf "\n"

singularity exec --nv --bind "\$PROJECT_DIR" --bind "/scratch/\$USER_NAME" "\$IMAGE" "\${CMD[@]}"
EOF

chmod 700 "$JOB_SCRIPT"
echo "[Launch] job script: $JOB_SCRIPT"

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
  "$JOB_SCRIPT"
