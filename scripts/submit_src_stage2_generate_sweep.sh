#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${USER:-$(id -un)}"

PARTITION="${PARTITION:-gpu-redhat}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-64000}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/scratch/$USER_NAME/runs/src_patch_noov_sweep}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER_NAME/runs/src_patch_noov_generate}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER_NAME/submission_snapshots}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-laser_src_stage2_generate_sweep}"
RUN_PREFIX="${RUN_PREFIX:-src_patch_noov_generate}"
JOB_PREFIX="${JOB_PREFIX:-src-pnoovg}"
CASE_FILTER="${CASE_FILTER:-}"

IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER_NAME/.pydeps/laser_src_generate_py}"

NUM_SAMPLES="${NUM_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-0}"
DEVICE="${DEVICE:-cuda}"
COEFF_SAMPLE_TEMPERATURE="${COEFF_SAMPLE_TEMPERATURE:-}"
COEFF_SAMPLE_MODE="${COEFF_SAMPLE_MODE:-}"
GENERATE_TAG="${GENERATE_TAG:-resample}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$SOURCE_RUN_ROOT" ]]; then
  echo "Missing source run root: $SOURCE_RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/cluster_logs" "$SNAPSHOT_ROOT"

snapshot_dir="$SNAPSHOT_ROOT/${SNAPSHOT_TAG}_${TIMESTAMP}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --include=/configs/wandb/ \
    --include=/configs/wandb/*** \
    --exclude=.git \
    --exclude=__pycache__ \
    --exclude=.pytest_cache \
    --exclude=.mypy_cache \
    --exclude=.ruff_cache \
    --exclude=.tmp \
    --exclude=.tmp_* \
    --exclude=wandb \
    --exclude=outputs \
    --exclude=scratch/.tmp \
    --exclude=scratch/.tmp_* \
    --exclude=scratch/cluster_logs \
    --exclude=scratch/resamples \
    --exclude='*.out' \
    --exclude='*.err' \
    "$ROOT_DIR/" "$snapshot_dir/"
else
  cp -R "$ROOT_DIR" "$snapshot_dir"
  rm -rf \
    "$snapshot_dir/.git" \
    "$snapshot_dir/__pycache__" \
    "$snapshot_dir/.pytest_cache" \
    "$snapshot_dir/.mypy_cache" \
    "$snapshot_dir/.ruff_cache" \
    "$snapshot_dir/.tmp" \
    "$snapshot_dir/wandb" \
    "$snapshot_dir/outputs"
fi

runner_script="$snapshot_dir/.run_src_stage2_generate_case.sh"
cat > "$runner_script" <<'EOF'
#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?}"
STAGE2_OUT="${STAGE2_OUT:?}"
RUN_NAME="${RUN_NAME:?}"
IMAGE="${IMAGE:?}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:?}"
NUM_SAMPLES="${NUM_SAMPLES:?}"
BATCH_SIZE="${BATCH_SIZE:?}"
TEMPERATURE="${TEMPERATURE:?}"
TOP_K="${TOP_K:?}"
DEVICE="${DEVICE:?}"
GENERATE_TAG="${GENERATE_TAG:?}"
COEFF_SAMPLE_TEMPERATURE="${COEFF_SAMPLE_TEMPERATURE:-}"
COEFF_SAMPLE_MODE="${COEFF_SAMPLE_MODE:-}"

USER_NAME="$(id -un)"

if ! command -v singularity >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  elif [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  fi
fi
command -v singularity >/dev/null 2>&1 || { echo "singularity_not_found" >&2; exit 1; }

mkdir -p "$PYTHONUSERBASE_DIR"

PYVER="$(
  singularity exec \
    --bind "$PROJECT_DIR" \
    --bind "/scratch/$USER_NAME" \
    --bind "/cache/home/$USER_NAME" \
    "$IMAGE" \
    python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python$PYVER/site-packages"

export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PROJECT_DIR:$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

check_imports='import hydra, lightning, torchvision, omegaconf, rich, torchmetrics, scipy'
if ! singularity exec \
    --bind "$PROJECT_DIR" \
    --bind "/scratch/$USER_NAME" \
    --bind "/cache/home/$USER_NAME" \
    "$IMAGE" \
    python3 -c "$check_imports" >/dev/null 2>&1; then
  lock_dir="${PYTHONUSERBASE_DIR}.install.lock"
  while ! mkdir "$lock_dir" 2>/dev/null; do
    sleep 10
  done
  cleanup_lock() {
    rmdir "$lock_dir" 2>/dev/null || true
  }
  trap cleanup_lock EXIT
  if ! singularity exec \
      --bind "$PROJECT_DIR" \
      --bind "/scratch/$USER_NAME" \
      --bind "/cache/home/$USER_NAME" \
      "$IMAGE" \
      python3 -c "$check_imports" >/dev/null 2>&1; then
    singularity exec \
      --bind "$PROJECT_DIR" \
      --bind "/scratch/$USER_NAME" \
      --bind "/cache/home/$USER_NAME" \
      "$IMAGE" \
      python3 -m pip install --user -r "$PROJECT_DIR/requirements.txt"
  fi
fi

nvidia-smi || true

shopt -s nullglob
stage2_ckpts=( "$STAGE2_OUT"/checkpoints/*/last.ckpt )
if ((${#stage2_ckpts[@]} == 0)); then
  stage2_ckpts=( "$STAGE2_OUT"/checkpoints/*/*.ckpt )
fi
if ((${#stage2_ckpts[@]} == 0)); then
  echo "No stage-2 checkpoint found under $STAGE2_OUT/checkpoints" >&2
  exit 1
fi
stage2_ckpt="$(ls -t "${stage2_ckpts[@]}" | head -n 1)"

token_caches=( "$STAGE2_OUT"/token_cache/*.pt )
if ((${#token_caches[@]} == 0)); then
  echo "No token cache found under $STAGE2_OUT/token_cache" >&2
  exit 1
fi
token_cache_path="$(ls -t "${token_caches[@]}" | head -n 1)"

output_dir="$STAGE2_OUT/generated/${GENERATE_TAG}"
generate_cmd=(
  python3 "$PROJECT_DIR/generate_ar.py"
  --ar-output-dir "$STAGE2_OUT"
  --stage2-checkpoint "$stage2_ckpt"
  --token-cache "$token_cache_path"
  --output-dir "$output_dir"
  --num-samples "$NUM_SAMPLES"
  --batch-size "$BATCH_SIZE"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --device "$DEVICE"
)
if [[ -n "${COEFF_SAMPLE_TEMPERATURE// }" ]]; then
  generate_cmd+=( --coeff-temperature "$COEFF_SAMPLE_TEMPERATURE" )
fi
if [[ -n "${COEFF_SAMPLE_MODE// }" ]]; then
  generate_cmd+=( --coeff-sample-mode "$COEFF_SAMPLE_MODE" )
fi

singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER_NAME" \
  --bind "/cache/home/$USER_NAME" \
  "$IMAGE" \
  "${generate_cmd[@]}"
EOF
chmod +x "$runner_script"

submitted=0
submitted_jobs=()
filter_token=",$(echo "$CASE_FILTER" | tr -d ' '),"

for run_dir in "$SOURCE_RUN_ROOT"/*; do
  if [[ ! -d "$run_dir" ]]; then
    continue
  fi
  case_name="$(basename "$run_dir")"
  if [[ "$case_name" == "cluster_logs" ]]; then
    continue
  fi
  if [[ -n "${CASE_FILTER// }" && "$filter_token" != *",$case_name,"* ]]; then
    continue
  fi

  submit_name="${RUN_PREFIX}_${case_name}"
  short_case="$(printf '%s' "$case_name" | cut -d_ -f4)"
  job_name="${JOB_PREFIX}-${short_case:-gen}"

  submit_output="$(
    sbatch \
      --partition="$PARTITION" \
      --requeue \
      --job-name="$job_name" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="$CPUS_PER_TASK" \
      --gres="gpu:${GPUS}" \
      --mem="$MEM_MB" \
      --time="$TIME_LIMIT" \
      --chdir="$snapshot_dir" \
      --output="$RUN_ROOT/cluster_logs/${submit_name}_%j.out" \
      --error="$RUN_ROOT/cluster_logs/${submit_name}_%j.err" \
      --export=ALL,PROJECT_DIR="$snapshot_dir",STAGE2_OUT="$run_dir/stage2",RUN_NAME="$submit_name",IMAGE="$IMAGE",PYTHONUSERBASE_DIR="$PYTHONUSERBASE_DIR",NUM_SAMPLES="$NUM_SAMPLES",BATCH_SIZE="$BATCH_SIZE",TEMPERATURE="$TEMPERATURE",TOP_K="$TOP_K",DEVICE="$DEVICE",GENERATE_TAG="$GENERATE_TAG",COEFF_SAMPLE_TEMPERATURE="$COEFF_SAMPLE_TEMPERATURE",COEFF_SAMPLE_MODE="$COEFF_SAMPLE_MODE" \
      "$runner_script"
  )"
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | awk '/Submitted batch job/{print $4}')"
  if [[ -n "$job_id" ]]; then
    submitted_jobs+=("$job_id")
  fi
  submitted=$((submitted + 1))
done

if ((submitted == 0)); then
  echo "No generate cases matched CASE_FILTER=$CASE_FILTER under $SOURCE_RUN_ROOT" >&2
  exit 1
fi

echo "snapshot_dir=$snapshot_dir"
echo "runner_script=$runner_script"
echo "submitted_cases=$submitted"
echo "job_ids=${submitted_jobs[*]}"
