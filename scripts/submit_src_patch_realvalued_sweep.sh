#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${USER:-$(id -un)}"

PARTITION="${PARTITION:-gpu-redhat}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-96000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

TOKEN_CACHE="${TOKEN_CACHE:-/scratch/$USER_NAME/runs/laser_patch100/20260318_083442/stage2/tokens_cache.pt}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER_NAME/runs/src_patch_realvalued_sweep}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER_NAME/submission_snapshots}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-laser_src_patch_realvalued_sweep}"
RUN_PREFIX="${RUN_PREFIX:-src_patch_rv}"
JOB_PREFIX="${JOB_PREFIX:-src-prv}"

IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER_NAME/.pydeps/laser_src_stage2_py}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER_NAME/.secrets/wandb_api_key}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-src-ar}"
WANDB_MODE="${WANDB_MODE:-online}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAMPLE_EVERY_N_STEPS="${SAMPLE_EVERY_N_STEPS:-1000}"
COEFF_HUBER_DELTA="${COEFF_HUBER_DELTA:-0.5}"
CASE_FILTER="${CASE_FILTER:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "$TOKEN_CACHE" ]]; then
  echo "Missing token cache: $TOKEN_CACHE" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/cluster_logs" "$SNAPSHOT_ROOT"

snapshot_dir="$SNAPSHOT_ROOT/${SNAPSHOT_TAG}_${TIMESTAMP}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
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
  rm -rf "$snapshot_dir/.git" "$snapshot_dir/__pycache__" "$snapshot_dir/outputs" "$snapshot_dir/wandb"
fi

runner_script="$snapshot_dir/.run_src_patch_realvalued_case.sh"
cat > "$runner_script" <<'EOF'
#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?}"
TOKEN_CACHE="${TOKEN_CACHE:?}"
OUT_DIR="${OUT_DIR:?}"
RUN_NAME="${RUN_NAME:?}"
IMAGE="${IMAGE:?}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:?}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:?}"
WANDB_PROJECT="${WANDB_PROJECT:?}"
WANDB_MODE="${WANDB_MODE:?}"
AUTOREGRESSIVE_COEFFS="${AUTOREGRESSIVE_COEFFS:?}"
COEFF_LOSS_TYPE="${COEFF_LOSS_TYPE:?}"
COEFF_HUBER_DELTA="${COEFF_HUBER_DELTA:?}"
D_MODEL="${D_MODEL:?}"
N_HEADS="${N_HEADS:?}"
N_LAYERS="${N_LAYERS:?}"
D_FF="${D_FF:?}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:?}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:?}"
NUM_WORKERS="${NUM_WORKERS:?}"
SAMPLE_EVERY_N_STEPS="${SAMPLE_EVERY_N_STEPS:?}"

USER_NAME="$(id -un)"

if [[ ! -f "$TOKEN_CACHE" ]]; then
  echo "Token cache not found on compute node: $TOKEN_CACHE" >&2
  exit 1
fi

if [[ "$WANDB_MODE" == "online" && ! -f "$WANDB_API_KEY_FILE" ]]; then
  echo "W&B key missing at $WANDB_API_KEY_FILE; falling back to offline mode."
  WANDB_MODE="offline"
fi

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

mkdir -p "$OUT_DIR" "$PYTHONUSERBASE_DIR"

if [[ -f "$WANDB_API_KEY_FILE" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\n' < "$WANDB_API_KEY_FILE")"
fi

PYVER="$(
  singularity exec \
    --bind "$PROJECT_DIR" \
    --bind "/scratch/$USER_NAME" \
    "$IMAGE" \
    python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python$PYVER/site-packages"

export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/scratch:$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE

check_imports='import hydra, lightning, torchvision, wandb, omegaconf, rich, torchmetrics, scipy'
if ! singularity exec \
    --bind "$PROJECT_DIR" \
    --bind "/scratch/$USER_NAME" \
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
      "$IMAGE" \
      python3 -c "$check_imports" >/dev/null 2>&1; then
    singularity exec \
      --bind "$PROJECT_DIR" \
      --bind "/scratch/$USER_NAME" \
      "$IMAGE" \
      python3 -m pip install --user -r "$PROJECT_DIR/requirements.txt"
  fi
fi

nvidia-smi || true

srun singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER_NAME" \
  "$IMAGE" \
  python3 "$PROJECT_DIR/train_ar.py" \
    "output_dir=$OUT_DIR" \
    "token_cache_path=$TOKEN_CACHE" \
    "wandb.project=$WANDB_PROJECT" \
    "wandb.name=$RUN_NAME" \
    "wandb.save_dir=$OUT_DIR/wandb" \
    "ar.type=sparse_spatial_depth" \
    "ar.autoregressive_coeffs=$AUTOREGRESSIVE_COEFFS" \
    "ar.coeff_loss_type=$COEFF_LOSS_TYPE" \
    "ar.coeff_huber_delta=$COEFF_HUBER_DELTA" \
    "ar.d_model=$D_MODEL" \
    "ar.n_heads=$N_HEADS" \
    "ar.n_layers=$N_LAYERS" \
    "ar.d_ff=$D_FF" \
    "train_ar.batch_size=$TRAIN_BATCH_SIZE" \
    "train_ar.max_epochs=$STAGE2_EPOCHS" \
    "train_ar.accelerator=gpu" \
    "train_ar.devices=1" \
    "train_ar.precision=16-mixed" \
    "train_ar.log_every_n_steps=25" \
    "train_ar.sample_every_n_steps=$SAMPLE_EVERY_N_STEPS" \
    "train_ar.sample_num_images=4" \
    "train_ar.sample_coeff_mode=mean" \
    "data.num_workers=$NUM_WORKERS"
EOF
chmod +x "$runner_script"

cases=(
  "ar_mse|true|mse|512|8|6|2048"
  "ar_huber|true|huber|512|8|6|2048"
  "support_mse|false|mse|512|8|6|2048"
  "support_huber|false|huber|512|8|6|2048"
)

submitted=0
submitted_jobs=()
filter_token=",$(echo "$CASE_FILTER" | tr -d ' '),"

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name autoregressive_coeffs coeff_loss_type d_model n_heads n_layers d_ff <<< "$case_spec"

  if [[ -n "${CASE_FILTER// }" && "$filter_token" != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  out_dir="$RUN_ROOT/$run_name"
  mkdir -p "$out_dir"

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
      --output="$RUN_ROOT/cluster_logs/${run_name}_%j.out" \
      --error="$RUN_ROOT/cluster_logs/${run_name}_%j.err" \
      --export=ALL,PROJECT_DIR="$snapshot_dir",TOKEN_CACHE="$TOKEN_CACHE",OUT_DIR="$out_dir",RUN_NAME="$run_name",IMAGE="$IMAGE",PYTHONUSERBASE_DIR="$PYTHONUSERBASE_DIR",WANDB_API_KEY_FILE="$WANDB_API_KEY_FILE",WANDB_PROJECT="$WANDB_PROJECT",WANDB_MODE="$WANDB_MODE",AUTOREGRESSIVE_COEFFS="$autoregressive_coeffs",COEFF_LOSS_TYPE="$coeff_loss_type",COEFF_HUBER_DELTA="$COEFF_HUBER_DELTA",D_MODEL="$d_model",N_HEADS="$n_heads",N_LAYERS="$n_layers",D_FF="$d_ff",TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE",STAGE2_EPOCHS="$STAGE2_EPOCHS",NUM_WORKERS="$NUM_WORKERS",SAMPLE_EVERY_N_STEPS="$SAMPLE_EVERY_N_STEPS" \
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
  echo "No sweep cases matched CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi

echo "snapshot_dir=$snapshot_dir"
echo "runner_script=$runner_script"
echo "submitted_cases=$submitted"
echo "job_ids=${submitted_jobs[*]}"
