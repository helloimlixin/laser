#!/usr/bin/env bash
# Submit a raw-waveform VCTK stage-2 prior from an existing stage-1 checkpoint.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL_FAMILY="${MODEL_FAMILY:-${1:-laser}}"
CASE_NAME="${CASE_NAME:-${2:-}}"
STAGE1_CKPT="${STAGE1_CKPT:-${3:-}}"
TOKEN_CACHE="${TOKEN_CACHE:-}"
ALLOW_MISSING_CKPT="${ALLOW_MISSING_CKPT:-false}"
CKPT_PATH="${CKPT_PATH:-}"

if [[ -z "$CASE_NAME" || ( -z "$STAGE1_CKPT" && -z "$TOKEN_CACHE" ) ]]; then
  echo "Usage: MODEL_FAMILY=<laser|vqvae> CASE_NAME=<name> STAGE1_CKPT=<path> $0" >&2
  echo "   or: TOKEN_CACHE=<path> MODEL_FAMILY=<laser|vqvae> CASE_NAME=<name> $0" >&2
  echo "   or: $0 <laser|vqvae> <case_name> <stage1_ckpt>" >&2
  exit 2
fi

case "$MODEL_FAMILY" in
  laser|vqvae) ;;
  *)
    echo "MODEL_FAMILY must be 'laser' or 'vqvae', got: $MODEL_FAMILY" >&2
    exit 2
    ;;
esac

if [[ -n "$TOKEN_CACHE" && ! -f "$TOKEN_CACHE" ]]; then
  echo "Token cache not found: $TOKEN_CACHE" >&2
  exit 1
fi

if [[ -z "$TOKEN_CACHE" && ! -f "$STAGE1_CKPT" && "$ALLOW_MISSING_CKPT" != "true" ]]; then
  echo "Stage-1 checkpoint not found: $STAGE1_CKPT" >&2
  echo "Set ALLOW_MISSING_CKPT=true only when the job has a dependency that will create it." >&2
  exit 1
fi

PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
PROJECT="${PROJECT:-laser}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/vctk_src_sweeps}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/datasets/VCTK-Corpus-0.92}"
AUDIO_TARGET_PEAK="${AUDIO_TARGET_PEAK:-0.95}"
AUDIO_TARGET_RMS="${AUDIO_TARGET_RMS:-0.12}"
AUDIO_MAX_GAIN="${AUDIO_MAX_GAIN:-8.0}"
AUDIO_MIN_CROP_RMS="${AUDIO_MIN_CROP_RMS:-0.03}"
AUDIO_CROP_ATTEMPTS="${AUDIO_CROP_ATTEMPTS:-64}"
AUDIO_FADE_SAMPLES="${AUDIO_FADE_SAMPLES:-1024}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-2}"
# Quantized coefficients are the safe default for non-variational LASER
# checkpoints. Raw real-valued coefficient caches (COEFF_BINS=0) need a
# variational coefficient head; otherwise generation collapses toward
# near-zero deterministic coefficient means.
COEFF_BINS="${COEFF_BINS:-512}"
COEFF_MAX="${COEFF_MAX:-16.0}"
COEFF_QUANTIZATION="${COEFF_QUANTIZATION:-uniform}"
COEFF_MU="${COEFF_MU:-0.0}"
AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_HEADS="${AR_N_HEADS:-8}"
AR_N_LAYERS="${AR_N_LAYERS:-6}"
AR_D_FF="${AR_D_FF:-1024}"
AR_LR="${AR_LR:-4.0e-4}"
AR_WARMUP_STEPS="${AR_WARMUP_STEPS:-750}"
AR_MIN_LR_RATIO="${AR_MIN_LR_RATIO:-0.05}"
COEFF_LOSS_WEIGHT="${COEFF_LOSS_WEIGHT:-1.0}"
COEFF_LOSS_TYPE="${COEFF_LOSS_TYPE:-huber}"
COEFF_HUBER_DELTA="${COEFF_HUBER_DELTA:-1.0}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.8}"
SAMPLE_TOP_K="${SAMPLE_TOP_K:-0}"
SAMPLE_COEFF_MODE="${SAMPLE_COEFF_MODE:-mean}"
SAMPLE_EVERY_N_EPOCHS="${SAMPLE_EVERY_N_EPOCHS:-1}"
SAMPLE_NUM_IMAGES="${SAMPLE_NUM_IMAGES:-8}"
SAMPLE_LOG_TO_WANDB="${SAMPLE_LOG_TO_WANDB:-true}"
GENERATION_METRIC_NUM_SAMPLES="${GENERATION_METRIC_NUM_SAMPLES:-$SAMPLE_NUM_IMAGES}"
COMPUTE_AUDIO_GENERATION_METRICS="${COMPUTE_AUDIO_GENERATION_METRICS:-true}"
RUN_TEST_AFTER_FIT="${RUN_TEST_AFTER_FIT:-false}"
SAVE_FINAL_SAMPLES_AFTER_FIT="${SAVE_FINAL_SAMPLES_AFTER_FIT:-false}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
DEPENDENCY="${DEPENDENCY:-}"
EXCLUDE="${EXCLUDE:-}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"

if [[ ! -d "$VCTK_DIR" ]]; then
  echo "VCTK directory not found: $VCTK_DIR" >&2
  exit 1
fi

safe_case="$(printf '%s' "$CASE_NAME" | sed 's/[^A-Za-z0-9_.-]/-/g')"
snapshot_dir="$SNAPSHOT_ROOT/laser_${MODEL_FAMILY}_vctk-waveform-${safe_case}-s2-${STAGE2_EPOCHS}_${STAMP}"
run_root="$RUN_ROOT_BASE/vctk-waveform-${MODEL_FAMILY}-${safe_case}-s2-${STAGE2_EPOCHS}-${STAMP}"
run_dir="$run_root/vctk"
stage2_dir="$run_dir/stage2"
token_cache="${TOKEN_CACHE:-$run_dir/token_cache.pt}"
run_script="$run_dir/run.sh"
sbatch_script="$run_dir/sbatch.sh"
log_base="$run_dir/vctk"
wandb_group="vctk-waveform-${MODEL_FAMILY}-${safe_case}-s2-${STAGE2_EPOCHS}-${STAMP}"

mkdir -p "$SNAPSHOT_ROOT" "$run_dir" "$stage2_dir"

EXCLUDES=(
  --exclude=.git
  --exclude=__pycache__
  --exclude=.pytest_cache
  --exclude=.mypy_cache
  --exclude=.ruff_cache
  --exclude=.tmp
  --exclude='.tmp_*'
  --exclude=cluster_logs
  --exclude=/wandb
  --exclude=runs
  --exclude='source_snapshot_*'
  --exclude='pre_variation_snapshot_*'
  --exclude='*.out'
  --exclude='*.err'
  --exclude='*.pyc'
  --exclude='*.pyo'
  --exclude='*.swp'
)

if command -v rsync >/dev/null 2>&1; then
  rsync -a "${EXCLUDES[@]}" "$PROJECT_DIR/" "$snapshot_dir/"
else
  mkdir -p "$snapshot_dir"
  tar -C "$PROJECT_DIR" \
    --exclude=.git \
    --exclude=__pycache__ \
    --exclude=.pytest_cache \
    --exclude=./wandb \
    --exclude=runs \
    -cf - . | tar -C "$snapshot_dir" -xf -
fi

cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export PYTHONUSERBASE="/scratch/$USER/.pydeps/laser_src_py311"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$snapshot_dir\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="\${WANDB_MODE:-online}"
export WANDB_RESUME="$WANDB_RESUME"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HYDRA_FULL_ERROR=1
export TMPDIR="/tmp/laser_\${SLURM_JOB_ID:-\$\$}"
export TEMP="\$TMPDIR"
export TMP="\$TMPDIR"

mkdir -p "\$TMPDIR" "$stage2_dir/wandb"

PYTHON_BIN="\${PYTHON_BIN:-\$(command -v python3 || command -v python || true)}"
if [[ -z "\$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

if ! "\$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: \$PYTHON_BIN must be Python >= 3.10" >&2
  exit 2
fi

"\$PYTHON_BIN" -m pip install --user --quiet \\
  numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \\
  torch-fidelity matplotlib lpips soundfile 2>/dev/null || true

"\$PYTHON_BIN" - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this job.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

cd "$snapshot_dir"

if [[ ! -f "$token_cache" ]]; then
  if [[ -z "$STAGE1_CKPT" ]]; then
    echo "No TOKEN_CACHE exists and STAGE1_CKPT is empty; cannot extract cache." >&2
    exit 1
  fi
  echo "=== Token cache extraction: $MODEL_FAMILY $CASE_NAME ==="
  "\$PYTHON_BIN" cache.py \\
    --stage1-checkpoint "$STAGE1_CKPT" \\
    --output-path "$token_cache" \\
    --dataset vctk \\
    --data-dir "$VCTK_DIR" \\
    --image-size 128 \\
    --batch-size "$CACHE_BATCH_SIZE" \\
    --num-workers "$CACHE_NUM_WORKERS" \\
    --seed 42 \\
    --model-type "$MODEL_FAMILY" \\
    --audio-representation waveform \\
    --audio-dc-remove \\
    --audio-peak-normalize \\
    --audio-target-peak "$AUDIO_TARGET_PEAK" \\
    --audio-rms-normalize \\
    --audio-target-rms "$AUDIO_TARGET_RMS" \\
    --audio-max-gain "$AUDIO_MAX_GAIN" \\
    --audio-min-crop-rms "$AUDIO_MIN_CROP_RMS" \\
    --audio-crop-attempts "$AUDIO_CROP_ATTEMPTS" \\
    --audio-fade-samples "$AUDIO_FADE_SAMPLES" \\
    --coeff-bins "$COEFF_BINS" \\
    --coeff-max "$COEFF_MAX" \\
    --coeff-quantization "$COEFF_QUANTIZATION" \\
    --coeff-mu "$COEFF_MU"
else
  echo "Using existing token cache: $token_cache"
fi

echo "=== Stage 2: $MODEL_FAMILY $CASE_NAME, ${STAGE2_EPOCHS} epochs ==="
"\$PYTHON_BIN" train_stage2_prior.py \\
  ckpt_path="$CKPT_PATH" \\
  token_cache_path="$token_cache" \\
  output_dir="$stage2_dir" \\
  seed=42 \\
  ar.type=sparse_spatial_depth \\
  ar.max_steps=-1 \\
  ar.d_model="$AR_D_MODEL" \\
  ar.n_heads="$AR_N_HEADS" \\
  ar.n_layers="$AR_N_LAYERS" \\
  ar.d_ff="$AR_D_FF" \\
  ar.learning_rate="$AR_LR" \\
  ar.warmup_steps="$AR_WARMUP_STEPS" \\
  ar.min_lr_ratio="$AR_MIN_LR_RATIO" \\
  ar.coeff_loss_type="$COEFF_LOSS_TYPE" \\
  ar.coeff_loss_weight="$COEFF_LOSS_WEIGHT" \\
  ar.coeff_huber_delta="$COEFF_HUBER_DELTA" \\
  train_ar.max_epochs="$STAGE2_EPOCHS" \\
  train_ar.batch_size="$STAGE2_BATCH_SIZE" \\
  train_ar.max_items=0 \\
  train_ar.limit_train_batches=1.0 \\
  train_ar.limit_val_batches=1.0 \\
  train_ar.limit_test_batches=1.0 \\
  train_ar.log_every_n_steps=50 \\
  train_ar.sample_every_n_epochs="$SAMPLE_EVERY_N_EPOCHS" \\
  train_ar.sample_log_to_wandb="$SAMPLE_LOG_TO_WANDB" \\
  train_ar.sample_num_images="$SAMPLE_NUM_IMAGES" \\
  train_ar.sample_temperature="$SAMPLE_TEMPERATURE" \\
  train_ar.sample_top_k="$SAMPLE_TOP_K" \\
  train_ar.sample_coeff_mode="$SAMPLE_COEFF_MODE" \\
  train_ar.generation_metric_num_samples="$GENERATION_METRIC_NUM_SAMPLES" \\
  train_ar.compute_generation_fid=false \\
  train_ar.compute_audio_generation_metrics="$COMPUTE_AUDIO_GENERATION_METRICS" \\
  train_ar.run_test_after_fit="$RUN_TEST_AFTER_FIT" \\
  train_ar.save_final_samples_after_fit="$SAVE_FINAL_SAMPLES_AFTER_FIT" \\
  train_ar.devices=1 \\
  train_ar.strategy=auto \\
  train_ar.precision=bf16-mixed \\
  train_ar.accelerator=gpu \\
  data.dataset=vctk \\
  data.data_dir="$VCTK_DIR" \\
  data.image_size=128 \\
  data.num_workers="$STAGE2_NUM_WORKERS" \\
  wandb.project="$PROJECT" \\
  wandb.group="$wandb_group" \\
  wandb.name="vctk-waveform-${MODEL_FAMILY}-${safe_case}-stage2-${STAGE2_EPOCHS}" \\
  wandb.tags="[vctk,waveform,${MODEL_FAMILY},stage2,${safe_case}]" \\
  wandb.append_timestamp=false \\
  wandb.id="$WANDB_ID" \\
  wandb.resume="$WANDB_RESUME" \\
  wandb.save_dir="$stage2_dir/wandb"
EOF
chmod +x "$run_script"

cat > "$sbatch_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
  fi
fi

CONTAINER_BIN=""
for candidate in singularity apptainer; do
  if command -v "\$candidate" >/dev/null 2>&1; then
    CONTAINER_BIN="\$candidate"
    break
  fi
done

if [[ -z "\$CONTAINER_BIN" ]]; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
  module load apptainer 2>/dev/null || true
  for candidate in singularity apptainer; do
    if command -v "\$candidate" >/dev/null 2>&1; then
      CONTAINER_BIN="\$candidate"
      break
    fi
  done
fi

if [[ -n "\$CONTAINER_BIN" ]]; then
  "\$CONTAINER_BIN" exec --nv \\
    --bind "$snapshot_dir" \\
    --bind "/scratch/$USER" \\
    --bind "/cache/home/$USER" \\
    --bind "$VCTK_DIR" \\
    --bind "$run_dir" \\
    --bind /dev/shm \\
    "$IMAGE" \\
    bash "$run_script"
else
  bash "$run_script"
fi
EOF
chmod +x "$sbatch_script"

SBATCH_ARGS=(
  --partition="$PARTITION"
  --job-name="${MODEL_FAMILY:0:2}-vctk-s2"
  --nodes=1
  --ntasks=1
  --cpus-per-task="$CPUS_PER_TASK"
  --gres=gpu:1
  --mem="$MEM_MB"
  --time="$TIME_LIMIT"
  --chdir="$snapshot_dir"
  --output="${log_base}_%j.out"
  --error="${log_base}_%j.err"
)
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi
if [[ -n "$EXCLUDE" ]]; then
  SBATCH_ARGS+=(--exclude="$EXCLUDE")
fi

job_id="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$sbatch_script")"

echo "Submitted VCTK waveform stage2 job: $job_id"
echo "model_family=$MODEL_FAMILY"
echo "case=$CASE_NAME"
echo "stage1_checkpoint=$STAGE1_CKPT"
echo "snapshot=$snapshot_dir"
echo "run_dir=$run_dir"
echo "stdout=${log_base}_${job_id}.out"
echo "stderr=${log_base}_${job_id}.err"
