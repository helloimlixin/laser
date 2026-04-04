#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${USER:-$(id -un)}"

PARTITION="${PARTITION:-gpu-redhat}"
GPUS="${GPUS:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

DATA_DIR="${DATA_DIR:-/cache/home/$USER_NAME/Projects/data/celeba_hq_256}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER_NAME/runs/src_patch_sweep}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER_NAME/submission_snapshots}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-laser_src_patch_sweep}"
RUN_PREFIX="${RUN_PREFIX:-src_patch}"
JOB_PREFIX="${JOB_PREFIX:-src-patch}"

IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER_NAME/.pydeps/laser_src_py}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER_NAME/.secrets/wandb_api_key}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
WANDB_MODE="${WANDB_MODE:-online}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-8}"
TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-2}"
TRAIN_PRECISION="${TRAIN_PRECISION:-16-mixed}"
AR_PRECISION="${AR_PRECISION:-16-mixed}"
TRAIN_AR_DEVICES="${TRAIN_AR_DEVICES:-$GPUS}"
TRAIN_AR_STRATEGY="${TRAIN_AR_STRATEGY:-}"
CASE_FILTER="${CASE_FILTER:-}"

STAGE1_LR="${STAGE1_LR:-2.5e-4}"
STAGE2_LR="${STAGE2_LR:-3e-4}"
MODEL_NUM_HIDDENS="${MODEL_NUM_HIDDENS:-128}"
MODEL_NUM_RESIDUAL_BLOCKS="${MODEL_NUM_RESIDUAL_BLOCKS:-2}"
MODEL_NUM_RESIDUAL_HIDDENS="${MODEL_NUM_RESIDUAL_HIDDENS:-32}"
MODEL_EMBEDDING_DIM="${MODEL_EMBEDDING_DIM:-4}"
PATCH_SIZE="${PATCH_SIZE:-8}"
PATCH_STRIDE="${PATCH_STRIDE:-4}"
PATCH_RECONSTRUCTION="${PATCH_RECONSTRUCTION:-hann}"
BOTTLENECK_COEF_MAX="${BOTTLENECK_COEF_MAX:-8.0}"
BOUNDED_OMP_REFINE_STEPS="${BOUNDED_OMP_REFINE_STEPS:-8}"

AR_D_MODEL="${AR_D_MODEL:-512}"
AR_N_HEADS="${AR_N_HEADS:-8}"
AR_N_LAYERS="${AR_N_LAYERS:-6}"
AR_D_FF="${AR_D_FF:-2048}"
AR_DROPOUT="${AR_DROPOUT:-0.1}"
AR_WARMUP_STEPS="${AR_WARMUP_STEPS:-1000}"
AR_MIN_LR_RATIO="${AR_MIN_LR_RATIO:-0.01}"
AR_WEIGHT_DECAY="${AR_WEIGHT_DECAY:-0.01}"
AR_COEFF_LOSS_TYPE="${AR_COEFF_LOSS_TYPE:-auto}"
STAGE2_SAMPLE_EVERY_N_STEPS="${STAGE2_SAMPLE_EVERY_N_STEPS:-250}"
STAGE2_SAMPLE_EVERY_N_EPOCHS="${STAGE2_SAMPLE_EVERY_N_EPOCHS:-0}"
STAGE2_SAMPLE_NUM_IMAGES="${STAGE2_SAMPLE_NUM_IMAGES:-16}"
STAGE2_SAMPLE_TEMPERATURE="${STAGE2_SAMPLE_TEMPERATURE:-1.0}"
STAGE2_SAMPLE_TOP_K="${STAGE2_SAMPLE_TOP_K:-0}"
STAGE2_SAMPLE_LOG_TO_WANDB="${STAGE2_SAMPLE_LOG_TO_WANDB:-false}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -z "$TRAIN_AR_STRATEGY" ]]; then
  if [[ "${TRAIN_AR_DEVICES}" == "1" ]]; then
    TRAIN_AR_STRATEGY="auto"
  else
    TRAIN_AR_STRATEGY="ddp"
  fi
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
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

runner_script="$snapshot_dir/.run_src_patch_case.sh"
cat > "$runner_script" <<'EOF'
#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?}"
RUN_DIR="${RUN_DIR:?}"
RUN_NAME="${RUN_NAME:?}"
IMAGE="${IMAGE:?}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:?}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:?}"
WANDB_PROJECT="${WANDB_PROJECT:?}"
WANDB_MODE="${WANDB_MODE:?}"
DATA_DIR="${DATA_DIR:?}"
IMAGE_SIZE="${IMAGE_SIZE:?}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:?}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:?}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:?}"
TOKEN_BATCH_SIZE="${TOKEN_BATCH_SIZE:?}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:?}"
NUM_WORKERS="${NUM_WORKERS:?}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:?}"
TRAIN_PRECISION="${TRAIN_PRECISION:?}"
AR_PRECISION="${AR_PRECISION:?}"
GPUS="${GPUS:?}"
TRAIN_AR_DEVICES="${TRAIN_AR_DEVICES:?}"
TRAIN_AR_STRATEGY="${TRAIN_AR_STRATEGY:?}"
STAGE1_LR="${STAGE1_LR:?}"
STAGE2_LR="${STAGE2_LR:?}"
MODEL_NUM_HIDDENS="${MODEL_NUM_HIDDENS:?}"
MODEL_NUM_RESIDUAL_BLOCKS="${MODEL_NUM_RESIDUAL_BLOCKS:?}"
MODEL_NUM_RESIDUAL_HIDDENS="${MODEL_NUM_RESIDUAL_HIDDENS:?}"
MODEL_EMBEDDING_DIM="${MODEL_EMBEDDING_DIM:?}"
PATCH_SIZE="${PATCH_SIZE:?}"
PATCH_STRIDE="${PATCH_STRIDE:?}"
PATCH_RECONSTRUCTION="${PATCH_RECONSTRUCTION:?}"
BOTTLENECK_COEF_MAX="${BOTTLENECK_COEF_MAX:?}"
BOUNDED_OMP_REFINE_STEPS="${BOUNDED_OMP_REFINE_STEPS:?}"
AR_D_MODEL="${AR_D_MODEL:?}"
AR_N_HEADS="${AR_N_HEADS:?}"
AR_N_LAYERS="${AR_N_LAYERS:?}"
AR_D_FF="${AR_D_FF:?}"
AR_DROPOUT="${AR_DROPOUT:?}"
AR_WARMUP_STEPS="${AR_WARMUP_STEPS:?}"
AR_MIN_LR_RATIO="${AR_MIN_LR_RATIO:?}"
AR_WEIGHT_DECAY="${AR_WEIGHT_DECAY:?}"
AR_COEFF_LOSS_TYPE="${AR_COEFF_LOSS_TYPE:?}"
STAGE2_SAMPLE_EVERY_N_STEPS="${STAGE2_SAMPLE_EVERY_N_STEPS:?}"
STAGE2_SAMPLE_EVERY_N_EPOCHS="${STAGE2_SAMPLE_EVERY_N_EPOCHS:?}"
STAGE2_SAMPLE_NUM_IMAGES="${STAGE2_SAMPLE_NUM_IMAGES:?}"
STAGE2_SAMPLE_TEMPERATURE="${STAGE2_SAMPLE_TEMPERATURE:?}"
STAGE2_SAMPLE_TOP_K="${STAGE2_SAMPLE_TOP_K:?}"
STAGE2_SAMPLE_LOG_TO_WANDB="${STAGE2_SAMPLE_LOG_TO_WANDB:?}"
CACHE_MODE="${CACHE_MODE:?}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:?}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:?}"
COEFF_VOCAB_SIZE="${COEFF_VOCAB_SIZE:-0}"
COEFF_QUANTIZATION="${COEFF_QUANTIZATION:-uniform}"
COEFF_MU="${COEFF_MU:-0.0}"

USER_NAME="$(id -un)"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory not found on compute node: $DATA_DIR" >&2
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

mkdir -p "$RUN_DIR" "$PYTHONUSERBASE_DIR"

if [[ -f "$WANDB_API_KEY_FILE" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\n' < "$WANDB_API_KEY_FILE")"
fi

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
export WANDB_MODE

check_imports='import hydra, lightning, torchvision, wandb, omegaconf, rich, torchmetrics, scipy'
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

STAGE1_OUT="$RUN_DIR/stage1"
STAGE2_OUT="$RUN_DIR/stage2"
mkdir -p "$STAGE1_OUT" "$STAGE2_OUT"

stage1_cmd=(
  python3 "$PROJECT_DIR/train.py"
  "output_dir=$STAGE1_OUT"
  "hydra.run.dir=$STAGE1_OUT/hydra"
  "model=laser"
  "data=celeba"
  "data.data_dir=$DATA_DIR"
  "data.batch_size=$STAGE1_BATCH_SIZE"
  "data.num_workers=$NUM_WORKERS"
  "data.image_size=$IMAGE_SIZE"
  "train.learning_rate=$STAGE1_LR"
  "train.max_epochs=$STAGE1_EPOCHS"
  "train.accelerator=gpu"
  "train.devices=$GPUS"
  "train.strategy=ddp"
  "train.precision=$TRAIN_PRECISION"
  "train.log_every_n_steps=25"
  "train.run_test_after_fit=false"
  "checkpoint.save_top_k=1"
  "model.num_hiddens=$MODEL_NUM_HIDDENS"
  "model.embedding_dim=$MODEL_EMBEDDING_DIM"
  "model.num_embeddings=$NUM_EMBEDDINGS"
  "model.sparsity_level=$SPARSITY_LEVEL"
  "model.num_residual_blocks=$MODEL_NUM_RESIDUAL_BLOCKS"
  "model.num_residual_hiddens=$MODEL_NUM_RESIDUAL_HIDDENS"
  "model.patch_based=true"
  "model.patch_size=$PATCH_SIZE"
  "model.patch_stride=$PATCH_STRIDE"
  "model.patch_reconstruction=$PATCH_RECONSTRUCTION"
  "model.coef_max=$BOTTLENECK_COEF_MAX"
  "model.bounded_omp_refine_steps=$BOUNDED_OMP_REFINE_STEPS"
  "model.perceptual_weight=0.0"
  "model.compute_fid=false"
  "model.log_images_every_n_steps=0"
  "wandb.project=$WANDB_PROJECT"
  "wandb.name=${RUN_NAME}_s1"
  "wandb.save_dir=$STAGE1_OUT/wandb"
)

singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER_NAME" \
  --bind "/cache/home/$USER_NAME" \
  "$IMAGE" \
  "${stage1_cmd[@]}"

shopt -s nullglob
stage1_ckpts=( "$STAGE1_OUT"/checkpoints/run_*/laser/last.ckpt )
if ((${#stage1_ckpts[@]} == 0)); then
  stage1_ckpts=( "$STAGE1_OUT"/checkpoints/run_*/laser/*.ckpt )
fi
if ((${#stage1_ckpts[@]} == 0)); then
  echo "No stage-1 checkpoint found under $STAGE1_OUT/checkpoints" >&2
  exit 1
fi
stage1_ckpt="$(ls -t "${stage1_ckpts[@]}" | head -n 1)"

token_cache_cmd=(
  python3 "$PROJECT_DIR/build_token_cache.py"
  --stage1_checkpoint "$stage1_ckpt"
  --dataset celeba
  --data_dir "$DATA_DIR"
  --split train
  --image_size "$IMAGE_SIZE"
  --batch_size "$TOKEN_BATCH_SIZE"
  --num_workers "$TOKEN_NUM_WORKERS"
  --coeff_max "$BOTTLENECK_COEF_MAX"
  --ar_output_dir "$STAGE2_OUT"
  --device cuda
  --cache_mode "$CACHE_MODE"
)
if [[ "$CACHE_MODE" == "quantized" ]]; then
  token_cache_cmd+=(
    --coeff_vocab_size "$COEFF_VOCAB_SIZE"
    --coeff_quantization "$COEFF_QUANTIZATION"
    --coeff_mu "$COEFF_MU"
  )
fi

singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER_NAME" \
  --bind "/cache/home/$USER_NAME" \
  "$IMAGE" \
  "${token_cache_cmd[@]}"

token_caches=( "$STAGE2_OUT"/token_cache/*.pt )
if ((${#token_caches[@]} == 0)); then
  echo "No token cache found under $STAGE2_OUT/token_cache" >&2
  exit 1
fi
token_cache_path="$(ls -t "${token_caches[@]}" | head -n 1)"

stage2_cmd=(
  python3 "$PROJECT_DIR/train_ar.py"
  "output_dir=$STAGE2_OUT"
  "token_cache_path=$token_cache_path"
  "data.dataset=celeba"
  "data.data_dir=$DATA_DIR"
  "data.image_size=$IMAGE_SIZE"
  "data.num_workers=$TOKEN_NUM_WORKERS"
  "wandb.project=$WANDB_PROJECT"
  "wandb.name=${RUN_NAME}_s2"
  "wandb.save_dir=$STAGE2_OUT/wandb"
  "ar.type=sparse_spatial_depth"
  "ar.d_model=$AR_D_MODEL"
  "ar.n_heads=$AR_N_HEADS"
  "ar.n_layers=$AR_N_LAYERS"
  "ar.d_ff=$AR_D_FF"
  "ar.dropout=$AR_DROPOUT"
  "ar.learning_rate=$STAGE2_LR"
  "ar.warmup_steps=$AR_WARMUP_STEPS"
  "ar.min_lr_ratio=$AR_MIN_LR_RATIO"
  "ar.weight_decay=$AR_WEIGHT_DECAY"
  "ar.coeff_loss_type=$AR_COEFF_LOSS_TYPE"
  "train_ar.batch_size=$STAGE2_BATCH_SIZE"
  "train_ar.max_epochs=$STAGE2_EPOCHS"
  "train_ar.accelerator=gpu"
  "train_ar.devices=$TRAIN_AR_DEVICES"
  "train_ar.strategy=$TRAIN_AR_STRATEGY"
  "train_ar.precision=$AR_PRECISION"
  "train_ar.log_every_n_steps=25"
  "train_ar.sample_every_n_steps=$STAGE2_SAMPLE_EVERY_N_STEPS"
  "train_ar.sample_every_n_epochs=$STAGE2_SAMPLE_EVERY_N_EPOCHS"
  "train_ar.sample_num_images=$STAGE2_SAMPLE_NUM_IMAGES"
  "train_ar.sample_temperature=$STAGE2_SAMPLE_TEMPERATURE"
  "train_ar.sample_top_k=$STAGE2_SAMPLE_TOP_K"
  "train_ar.sample_log_to_wandb=$STAGE2_SAMPLE_LOG_TO_WANDB"
)

singularity exec --nv \
  --bind "$PROJECT_DIR" \
  --bind "/scratch/$USER_NAME" \
  --bind "/cache/home/$USER_NAME" \
  "$IMAGE" \
  "${stage2_cmd[@]}"
EOF
chmod +x "$runner_script"

cases=(
  "base|real_valued|4096|16|0|uniform|0.0"
  "widek|real_valued|4096|24|0|uniform|0.0"
  "widea|real_valued|6144|16|0|uniform|0.0"
  "qbase|quantized|4096|16|512|uniform|0.0"
)

submitted=0
submitted_jobs=()
filter_token=",$(echo "$CASE_FILTER" | tr -d ' '),"

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name cache_mode num_embeddings sparsity_level coeff_vocab_size coeff_quantization coeff_mu <<< "$case_spec"

  if [[ -n "${CASE_FILTER// }" && "$filter_token" != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}_${STAGE1_EPOCHS}s1_${STAGE2_EPOCHS}s2_p${PATCH_SIZE}s${PATCH_STRIDE}_a${num_embeddings}_k${sparsity_level}"
  job_name="${JOB_PREFIX}-${case_name}"
  run_dir="$RUN_ROOT/$run_name"
  mkdir -p "$run_dir"

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
      --export=ALL,PROJECT_DIR="$snapshot_dir",RUN_DIR="$run_dir",RUN_NAME="$run_name",IMAGE="$IMAGE",PYTHONUSERBASE_DIR="$PYTHONUSERBASE_DIR",WANDB_API_KEY_FILE="$WANDB_API_KEY_FILE",WANDB_PROJECT="$WANDB_PROJECT",WANDB_MODE="$WANDB_MODE",DATA_DIR="$DATA_DIR",IMAGE_SIZE="$IMAGE_SIZE",STAGE1_EPOCHS="$STAGE1_EPOCHS",STAGE2_EPOCHS="$STAGE2_EPOCHS",STAGE1_BATCH_SIZE="$STAGE1_BATCH_SIZE",TOKEN_BATCH_SIZE="$TOKEN_BATCH_SIZE",STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE",NUM_WORKERS="$NUM_WORKERS",TOKEN_NUM_WORKERS="$TOKEN_NUM_WORKERS",TRAIN_PRECISION="$TRAIN_PRECISION",AR_PRECISION="$AR_PRECISION",GPUS="$GPUS",TRAIN_AR_DEVICES="$TRAIN_AR_DEVICES",TRAIN_AR_STRATEGY="$TRAIN_AR_STRATEGY",STAGE1_LR="$STAGE1_LR",STAGE2_LR="$STAGE2_LR",MODEL_NUM_HIDDENS="$MODEL_NUM_HIDDENS",MODEL_NUM_RESIDUAL_BLOCKS="$MODEL_NUM_RESIDUAL_BLOCKS",MODEL_NUM_RESIDUAL_HIDDENS="$MODEL_NUM_RESIDUAL_HIDDENS",MODEL_EMBEDDING_DIM="$MODEL_EMBEDDING_DIM",PATCH_SIZE="$PATCH_SIZE",PATCH_STRIDE="$PATCH_STRIDE",PATCH_RECONSTRUCTION="$PATCH_RECONSTRUCTION",BOTTLENECK_COEF_MAX="$BOTTLENECK_COEF_MAX",BOUNDED_OMP_REFINE_STEPS="$BOUNDED_OMP_REFINE_STEPS",AR_D_MODEL="$AR_D_MODEL",AR_N_HEADS="$AR_N_HEADS",AR_N_LAYERS="$AR_N_LAYERS",AR_D_FF="$AR_D_FF",AR_DROPOUT="$AR_DROPOUT",AR_WARMUP_STEPS="$AR_WARMUP_STEPS",AR_MIN_LR_RATIO="$AR_MIN_LR_RATIO",AR_WEIGHT_DECAY="$AR_WEIGHT_DECAY",AR_COEFF_LOSS_TYPE="$AR_COEFF_LOSS_TYPE",STAGE2_SAMPLE_EVERY_N_STEPS="$STAGE2_SAMPLE_EVERY_N_STEPS",STAGE2_SAMPLE_EVERY_N_EPOCHS="$STAGE2_SAMPLE_EVERY_N_EPOCHS",STAGE2_SAMPLE_NUM_IMAGES="$STAGE2_SAMPLE_NUM_IMAGES",STAGE2_SAMPLE_TEMPERATURE="$STAGE2_SAMPLE_TEMPERATURE",STAGE2_SAMPLE_TOP_K="$STAGE2_SAMPLE_TOP_K",STAGE2_SAMPLE_LOG_TO_WANDB="$STAGE2_SAMPLE_LOG_TO_WANDB",CACHE_MODE="$cache_mode",NUM_EMBEDDINGS="$num_embeddings",SPARSITY_LEVEL="$sparsity_level",COEFF_VOCAB_SIZE="$coeff_vocab_size",COEFF_QUANTIZATION="$coeff_quantization",COEFF_MU="$coeff_mu" \
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
