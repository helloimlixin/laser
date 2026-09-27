#!/bin/bash
set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SELF="${SELF:-$PROJECT_DIR/scripts/submit_cc3m_x3h5cl0h_text2image.sh}"

set_defaults() {
  TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
  PARTITION="${PARTITION:-gpu-redhat}"
  CONSTRAINT="${CONSTRAINT:-adalovelace}"
  EXCLUDE_NODES="${EXCLUDE_NODES:-}"
  DEPENDENCY="${DEPENDENCY:-}"
  NODES="${NODES:-4}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-24}"
  MEM_MB="${MEM_MB:-250000}"
  TIME_LIMIT="${TIME_LIMIT:-72:00:00}"

  DATA_ROOT="${CC3M_ROOT:-/scratch/$USER/Projects/data/cc3m}"
  STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-/scratch/$USER/runs/imagenet_x3h5cl0h_strict_bottleneck_sweep/imagenet-x3h5cl0h-strict-bottleneck-sweep-20260719_014434/k2-a16384/in256-rqvae-laser-8x8-a16384-k2/19072026_024726}"
  STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$STAGE1_RUN_DIR/best_rfid_slot3_model.pt}"
  STAGE1_CONFIG="${STAGE1_CONFIG:-$STAGE1_RUN_DIR/config.yaml}"
  STAGE1_SOURCE_DIR="${STAGE1_SOURCE_DIR:-$STAGE1_RUN_DIR}"
  STAGE1_WANDB_RUN="${STAGE1_WANDB_RUN:-helloimlixin-rutgers/laser/x3h5cl0h-a16384-k2-20260719-014434}"

  RECIPE_NAME="${RECIPE_NAME:-cc3m-text2image-imagenet-x3h5cl0h-a16384-k2}"
  JOB_NAME="${JOB_NAME:-cc3m-x3-s2}"
  RUN_NAME="${RUN_NAME:-$RECIPE_NAME-$TS}"
  RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser/$RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/stage2}"
  LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/$RUN_NAME}"
  RUN_LOCK="${RUN_LOCK:-$RUN_DIR/active-training.lock}"
  TOKEN_CACHE="${TOKEN_CACHE:-$PROJECT_DIR/outputs/cc3m_x3h5cl0h_a16384k2_text2image/token_cache/cc3m_train_imagenet_x3h5cl0h_a16384k2_q128_rq_bpe16k_text32.pt}"

  ATOM_VOCAB_SIZE="${ATOM_VOCAB_SIZE:-16384}"
  SPARSITY_LEVEL="${SPARSITY_LEVEL:-2}"
  EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
  COEFF_BINS="${COEFF_BINS:-128}"
  TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-32}"
  TEXT_TOKENIZER="${TEXT_TOKENIZER:-bpe16k_huggingface}"
  BPE_DROPOUT="${BPE_DROPOUT:-0.1}"
  CACHE_MODE="${CACHE_MODE:-quantized}"
  if [[ "$CACHE_MODE" != "quantized" && "$CACHE_MODE" != "real_valued" ]]; then
    echo "CACHE_MODE must be quantized or real_valued, got $CACHE_MODE" >&2
    exit 1
  fi
  CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-32}"
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-8}"
  CACHE_MAX_ITEMS="${CACHE_MAX_ITEMS:-0}"

  TARGET_EPOCHS="${TARGET_EPOCHS:-100}"
  TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-2048}"
  S2_BATCH_SIZE="${S2_BATCH_SIZE:-32}"
  S2_NUM_WORKERS="${S2_NUM_WORKERS:-8}"
  S2_MAX_STEPS="${S2_MAX_STEPS:--1}"
  S2_D_MODEL="${S2_D_MODEL:-1280}"
  S2_LAYERS="${S2_LAYERS:-26}"
  S2_DEPTH_LAYERS="${S2_DEPTH_LAYERS:-4}"
  S2_HEADS="${S2_HEADS:-20}"
  S2_FF="${S2_FF:-5120}"
  S2_LR="${S2_LR:-5e-4}"
  S2_WEIGHT_DECAY="${S2_WEIGHT_DECAY:-1e-4}"
  S2_WARMUP_STEPS="${S2_WARMUP_STEPS:-0}"
  S2_MIN_LR_RATIO="${S2_MIN_LR_RATIO:-0.0}"
  S2_VAL_BATCHES="${S2_VAL_BATCHES:-1.0}"
  S2_STRATEGY="${S2_STRATEGY:-ddp_find_unused_parameters_true}"
  SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-1}"
  RECOVERY_SAVE_STEP_FREQ="${RECOVERY_SAVE_STEP_FREQ:-500}"
  CHECKPOINT_UPLOAD_EVERY="${CHECKPOINT_UPLOAD_EVERY:-1}"
  SAMPLE_EVERY_N_STEPS="${SAMPLE_EVERY_N_STEPS:-0}"
  SAMPLE_EVERY_N_EPOCHS="${SAMPLE_EVERY_N_EPOCHS:-1}"
  SAMPLE_NUM_IMAGES="${SAMPLE_NUM_IMAGES:-64}"
  SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-1.0}"
  SAMPLE_TOP_K="${SAMPLE_TOP_K:-$ATOM_VOCAB_SIZE}"
  SAMPLE_TOP_P="${SAMPLE_TOP_P:-0.7}"
  ATOM_LABEL_SMOOTHING="${ATOM_LABEL_SMOOTHING:-0.0}"
  ATOM_COVERAGE_WEIGHT="${ATOM_COVERAGE_WEIGHT:-0.0}"

  WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  WANDB_ID="${WANDB_ID:-cc3mx3$(date +%m%d%H%M%S)}"
  WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
  WANDB_GROUP="${WANDB_GROUP:-$RECIPE_NAME}"

  IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
  PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
  MASTER_PORT="${MASTER_PORT:-$((21000 + (${SLURM_JOB_ID:-0} % 20000)))}"

  export TS PARTITION CONSTRAINT EXCLUDE_NODES DEPENDENCY NODES GPUS_PER_NODE CPUS_PER_TASK MEM_MB TIME_LIMIT
  export DATA_ROOT STAGE1_RUN_DIR STAGE1_CHECKPOINT STAGE1_CONFIG STAGE1_SOURCE_DIR STAGE1_WANDB_RUN
  export RECIPE_NAME JOB_NAME RUN_NAME RUN_DIR OUTPUT_DIR LOCAL_SCRATCH_ROOT RUN_LOCK TOKEN_CACHE
  export ATOM_VOCAB_SIZE SPARSITY_LEVEL EMBEDDING_DIM COEFF_BINS TEXT_MAX_LENGTH TEXT_TOKENIZER BPE_DROPOUT CACHE_MODE
  export CACHE_BATCH_SIZE CACHE_NUM_WORKERS CACHE_MAX_ITEMS
  export TARGET_EPOCHS TOTAL_BATCH_SIZE S2_BATCH_SIZE S2_NUM_WORKERS S2_MAX_STEPS
  export S2_D_MODEL S2_LAYERS S2_DEPTH_LAYERS S2_HEADS S2_FF S2_LR S2_WEIGHT_DECAY S2_WARMUP_STEPS S2_MIN_LR_RATIO S2_VAL_BATCHES S2_STRATEGY
  export SAVE_CKPT_FREQ RECOVERY_SAVE_STEP_FREQ CHECKPOINT_UPLOAD_EVERY SAMPLE_EVERY_N_STEPS SAMPLE_EVERY_N_EPOCHS
  export SAMPLE_NUM_IMAGES SAMPLE_TEMPERATURE SAMPLE_TOP_K SAMPLE_TOP_P ATOM_LABEL_SMOOTHING ATOM_COVERAGE_WEIGHT
  export WANDB_ENTITY WANDB_PROJECT WANDB_ID WANDB_NAME WANDB_GROUP
  export IMAGE PYDEPS MASTER_PORT PROJECT_DIR SELF
}

submit_job() {
  set_defaults
  if [[ ! -d "$DATA_ROOT" || ! -d "$DATA_ROOT/wds" ]]; then
    echo "CC3M WebDataset shards not found under $DATA_ROOT" >&2
    exit 1
  fi
  if [[ ! -f "$STAGE1_CHECKPOINT" || ! -f "$STAGE1_CONFIG" || ! -d "$STAGE1_SOURCE_DIR/rqvae" ]]; then
    echo "Stage-1 checkpoint/config/source missing under $STAGE1_RUN_DIR" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$(dirname "$TOKEN_CACHE")"

  echo "=== CC3M text-to-image stage-2 from ImageNet x3h5cl0h a16384/k2 ==="
  echo "  run_dir=$RUN_DIR"
  echo "  wandb=$WANDB_ENTITY/$WANDB_PROJECT id=$WANDB_ID name=$WANDB_NAME"
  echo "  resources: partition=$PARTITION constraint=$CONSTRAINT nodes=$NODES gpus_per_node=$GPUS_PER_NODE world_size=$((NODES * GPUS_PER_NODE))"
  echo "  data=$DATA_ROOT"
  echo "  stage1_run=$STAGE1_WANDB_RUN"
  echo "  stage1_checkpoint=$STAGE1_CHECKPOINT"
  echo "  token_cache=$TOKEN_CACHE"
  echo "  cache_mode=$CACHE_MODE"
  echo "  tokenizer=$TEXT_TOKENIZER context=$TEXT_MAX_LENGTH bpe_dropout=$BPE_DROPOUT"
  echo "  architecture=8x8x4 d_model=$S2_D_MODEL layers=$S2_LAYERS+$S2_DEPTH_LAYERS heads=$S2_HEADS"
  echo "  optimization=adamW lr=$S2_LR weight_decay=$S2_WEIGHT_DECAY local_batch=$S2_BATCH_SIZE total_batch=$TOTAL_BATCH_SIZE epochs=$TARGET_EPOCHS"
  echo "  sampling=temperature=$SAMPLE_TEMPERATURE top_k=$SAMPLE_TOP_K top_p=$SAMPLE_TOP_P"

  sbatch_args=(
    --partition="$PARTITION"
    --job-name="$JOB_NAME"
    --nodes="$NODES"
    --ntasks="$NODES"
    --ntasks-per-node=1
    --cpus-per-task="$CPUS_PER_TASK"
    --gres="gpu:$GPUS_PER_NODE"
    --mem="$MEM_MB"
    --time="$TIME_LIMIT"
    --chdir="$PROJECT_DIR"
    --output="$RUN_DIR/slurm-%j.out"
    --error="$RUN_DIR/slurm-%j.err"
    --requeue
  )
  if [[ -n "$CONSTRAINT" ]]; then
    sbatch_args+=(--constraint="$CONSTRAINT")
  fi
  if [[ -n "$EXCLUDE_NODES" ]]; then
    sbatch_args+=(--exclude="$EXCLUDE_NODES")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    sbatch_args+=(--dependency="$DEPENDENCY")
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN sbatch'
    printf ' %q' "${sbatch_args[@]}" "$SELF" --worker
    printf '\n'
    exit 0
  fi
  sbatch "${sbatch_args[@]}" "$SELF" --worker
}

container_prefix() {
  if ! command -v module >/dev/null 2>&1; then
    if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
      set +u; source /usr/share/lmod/lmod/init/bash; set -u
    elif [[ -f /usr/share/Modules/init/bash ]]; then
      set +u; source /usr/share/Modules/init/bash; set -u
    fi
  fi
  if ! command -v singularity >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
  fi
  if command -v singularity >/dev/null 2>&1; then
    CONTAINER=(
      singularity exec --nv
      --bind "$PROJECT_DIR"
      --bind "/scratch/$USER"
      --bind "$DATA_ROOT"
      --bind "$RUN_DIR"
      --bind "$STAGE1_RUN_DIR"
      --bind /mnt/scratch
      --bind /dev/shm
      "$IMAGE"
    )
  else
    echo "Warning: singularity not found; running bare." >&2
    CONTAINER=()
  fi
}

run_worker() {
  set_defaults
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$LOCAL_SCRATCH_ROOT"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$RUN_LOCK"
    if ! flock -n 9; then
      echo "Another CC3M x3h5cl0h training allocation already holds $RUN_LOCK; exiting backup job."
      exit 0
    fi
    echo "Acquired run lock: $RUN_LOCK"
  else
    echo "Warning: flock not found; backup jobs may need manual cancellation if multiple allocations start." >&2
  fi
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  export MASTER_ADDR MASTER_PORT

  echo "=== SLURM allocation ==="
  echo "  job=$SLURM_JOB_ID nodes=$SLURM_NNODES nodelist=$SLURM_NODELIST"
  echo "  master=$MASTER_ADDR port=$MASTER_PORT"
  echo "  run_dir=$RUN_DIR"

  container_prefix
  "${CONTAINER[@]}" bash "$SELF" --inside setup
  "${CONTAINER[@]}" bash "$SELF" --inside cache
  "${CONTAINER[@]}" bash "$SELF" --inside validate
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside train
}

ensure_runtime() {
  export PYTHONUSERBASE="$PYDEPS"
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:$PROJECT_DIR/third_party/rq-vae-transformer:${PYTHONPATH:-}"
  export WANDB_ENTITY WANDB_PROJECT
  export WANDB_DIR="${WANDB_DIR:-$LOCAL_SCRATCH_ROOT/wandb/run}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$LOCAL_SCRATCH_ROOT/wandb/cache}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$LOCAL_SCRATCH_ROOT/wandb/data}"
  export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$LOCAL_SCRATCH_ROOT/wandb/artifacts}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH_ROOT/cache}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
  export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
  export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"
  export TEMP="$TMPDIR"
  export TMP="$TMPDIR"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export HYDRA_FULL_ERROR=1
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  unset NCCL_ASYNC_ERROR_HANDLING
  mkdir -p "$PYTHONUSERBASE" "$LOCAL_SCRATCH_ROOT" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
    "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
    "$HF_HOME" "$TORCH_HOME/checkpoints" "$PIP_CACHE_DIR" "$TMPDIR" "$OUTPUT_DIR" \
    "$(dirname "$TOKEN_CACHE")"
}

install_deps() {
  ensure_runtime
  install_cmd=(
    python -m pip install --user --quiet
    "numpy<2" scipy "wandb==0.16.6" lightning omegaconf hydra-core rich
    "torchmetrics[image]" torch-fidelity matplotlib lpips tqdm webdataset tokenizers
  )
  if command -v flock >/dev/null 2>&1; then
    (
      flock 9
      "${install_cmd[@]}" 2>/dev/null || true
    ) 9>"$PYTHONUSERBASE/.install.lock"
  else
    "${install_cmd[@]}" 2>/dev/null || true
  fi
}

setup_inputs() {
  install_deps
  python - "$STAGE1_CHECKPOINT" "$STAGE1_CONFIG" "$STAGE1_SOURCE_DIR" "$DATA_ROOT" "$RUN_DIR/stage1_source.json" <<'PY'
import json
import sys
from pathlib import Path

ckpt, config, source, data, out = [Path(arg).expanduser().resolve() for arg in sys.argv[1:6]]
if not ckpt.is_file():
    raise SystemExit(f"missing stage-1 checkpoint: {ckpt}")
if not config.is_file():
    raise SystemExit(f"missing stage-1 config: {config}")
if not (source / "rqvae").is_dir():
    raise SystemExit(f"missing copied rqvae source package: {source / 'rqvae'}")
if not (data / "wds").is_dir():
    raise SystemExit(f"missing CC3M wds dir: {data / 'wds'}")
payload = {
    "stage1_wandb_run": "helloimlixin-rutgers/laser/x3h5cl0h-a16384-k2-20260719-014434",
    "stage1_checkpoint": str(ckpt),
    "stage1_config": str(config),
    "stage1_source_dir": str(source),
    "cc3m_root": str(data),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
PY
}

validate_cache() {
  install_deps
  python - "$TOKEN_CACHE" "$STAGE1_CHECKPOINT" "$STAGE1_CONFIG" "$STAGE1_SOURCE_DIR" "$CACHE_MODE" <<'PY'
import json
import sys
from pathlib import Path

import torch

cache_path, ckpt, config, source = [Path(arg).expanduser().resolve() for arg in sys.argv[1:5]]
cache_mode = str(sys.argv[5]).strip().lower()
if not cache_path.is_file():
    raise SystemExit(f"token cache missing: {cache_path}")
try:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
except TypeError:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
meta = dict(cache.get("meta", {}) or {})
tokens = cache.get("tokens_flat")
coeffs = cache.get("coeffs_flat")
text = cache.get("text_tokens")
mask = cache.get("text_mask")
errors = []
if not torch.is_tensor(tokens) or tokens.ndim != 2 or int(tokens.size(0)) <= 0:
    errors.append("tokens_flat must be a non-empty rank-2 tensor")
expected_shape = (8, 8, 2) if cache_mode == "real_valued" else (8, 8, 4)
if tuple(cache.get("shape", ())) != expected_shape:
    errors.append(f"unexpected token grid shape: {cache.get('shape')!r}, expected {expected_shape}")
if not torch.is_tensor(text) or tuple(text.shape) != (int(tokens.size(0)), 32):
    errors.append("text_tokens must have shape [num_items, 32]")
if not torch.is_tensor(mask) or tuple(mask.shape) != tuple(text.shape):
    errors.append("text_mask must match text_tokens")
if int(meta.get("num_atoms", 0)) != 16384:
    errors.append(f"num_atoms={meta.get('num_atoms')} expected 16384")
if int(meta.get("sparsity_level", 0)) != 2:
    errors.append(f"sparsity_level={meta.get('sparsity_level')} expected 2")
if cache_mode == "real_valued":
    if not torch.is_tensor(tokens) or not torch.is_tensor(coeffs) or tuple(coeffs.shape) != tuple(tokens.shape):
        errors.append("compound-pair cache requires coeffs_flat matching tokens_flat")
    elif not bool(torch.isfinite(coeffs).all()):
        errors.append("coeffs_flat contains non-finite values")
    if bool(meta.get("quantize_sparse_coeffs", True)):
        errors.append("compound-pair cache must retain real-valued coefficients")
else:
    if coeffs is not None:
        errors.append("quantized cache unexpectedly contains coeffs_flat")
    if int(meta.get("coeff_vocab_size", 0)) != 128:
        errors.append(f"coeff_vocab_size={meta.get('coeff_vocab_size')} expected 128")
if meta.get("text_tokenizer") != "rq_bpe16k":
    errors.append(f"text_tokenizer={meta.get('text_tokenizer')!r} expected rq_bpe16k")
if meta.get("stage1_model_type") != "upstream_rqvae_laser":
    errors.append(f"stage1_model_type={meta.get('stage1_model_type')!r} expected upstream_rqvae_laser")
if str(Path(meta.get("stage1_checkpoint", "")).expanduser().resolve()) != str(ckpt):
    errors.append("stage1_checkpoint metadata mismatch")
if str(Path(meta.get("stage1_config", "")).expanduser().resolve()) != str(config):
    errors.append("stage1_config metadata mismatch")
if str(Path(meta.get("stage1_source_dir", "")).expanduser().resolve()) != str(source):
    errors.append("stage1_source_dir metadata mismatch")
report = {
    "passed": not errors,
    "errors": errors,
    "cache_path": str(cache_path),
    "num_items": int(tokens.size(0)) if torch.is_tensor(tokens) else 0,
    "token_shape": list(cache.get("shape", ())),
    "cache_mode": cache_mode,
    "compound_sequence_length": 128 if cache_mode == "real_valued" else None,
    "separate_token_sequence_length": 256 if cache_mode == "quantized" else None,
    "text_shape": list(text.shape) if torch.is_tensor(text) else None,
    "stage1_checkpoint": str(ckpt),
}
report_path = cache_path.with_suffix(".validation.json")
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if errors:
    raise SystemExit("token cache validation failed: " + "; ".join(errors))
print(f"validated CC3M token cache: {report_path} items={report['num_items']}", flush=True)
PY
}

build_cache() {
  install_deps
  if [[ -f "$TOKEN_CACHE" && "${CACHE_FORCE:-0}" != "1" ]]; then
    if validate_cache; then
      echo "Reusing validated token cache: $TOKEN_CACHE"
      return
    fi
    echo "Existing token cache failed validation; rebuilding: $TOKEN_CACHE" >&2
  fi
  export CUDA_VISIBLE_DEVICES="${CACHE_CUDA_VISIBLE_DEVICES:-0}"
  echo "=== Building CC3M token cache before training ==="
  echo "  output=$TOKEN_CACHE"
  echo "  stage1=$STAGE1_CHECKPOINT"
  python "$PROJECT_DIR/scripts/tools/build_token_cache.py" \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --stage1_format upstream_rqvae_laser \
    --stage1_config "$STAGE1_CONFIG" \
    --stage1_source_dir "$STAGE1_SOURCE_DIR" \
    --dataset cc3m \
    --data_dir "$DATA_ROOT" \
    --split train \
    --cache_mode "$CACHE_MODE" \
    --image_size 256 \
    --batch_size "$CACHE_BATCH_SIZE" \
    --num_workers "$CACHE_NUM_WORKERS" \
    --coeff_vocab_size "$COEFF_BINS" \
    --coeff_max auto \
    --coeff_quantization quantile \
    --coeff_calibration_percentile 99.5 \
    --text_max_length "$TEXT_MAX_LENGTH" \
    --text_tokenizer "$TEXT_TOKENIZER" \
    --max_items "$CACHE_MAX_ITEMS" \
    --device auto \
    --output "$TOKEN_CACHE"
  validate_cache
}

latest_stage2_checkpoint() {
  find "$OUTPUT_DIR/checkpoints" -type f -name "*.ckpt" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print; }'
}

train_stage2() {
  install_deps
  validate_cache
  unset CUDA_VISIBLE_DEVICES
  export NODE_RANK="${SLURM_PROCID:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
  export MASTER_PORT
  world_size=$((NODES * GPUS_PER_NODE))
  denom=$((S2_BATCH_SIZE * world_size))
  if (( denom <= 0 )); then
    echo "invalid batch/world size" >&2
    exit 1
  fi
  accumulate=$((TOTAL_BATCH_SIZE / denom))
  if (( accumulate < 1 )); then
    accumulate=1
  fi
  actual_global=$((S2_BATCH_SIZE * world_size * accumulate))
  if (( actual_global != TOTAL_BATCH_SIZE )); then
    echo "Warning: actual global batch $actual_global differs from requested $TOTAL_BATCH_SIZE" >&2
  fi

  echo "=== Starting CC3M text-to-image stage-2 training ==="
  echo "  node_rank=$NODE_RANK host=$(hostname)"
  echo "  devices_per_node=$GPUS_PER_NODE num_nodes=$NODES world_size=$world_size"
  echo "  local_batch=$S2_BATCH_SIZE accumulate=$accumulate global_batch=$actual_global"
  echo "  token_cache=$TOKEN_CACHE"
  echo "  cache_mode=$CACHE_MODE"

  train_args=(
    stage2
    "token_cache_path=$TOKEN_CACHE"
    "output_dir=$OUTPUT_DIR"
    "seed=42"
    "token_cache.build=false"
    "data.dataset=cc3m"
    "data.data_dir=$DATA_ROOT"
    "data.image_size=256"
    "data.num_workers=$S2_NUM_WORKERS"
    "ar.type=sparse_spatial_depth"
    "ar.autoregressive_coeffs=true"
    "ar.text_conditional=true"
    "ar.text_conditioning_mode=rq_prefix"
    "ar.text_prefix_length=$TEXT_MAX_LENGTH"
    "ar.text_loss_weight=0.1"
    "ar.image_loss_weight=0.9"
    "ar.n_global_spatial_tokens=0"
    "ar.d_model=$S2_D_MODEL"
    "ar.n_heads=$S2_HEADS"
    "ar.n_layers=$S2_LAYERS"
    "ar.n_depth_layers=$S2_DEPTH_LAYERS"
    "ar.d_ff=$S2_FF"
    "ar.dropout=0.1"
    "ar.learning_rate=$S2_LR"
    "ar.weight_decay=$S2_WEIGHT_DECAY"
    "ar.optimizer_beta1=0.9"
    "ar.optimizer_beta2=0.95"
    "ar.warmup_steps=$S2_WARMUP_STEPS"
    "ar.max_steps=$S2_MAX_STEPS"
    "ar.min_lr_ratio=$S2_MIN_LR_RATIO"
    "ar.atom_loss_weight=1.0"
    "ar.coeff_loss_weight=1.0"
    "ar.atom_label_smoothing=$ATOM_LABEL_SMOOTHING"
    "ar.atom_coverage_weight=$ATOM_COVERAGE_WEIGHT"
    "ar.coeff_loss_type=huber"
    "ar.coeff_huber_delta=0.25"
    "ar.coeff_head_hidden_mult=2.0"
    "ar.coeff_head_depth=2"
    "ar.coeff_head_dropout=0.05"
    "ar.sample_coeff_mode=mean"
    "train_ar.max_epochs=$TARGET_EPOCHS"
    "train_ar.batch_size=$S2_BATCH_SIZE"
    "train_ar.accumulate_grad_batches=$accumulate"
    "train_ar.max_items=0"
    "train_ar.limit_train_batches=1.0"
    "train_ar.limit_val_batches=$S2_VAL_BATCHES"
    "train_ar.limit_test_batches=0"
    "train_ar.val_check_interval=1.0"
    "train_ar.validation_split=0.05"
    "train_ar.test_split=0.00"
    "train_ar.log_every_n_steps=20"
    "train_ar.devices=$GPUS_PER_NODE"
    "train_ar.num_nodes=$NODES"
    "train_ar.strategy=$S2_STRATEGY"
    "train_ar.precision=bf16-mixed"
    "train_ar.deterministic=false"
    "train_ar.gradient_clip_val=1.0"
    "train_ar.checkpoint_save_top_k=3"
    "train_ar.checkpoint_save_last=true"
    "train_ar.checkpoint_every_n_epochs=$SAVE_CKPT_FREQ"
    "train_ar.checkpoint_every_n_train_steps=$RECOVERY_SAVE_STEP_FREQ"
    "train_ar.checkpoint_monitor=val/loss"
    "train_ar.checkpoint_mode=min"
    "+train_ar.checkpoint_upload_to_wandb=true"
    "+train_ar.checkpoint_upload_every_n_epochs=$CHECKPOINT_UPLOAD_EVERY"
    "train_ar.sample_every_n_steps=$SAMPLE_EVERY_N_STEPS"
    "train_ar.sample_every_n_epochs=$SAMPLE_EVERY_N_EPOCHS"
    "train_ar.sample_num_images=$SAMPLE_NUM_IMAGES"
    "train_ar.sample_temperature=$SAMPLE_TEMPERATURE"
    "train_ar.sample_top_k=$SAMPLE_TOP_K"
    "train_ar.sample_top_p=$SAMPLE_TOP_P"
    "train_ar.sample_coeff_mode=mean"
    "train_ar.sample_log_to_wandb=true"
    "train_ar.sample_text_prompts=[\"a red sports car parked on a city street\",\"a small dog running through green grass\",\"a plate of fresh fruit on a wooden table\",\"a bedroom with a large window and white sheets\",\"eiffel tower on a desert\",\"a painting by vincent van gogh\",\"a person riding a bicycle near the ocean\",\"a bowl of soup on a kitchen counter\"]"
    "train_ar.compute_generation_fid=false"
    "train_ar.generation_metric_num_samples=0"
    "train_ar.run_test_after_fit=false"
    "train_ar.save_final_samples_after_fit=false"
    "wandb.project=$WANDB_PROJECT"
    "wandb.name=$WANDB_NAME"
    "wandb.id=$WANDB_ID"
    "wandb.resume=allow"
    "wandb.group=$WANDB_GROUP"
    "wandb.tags=[stage2,cc3m,laser,text_conditional,rq_prefix,imagenet_tokenizer,x3h5cl0h,a16384,k2,$CACHE_MODE]"
    "wandb.append_timestamp=false"
    "wandb.save_dir=$WANDB_DIR"
  )
  latest_ckpt="$(latest_stage2_checkpoint || true)"
  if [[ -n "$latest_ckpt" && -f "$latest_ckpt" ]]; then
    echo "Resuming local stage-2 checkpoint: $latest_ckpt"
    train_args+=("ckpt_path=$latest_ckpt")
  fi
  exec python "$PROJECT_DIR/train.py" "${train_args[@]}"
}

case "${1:-}" in
  "")
    submit_job
    ;;
  --worker)
    run_worker
    ;;
  --inside)
    set_defaults
    case "${2:-}" in
      setup)
        setup_inputs
        ;;
      cache)
        build_cache
        ;;
      validate)
        validate_cache
        ;;
      train)
        train_stage2
        ;;
      *)
        echo "usage: $SELF --inside {setup|cache|validate|train}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "usage: $SELF [--worker|--inside {setup|cache|validate|train}]" >&2
    exit 2
    ;;
esac
