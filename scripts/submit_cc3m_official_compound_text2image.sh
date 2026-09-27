#!/bin/bash
set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SELF="${SELF:-$PROJECT_DIR/scripts/submit_cc3m_official_compound_text2image.sh}"

set_defaults() {
  TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
  PARTITION="${PARTITION:-gpu-redhat}"
  CONSTRAINT="${CONSTRAINT:-adalovelace}"
  NODES="${NODES:-4}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-24}"
  MEM_MB="${MEM_MB:-250000}"
  TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
  WORLD_SIZE=$((NODES * GPUS_PER_NODE))

  DATA_ROOT="${CC3M_ROOT:-/scratch/$USER/Projects/data/cc3m}"
  STAGE1_DIR="${STAGE1_DIR:-/scratch/$USER/runs/imagenet_x3h5cl0h_strict_bottleneck_sweep/imagenet-x3h5cl0h-strict-bottleneck-sweep-20260719_014434/k2-a16384/in256-rqvae-laser-8x8-a16384-k2/19072026_024726}"
  STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$STAGE1_DIR/best_rfid_slot3_model.pt}"
  TOKEN_CACHE="${TOKEN_CACHE:-$PROJECT_DIR/outputs/cc3m_x3h5cl0h_a16384k2_text2image/token_cache/cc3m_train_imagenet_x3h5cl0h_a16384k2_q128_rq_bpe16k_text32.pt}"

  RUN_NAME="${RUN_NAME:-cc3m-official-rqt650m-compound-a16384-k2-${WORLD_SIZE}gpu-$TS}"
  RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser/$RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/stage2}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
  SOURCE_SNAPSHOT="${SOURCE_SNAPSHOT:-$RUN_DIR/source_snapshot}"
  LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/$RUN_NAME}"

  TARGET_EPOCHS="${TARGET_EPOCHS:-100}"
  STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
  TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-2048}"
  LR="${LR:-0.0005}"
  LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
  MIN_LR="${MIN_LR:-0.0}"
  LR_SCHEDULE_EPOCHS="${LR_SCHEDULE_EPOCHS:-100}"
  TEXT_LOSS_WEIGHT="${TEXT_LOSS_WEIGHT:-0.1}"
  IMAGE_LOSS_WEIGHT="${IMAGE_LOSS_WEIGHT:-0.9}"
  SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-1}"
  SAVE_STEP_FREQ="${SAVE_STEP_FREQ:-500}"
  SAMPLE_GRID_EVERY="${SAMPLE_GRID_EVERY:-500}"
  SAMPLE_GRID_ON_START="${SAMPLE_GRID_ON_START:-1}"
  FID_EVERY="${FID_EVERY:-1}"
  FID_REFERENCE_FULL_DATASET="${FID_REFERENCE_FULL_DATASET:-1}"
  ATOM_TOP_K="${ATOM_TOP_K:-1024}"
  ATOM_TOP_P="${ATOM_TOP_P:-0.9}"
  COEFF_TOP_K="${COEFF_TOP_K:-1024}"
  COEFF_TOP_P="${COEFF_TOP_P:-0.9}"
  CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-/scratch/$USER/.cache/clip}"

  WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  WANDB_ID="${WANDB_ID:-cc3mcmp$(date +%m%d%H%M%S)}"
  WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"

  IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
  PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
  MASTER_PORT="${MASTER_PORT:-$((22000 + (${SLURM_JOB_ID:-0} % 18000)))}"

  if (( WORLD_SIZE != 8 && WORLD_SIZE != 16 )); then
    echo "CC3M official run requires exactly 8 or 16 GPUs, got $WORLD_SIZE" >&2
    exit 1
  fi
  if (( TOTAL_BATCH_SIZE % (STAGE2_BATCH_SIZE * WORLD_SIZE) != 0 )); then
    echo "total batch $TOTAL_BATCH_SIZE is not divisible by local batch x world size" >&2
    exit 1
  fi

  export TS PARTITION CONSTRAINT NODES GPUS_PER_NODE CPUS_PER_TASK MEM_MB TIME_LIMIT WORLD_SIZE
  export DATA_ROOT STAGE1_DIR STAGE1_CHECKPOINT TOKEN_CACHE
  export RUN_NAME RUN_DIR OUTPUT_DIR CHECKPOINT_DIR SOURCE_SNAPSHOT LOCAL_SCRATCH_ROOT
  export TARGET_EPOCHS STAGE2_BATCH_SIZE TOTAL_BATCH_SIZE LR LR_SCHEDULE MIN_LR LR_SCHEDULE_EPOCHS
  export TEXT_LOSS_WEIGHT IMAGE_LOSS_WEIGHT SAVE_CKPT_FREQ SAVE_STEP_FREQ SAMPLE_GRID_EVERY
  export SAMPLE_GRID_ON_START FID_EVERY FID_REFERENCE_FULL_DATASET
  export ATOM_TOP_K ATOM_TOP_P COEFF_TOP_K COEFF_TOP_P CLIP_CACHE_DIR
  export WANDB_ENTITY WANDB_PROJECT WANDB_ID WANDB_NAME IMAGE PYDEPS MASTER_PORT PROJECT_DIR SELF
}

freeze_source() {
  mkdir -p "$SOURCE_SNAPSHOT/scripts"
  if [[ ! -f "$SOURCE_SNAPSHOT/scripts/train_official_rqtransformer_laser_stage2.py" ]]; then
    cp "$PROJECT_DIR/scripts/train_official_rqtransformer_laser_stage2.py" "$SOURCE_SNAPSHOT/scripts/"
    cp -a "$PROJECT_DIR/src" "$SOURCE_SNAPSHOT/src"
    cp "$PROJECT_DIR/third_party/rq-vae-transformer/configs/cc3m/cc3m-rqtransformer-8x8x4-650M.yaml" \
      "$SOURCE_SNAPSHOT/official-cc3m-rqtransformer-8x8x4-650M.yaml"
  fi
  sha256sum \
    "$SOURCE_SNAPSHOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    "$SOURCE_SNAPSHOT/official-cc3m-rqtransformer-8x8x4-650M.yaml" \
    > "$SOURCE_SNAPSHOT/SHA256SUMS"
}

submit_job() {
  set_defaults
  [[ -d "$DATA_ROOT/wds" ]] || { echo "CC3M WebDataset missing: $DATA_ROOT/wds" >&2; exit 1; }
  [[ -f "$STAGE1_CHECKPOINT" ]] || { echo "stage-1 checkpoint missing: $STAGE1_CHECKPOINT" >&2; exit 1; }
  [[ -f "$TOKEN_CACHE" ]] || { echo "CC3M token cache missing: $TOKEN_CACHE" >&2; exit 1; }
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"
  freeze_source

  printf '%s\n' \
    "CC3M official compound RQTransformer" \
    "  resources=$WORLD_SIZE GPUs ($NODES nodes x $GPUS_PER_NODE), $PARTITION/$CONSTRAINT" \
    "  run_dir=$RUN_DIR" \
    "  frozen_source=$SOURCE_SNAPSHOT" \
    "  cache=$TOKEN_CACHE" \
    "  compound_grid=8x8x2 (128 paired events)" \
    "  architecture=1280d body26 head4 heads20 text16k/context32" \
    "  batch=$STAGE2_BATCH_SIZE total_batch=$TOTAL_BATCH_SIZE loss=text${TEXT_LOSS_WEIGHT}/image${IMAGE_LOSS_WEIGHT}" \
    "  wandb=https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/runs/$WANDB_ID"

  args=(
    --partition="$PARTITION" --constraint="$CONSTRAINT" --job-name="cc3mcmp-s2"
    --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1
    --cpus-per-task="$CPUS_PER_TASK" --gres="gpu:$GPUS_PER_NODE"
    --mem="$MEM_MB" --time="$TIME_LIMIT" --chdir="$PROJECT_DIR"
    --output="$RUN_DIR/slurm-%j.out" --error="$RUN_DIR/slurm-%j.err" --requeue
  )
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY_RUN sbatch'; printf ' %q' "${args[@]}" "$SELF" --worker; printf '\n'
  else
    sbatch "${args[@]}" "$SELF" --worker
  fi
}

container_prefix() {
  if ! command -v module >/dev/null 2>&1; then
    set +u
    [[ -f /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash
    set -u
  fi
  command -v singularity >/dev/null 2>&1 || module load singularity 2>/dev/null || true
  CONTAINER=(singularity exec --nv --bind /scratch --bind /mnt/scratch --bind /dev/shm "$IMAGE")
}

ensure_runtime() {
  export PYTHONUSERBASE="$PYDEPS"
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONPATH="$SOURCE_SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
  export WANDB_ENTITY WANDB_PROJECT
  export WANDB_DIR="$LOCAL_SCRATCH_ROOT/wandb/run"
  export WANDB_CACHE_DIR="$LOCAL_SCRATCH_ROOT/wandb/cache"
  export WANDB_DATA_DIR="$LOCAL_SCRATCH_ROOT/wandb/data"
  export WANDB_ARTIFACT_DIR="$LOCAL_SCRATCH_ROOT/wandb/artifacts"
  export XDG_CACHE_HOME="$LOCAL_SCRATCH_ROOT/cache"
  export MPLCONFIGDIR="$LOCAL_SCRATCH_ROOT/cache/matplotlib"
  export TORCH_HOME="$LOCAL_SCRATCH_ROOT/cache/torch"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  unset NCCL_ASYNC_ERROR_HANDLING
  mkdir -p "$PYDEPS" "$CHECKPOINT_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
    "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
    "$TORCH_HOME" "$CLIP_CACHE_DIR"
}

install_deps() {
  ensure_runtime
  install=(python -m pip install --user --quiet "numpy<2" scipy "wandb==0.16.6"
    omegaconf hydra-core matplotlib pillow tqdm "tokenizers==0.23.1"
    "torchmetrics==1.9.0" "torch-fidelity==0.4.0" "openai-clip==1.0.1" ftfy)
  (flock 9; "${install[@]}") 9>"$PYDEPS/.install.lock"
}

refresh_stage2_entrypoint() {
  # A requeued continuation must retain the checkpoint's frozen source tree,
  # while explicitly requested sampler/evaluator fixes need to enter that
  # tree before Python starts. The lock makes this safe across all node tasks.
  (
    flock 9
    cp "$PROJECT_DIR/scripts/train_official_rqtransformer_laser_stage2.py" \
      "$SOURCE_SNAPSHOT/scripts/train_official_rqtransformer_laser_stage2.py"
    sha256sum \
      "$SOURCE_SNAPSHOT/scripts/train_official_rqtransformer_laser_stage2.py" \
      "$SOURCE_SNAPSHOT/official-cc3m-rqtransformer-8x8x4-650M.yaml" \
      > "$SOURCE_SNAPSHOT/SHA256SUMS"
  ) 9>"$SOURCE_SNAPSHOT/.entrypoint-refresh.lock"
}

validate_inputs() {
  install_deps
  python - "$TOKEN_CACHE" "$STAGE1_CHECKPOINT" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
import torch

cache_path, checkpoint_path, run_dir = map(Path, sys.argv[1:])
cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
meta = dict(cache.get("meta", {}))
errors = []
if tuple(cache.get("shape", ())) != (8, 8, 4):
    errors.append("source cache is not the validated interleaved 8x8x4 representation")
if tuple(cache["tokens_flat"].shape[1:]) != (256,):
    errors.append("source cache rows are not 256 interleaved scalar tokens")
if tuple(cache["text_tokens"].shape) != (len(cache["tokens_flat"]), 32):
    errors.append("text-token cache is not aligned [N,32]")
if int(meta.get("num_atoms", 0)) != 16384 or int(meta.get("sparsity_level", 0)) != 2:
    errors.append("cache is not a16384/k2")
if int(meta.get("text_vocab_size", 0)) != 16384 or meta.get("text_tokenizer") != "rq_bpe16k":
    errors.append("cache does not use the official BPE16K tokenizer")
if str(Path(meta.get("stage1_checkpoint", "")).resolve()) != str(checkpoint_path.resolve()):
    errors.append("stage-1 checkpoint metadata mismatch")
report = {
    "passed": not errors,
    "errors": errors,
    "source_cache": str(cache_path.resolve()),
    "num_items": len(cache["tokens_flat"]),
    "source_scalar_grid": [8, 8, 4],
    "training_compound_grid": [8, 8, 2],
    "training_sequence_events": 128,
    "atom_vocab_size": 16384,
    "text_vocab_size": 16384,
    "text_context_length": 32,
    "code_lineage": "ffhqcmp0804205803 / vendored KakaoBrain RQTransformer",
}
target = run_dir / "compound_cache_validation.json"
target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True), flush=True)
if errors:
    raise SystemExit("; ".join(errors))
PY
}

train_stage2() {
  install_deps
  refresh_stage2_entrypoint
  fid_reference_flag="--no-fid-reference-full-dataset"
  if [[ "$FID_REFERENCE_FULL_DATASET" == 1 ]]; then
    fid_reference_flag="--fid-reference-full-dataset"
  fi
  sample_start_flag="--no-sample-grid-on-start"
  if [[ "$SAMPLE_GRID_ON_START" == 1 ]]; then
    sample_start_flag="--sample-grid-on-start"
  fi
  python -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$SOURCE_SNAPSHOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    --checkpoint "$STAGE1_CHECKPOINT" --data "$DATA_ROOT" --dataset cc3m-cache \
    --token-cache "$TOKEN_CACHE" --output "$OUTPUT_DIR" --checkpoint-dir "$CHECKPOINT_DIR" \
    --architecture cc3m-650m --compound-tokens \
    --compound-micro-transformer-layers 2 --compound-depth-specific-coeff-heads \
    --compound-distribution-geometry --geometry-loss-weight 0.05 \
    --geometry-start-epoch 2 --geometry-warmup-epochs 3 --geometry-top-k 4 \
    --atom-loss-weight 1.5 --num-atoms 16384 --coeff-vocab-size 2048 \
    --coeff-max 20 --coeff-scale 6.4 \
    --epochs "$TARGET_EPOCHS" --batch-size "$STAGE2_BATCH_SIZE" \
    --total-batch-size "$TOTAL_BATCH_SIZE" --text-loss-weight "$TEXT_LOSS_WEIGHT" \
    --image-loss-weight "$IMAGE_LOSS_WEIGHT" --lr "$LR" --lr-schedule "$LR_SCHEDULE" \
    --lr-schedule-epochs "$LR_SCHEDULE_EPOCHS" --min-lr "$MIN_LR" \
    --atom-temperature 1.0 --atom-top-k "$ATOM_TOP_K" --atom-top-p "$ATOM_TOP_P" \
    --coeff-temperature 1.0 --coeff-top-k "$COEFF_TOP_K" --coeff-top-p "$COEFF_TOP_P" \
    --fid-every "$FID_EVERY" "$fid_reference_flag" \
    --clip-cache-dir "$CLIP_CACHE_DIR" --save-ckpt-freq "$SAVE_CKPT_FREQ" \
    --save-step-freq "$SAVE_STEP_FREQ" --sample-grid-every "$SAMPLE_GRID_EVERY" \
    "$sample_start_flag" \
    --wandb-project "$WANDB_PROJECT" --wandb-id "$WANDB_ID" --wandb-name "$WANDB_NAME"
}

worker() {
  set_defaults
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$LOCAL_SCRATCH_ROOT"
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  export MASTER_ADDR
  container_prefix
  "${CONTAINER[@]}" bash "$SELF" --inside validate
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside train
}

case "${1:-submit}" in
  submit) submit_job ;;
  --worker) worker ;;
  --inside)
    shift
    set_defaults
    case "${1:-}" in
      validate) validate_inputs ;;
      train) train_stage2 ;;
      *) echo "unknown inside phase: ${1:-}" >&2; exit 2 ;;
    esac
    ;;
  *) echo "usage: $SELF [submit|--worker|--inside validate|train]" >&2; exit 2 ;;
esac
