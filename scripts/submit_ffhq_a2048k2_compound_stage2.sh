#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SELF="${SELF:-$PROJECT_DIR/scripts/submit_ffhq_a2048k2_compound_stage2.sh}"

set_defaults() {
  TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
  PARTITION="${PARTITION:-gpu-redhat}"
  CONSTRAINT="${CONSTRAINT:-adalovelace}"
  NODES="${NODES:-4}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
  WORLD_SIZE=$((NODES * GPUS_PER_NODE))
  CPUS_PER_TASK="${CPUS_PER_TASK:-24}"
  MEM_MB="${MEM_MB:-250000}"
  TIME_LIMIT="${TIME_LIMIT:-72:00:00}"

  DATA_ROOT="${FFHQ_ROOT:-/scratch/$USER/Projects/data/ffhq}"
  STAGE1_DIR="${STAGE1_DIR:-/scratch/$USER/runs/ffhq_celebahq_rqvae_strict_dict_sweep/ffhq-celebahq-rqvae-strict-dict-sweep-20260720_145706/ffhq-a2048-k2/ffhq256-rqvae-laser-8x8-a2048-k2/21072026_110546}"
  STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$STAGE1_DIR/best_rfid_slot3_model.pt}"
  STAGE1_CONFIG="${STAGE1_CONFIG:-$STAGE1_DIR/config.yaml}"
  STAGE1_POLICY="${STAGE1_POLICY:-$STAGE1_DIR/checkpoint_policy.json}"
  RUN_NAME="${RUN_NAME:-ffhq-official-rqt350m-laser-a2048-k2-compound-${WORLD_SIZE}gpu-$TS}"
  RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser/$RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/stage2}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
  TOKEN_CACHE="${TOKEN_CACHE:-$RUN_DIR/token_cache/ffhq_train_a2048k2_compound_pairs.pt}"
  AUDIT_REPORT="${AUDIT_REPORT:-$RUN_DIR/stage1_audit.json}"

  TARGET_EPOCHS="${TARGET_EPOCHS:-200}"
  TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
  if (( TOTAL_BATCH_SIZE % WORLD_SIZE != 0 )); then
    echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE is not divisible by WORLD_SIZE=$WORLD_SIZE" >&2
    exit 1
  fi
  STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-$((TOTAL_BATCH_SIZE / WORLD_SIZE))}"
  CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-64}"
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-8}"
  FID_EVERY="${FID_EVERY:-5}"
  FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-50000}"
  FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}"
  FID_REFERENCE_FULL_DATASET="${FID_REFERENCE_FULL_DATASET:-1}"
  COMPUTE_INCEPTION_SCORE="${COMPUTE_INCEPTION_SCORE:-0}"
  SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
  SAVE_STEP_FREQ="${SAVE_STEP_FREQ:-500}"
  SAMPLE_GRID_EVERY="${SAMPLE_GRID_EVERY:-500}"

  WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  WANDB_ID="${WANDB_ID:-ffhqcmp$(date +%m%d%H%M%S)}"
  WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
  IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
  PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
  MASTER_PORT="${MASTER_PORT:-$((22000 + (${SLURM_JOB_ID:-0} % 18000)))}"
  CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-$MASTER_PORT}"
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-$((MASTER_PORT + 1))}"
  LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/$RUN_NAME}"

  export TS PARTITION CONSTRAINT NODES GPUS_PER_NODE WORLD_SIZE CPUS_PER_TASK MEM_MB TIME_LIMIT
  export DATA_ROOT STAGE1_DIR STAGE1_CHECKPOINT STAGE1_CONFIG STAGE1_POLICY
  export RUN_NAME RUN_DIR OUTPUT_DIR CHECKPOINT_DIR TOKEN_CACHE AUDIT_REPORT
  export TARGET_EPOCHS TOTAL_BATCH_SIZE STAGE2_BATCH_SIZE CACHE_BATCH_SIZE CACHE_NUM_WORKERS
  export FID_EVERY FID_NUM_SAMPLES FID_BATCH_SIZE FID_REFERENCE_FULL_DATASET
  export COMPUTE_INCEPTION_SCORE SAVE_CKPT_FREQ SAVE_STEP_FREQ SAMPLE_GRID_EVERY
  export WANDB_ENTITY WANDB_PROJECT WANDB_ID WANDB_NAME IMAGE PYDEPS
  export MASTER_PORT CACHE_MASTER_PORT TRAIN_MASTER_PORT LOCAL_SCRATCH_ROOT PROJECT_DIR SELF
}

submit_job() {
  set_defaults
  [[ -d "$DATA_ROOT" ]] || { echo "FFHQ directory missing: $DATA_ROOT" >&2; exit 1; }
  [[ -f "$STAGE1_CHECKPOINT" ]] || { echo "Stage-1 checkpoint missing: $STAGE1_CHECKPOINT" >&2; exit 1; }
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$(dirname "$TOKEN_CACHE")"
  printf '%s\n' \
    "FFHQ compound pipeline" \
    "  resources=$WORLD_SIZE GPUs ($NODES nodes x $GPUS_PER_NODE), $PARTITION/$CONSTRAINT" \
    "  run_dir=$RUN_DIR" \
    "  stage1=$STAGE1_CHECKPOINT" \
    "  cache=$TOKEN_CACHE" \
    "  wandb=https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/runs/$WANDB_ID"
  args=(
    --partition="$PARTITION" --constraint="$CONSTRAINT" --job-name="ffhqcmp-s2"
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
  command -v apptainer >/dev/null 2>&1 || module load singularity 2>/dev/null || true
  if command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
  else
    runtime=singularity
  fi
  CONTAINER=("$runtime" exec --nv --bind /scratch --bind /mnt/scratch --bind /dev/shm "$IMAGE")
}

ensure_runtime() {
  export PYTHONUSERBASE="$PYDEPS"
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export WANDB_ENTITY WANDB_PROJECT
  export WANDB_DIR="$LOCAL_SCRATCH_ROOT/wandb/run"
  export WANDB_CACHE_DIR="$LOCAL_SCRATCH_ROOT/wandb/cache"
  export WANDB_DATA_DIR="$LOCAL_SCRATCH_ROOT/wandb/data"
  export WANDB_ARTIFACT_DIR="$LOCAL_SCRATCH_ROOT/wandb/artifacts"
  export XDG_CACHE_HOME="$LOCAL_SCRATCH_ROOT/cache"
  export MPLCONFIGDIR="$LOCAL_SCRATCH_ROOT/cache/matplotlib"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  unset NCCL_ASYNC_ERROR_HANDLING
  mkdir -p "$PYDEPS" "$CHECKPOINT_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
    "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
}

install_deps() {
  ensure_runtime
  install=(python -m pip install --user --quiet "numpy<2" scipy "wandb==0.16.6"
    lightning omegaconf hydra-core rich "torchmetrics[image]" torch-fidelity
    matplotlib lpips tqdm pillow)
  (flock 9; "${install[@]}" 2>/dev/null || true) 9>"$PYDEPS/.install.lock"
}

inside() {
  phase="$1"
  ensure_runtime
  case "$phase" in
    setup)
      install_deps
      python scripts/tools/audit_ffhq_stage1.py \
        --checkpoint "$STAGE1_CHECKPOINT" --config "$STAGE1_CONFIG" \
        --policy "$STAGE1_POLICY" --output "$AUDIT_REPORT"
      ;;
    cache)
      if [[ -f "$TOKEN_CACHE" ]] && python - "$TOKEN_CACHE" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
assert p["meta"]["format"] == "laser_compound_pairs_v1"
assert p["meta"]["validation_passed"] is True
assert len(p["atoms"]) == 70000
PY
      then
        echo "Reusing validated 70K cache $TOKEN_CACHE"
      else
        python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
          --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" \
          --master_port="$CACHE_MASTER_PORT" \
          scripts/tools/build_official_ffhq_token_cache.py \
          --checkpoint "$STAGE1_CHECKPOINT" --data "$DATA_ROOT" --output "$TOKEN_CACHE" \
          --batch-size "$CACHE_BATCH_SIZE" --num-workers "$CACHE_NUM_WORKERS" \
          --num-atoms 2048 --coeff-vocab-size 2048 --coeff-max 3.0 \
          --calibration-quantile 0.995 --verify-samples 256
      fi
      ;;
    validate)
      python - "$TOKEN_CACHE" "$AUDIT_REPORT" <<'PY'
import json, sys, torch
cache_path, audit_path = sys.argv[1:]
audit = json.load(open(audit_path))
cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
assert audit["passed"] is True
assert cache["meta"]["validation_passed"] is True
assert cache["meta"]["shape"] == [8, 8, 2]
assert len(cache["atoms"]) == 70000
print(json.dumps({"stage1_audit": audit, "cache_meta": cache["meta"]}, indent=2))
PY
      ;;
    train)
      evaluation_args=()
      if [[ "$FID_REFERENCE_FULL_DATASET" == 1 ]]; then
        evaluation_args+=(--fid-reference-full-dataset)
      else
        evaluation_args+=(--no-fid-reference-full-dataset)
      fi
      if [[ "$COMPUTE_INCEPTION_SCORE" == 1 ]]; then
        evaluation_args+=(--inception-score)
      else
        evaluation_args+=(--no-inception-score)
      fi
      python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
        --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" \
        --master_port="$TRAIN_MASTER_PORT" \
        scripts/train_official_rqtransformer_laser_stage2.py \
        --checkpoint "$STAGE1_CHECKPOINT" --data "$DATA_ROOT" --dataset ffhq-flat \
        --token-cache "$TOKEN_CACHE" --output "$OUTPUT_DIR" \
        --checkpoint-dir "$CHECKPOINT_DIR" --architecture ffhq-350m \
        --epochs "$TARGET_EPOCHS" --batch-size "$STAGE2_BATCH_SIZE" \
        --total-batch-size "$TOTAL_BATCH_SIZE" --num-atoms 2048 \
        --coeff-vocab-size 2048 --coeff-max 3.0 --stage1-attn-resolutions 16 \
        --compound-tokens --compound-micro-transformer-layers 2 \
        --compound-depth-specific-coeff-heads --compound-distribution-geometry \
        --atom-loss-weight 1.5 --geometry-loss-weight 0.05 \
        --geometry-start-epoch 2 --geometry-warmup-epochs 3 --geometry-top-k 4 \
        --atom-temperature 1.0 --atom-top-k 250 --atom-top-p 1.0 \
        --coeff-temperature 1.0 --coeff-top-p 0.85 \
        --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 200 --min-lr 0 \
        --wandb-project "$WANDB_PROJECT" --wandb-id "$WANDB_ID" --wandb-name "$WANDB_NAME" \
        --fid-num-samples "$FID_NUM_SAMPLES" --fid-batch-size "$FID_BATCH_SIZE" \
        "${evaluation_args[@]}" \
        --fid-every "$FID_EVERY" --save-ckpt-freq "$SAVE_CKPT_FREQ" \
        --save-step-freq "$SAVE_STEP_FREQ" --sample-grid-every "$SAMPLE_GRID_EVERY" \
        --upload-checkpoints
      ;;
    *) echo "Unknown inside phase: $phase" >&2; exit 2 ;;
  esac
}

worker() {
  set_defaults
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$(dirname "$TOKEN_CACHE")"
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  export MASTER_ADDR
  container_prefix
  "${CONTAINER[@]}" bash "$SELF" --inside setup
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside cache
  "${CONTAINER[@]}" bash "$SELF" --inside validate
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside train
}

case "${1:-submit}" in
  submit) submit_job ;;
  --worker) worker ;;
  --inside) shift; set_defaults; inside "$@" ;;
  *) echo "Usage: $0 [submit|--worker|--inside PHASE]" >&2; exit 2 ;;
esac
