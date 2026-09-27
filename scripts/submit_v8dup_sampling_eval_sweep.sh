#!/bin/bash
set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SELF="${SELF:-$PROJECT_DIR/scripts/submit_v8dup_sampling_eval_sweep.sh}"

set_defaults() {
  TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
  PARTITION="${PARTITION:-gpu-redhat}"
  CONSTRAINT="${CONSTRAINT:-adalovelace}"
  EXCLUDE_NODES="${EXCLUDE_NODES:-}"
  DEPENDENCY="${DEPENDENCY:-}"
  NODES="${NODES:-4}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
  MEM_MB="${MEM_MB:-250000}"
  TIME_LIMIT="${TIME_LIMIT:-72:00:00}"

  DATA_ROOT="${IMAGENET_ROOT:-/scratch/$USER/Projects/data/imagenet}"
  REAL_SPLIT="${REAL_SPLIT:-train}"
  REAL_NUM_SAMPLES="${REAL_NUM_SAMPLES:-0}"
  REAL_STATS="${REAL_STATS:-}"
  FID_WEIGHTS_FILE="${FID_WEIGHTS_FILE:-}"
  RUN_NAME="${RUN_NAME:-v8dup0731113220-full-imagenet-sampling-sweep-$TS}"
  RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser_sampling_sweeps/$RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/results}"
  LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser_sampling_sweeps/$RUN_NAME}"

  STAGE1_ARTIFACT="${STAGE1_ARTIFACT:-helloimlixin-rutgers/laser/x3h5cl0h-stage1-checkpoint:latest}"
  STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/scratch/$USER/runs/imagenet_x3h5cl0h_strict_bottleneck_sweep/imagenet-x3h5cl0h-strict-bottleneck-sweep-20260719_014434/k2-a16384/in256-rqvae-laser-8x8-a16384-k2/19072026_024726/best_rfid_slot3_model.pt}"
  STAGE2_ARTIFACT="${STAGE2_ARTIFACT:-helloimlixin-rutgers/laser/v8dup0731113220-checkpoint:latest}"
  STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-$LOCAL_SCRATCH_ROOT/source/v8dup0731113220-checkpoint/last.pt}"

  NUM_SAMPLES="${NUM_SAMPLES:-50000}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
  TEMPERATURES="${TEMPERATURES:-0.90,0.95,1.00,1.05}"
  TOP_PS="${TOP_PS:-0.90,0.92,0.95}"
  TOP_KS="${TOP_KS:-full}"
  REJECTION_FACTORS="${REJECTION_FACTORS:-2,4}"
  REJECTION_BASE="${REJECTION_BASE:-1.0,0.92,full}"
  REJECTION_CLASSIFIER="${REJECTION_CLASSIFIER:-resnet50}"
  SKIP_RECONSTRUCTION_FID="${SKIP_RECONSTRUCTION_FID:-1}"

  WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  WANDB_ID="${WANDB_ID:-v8eval$(date +%m%d%H%M%S)}"
  WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
  WANDB_GROUP="${WANDB_GROUP:-v8dup0731113220-sampling-eval}"
  WANDB_MODE="${WANDB_MODE:-online}"

  IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
  PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
  MASTER_PORT="${MASTER_PORT:-$((24000 + (${SLURM_JOB_ID:-0} % 20000)))}"

  export TS PARTITION CONSTRAINT EXCLUDE_NODES DEPENDENCY NODES GPUS_PER_NODE CPUS_PER_TASK MEM_MB TIME_LIMIT
  export DATA_ROOT REAL_SPLIT REAL_NUM_SAMPLES REAL_STATS FID_WEIGHTS_FILE RUN_NAME RUN_DIR OUTPUT_DIR LOCAL_SCRATCH_ROOT
  export STAGE1_ARTIFACT STAGE1_CHECKPOINT STAGE2_ARTIFACT STAGE2_CHECKPOINT
  export NUM_SAMPLES EVAL_BATCH_SIZE TEMPERATURES TOP_PS TOP_KS REJECTION_FACTORS REJECTION_BASE
  export REJECTION_CLASSIFIER SKIP_RECONSTRUCTION_FID
  export WANDB_ENTITY WANDB_PROJECT WANDB_ID WANDB_NAME WANDB_GROUP WANDB_MODE
  export IMAGE PYDEPS MASTER_PORT PROJECT_DIR SELF
}

submit_job() {
  set_defaults
  if [[ -z "$REAL_STATS" && ! -d "$DATA_ROOT/$REAL_SPLIT" ]]; then
    echo "ImageNet $REAL_SPLIT not found under $DATA_ROOT" >&2
    exit 1
  fi
  if [[ -n "$REAL_STATS" && ! -f "$REAL_STATS" ]]; then
    echo "FID statistics file not found: $REAL_STATS" >&2
    exit 1
  fi
  if [[ -n "$FID_WEIGHTS_FILE" && ! -f "$FID_WEIGHTS_FILE" ]]; then
    echo "FID Inception weights not found: $FID_WEIGHTS_FILE" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$(dirname "$STAGE1_CHECKPOINT")"

  echo "=== v8dup full ImageNet sampling sweep ==="
  echo "  run_dir=$RUN_DIR"
  echo "  wandb=$WANDB_ENTITY/$WANDB_PROJECT id=$WANDB_ID name=$WANDB_NAME"
  echo "  resources: partition=$PARTITION constraint=$CONSTRAINT nodes=$NODES gpus_per_node=$GPUS_PER_NODE world_size=$((NODES * GPUS_PER_NODE))"
  echo "  data=$DATA_ROOT"
  echo "  real_fid_reference=${REAL_STATS:-$DATA_ROOT/$REAL_SPLIT} real_num_samples=$REAL_NUM_SAMPLES (0=entire split)"
  echo "  fid_inception_weights=${FID_WEIGHTS_FILE:-download-on-demand}"
  echo "  stage1=$STAGE1_CHECKPOINT"
  echo "  stage2_artifact=$STAGE2_ARTIFACT"
  echo "  stage2_checkpoint=$STAGE2_CHECKPOINT"
  echo "  sweep: n=$NUM_SAMPLES batch=$EVAL_BATCH_SIZE temps=$TEMPERATURES top_ps=$TOP_PS top_ks=$TOP_KS rejection=$REJECTION_FACTORS base=$REJECTION_BASE"

  sbatch_args=(
    --partition="$PARTITION"
    --job-name="v8-samp-eval"
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
  if [[ "${TEST_ONLY:-0}" == "1" ]]; then
    sbatch --test-only "${sbatch_args[@]}" "$SELF" --worker
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
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  export MASTER_ADDR MASTER_PORT

  echo "=== SLURM allocation ==="
  echo "  job=$SLURM_JOB_ID nodes=$SLURM_NNODES nodelist=$SLURM_NODELIST"
  echo "  master=$MASTER_ADDR port=$MASTER_PORT"
  echo "  run_dir=$RUN_DIR"

  container_prefix
  "${CONTAINER[@]}" bash "$SELF" --inside setup
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside checkpoint
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside eval
}

ensure_runtime() {
  export PYTHONUSERBASE="$PYDEPS"
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export WANDB_ENTITY WANDB_PROJECT WANDB_MODE
  export WANDB_DIR="${WANDB_DIR:-$LOCAL_SCRATCH_ROOT/wandb/run}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$LOCAL_SCRATCH_ROOT/wandb/cache}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$LOCAL_SCRATCH_ROOT/wandb/data}"
  export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$LOCAL_SCRATCH_ROOT/wandb/artifacts}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH_ROOT/cache}"
  export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  unset NCCL_ASYNC_ERROR_HANDLING
  mkdir -p "$PYTHONUSERBASE" "$LOCAL_SCRATCH_ROOT" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
    "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME/checkpoints" \
    "$MPLCONFIGDIR" "$(dirname "$STAGE2_CHECKPOINT")"
  if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    echo "W&B run dir: $WANDB_DIR" >&2
    echo "W&B data/staging root: $WANDB_DATA_DIR" >&2
    echo "Torch cache: $TORCH_HOME" >&2
    df -h "$LOCAL_SCRATCH_ROOT" >&2 || true
  fi
}

install_deps() {
  ensure_runtime
  install_cmd=(
    python -m pip install --user --quiet
    "numpy<2" scipy "wandb==0.16.6" lightning omegaconf hydra-core rich
    "torchmetrics[image]" torch-fidelity matplotlib lpips tqdm
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

download_stage1() {
  install_deps
  python - "$STAGE1_ARTIFACT" "$STAGE1_CHECKPOINT" <<'PY'
import shutil
import sys
from pathlib import Path

import wandb

artifact_name, target_arg = sys.argv[1:3]
target = Path(target_arg).expanduser().resolve()
if target.is_file():
    print(f"reusing {target}", flush=True)
    raise SystemExit(0)
target.parent.mkdir(parents=True, exist_ok=True)
api = wandb.Api(timeout=180)
artifact = api.artifact(artifact_name, type="model")
downloaded = Path(artifact.download(root=str(target.parent / f".download-{artifact.name.replace(':', '-')}")))
candidates = [downloaded / "best_rfid_slot3_model.pt", downloaded / "epoch5_model.pt"]
candidates.extend(path for path in downloaded.rglob("*") if path.is_file() and path.suffix in {".pt", ".ckpt"})
for candidate in candidates:
    if candidate.is_file():
        shutil.copy2(candidate, target)
        print(f"downloaded {artifact_name} -> {target}", flush=True)
        break
else:
    raise FileNotFoundError(f"No checkpoint file found in {artifact_name} under {downloaded}")
PY
}

download_stage2() {
  install_deps
  python - "$STAGE2_ARTIFACT" "$STAGE2_CHECKPOINT" "$OUTPUT_DIR/source_checkpoint_artifact.json" <<'PY'
import json
import os
import re
import shutil
import sys
from pathlib import Path

import wandb

artifact_name, target_arg, spec_arg = sys.argv[1:4]
target = Path(target_arg).expanduser().resolve()
target.parent.mkdir(parents=True, exist_ok=True)
api = wandb.Api(timeout=180)
artifact = api.artifact(artifact_name, type="model")
safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact.version)
marker = target.parent / ".source_artifact.json"
desired = {
    "requested": artifact_name,
    "resolved": artifact_name.rsplit(":", 1)[0] + f":{artifact.version}",
    "version": artifact.version,
    "metadata": dict(artifact.metadata or {}),
}
is_node_zero = os.environ.get("SLURM_PROCID", "0") == "0"
if target.is_file() and marker.is_file():
    current = json.loads(marker.read_text())
    if current.get("version") == desired["version"]:
        print(f"reusing {target} from {desired['resolved']}", flush=True)
        if is_node_zero:
            Path(spec_arg).parent.mkdir(parents=True, exist_ok=True)
            Path(spec_arg).write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
        raise SystemExit(0)
download_root = target.parent / f".download-v8dup-checkpoint-{safe_version}"
downloaded = Path(artifact.download(root=str(download_root)))
candidates = [downloaded / "last.pt"]
candidates.extend(path for path in downloaded.rglob("*") if path.is_file() and path.suffix in {".pt", ".ckpt"})
for candidate in candidates:
    if candidate.is_file():
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(candidate, tmp)
        os.replace(tmp, target)
        marker.write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
        if is_node_zero:
            Path(spec_arg).parent.mkdir(parents=True, exist_ok=True)
            Path(spec_arg).write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(download_root, ignore_errors=True)
        print(f"downloaded {desired['resolved']} -> {target} metadata={desired['metadata']}", flush=True)
        break
else:
    raise FileNotFoundError(f"No checkpoint file found in {desired['resolved']} under {downloaded}")
PY
}

run_eval() {
  install_deps
  if [[ -n "$FID_WEIGHTS_FILE" ]]; then
    fid_weights_target="$TORCH_HOME/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth"
    mkdir -p "$(dirname "$fid_weights_target")"
    if [[ ! -f "$fid_weights_target" ]]; then
      cp "$FID_WEIGHTS_FILE" "$fid_weights_target.tmp"
      mv "$fid_weights_target.tmp" "$fid_weights_target"
    fi
  fi
  extra_args=()
  if [[ -n "$REAL_STATS" ]]; then
    extra_args+=(--real-stats "$REAL_STATS")
  fi
  if [[ "$SKIP_RECONSTRUCTION_FID" == "0" || "$SKIP_RECONSTRUCTION_FID" == "false" ]]; then
    extra_args+=(--no-skip-reconstruction-fid)
  fi
  python -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="${SLURM_JOB_ID}_v8_sampling_eval" \
    --node_rank="$SLURM_PROCID" \
    "$PROJECT_DIR/scripts/evaluate_laser_full_imagenet.py" \
    --stage1 "$STAGE1_CHECKPOINT" \
    --stage2 "$STAGE2_CHECKPOINT" \
    --data "$DATA_ROOT" \
    --real-split "$REAL_SPLIT" \
    --real-num-samples "$REAL_NUM_SAMPLES" \
    --output "$OUTPUT_DIR" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-samples "$NUM_SAMPLES" \
    --temperatures "$TEMPERATURES" \
    --top-ps "$TOP_PS" \
    --top-ks "$TOP_KS" \
    --rejection-factors "$REJECTION_FACTORS" \
    --rejection-base "$REJECTION_BASE" \
    --rejection-classifier "$REJECTION_CLASSIFIER" \
    --wandb \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-id "$WANDB_ID" \
    --wandb-name "$WANDB_NAME" \
    --wandb-group "$WANDB_GROUP" \
    "${extra_args[@]}"
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
        download_stage1
        ;;
      checkpoint)
        download_stage2
        ;;
      eval)
        run_eval
        ;;
      *)
        echo "usage: $SELF --inside {setup|checkpoint|eval}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "usage: $SELF [--worker|--inside {setup|checkpoint|eval}]" >&2
    exit 2
    ;;
esac
