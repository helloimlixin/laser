#!/bin/bash
set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SELF="${SELF:-$PROJECT_DIR/scripts/submit_v8dup_class_grid_sampling.sh}"

set_defaults() {
  TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
  PARTITION="${PARTITION:-cgpu-redhat}"
  CONSTRAINT="${CONSTRAINT:-volta}"
  NODES="${NODES:-2}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
  MEM_MB="${MEM_MB:-180000}"
  TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
  DEPENDENCY="${DEPENDENCY:-}"

  RUN_NAME="${RUN_NAME:-v8dup0731113220-v15-class-grids-$TS}"
  RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser_class_grids/$RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/grids}"
  LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser_class_grids/$RUN_NAME}"

  STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/scratch/$USER/runs/imagenet_x3h5cl0h_strict_bottleneck_sweep/imagenet-x3h5cl0h-strict-bottleneck-sweep-20260719_014434/k2-a16384/in256-rqvae-laser-8x8-a16384-k2/19072026_024726/best_rfid_slot3_model.pt}"
  STAGE2_ARTIFACT="${STAGE2_ARTIFACT:-helloimlixin-rutgers/laser/v8dup0731113220-checkpoint:v15}"
  STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-$LOCAL_SCRATCH_ROOT/source/v8dup0731113220-checkpoint/last.pt}"
  SOURCE_ARTIFACT_MANIFEST="${SOURCE_ARTIFACT_MANIFEST:-$RUN_DIR/source_checkpoint_artifact.json}"

  SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-64}"
  GRID_COLUMNS="${GRID_COLUMNS:-8}"
  SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
  TEMPERATURE="${TEMPERATURE:-1.0}"
  TOP_P="${TOP_P:-0.92}"
  TOP_K="${TOP_K:-0}"
  SEED="${SEED:-1234}"

  WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  WANDB_ID="${WANDB_ID:-v8grid$(date +%m%d%H%M%S)}"
  WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
  WANDB_GROUP="${WANDB_GROUP:-v8dup0731113220-class-grids}"
  WANDB_MODE="${WANDB_MODE:-online}"

  IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
  PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
  MASTER_PORT="${MASTER_PORT:-$((24000 + (${SLURM_JOB_ID:-0} % 20000)))}"

  export TS PARTITION CONSTRAINT NODES GPUS_PER_NODE CPUS_PER_TASK MEM_MB TIME_LIMIT DEPENDENCY
  export RUN_NAME RUN_DIR OUTPUT_DIR LOCAL_SCRATCH_ROOT STAGE1_CHECKPOINT STAGE2_ARTIFACT STAGE2_CHECKPOINT SOURCE_ARTIFACT_MANIFEST
  export SAMPLES_PER_CLASS GRID_COLUMNS SAMPLE_BATCH_SIZE TEMPERATURE TOP_P TOP_K SEED
  export WANDB_ENTITY WANDB_PROJECT WANDB_ID WANDB_NAME WANDB_GROUP WANDB_MODE
  export IMAGE PYDEPS MASTER_PORT PROJECT_DIR SELF
}

submit_job() {
  set_defaults
  if [[ ! -f "$STAGE1_CHECKPOINT" ]]; then
    echo "Missing stage-1 checkpoint: $STAGE1_CHECKPOINT" >&2
    exit 2
  fi
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR"
  args=(
    --partition="$PARTITION"
    --job-name="v8-class-grid"
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
    args+=(--constraint="$CONSTRAINT")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    args+=(--dependency="$DEPENDENCY")
  fi
  echo "run_dir=$RUN_DIR artifact=$STAGE2_ARTIFACT wandb=$WANDB_ENTITY/$WANDB_PROJECT/$WANDB_ID"
  echo "classes=17 samples_per_class=$SAMPLES_PER_CLASS grid=${GRID_COLUMNS}x$((SAMPLES_PER_CLASS / GRID_COLUMNS)) padding=0"
  echo "sampling=temperature:$TEMPERATURE top_p:$TOP_P top_k:${TOP_K} (0=full)"
  if [[ "${TEST_ONLY:-0}" == "1" ]]; then
    sbatch --test-only "${args[@]}" "$SELF" --worker
  elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN sbatch'
    printf ' %q' "${args[@]}" "$SELF" --worker
    printf '\n'
  else
    sbatch "${args[@]}" "$SELF" --worker
  fi
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
      --bind "$RUN_DIR"
      --bind /mnt/scratch
      --bind /dev/shm
      "$IMAGE"
    )
  else
    echo "Warning: singularity unavailable; running bare" >&2
    CONTAINER=()
  fi
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
    "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
    "$MPLCONFIGDIR" "$(dirname "$STAGE2_CHECKPOINT")"
}

install_deps() {
  ensure_runtime
  if python -c 'import lightning, omegaconf, torchmetrics, wandb' >/dev/null 2>&1; then
    return
  fi
  install_cmd=(
    python -m pip install --user --quiet
    "numpy<2" scipy "wandb==0.16.6" lightning omegaconf hydra-core rich
    "torchmetrics[image]" torch-fidelity matplotlib lpips tqdm
  )
  if command -v flock >/dev/null 2>&1; then
    (flock 9; "${install_cmd[@]}") 9>"$PYTHONUSERBASE/.install.lock"
  else
    "${install_cmd[@]}"
  fi
}

download_stage2() {
  install_deps
  python - "$STAGE2_ARTIFACT" "$STAGE2_CHECKPOINT" "$SOURCE_ARTIFACT_MANIFEST" <<'PY'
import json
import os
import re
import shutil
import sys
from pathlib import Path

import wandb

artifact_name, target_arg, manifest_arg = sys.argv[1:4]
target = Path(target_arg).expanduser().resolve()
target.parent.mkdir(parents=True, exist_ok=True)
api = wandb.Api(timeout=180)
artifact = api.artifact(artifact_name, type="model")
desired = {
    "requested": artifact_name,
    "resolved": artifact_name.rsplit(":", 1)[0] + f":{artifact.version}",
    "version": artifact.version,
    "digest": artifact.digest,
    "metadata": dict(artifact.metadata or {}),
}
marker = target.parent / ".source_artifact.json"
node_rank = int(os.environ.get("SLURM_PROCID", "0"))
if target.is_file() and marker.is_file():
    current = json.loads(marker.read_text())
    if current.get("digest") == desired["digest"]:
        if node_rank == 0:
            Path(manifest_arg).write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
        print(f"reusing {target} from {desired['resolved']}", flush=True)
        raise SystemExit(0)
safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact.version)
downloaded = Path(artifact.download(root=str(target.parent / f".download-{safe_version}")))
candidates = [downloaded / "last.pt"]
candidates.extend(path for path in downloaded.rglob("*") if path.is_file() and path.suffix in {".pt", ".ckpt"})
for candidate in candidates:
    if not candidate.is_file():
        continue
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(candidate, temporary)
    os.replace(temporary, target)
    marker.write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
    if node_rank == 0:
        Path(manifest_arg).write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
    print(f"downloaded {desired['resolved']} -> {target}", flush=True)
    break
else:
    raise FileNotFoundError(f"No checkpoint in {desired['resolved']} under {downloaded}")
PY
}

sample_grids() {
  install_deps
  python -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$SLURM_PROCID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="${SLURM_JOB_ID}_v8_class_grids" \
    "$PROJECT_DIR/scripts/sample_v8dup_class_grids.py" \
      --stage1 "$STAGE1_CHECKPOINT" \
      --stage2 "$STAGE2_CHECKPOINT" \
      --source-artifact-manifest "$SOURCE_ARTIFACT_MANIFEST" \
      --output "$OUTPUT_DIR" \
      --samples-per-class "$SAMPLES_PER_CLASS" \
      --grid-columns "$GRID_COLUMNS" \
      --batch-size "$SAMPLE_BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --seed "$SEED" \
      --expected-epoch 100 \
      --wandb \
      --wandb-entity "$WANDB_ENTITY" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-id "$WANDB_ID" \
      --wandb-name "$WANDB_NAME" \
      --wandb-group "$WANDB_GROUP"
}

run_worker() {
  set_defaults
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$LOCAL_SCRATCH_ROOT"
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  export MASTER_ADDR MASTER_PORT
  echo "job=$SLURM_JOB_ID nodes=$SLURM_NNODES gpus=$((SLURM_NNODES * GPUS_PER_NODE)) nodelist=$SLURM_NODELIST"
  echo "artifact=$STAGE2_ARTIFACT output=$OUTPUT_DIR wandb_id=$WANDB_ID"
  container_prefix
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside checkpoint
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside sample
}

case "${1:-}" in
  "") submit_job ;;
  --worker) run_worker ;;
  --inside)
    set_defaults
    case "${2:-}" in
      checkpoint) download_stage2 ;;
      sample) sample_grids ;;
      *) echo "usage: $SELF --inside {checkpoint|sample}" >&2; exit 2 ;;
    esac
    ;;
  *) echo "usage: $SELF [--worker|--inside {checkpoint|sample}]" >&2; exit 2 ;;
esac
