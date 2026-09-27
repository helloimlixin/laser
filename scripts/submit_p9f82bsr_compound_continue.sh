#!/bin/bash
set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SELF="${SELF:-$PROJECT_DIR/scripts/submit_p9f82bsr_compound_continue.sh}"

set_defaults() {
  TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
  PARTITION="${PARTITION:-gpu-redhat}"
  CONSTRAINT="${CONSTRAINT:-adalovelace}"
  NODES="${NODES:-8}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-24}"
  MEM_MB="${MEM_MB:-250000}"
  TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
  WORLD_SIZE=$((NODES * GPUS_PER_NODE))

  DATA_ROOT="${IMAGENET_ROOT:-/scratch/$USER/Projects/data/imagenet}"
  RUN_NAME="${RUN_NAME:-imagenet-official-rqtransformer-laser-p9f82bsr-compound-continue-${WORLD_SIZE}gpu-$TS}"
  RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser/$RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/stage2}"
  LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/$RUN_NAME}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$LOCAL_SCRATCH_ROOT/checkpoints}"

  STAGE1_ARTIFACT="${STAGE1_ARTIFACT:-helloimlixin-rutgers/laser/x3h5cl0h-stage1-checkpoint:latest}"
  SOURCE_STAGE2_ARTIFACT="${SOURCE_STAGE2_ARTIFACT:-helloimlixin-rutgers/laser/p9f82bsr-checkpoint:latest}"
  STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/scratch/$USER/runs/imagenet_x3h5cl0h_strict_bottleneck_sweep/imagenet-x3h5cl0h-strict-bottleneck-sweep-20260719_014434/k2-a16384/in256-rqvae-laser-8x8-a16384-k2/19072026_024726/best_rfid_slot3_model.pt}"
  SOURCE_RESUME_CHECKPOINT="${SOURCE_RESUME_CHECKPOINT:-$LOCAL_SCRATCH_ROOT/source/p9f82bsr-checkpoint/last.pt}"
  TOKEN_CACHE="${TOKEN_CACHE:-$PROJECT_DIR/outputs/p9f82bsr_compound_continue/token_cache/imagenet_train_compound_pairs.pt}"

  TARGET_EPOCHS="${TARGET_EPOCHS:-100}"
  STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-64}"
  TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-2048}"
  LR="${LR:-0.0005}"
  LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
  MIN_LR="${MIN_LR:-0.0}"
  LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-0.0}"
  LR_SCHEDULE_EPOCHS="${LR_SCHEDULE_EPOCHS:-$TARGET_EPOCHS}"
  CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-32}"
  CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-4}"
  FID_EVERY="${FID_EVERY:-5}"
  FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-50000}"
  FID_BATCH_SIZE="${FID_BATCH_SIZE:-16}"
  SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
  SAMPLE_GRID_EVERY="${SAMPLE_GRID_EVERY:-500}"

  WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  WANDB_ID="${WANDB_ID:-p9dup$(date +%m%d%H%M%S)}"
  WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
  SELF_STAGE2_ARTIFACT="${SELF_STAGE2_ARTIFACT:-$WANDB_ENTITY/$WANDB_PROJECT/$WANDB_ID-checkpoint:latest}"
  PREFER_SELF_STAGE2_ARTIFACT="${PREFER_SELF_STAGE2_ARTIFACT:-1}"
  REQUIRE_STAGE2_ARTIFACT="${REQUIRE_STAGE2_ARTIFACT:-1}"
  RESUME_WAIT_SECONDS="${RESUME_WAIT_SECONDS:-21600}"
  RESUME_POLL_SECONDS="${RESUME_POLL_SECONDS:-300}"

  IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
  PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
  MASTER_PORT="${MASTER_PORT:-$((22000 + (${SLURM_JOB_ID:-0} % 18000)))}"
  CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-$MASTER_PORT}"
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-$((MASTER_PORT + 1))}"

  export TS PARTITION CONSTRAINT NODES GPUS_PER_NODE CPUS_PER_TASK MEM_MB TIME_LIMIT WORLD_SIZE
  export DATA_ROOT RUN_NAME RUN_DIR OUTPUT_DIR LOCAL_SCRATCH_ROOT CHECKPOINT_DIR
  export STAGE1_ARTIFACT SOURCE_STAGE2_ARTIFACT STAGE1_CHECKPOINT SOURCE_RESUME_CHECKPOINT TOKEN_CACHE
  export TARGET_EPOCHS STAGE2_BATCH_SIZE TOTAL_BATCH_SIZE LR LR_SCHEDULE MIN_LR LR_WARMUP_EPOCHS LR_SCHEDULE_EPOCHS CACHE_BATCH_SIZE CACHE_NUM_WORKERS
  export FID_EVERY FID_NUM_SAMPLES FID_BATCH_SIZE SAVE_CKPT_FREQ SAMPLE_GRID_EVERY
  export WANDB_ENTITY WANDB_PROJECT WANDB_ID WANDB_NAME SELF_STAGE2_ARTIFACT PREFER_SELF_STAGE2_ARTIFACT
  export REQUIRE_STAGE2_ARTIFACT RESUME_WAIT_SECONDS RESUME_POLL_SECONDS
  export IMAGE PYDEPS MASTER_PORT CACHE_MASTER_PORT TRAIN_MASTER_PORT PROJECT_DIR SELF
}

submit_job() {
  set_defaults
  if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/val" ]]; then
    echo "ImageNet train/val not found under $DATA_ROOT" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$(dirname "$TOKEN_CACHE")" "$(dirname "$STAGE1_CHECKPOINT")"

  echo "=== p9f82bsr compound continuation ==="
  echo "  run_dir=$RUN_DIR"
  echo "  wandb=$WANDB_ENTITY/$WANDB_PROJECT id=$WANDB_ID name=$WANDB_NAME"
  echo "  resources: partition=$PARTITION constraint=$CONSTRAINT nodes=$NODES gpus_per_node=$GPUS_PER_NODE world_size=$WORLD_SIZE"
  echo "  data=$DATA_ROOT"
  echo "  stage1=$STAGE1_CHECKPOINT"
  echo "  source_artifact=$SOURCE_STAGE2_ARTIFACT"
  echo "  self_artifact=$SELF_STAGE2_ARTIFACT prefer_self=$PREFER_SELF_STAGE2_ARTIFACT"
  echo "  source_resume=$SOURCE_RESUME_CHECKPOINT"
  echo "  token_cache=$TOKEN_CACHE"
  echo "  lr=$LR schedule=$LR_SCHEDULE min_lr=$MIN_LR warmup_epochs=$LR_WARMUP_EPOCHS schedule_epochs=$LR_SCHEDULE_EPOCHS"
  echo "  resume_wait=${RESUME_WAIT_SECONDS}s require_resume=$REQUIRE_STAGE2_ARTIFACT"

  sbatch_args=(
    --partition="$PARTITION"
    --job-name="p9cmp-s2"
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
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  export MASTER_ADDR MASTER_PORT CACHE_MASTER_PORT TRAIN_MASTER_PORT

  echo "=== SLURM allocation ==="
  echo "  job=$SLURM_JOB_ID nodes=$SLURM_NNODES nodelist=$SLURM_NODELIST"
  echo "  master=$MASTER_ADDR cache_port=$CACHE_MASTER_PORT train_port=$TRAIN_MASTER_PORT"
  echo "  run_dir=$RUN_DIR"

  container_prefix
  "${CONTAINER[@]}" bash "$SELF" --inside setup

  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside cache

  "${CONTAINER[@]}" bash "$SELF" --inside validate

  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside resume

  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    "${CONTAINER[@]}" bash "$SELF" --inside train
}

ensure_runtime() {
  export PYTHONUSERBASE="$PYDEPS"
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export WANDB_ENTITY WANDB_PROJECT
  export WANDB_DIR="${WANDB_DIR:-$LOCAL_SCRATCH_ROOT/wandb/run}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$LOCAL_SCRATCH_ROOT/wandb/cache}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$LOCAL_SCRATCH_ROOT/wandb/data}"
  export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$LOCAL_SCRATCH_ROOT/wandb/artifacts}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH_ROOT/cache}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCAL_SCRATCH_ROOT/cache/matplotlib}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  unset NCCL_ASYNC_ERROR_HANDLING
  mkdir -p "$PYTHONUSERBASE" "$LOCAL_SCRATCH_ROOT" "$CHECKPOINT_DIR" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" \
    "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
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

setup_stage1() {
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
download_root = target.parent / f".download-{artifact.name.replace(':', '-')}"
downloaded = Path(artifact.download(root=str(download_root)))
candidates = [downloaded / "best_rfid_slot3_model.pt", downloaded / "epoch5_model.pt"]
candidates.extend(path for path in downloaded.rglob("*") if path.is_file() and path.suffix in {".pt", ".ckpt"})
for candidate in candidates:
    if candidate.is_file():
        if candidate.resolve() != target:
            shutil.copy2(candidate, target)
        print(f"downloaded {artifact_name} -> {target}", flush=True)
        break
else:
    raise FileNotFoundError(f"No stage-1 checkpoint found in {artifact_name} under {downloaded}")
PY
}

resolve_resume_checkpoint() {
  install_deps
  python - "$SOURCE_STAGE2_ARTIFACT" "$SELF_STAGE2_ARTIFACT" "$PREFER_SELF_STAGE2_ARTIFACT" \
    "$SOURCE_RESUME_CHECKPOINT" "$RUN_DIR/source_resume_artifact.json" \
    "$REQUIRE_STAGE2_ARTIFACT" "$RESUME_WAIT_SECONDS" "$RESUME_POLL_SECONDS" <<'PY'
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import wandb

(
    source_ref,
    self_ref,
    prefer_self,
    target_arg,
    spec_arg,
    require_resume,
    wait_seconds,
    poll_seconds,
) = sys.argv[1:9]
target = Path(target_arg).expanduser().resolve()
target.parent.mkdir(parents=True, exist_ok=True)
spec_path = Path(spec_arg)
require_resume = require_resume not in {"0", "false", "False", ""}
deadline = time.monotonic() + max(0, int(wait_seconds))
poll = max(10, int(poll_seconds))
api = wandb.Api(timeout=180)


def candidate_refs():
    refs = []
    if prefer_self not in {"0", "false", "False", ""} and self_ref:
        refs.append(self_ref)
    refs.append(source_ref)
    if self_ref and self_ref not in refs:
        refs.append(self_ref)
    return refs


def try_resolve():
    errors = []
    for ref in candidate_refs():
        try:
            artifact = api.artifact(ref, type="model")
            entry = artifact.manifest.entries.get("last.pt")
            if entry is None:
                raise FileNotFoundError(f"{ref} does not contain last.pt")
            base = ref.rsplit(":", 1)[0]
            desired = {
                "requested": ref,
                "fallback": source_ref,
                "preferred_self": self_ref,
                "resolved": f"{base}:{artifact.version}",
                "version": artifact.version,
                "manifest_digest": getattr(entry, "digest", None),
                "manifest_size": getattr(entry, "size", None),
                "metadata": dict(artifact.metadata or {}),
            }
            return artifact, desired
        except Exception as exc:
            errors.append(f"{ref}: {exc}")
    return None, errors


while True:
    artifact, result = try_resolve()
    if artifact is not None:
        desired = result
        break
    if time.monotonic() >= deadline:
        if require_resume:
            raise RuntimeError("Could not resolve a stage-2 resume artifact:\n" + "\n".join(result))
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps({"resolved": None, "errors": result}, indent=2, sort_keys=True) + "\n")
        print("No resume artifact found; scratch training allowed", flush=True)
        raise SystemExit(0)
    print("Waiting for stage-2 resume artifact:\n" + "\n".join(result), flush=True)
    time.sleep(poll)

marker = target.parent / ".source_artifact.json"
if target.is_file() and marker.is_file():
    current = json.loads(marker.read_text())
    if (
        current.get("version") == desired["version"]
        and current.get("manifest_digest") == desired["manifest_digest"]
    ):
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
        print(f"reusing {target} from {desired['resolved']}", flush=True)
        raise SystemExit(0)

safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", desired["resolved"])
download_root = target.parent / f".download-{safe_name}"
downloaded = Path(artifact.download(root=str(download_root)))
candidates = [downloaded / "last.pt"]
candidates.extend(path for path in downloaded.rglob("*") if path.is_file() and path.suffix in {".pt", ".ckpt"})
for candidate in candidates:
    if candidate.is_file():
        if candidate.resolve() != target:
            tmp = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(candidate, tmp)
            os.replace(tmp, target)
        marker.write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(desired, indent=2, sort_keys=True) + "\n")
        print(
            f"downloaded {desired['resolved']} -> {target} "
            f"metadata={desired['metadata']}",
            flush=True,
        )
        break
else:
    raise FileNotFoundError(f"No checkpoint file found in {desired['resolved']} under {downloaded}")
PY
}

validate_cache() {
  install_deps
  python - "$TOKEN_CACHE" <<'PY'
import json
import sys
from pathlib import Path

cache = Path(sys.argv[1])
report_path = cache.with_suffix(".validation.json")
if not cache.is_file():
    raise SystemExit(f"token cache missing: {cache}")
if not report_path.is_file():
    raise SystemExit(f"token cache validation missing: {report_path}")
report = json.loads(report_path.read_text())
if not report.get("passed"):
    raise SystemExit(f"token cache validation failed: {report}")
if int(report.get("compound_sequence_length", 0)) != 128:
    raise SystemExit(f"token cache is not compound-pair format: {report}")
print(f"validated compound token cache: {report_path}", flush=True)
PY
}

build_cache() {
  install_deps
  if [[ -f "$TOKEN_CACHE" && -f "${TOKEN_CACHE%.pt}.validation.json" ]]; then
    validate_cache
    exit 0
  fi
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$CACHE_MASTER_PORT" \
    --rdzv_id="${SLURM_JOB_ID}_cache" \
    --node_rank="$SLURM_PROCID" \
    "$PROJECT_DIR/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$STAGE1_CHECKPOINT" \
    --data "$DATA_ROOT" \
    --output "$TOKEN_CACHE" \
    --batch-size "$CACHE_BATCH_SIZE" \
    --num-workers "$CACHE_NUM_WORKERS" \
    --compound
}

train_stage2() {
  install_deps
  validate_cache
  resume_checkpoint="$SOURCE_RESUME_CHECKPOINT"
  if [[ "$CHECKPOINT_DIR" != /mnt/scratch/* && -f "$CHECKPOINT_DIR/last.pt" ]]; then
    resume_checkpoint="$CHECKPOINT_DIR/last.pt"
  fi
  echo "using resume checkpoint: $resume_checkpoint"
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$TRAIN_MASTER_PORT" \
    --rdzv_id="${SLURM_JOB_ID}_train" \
    --node_rank="$SLURM_PROCID" \
    "$PROJECT_DIR/scripts/train_official_rqtransformer_laser_stage2.py" \
    --checkpoint "$STAGE1_CHECKPOINT" \
    --resume-checkpoint "$resume_checkpoint" \
    --token-cache "$TOKEN_CACHE" \
    --compound-tokens \
    --data "$DATA_ROOT" \
    --output "$OUTPUT_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --epochs "$TARGET_EPOCHS" \
    --batch-size "$STAGE2_BATCH_SIZE" \
    --total-batch-size "$TOTAL_BATCH_SIZE" \
    --num-atoms 16384 \
    --coeff-vocab-size 2048 \
    --coeff-max 20 \
    --coeff-scale 6.4 \
    --lr "$LR" \
    --lr-schedule "$LR_SCHEDULE" \
    --min-lr "$MIN_LR" \
    --lr-warmup-epochs "$LR_WARMUP_EPOCHS" \
    --lr-schedule-epochs "$LR_SCHEDULE_EPOCHS" \
    --fid-num-samples "$FID_NUM_SAMPLES" \
    --fid-batch-size "$FID_BATCH_SIZE" \
    --fid-every "$FID_EVERY" \
    --save-ckpt-freq "$SAVE_CKPT_FREQ" \
    --sample-grid-every "$SAMPLE_GRID_EVERY" \
    --upload-checkpoints \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-id "$WANDB_ID" \
    --wandb-name "$WANDB_NAME"
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
        setup_stage1
        ;;
      cache)
        build_cache
        ;;
      validate)
        validate_cache
        ;;
      resume)
        resolve_resume_checkpoint
        ;;
      train)
        train_stage2
        ;;
      *)
        echo "unknown inside phase: ${2:-}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "usage: $SELF [--worker|--inside phase]" >&2
    exit 2
    ;;
esac
