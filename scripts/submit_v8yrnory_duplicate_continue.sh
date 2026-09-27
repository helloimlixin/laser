#!/bin/bash
set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SELF="${SELF:-$PROJECT_DIR/scripts/submit_v8yrnory_duplicate_continue.sh}"

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

  DATA_ROOT="${IMAGENET_ROOT:-/scratch/$USER/Projects/data/imagenet}"
  RUN_NAME="${RUN_NAME:-imagenet-official-rqtransformer-laser-v8yrnory-continue-8gpu-$TS}"
  RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser/$RUN_NAME}"
  OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/stage2}"
  LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/$RUN_NAME}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$LOCAL_SCRATCH_ROOT/checkpoints}"

  STAGE1_ARTIFACT="${STAGE1_ARTIFACT:-helloimlixin-rutgers/laser/x3h5cl0h-stage1-checkpoint:latest}"
  SOURCE_STAGE2_ARTIFACT="${SOURCE_STAGE2_ARTIFACT:-helloimlixin-rutgers/laser/v8yrnory-checkpoint:latest}"
  STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$PROJECT_DIR/outputs/imagenet_x3h5cl0h_stage2/stage1_checkpoint/best_rfid_slot3_model.pt}"
  SOURCE_RESUME_CHECKPOINT="${SOURCE_RESUME_CHECKPOINT:-$LOCAL_SCRATCH_ROOT/source/v8yrnory-checkpoint-v0/last.pt}"
  TOKEN_CACHE="${TOKEN_CACHE:-$PROJECT_DIR/outputs/swgbasnb_cached_duplicate/token_cache/imagenet_train_sparse_components.pt}"

  TARGET_EPOCHS="${TARGET_EPOCHS:-100}"
  STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
  TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-2048}"
  LR="${LR:-0.0005}"
  LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
  MIN_LR="${MIN_LR:-0.0}"
  LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-0.0}"
  LR_SCHEDULE_EPOCHS="${LR_SCHEDULE_EPOCHS:-$TARGET_EPOCHS}"
  CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-128}"
  FID_EVERY="${FID_EVERY:-5}"
  FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-50000}"
  # The 1400M model remains resident during evaluation; 96-image Inception
  # batches caused CUDNN_STATUS_INTERNAL_ERROR on the 40 GB workers.
  FID_BATCH_SIZE="${FID_BATCH_SIZE:-16}"
  SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-5}"
  SAMPLE_GRID_EVERY="${SAMPLE_GRID_EVERY:-500}"

  WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
  WANDB_PROJECT="${WANDB_PROJECT:-laser}"
  WANDB_ID="${WANDB_ID:-v8dup$(date +%m%d%H%M%S)}"
  WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
  SELF_STAGE2_ARTIFACT="${SELF_STAGE2_ARTIFACT:-$WANDB_ENTITY/$WANDB_PROJECT/$WANDB_ID-checkpoint:latest}"
  PREFER_SELF_STAGE2_ARTIFACT="${PREFER_SELF_STAGE2_ARTIFACT:-1}"

  IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
  PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
  MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"
  CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-$MASTER_PORT}"
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-$((MASTER_PORT + 1))}"

  export TS PARTITION CONSTRAINT EXCLUDE_NODES DEPENDENCY NODES GPUS_PER_NODE CPUS_PER_TASK MEM_MB TIME_LIMIT
  export DATA_ROOT RUN_NAME RUN_DIR OUTPUT_DIR LOCAL_SCRATCH_ROOT CHECKPOINT_DIR
  export STAGE1_ARTIFACT SOURCE_STAGE2_ARTIFACT STAGE1_CHECKPOINT SOURCE_RESUME_CHECKPOINT TOKEN_CACHE
  export TARGET_EPOCHS STAGE2_BATCH_SIZE TOTAL_BATCH_SIZE LR LR_SCHEDULE MIN_LR LR_WARMUP_EPOCHS LR_SCHEDULE_EPOCHS CACHE_BATCH_SIZE
  export FID_EVERY FID_NUM_SAMPLES FID_BATCH_SIZE SAVE_CKPT_FREQ SAMPLE_GRID_EVERY
  export WANDB_ENTITY WANDB_PROJECT WANDB_ID WANDB_NAME SELF_STAGE2_ARTIFACT PREFER_SELF_STAGE2_ARTIFACT
  export IMAGE PYDEPS
  export MASTER_PORT CACHE_MASTER_PORT TRAIN_MASTER_PORT PROJECT_DIR SELF
}

submit_job() {
  set_defaults
  if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/val" ]]; then
    echo "ImageNet train/val not found under $DATA_ROOT" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$(dirname "$TOKEN_CACHE")" \
    "$(dirname "$STAGE1_CHECKPOINT")"

  echo "=== v8yrnory duplicate continuation ==="
  echo "  run_dir=$RUN_DIR"
  echo "  wandb=$WANDB_ENTITY/$WANDB_PROJECT id=$WANDB_ID name=$WANDB_NAME"
  echo "  resources: partition=$PARTITION constraint=$CONSTRAINT nodes=$NODES gpus_per_node=$GPUS_PER_NODE world_size=$((NODES * GPUS_PER_NODE))"
  echo "  data=$DATA_ROOT"
  echo "  stage1=$STAGE1_CHECKPOINT"
  echo "  source_resume=$SOURCE_RESUME_CHECKPOINT"
  echo "  self_resume_artifact=$SELF_STAGE2_ARTIFACT prefer_self=$PREFER_SELF_STAGE2_ARTIFACT"
  echo "  token_cache=$TOKEN_CACHE"
  echo "  lr=$LR schedule=$LR_SCHEDULE min_lr=$MIN_LR warmup_epochs=$LR_WARMUP_EPOCHS schedule_epochs=$LR_SCHEDULE_EPOCHS"

  sbatch_args=(
    --partition="$PARTITION"
    --job-name="v8dup-s2"
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
  # W&B stages checkpoint-sized artifacts before upload; keep every large
  # transient W&B path on node-local scratch instead of quota-limited GPFS.
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$LOCAL_SCRATCH_ROOT/wandb/data}"
  export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$LOCAL_SCRATCH_ROOT/wandb/artifacts}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH_ROOT/cache}"
  export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  unset NCCL_ASYNC_ERROR_HANDLING
  mkdir -p "$PYTHONUSERBASE" "$LOCAL_SCRATCH_ROOT" "$CHECKPOINT_DIR" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" \
    "$WANDB_ARTIFACT_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME/checkpoints" \
    "$TORCH_HOME/kernels"
  if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    echo "W&B run dir: $WANDB_DIR" >&2
    echo "W&B data/staging root: $WANDB_DATA_DIR" >&2
    echo "W&B artifact cache: $WANDB_ARTIFACT_DIR" >&2
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
  if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    python - <<'PY' || true
import os
from wandb.sdk.artifacts.staging import get_staging_dir
print(f"W&B resolved artifact staging: {get_staging_dir()}", flush=True)
for key in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_ARTIFACT_DIR"):
    print(f"{key}={os.environ.get(key, '')}", flush=True)
PY
  fi
}

download_artifacts() {
  install_deps
  python - "$STAGE1_ARTIFACT" "$STAGE1_CHECKPOINT" "$SOURCE_STAGE2_ARTIFACT" \
    "$SELF_STAGE2_ARTIFACT" "$PREFER_SELF_STAGE2_ARTIFACT" \
    "$RUN_DIR/source_resume_artifact.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import wandb

(
    stage1_artifact,
    stage1_target,
    source_stage2_artifact,
    self_stage2_artifact,
    prefer_self_stage2_artifact,
    source_spec_path,
) = sys.argv[1:7]
api = wandb.Api(timeout=180)


def fetch(artifact_name: str, target: Path, preferred_names):
    target = target.expanduser().resolve()
    if target.is_file():
        print(f"reusing {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    artifact = api.artifact(artifact_name, type="model")
    download_root = target.parent / f".download-{artifact.name.replace(':', '-')}"
    downloaded = Path(artifact.download(root=str(download_root)))
    candidates = [downloaded / name for name in preferred_names]
    candidates.extend(path for path in downloaded.rglob("*") if path.is_file() and path.suffix in {".pt", ".ckpt"})
    for candidate in candidates:
        if candidate.is_file():
            if candidate.resolve() != target:
                shutil.copy2(candidate, target)
            print(f"downloaded {artifact_name} -> {target}", flush=True)
            return
    raise FileNotFoundError(f"No checkpoint file found in {artifact_name} under {downloaded}")


def resolve_stage2_artifact(source_ref: str, self_ref: str, prefer_self: str):
    refs = []
    if prefer_self not in {"0", "false", "False", ""} and self_ref:
        refs.append(self_ref)
    refs.append(source_ref)
    if self_ref and self_ref not in refs:
        refs.append(self_ref)
    errors = []
    for ref in refs:
        try:
            artifact = api.artifact(ref, type="model")
            if "last.pt" not in artifact.manifest.entries:
                raise FileNotFoundError(f"{ref} does not contain last.pt")
            return ref, artifact
        except Exception as exc:
            errors.append(f"{ref}: {exc}")
    raise RuntimeError("Could not resolve a stage-2 resume artifact:\n" + "\n".join(errors))


fetch(stage1_artifact, Path(stage1_target), ["best_rfid_slot3_model.pt", "epoch5_model.pt"])
source_requested, source = resolve_stage2_artifact(
    source_stage2_artifact, self_stage2_artifact, prefer_self_stage2_artifact
)
source_base = source_requested.rsplit(":", 1)[0]
entry = source.manifest.entries.get("last.pt")
spec = {
    "requested": source_requested,
    "fallback": source_stage2_artifact,
    "preferred_self": self_stage2_artifact,
    "resolved": f"{source_base}:{source.version}",
    "version": source.version,
    "manifest_digest": None if entry is None else getattr(entry, "digest", None),
    "manifest_size": None if entry is None else getattr(entry, "size", None),
    "metadata": dict(source.metadata or {}),
}
spec_path = Path(source_spec_path)
spec_path.parent.mkdir(parents=True, exist_ok=True)
spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
print(
    f"resolved source resume artifact: {spec['resolved']} "
    f"metadata={spec['metadata']}",
    flush=True,
)
PY
}

download_resume_checkpoint() {
  install_deps
  python - "$SOURCE_STAGE2_ARTIFACT" "$SOURCE_RESUME_CHECKPOINT" \
    "$RUN_DIR/source_resume_artifact.json" "$CHECKPOINT_DIR" <<'PY'
import json
import os
import re
import shutil
import sys
from pathlib import Path

import wandb

artifact_name, target_arg, spec_arg, checkpoint_dir_arg = sys.argv[1:5]
target = Path(target_arg).expanduser().resolve()
target.parent.mkdir(parents=True, exist_ok=True)
checkpoint_dir = Path(checkpoint_dir_arg).expanduser().resolve()
checkpoint_dir.mkdir(parents=True, exist_ok=True)
api = wandb.Api(timeout=180)
spec_path = Path(spec_arg)
if spec_path.is_file():
    spec = json.loads(spec_path.read_text())
    artifact_ref = spec.get("resolved") or artifact_name
else:
    artifact_ref = artifact_name
    spec = {"requested": artifact_name}
artifact = api.artifact(artifact_ref, type="model")
entry = artifact.manifest.entries.get("last.pt")
desired = {
    "requested": spec.get("requested", artifact_name),
    "resolved": artifact_ref.rsplit(":", 1)[0] + f":{artifact.version}",
    "version": artifact.version,
    "manifest_digest": None if entry is None else getattr(entry, "digest", None),
    "manifest_size": None if entry is None else getattr(entry, "size", None),
    "metadata": dict(artifact.metadata or {}),
}
marker = target.parent / ".source_artifact.json"
if target.is_file() and marker.is_file():
    current = json.loads(marker.read_text())
    if (
        current.get("version") == desired["version"]
        and current.get("manifest_digest") == desired["manifest_digest"]
    ):
        print(f"reusing {target} from {desired['resolved']}", flush=True)
        raise SystemExit(0)
safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", desired["version"])
download_root = target.parent / f".download-v8yrnory-checkpoint-{safe_version}"
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
        # Restore the source artifact's retained best checkpoints as well as
        # last.pt, rebasing them into this run's bounded checkpoint directory.
        def install_local(source, destination):
            tmp = destination.with_suffix(destination.suffix + ".tmp")
            tmp.unlink(missing_ok=True)
            try:
                os.link(source, tmp)
            except OSError:
                shutil.copy2(source, tmp)
            os.replace(tmp, destination)

        for retained in downloaded.rglob("best_fid_*.pt"):
            restored = checkpoint_dir / retained.name
            install_local(retained, restored)
        local_last = checkpoint_dir / "last.pt"
        install_local(target, local_last)
        shutil.rmtree(download_root, ignore_errors=True)
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
print(f"validated token cache: {report_path}", flush=True)
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
    --batch-size "$CACHE_BATCH_SIZE"
}

train_stage2() {
  install_deps
  validate_cache
  resume_checkpoint="$SOURCE_RESUME_CHECKPOINT"
  echo "using artifact-resolved resume checkpoint: $resume_checkpoint"
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
        download_artifacts
        ;;
      cache)
        build_cache
        ;;
      validate)
        validate_cache
        ;;
      resume)
        download_resume_checkpoint
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
