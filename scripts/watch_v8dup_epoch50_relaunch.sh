#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/xl598/Projects/laser}"
RUN_NAME="${RUN_NAME:-imagenet-official-rqtransformer-laser-v8yrnory-continue-8gpu-20260731_113220}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser/$RUN_NAME}"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/$RUN_NAME}"
WATCH_JOB_ID="${WATCH_JOB_ID:-59903502}"
TARGET_EPOCH="${TARGET_EPOCH:-50}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RELAUNCH_CONSTRAINT="${RELAUNCH_CONSTRAINT:-adalovelace}"
WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_ID="${WANDB_ID:-v8dup0731113220}"
SELF_STAGE2_ARTIFACT="${SELF_STAGE2_ARTIFACT:-$WANDB_ENTITY/$WANDB_PROJECT/$WANDB_ID-checkpoint:latest}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/scratch/$USER/runs/imagenet_x3h5cl0h_strict_bottleneck_sweep/imagenet-x3h5cl0h-strict-bottleneck-sweep-20260719_014434/k2-a16384/in256-rqvae-laser-8x8-a16384-k2/19072026_024726/best_rfid_slot3_model.pt}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"

OUT_LOG="${OUT_LOG:-$RUN_DIR/slurm-$WATCH_JOB_ID.out}"
WATCH_LOG="${WATCH_LOG:-$RUN_DIR/relaunch-after-epoch${TARGET_EPOCH}.log}"
STATE_FILE="${STATE_FILE:-$RUN_DIR/relaunch-after-epoch${TARGET_EPOCH}.state}"
LOCK_FILE="${LOCK_FILE:-$RUN_DIR/relaunch-after-epoch${TARGET_EPOCH}.lock}"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$WATCH_LOG") 2>&1

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

write_state() {
  printf '%s\n' "$*" > "$STATE_FILE"
}

job_state() {
  squeue -h -j "$WATCH_JOB_ID" -o "%T" | head -n 1
}

job_is_live() {
  [[ -n "$(job_state)" ]]
}

epoch_log_ready() {
  [[ -f "$OUT_LOG" ]] && grep -Eq "Epoch ${TARGET_EPOCH}: .*saved .*last\\.pt|Epoch ${TARGET_EPOCH}: FID=.*saved" "$OUT_LOG"
}

load_singularity() {
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
}

wandb_artifact_ready() {
  load_singularity
  if ! command -v singularity >/dev/null 2>&1; then
    log "singularity unavailable for W&B artifact check"
    return 1
  fi
  singularity exec \
    --bind "$PROJECT_DIR" \
    --bind "/scratch/$USER" \
    --bind "$RUN_DIR" \
    --bind /mnt/scratch \
    "$IMAGE" \
    bash -lc "
      export PYTHONUSERBASE='$PYDEPS'
      export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
      export PYTHONPATH='$PROJECT_DIR'
      python - '$SELF_STAGE2_ARTIFACT' '$TARGET_EPOCH' <<'PY'
import json
import sys

import wandb

artifact_ref = sys.argv[1]
target_epoch = int(sys.argv[2])
api = wandb.Api(timeout=180)
artifact = api.artifact(artifact_ref, type='model')
metadata = dict(artifact.metadata or {})
epoch = metadata.get('epoch')
if epoch is None or int(epoch) < target_epoch:
    raise SystemExit(f'{artifact_ref} metadata epoch {epoch} < {target_epoch}')
entry = artifact.manifest.entries.get('last.pt')
if entry is None:
    raise SystemExit(f'{artifact_ref} has no last.pt')
print(json.dumps({
    'artifact': artifact_ref,
    'version': artifact.version,
    'epoch': int(epoch),
    'step': metadata.get('step'),
    'fid': metadata.get('fid'),
    'last_pt_digest': getattr(entry, 'digest', None),
    'last_pt_size': getattr(entry, 'size', None),
}, sort_keys=True))
PY
    "
}

submit_relaunch() {
  cd "$PROJECT_DIR"
  log "submitting patched relaunch with constraint=$RELAUNCH_CONSTRAINT"
  local submit_output
  submit_output=$(
    RUN_NAME="$RUN_NAME" \
    WANDB_ID="$WANDB_ID" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    CONSTRAINT="$RELAUNCH_CONSTRAINT" \
    STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
    STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT" \
    bash "$PROJECT_DIR/scripts/submit_v8yrnory_duplicate_continue.sh" 2>&1
  )
  printf '%s\n' "$submit_output"
  local new_job
  new_job=$(awk '/Submitted batch job/ {print $4}' <<<"$submit_output" | tail -n 1)
  if [[ -z "$new_job" ]]; then
    log "failed to parse new job id"
    write_state "failed_to_parse_new_job"
    return 1
  fi
  log "submitted patched relaunch job $new_job"
  write_state "submitted $new_job"
}

main() {
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "another watcher already holds $LOCK_FILE"
    exit 0
  fi

  log "watching job=$WATCH_JOB_ID target_epoch=$TARGET_EPOCH artifact=$SELF_STAGE2_ARTIFACT"
  log "run_dir=$RUN_DIR"
  write_state "watching $WATCH_JOB_ID target_epoch $TARGET_EPOCH"

  while ! epoch_log_ready; do
    if ! job_is_live; then
      log "watched job is no longer in squeue before epoch $TARGET_EPOCH was logged"
      write_state "stopped_before_epoch_${TARGET_EPOCH}"
      exit 1
    fi
    log "waiting for epoch $TARGET_EPOCH completion; current_state=$(job_state)"
    sleep "$POLL_SECONDS"
  done

  log "epoch $TARGET_EPOCH completion is present in $OUT_LOG"
  write_state "epoch_${TARGET_EPOCH}_logged"

  until wandb_artifact_ready; do
    if ! job_is_live; then
      log "watched job stopped before W&B artifact became ready"
      write_state "stopped_before_wandb_epoch_${TARGET_EPOCH}"
      exit 1
    fi
    log "waiting for W&B artifact $SELF_STAGE2_ARTIFACT to report epoch >= $TARGET_EPOCH"
    sleep "$POLL_SECONDS"
  done

  log "W&B artifact is ready for epoch $TARGET_EPOCH"
  write_state "wandb_epoch_${TARGET_EPOCH}_ready"

  if job_is_live; then
    log "cancelling old job $WATCH_JOB_ID"
    scancel "$WATCH_JOB_ID"
    for _ in $(seq 1 60); do
      if ! job_is_live; then
        break
      fi
      sleep 10
    done
  fi

  if job_is_live; then
    log "old job $WATCH_JOB_ID is still visible after cancellation wait"
    write_state "cancel_wait_failed"
    exit 1
  fi

  submit_relaunch
}

main "$@"
