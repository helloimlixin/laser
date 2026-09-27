#!/bin/bash
set -euo pipefail

L40S_JOB_ID="${L40S_JOB_ID:-}"
A100_JOB_ID="${A100_JOB_ID:-}"
CANDIDATE_JOB_IDS="${CANDIDATE_JOB_IDS:-}"
CANDIDATE_LABELS="${CANDIDATE_LABELS:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/xl598/Projects/laser}"
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_DIR/scripts/watch_v8dup_candidate_race.sh}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/runs/laser/v8dup-candidate-race}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/candidate-race.log}"
STATE_FILE="${STATE_FILE:-$RUN_DIR/candidate-race.state}"
RESUBMIT_ON_SIGNAL="${RESUBMIT_ON_SIGNAL:-0}"
WATCH_PARTITION="${WATCH_PARTITION:-main-redhat}"
WATCH_TIME_LIMIT="${WATCH_TIME_LIMIT:-72:00:00}"
WATCH_MEM_MB="${WATCH_MEM_MB:-1000}"
WATCH_JOB_NAME="${WATCH_JOB_NAME:-v8dup-race-watch}"
WATCH_SIGNAL_SECONDS="${WATCH_SIGNAL_SECONDS:-300}"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

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
  local job_id="$1"
  squeue -h -j "$job_id" -o "%T" | head -n 1
}

job_start_time() {
  local job_id="$1"
  scontrol show job -o "$job_id" 2>/dev/null \
    | tr ' ' '\n' \
    | awk -F= '$1 == "StartTime" {print $2; exit}'
}

label_for_index() {
  local index="$1"
  local label="${LABELS[$index]:-}"
  if [[ -n "$label" ]]; then
    printf '%s' "$label"
  else
    printf 'job-%s' "${JOBS[$index]}"
  fi
}

cancel_job() {
  local job_id="$1"
  local reason="$2"
  log "cancelling job $job_id: $reason"
  scancel "$job_id" || true
}

resubmit_self() {
  local submit_output
  submit_output=$(
    sbatch \
      --partition="$WATCH_PARTITION" \
      --job-name="$WATCH_JOB_NAME" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=1 \
      --mem="$WATCH_MEM_MB" \
      --time="$WATCH_TIME_LIMIT" \
      --signal="B:USR1@$WATCH_SIGNAL_SECONDS" \
      --chdir="$PROJECT_DIR" \
      --output="$RUN_DIR/slurm-%j.out" \
      --error="$RUN_DIR/slurm-%j.err" \
      --export=ALL,L40S_JOB_ID="$L40S_JOB_ID",A100_JOB_ID="$A100_JOB_ID",CANDIDATE_JOB_IDS="$CANDIDATE_JOB_IDS",CANDIDATE_LABELS="$CANDIDATE_LABELS",POLL_SECONDS="$POLL_SECONDS",PROJECT_DIR="$PROJECT_DIR",SCRIPT_PATH="$SCRIPT_PATH",RUN_DIR="$RUN_DIR",LOG_FILE="$LOG_FILE",STATE_FILE="$STATE_FILE",RESUBMIT_ON_SIGNAL="$RESUBMIT_ON_SIGNAL",WATCH_PARTITION="$WATCH_PARTITION",WATCH_TIME_LIMIT="$WATCH_TIME_LIMIT",WATCH_MEM_MB="$WATCH_MEM_MB",WATCH_JOB_NAME="$WATCH_JOB_NAME",WATCH_SIGNAL_SECONDS="$WATCH_SIGNAL_SECONDS" \
      "$SCRIPT_PATH" 2>&1
  )
  log "$submit_output"
  write_state "resubmitted_watcher $submit_output"
}

on_usr1() {
  log "received USR1 before time limit"
  if [[ "$RESUBMIT_ON_SIGNAL" == "1" ]]; then
    resubmit_self
  fi
  exit 0
}

main() {
  trap on_usr1 USR1
  if [[ -n "$CANDIDATE_JOB_IDS" ]]; then
    read -r -a JOBS <<<"$CANDIDATE_JOB_IDS"
    read -r -a LABELS <<<"$CANDIDATE_LABELS"
  else
    if [[ -z "$L40S_JOB_ID" || -z "$A100_JOB_ID" ]]; then
      echo "set CANDIDATE_JOB_IDS or both L40S_JOB_ID and A100_JOB_ID" >&2
      exit 2
    fi
    JOBS=("$L40S_JOB_ID" "$A100_JOB_ID")
    LABELS=("l40s" "a100")
  fi
  if (( ${#JOBS[@]} < 2 )); then
    echo "watcher needs at least two candidate jobs" >&2
    exit 2
  fi

  log "watching candidate race: jobs=${JOBS[*]} labels=${LABELS[*]:-} poll=${POLL_SECONDS}s"
  write_state "watching jobs=${JOBS[*]}"

  while true; do
    local states=()
    local running_indexes=()
    local live_count=0
    local state_line="states:"
    local i
    for i in "${!JOBS[@]}"; do
      local state
      state="$(job_state "${JOBS[$i]}")"
      states+=("$state")
      state_line+=" $(label_for_index "$i")=${state:-not-in-squeue}"
      if [[ "$state" == "PENDING" || "$state" == "RUNNING" ]]; then
        live_count=$((live_count + 1))
      fi
      if [[ "$state" == "RUNNING" ]]; then
        running_indexes+=("$i")
      fi
    done
    log "$state_line"

    if (( ${#running_indexes[@]} > 0 )); then
      local winner="${running_indexes[0]}"
      local winner_start
      winner_start="$(job_start_time "${JOBS[$winner]}")"
      for i in "${running_indexes[@]}"; do
        local start_time
        start_time="$(job_start_time "${JOBS[$i]}")"
        if [[ -n "$start_time" && ( -z "$winner_start" || "$start_time" < "$winner_start" ) ]]; then
          winner="$i"
          winner_start="$start_time"
        fi
      done

      for i in "${!JOBS[@]}"; do
        if [[ "$i" == "$winner" ]]; then
          continue
        fi
        if [[ "${states[$i]}" == "PENDING" || "${states[$i]}" == "RUNNING" ]]; then
          cancel_job "${JOBS[$i]}" "$(label_for_index "$winner") replacement started first"
        fi
      done
      write_state "winner=$(label_for_index "$winner") job=${JOBS[$winner]}"
      exit 0
    fi

    if (( live_count == 0 )); then
      log "no candidate is pending or running; exiting"
      write_state "no_candidates_visible"
      exit 0
    fi

    sleep "$POLL_SECONDS"
  done
}

main "$@"
