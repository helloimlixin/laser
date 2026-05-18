#!/bin/bash

: "${PARTITION:=gpu-redhat}"
: "${RUN_ROOT:=/cache/home/$USER/runs/chq256_pg30}"
: "${DATA_DIR:=/scratch/$USER/datasets/celebahq_packed_256}"
: "${CASES:=p4s4,p8s8}"
: "${BINS_LIST:=256,512}"

: "${IMG:=256}"
: "${ATOMS:=4096}"
: "${K:=16}"
: "${EMB:=4}"
: "${NHID:=128}"
: "${RBLK:=2}"
: "${RHID:=32}"
: "${CMAX:=20.0}"

: "${S1_EPOCHS:=30}"
: "${S1_BSZ:=4}"
: "${S1_WORKERS:=8}"
: "${S1_LR:=2e-4}"

: "${S2_EPOCHS:=30}"
: "${S2_BSZ:=2}"
: "${S2_WORKERS:=4}"
: "${S2_LR:=1e-4}"
: "${D_MODEL:=256}"
: "${HEADS:=8}"
: "${LAYERS:=8}"
: "${D_FF:=768}"
: "${WIN:=64}"

: "${WANDB_PROJECT:=laser}"

pg_tag() {
  printf '%s\n' "${TAG:?TAG is required}"
}

pg_patch() {
  printf '%s\n' "${PATCH:?PATCH is required}"
}

pg_stride() {
  printf '%s\n' "${STRIDE:?STRIDE is required}"
}

pg_meta_dir() {
  printf '%s\n' "$RUN_ROOT/meta"
}

pg_s1_out() {
  printf '%s\n' "$RUN_ROOT/s1"
}

pg_s2_out() {
  local bins="$1"
  printf '%s\n' "$RUN_ROOT/s2_b${bins}"
}

pg_cache_pt() {
  local bins="$1"
  printf '%s\n' "$RUN_ROOT/cache/tok_q${bins}.pt"
}

pg_s1_ckpt_ref() {
  printf '%s\n' "$(pg_meta_dir)/s1_ckpt.txt"
}

pg_wandb_dir() {
  printf '%s\n' "$RUN_ROOT/wandb"
}

pg_mkdirs() {
  mkdir -p "$RUN_ROOT/cache" "$(pg_meta_dir)" "$(pg_wandb_dir)"
}

pg_bins() {
  local bins=()
  IFS=',' read -r -a bins <<< "$BINS_LIST"
  printf '%s\n' "${bins[@]}"
}

pg_bin_at() {
  local idx="$1"
  local bins=()
  IFS=',' read -r -a bins <<< "$BINS_LIST"
  if (( idx < 0 || idx >= ${#bins[@]} )); then
    echo "bad_array_index idx=$idx count=${#bins[@]}" >&2
    return 1
  fi
  printf '%s\n' "${bins[$idx]}"
}

pg_array_spec() {
  local bins=()
  IFS=',' read -r -a bins <<< "$BINS_LIST"
  if ((${#bins[@]} == 0)); then
    echo "BINS_LIST must contain at least one value." >&2
    return 1
  fi
  printf '0-%d\n' "$((${#bins[@]} - 1))"
}

pg_write_ref() {
  local path="$1"
  local value="$2"
  mkdir -p "$(dirname "$path")"
  printf '%s\n' "$value" > "$path"
}

pg_read_ref() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "missing_ref_file $path" >&2
    return 1
  }
  cat "$path"
}
