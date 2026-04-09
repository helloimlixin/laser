#!/bin/bash

: "${PARTITION:=gpu}"
: "${RUN_ROOT:=/cache/home/$USER/runs/chq256_p4gwin}"
: "${DATA_DIR:=/scratch/$USER/datasets/celebahq_packed_256}"
: "${WIN_LIST:=16,32,64,128}"
: "${GST_LIST:=0,4,8,16}"

: "${IMG:=256}"
: "${PATCH:=4}"
: "${STRIDE:=4}"
: "${ATOMS:=4096}"
: "${K:=16}"
: "${BINS:=512}"
: "${CMAX:=8.0}"

: "${S1_EPOCHS:=40}"
: "${S1_GPUS:=3}"
: "${S1_CPUS:=24}"
: "${S1_MEM_MB:=96000}"
: "${S1_BSZ:=4}"
: "${S1_WORKERS:=8}"
: "${S1_LR:=2e-4}"
: "${S1_DICT_LR:=2.5e-4}"
: "${S1_WARMUP_STEPS:=500}"
: "${S1_MIN_LR_RATIO:=0.01}"
: "${S1_LOG_EVERY:=25}"
: "${S1_VAL_INTERVAL:=1.0}"
: "${S1_IMG_EVERY:=200}"
: "${S1_DIAG_EVERY:=100}"
: "${S1_LATENT_VIS:=true}"
: "${S1_BOTTLENECK_W:=1.0}"
: "${S1_PERCEPTUAL_W:=0.0}"
: "${S1_SPARSITY_REG_W:=0.01}"
: "${S1_COHERENCE_W:=0.0}"
: "${S1_BOUNDED_OMP:=8}"
: "${EMB:=4}"
: "${NHID:=128}"
: "${RBLK:=2}"
: "${RHID:=32}"

: "${S2_EPOCHS:=40}"
: "${S2_GPUS:=3}"
: "${S2_CPUS:=24}"
: "${S2_MEM_MB:=96000}"
: "${S2_BSZ:=2}"
: "${S2_WORKERS:=4}"
: "${S2_LR:=1e-4}"
: "${S2_VAL_INTERVAL:=0.25}"
: "${S2_SAMPLE_STEP_EVERY:=500}"
: "${S2_SAMPLE_EPOCH_EVERY:=5}"
: "${S2_SAMPLE_IMAGES:=16}"
: "${D_MODEL:=256}"
: "${HEADS:=8}"
: "${LAYERS:=8}"
: "${D_FF:=768}"

p4g_meta_dir() {
  printf '%s\n' "$RUN_ROOT/meta"
}

p4g_s1_out() {
  printf '%s\n' "$RUN_ROOT/s1"
}

p4g_s2_out() {
  local win="$1"
  local gst="${2:-0}"
  if [[ -n "$gst" && "$gst" != "0" ]]; then
    printf '%s\n' "$RUN_ROOT/s2_w${win}_g${gst}"
    return
  fi
  printf '%s\n' "$RUN_ROOT/s2_w${win}"
}

p4g_cache_pt() {
  printf '%s\n' "$RUN_ROOT/cache/tok_q${BINS}.pt"
}

p4g_s1_ckpt_ref() {
  printf '%s\n' "$(p4g_meta_dir)/s1_ckpt.txt"
}

p4g_cache_ref() {
  printf '%s\n' "$(p4g_meta_dir)/cache_pt.txt"
}

p4g_wandb_dir() {
  printf '%s\n' "$RUN_ROOT/wandb"
}

p4g_mkdirs() {
  mkdir -p "$RUN_ROOT/cache" "$(p4g_meta_dir)" "$(p4g_wandb_dir)"
}

p4g_windows() {
  local wins=()
  IFS=',' read -r -a wins <<< "$WIN_LIST"
  printf '%s\n' "${wins[@]}"
}

p4g_window_at() {
  local idx="$1"
  local wins=()
  IFS=',' read -r -a wins <<< "$WIN_LIST"
  if (( idx < 0 || idx >= ${#wins[@]} )); then
    echo "bad_array_index idx=$idx count=${#wins[@]}" >&2
    return 1
  fi
  printf '%s\n' "${wins[$idx]}"
}

p4g_array_spec() {
  local wins=()
  IFS=',' read -r -a wins <<< "$WIN_LIST"
  if ((${#wins[@]} == 0)); then
    echo "WIN_LIST must contain at least one value." >&2
    return 1
  fi
  printf '0-%d\n' "$((${#wins[@]} - 1))"
}

p4g_gst_at() {
  local idx="$1"
  local vals=()
  IFS=',' read -r -a vals <<< "$GST_LIST"
  if (( idx < 0 || idx >= ${#vals[@]} )); then
    echo "bad_array_index idx=$idx count=${#vals[@]}" >&2
    return 1
  fi
  printf '%s\n' "${vals[$idx]}"
}

p4g_gst_array_spec() {
  local vals=()
  IFS=',' read -r -a vals <<< "$GST_LIST"
  if ((${#vals[@]} == 0)); then
    echo "GST_LIST must contain at least one value." >&2
    return 1
  fi
  printf '0-%d\n' "$((${#vals[@]} - 1))"
}

p4g_write_ref() {
  local path="$1"
  local value="$2"
  mkdir -p "$(dirname "$path")"
  printf '%s\n' "$value" > "$path"
}

p4g_read_ref() {
  local path="$1"
  [[ -f "$path" ]] || {
    echo "missing_ref_file $path" >&2
    return 1
  }
  cat "$path"
}
