#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/p4g.sh"

: "${WIN:=32}"

p4g_mkdirs

if [[ -n "${S1_CKPT:-}" ]]; then
  p4g_write_ref "$(p4g_s1_ckpt_ref)" "$S1_CKPT"
fi
if [[ -n "${CACHE_PT:-}" ]]; then
  p4g_write_ref "$(p4g_cache_ref)" "$CACHE_PT"
fi

test -f "$(p4g_s1_ckpt_ref)"
test -f "$(p4g_cache_ref)"

array_spec="$(p4g_gst_array_spec)"

common_export="ALL,PARTITION=$PARTITION,RUN_ROOT=$RUN_ROOT,WIN=$WIN,GST_LIST=$GST_LIST,S2_EPOCHS=$S2_EPOCHS,S2_GPUS=$S2_GPUS,S2_CPUS=$S2_CPUS,S2_MEM_MB=$S2_MEM_MB,S2_BSZ=$S2_BSZ,S2_WORKERS=$S2_WORKERS,S2_LR=$S2_LR,S2_VAL_INTERVAL=$S2_VAL_INTERVAL,S2_SAMPLE_STEP_EVERY=$S2_SAMPLE_STEP_EVERY,S2_SAMPLE_EPOCH_EVERY=$S2_SAMPLE_EPOCH_EVERY,S2_SAMPLE_IMAGES=$S2_SAMPLE_IMAGES,ATOMS=$ATOMS,K=$K,PATCH=$PATCH,STRIDE=$STRIDE,BINS=$BINS,CMAX=$CMAX,D_MODEL=$D_MODEL,HEADS=$HEADS,LAYERS=$LAYERS,D_FF=$D_FF"

stage2_id="$(sbatch --parsable --partition="$PARTITION" --array="$array_spec" --gres="gpu:${S2_GPUS}" --cpus-per-task="$S2_CPUS" --mem="$S2_MEM_MB" --export="$common_export" "$ROOT_DIR/scripts/job_p4s2.sbatch")"

echo "stage2_job=$stage2_id"
echo "run_root=$RUN_ROOT"
echo "window=$WIN"
echo "global_tokens=${GST_LIST//,/ }"
