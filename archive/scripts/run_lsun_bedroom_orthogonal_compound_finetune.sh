#!/usr/bin/env bash
# Causal ordered-orthogonal coordinate fork of the Bedroom compound prior.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/tmp/laser-b300-venv/bin/python}"
BASELINE_ROOT="$ROOT/outputs/lsun-bedroom-a16384-k4-official-stage2-20260818-231500"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-bedroom-a16384-k4-orthogonal-stage2-20260821}"
STAGE1="$ROOT/outputs/lsun-bedroom-a16384-k4-imagenet5.08-ft1e-20260818-084230/bedroom256-rqvae-laser-8x8-a16384-k4-finetune/18082026_084335/best_rfid_slot1_model.pt"
STAGE2_INIT="$BASELINE_ROOT/stage2/checkpoints/fid_manual_step_0041500.pt"
SOURCE_CACHE="$BASELINE_ROOT/token_cache/lsun_bedroom_train_a16384k4_compound_pairs.pt"
ORTHOGONAL_CACHE="$RUN_ROOT/token_cache/lsun_bedroom_train_a16384k4_orthogonal_compound.pt"
CONTINUOUS_RFID="$RUN_ROOT/token_cache/rfid_lsun_bedroom_train50000_orthogonal_continuous.json"
QUANTIZED_RFID="$RUN_ROOT/token_cache/rfid_lsun_bedroom_train50000_orthogonal_quantized.json"
DATA="$BASELINE_ROOT/data/lsun"
FID_STATS="$ROOT/third_party/rq-vae-transformer/assets/fid_stats/lsun_256_bedroom.npz"
OUTPUT="$RUN_ROOT/stage2"
WANDB_ID="${WANDB_RUN_ID:-lsunbedrooma16384k4orthogonal-20260821}"
WANDB_MODE="${WANDB_MODE:-online}"
TARGET_EPOCHS="${TARGET_EPOCHS:-10}"
TRAIN_LR="${TRAIN_LR:-0.0001}"
LR_SCHEDULE_EPOCHS="${LR_SCHEDULE_EPOCHS:-10}"
LR_SCHEDULE_RESTART_ID="${LR_SCHEDULE_RESTART_ID:-}"
WANDB_NAME="${WANDB_NAME:-lsun-bedroom-orthogonal-compound-ft10-20260821}"
STATUS="$RUN_ROOT/status.tsv"

for required in "$PYTHON_BIN" "$STAGE1" "$STAGE2_INIT" "$SOURCE_CACHE" \
  "$DATA/bedroom/bedroom_train_lmdb/data.mdb" "$FID_STATS"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/token_cache" "$OUTPUT/checkpoints"
printf '%s\n' "$$" > "$RUN_ROOT/pipeline.pid"
[[ -s "$STATUS" ]] || printf 'time_utc\tphase\tstate\tdetail\n' > "$STATUS"
status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$1" "$2" "${3:-}" >> "$STATUS"
}
active_phase=driver
on_exit() {
  code="$?"
  (( code == 0 )) || status "$active_phase" failed "pipeline exit=$code"
}
trap on_exit EXIT

export PYTHONPATH="$ROOT/third_party/rq-vae-transformer:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$ROOT/wandb"
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

active_phase=orthogonal_cache
if [[ ! -f "$ORTHOGONAL_CACHE" ]]; then
  status "$active_phase" starting "convert full cache to ordered-orthogonal coordinates"
  "$PYTHON_BIN" "$ROOT/scripts/tools/build_orthogonal_compound_cache.py" \
    --input "$SOURCE_CACHE" --output "$ORTHOGONAL_CACHE" \
    --checkpoint "$STAGE1" --device cuda:0 --chunk-rows 512 \
    --scale-percentile 100 --verify-sites 65536 \
    2>&1 | tee "$RUN_ROOT/logs/orthogonal-cache.log"
  status "$active_phase" complete "cache=$ORTHOGONAL_CACHE"
fi

run_rfid() {
  mode="$1"
  output="$2"
  if [[ -f "$output" ]]; then return; fi
  active_phase="orthogonal_rfid_$mode"
  status "$active_phase" starting "50000 paired reconstructions"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/scripts/evaluate_upstream_laser_rfid.py" \
    --checkpoint "$STAGE1" --data "$DATA" --dataset lsun_bedroom \
    --token-cache "$ORTHOGONAL_CACHE" --cache-coeff-mode "$mode" \
    --output "$output" --num-images 50000 --batch-size 256 \
    --num-atoms 16384 --sparsity-level 4 --coeff-vocab-size 2048 \
    --backend native --wandb-mode disabled \
    2>&1 | tee "$RUN_ROOT/logs/orthogonal-rfid-$mode.log"
  status "$active_phase" complete "result=$output"
}
run_rfid continuous "$CONTINUOUS_RFID"
run_rfid quantized "$QUANTIZED_RFID"

common_args=(
  --checkpoint "$STAGE1" --data "$DATA" --dataset lsun_bedroom
  --model-preset lsun-bedroom-600m --token-cache "$ORTHOGONAL_CACHE"
  --orthogonal-compound-tokens --coeff-target-mode hard
  --num-atoms 16384 --sparsity-level 4 --coeff-vocab-size 2048
  --coeff-max 3 --coeff-scale 6.4
  --atom-loss-weight 1
)

active_phase=orthogonal_smoke
if [[ ! -f "$RUN_ROOT/.smoke_complete" ]]; then
  status "$active_phase" starting "strict weight transfer plus one optimizer step"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    "${common_args[@]}" --init-stage2-checkpoint "$STAGE2_INIT" \
    --output "$RUN_ROOT/smoke" --checkpoint-dir "$RUN_ROOT/smoke/checkpoints" \
    --distributed-backend ddp --epochs 10 --batch-size 2 --total-batch-size 2 \
    --max-optimizer-steps 1 --smoke-test --lr 0.0001 \
    --lr-schedule cosine --lr-schedule-epochs 10 --min-lr 0 \
    --fid-every 0 --save-step-freq 0 --sample-grid-every 0 \
    --wandb-mode disabled \
    2>&1 | tee "$RUN_ROOT/logs/orthogonal-smoke.log"
  touch "$RUN_ROOT/.smoke_complete"
  status "$active_phase" complete "one optimizer step passed"
fi

active_phase=orthogonal_stage2
status "$active_phase" starting "target_epoch=$TARGET_EPOCHS lr=$TRAIN_LR schedule_epochs=$LR_SCHEDULE_EPOCHS restart_id=${LR_SCHEDULE_RESTART_ID:-none}"
init_args=()
if [[ ! -f "$OUTPUT/checkpoints/last.pt" ]]; then
  init_args=(--init-stage2-checkpoint "$STAGE2_INIT")
fi
schedule_restart_args=()
if [[ -n "$LR_SCHEDULE_RESTART_ID" ]]; then
  schedule_restart_args=(--lr-schedule-restart-id "$LR_SCHEDULE_RESTART_ID")
fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  "${common_args[@]}" \
  --cache-rfid-preflight "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  --output "$OUTPUT" --checkpoint-dir "$OUTPUT/checkpoints" \
  "${init_args[@]}" \
  --distributed-backend ddp --epochs "$TARGET_EPOCHS" --batch-size 32 --total-batch-size 2048 \
  --lr "$TRAIN_LR" --lr-schedule cosine --lr-schedule-epochs "$LR_SCHEDULE_EPOCHS" --min-lr 0 \
  "${schedule_restart_args[@]}" \
  --atom-temperature 1 --atom-top-k 250 --atom-top-p 1 \
  --coeff-temperature 1 --coeff-top-k 250 --coeff-top-p 1 \
  --fid-num-samples 50000 --fid-batch-size 250 --fid-every 10 \
  --metric-backend original-rqvae --fid-reference-stats "$FID_STATS" \
  --save-ckpt-freq 10 --save-step-freq 500 \
  --sample-grid-every 1000 --sample-grid-size 64 \
  --sample-grid-batch-size 8 --sample-grid-samples-per-class 8 \
  --resume --wandb-mode "$WANDB_MODE" --wandb-entity helloimlixin-rutgers \
  --wandb-project laser --wandb-id "$WANDB_ID" \
  --wandb-name "$WANDB_NAME" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
