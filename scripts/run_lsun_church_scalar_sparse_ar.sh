#!/usr/bin/env bash
# Legacy fully autoregressive LASER stream for LSUN Church:
# atom_1, coeff_1, ..., atom_4, coeff_4 at every 8x8 spatial site.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/tmp/laser-b300-venv/bin/python}"
STAGE1="$ROOT/outputs/lsun-church-a16384-k4-fair-stage1-b256-20260815-064515/church256-rqvae-laser-8x8-a16384-k4-fair/15082026_064530/best_rfid_slot1_model.pt"
BASELINE_ROOT="$ROOT/outputs/lsun-church-a16384-k4-compound-v5-stage2-20260816-052724"
DATA="$BASELINE_ROOT/data/lsun"
TOKEN_CACHE="$BASELINE_ROOT/token_cache/lsun_church_train_a16384k4_compound_pairs.pt"
CONTINUOUS_RFID="$ROOT/outputs/lsun-church-a16384-k4-cache-rfid-preflight/continuous.json"
QUANTIZED_RFID="$ROOT/outputs/lsun-church-a16384-k4-cache-rfid-preflight/quantized.json"
FID_STATS="$ROOT/third_party/rq-vae-transformer/assets/fid_stats/lsun_256_church.npz"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-church-a16384-k4-scalar-ar-stage2-20260822}"
OUTPUT="$RUN_ROOT/stage2"
WANDB_ID="${WANDB_RUN_ID:-lsunchurcha16384k4scalarar-20260822}"
WANDB_MODE="${WANDB_MODE:-online}"
STATUS="$RUN_ROOT/status.tsv"

for required in "$PYTHON_BIN" "$STAGE1" "$TOKEN_CACHE" \
  "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  "$DATA/church/church_outdoor_train_lmdb/data.mdb" "$FID_STATS"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

mkdir -p "$RUN_ROOT/logs" "$OUTPUT/checkpoints"
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

common_args=(
  --checkpoint "$STAGE1" --data "$DATA" --dataset lsun_church
  --model-preset lsun-church-350m --token-cache "$TOKEN_CACHE"
  --num-atoms 16384 --sparsity-level 4 --coeff-vocab-size 2048
  --coeff-max 3 --coeff-scale 6.4
  --coeff-target-mode soft --coeff-target-temperature 0.5
)

active_phase=scalar_generation_smoke
if [[ ! -f "$RUN_ROOT/.generation_smoke_complete" ]]; then
  status "$active_phase" starting "generate and decode all 512 alternating scalar slots"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    "${common_args[@]}" \
    --output "$RUN_ROOT/generation-smoke" \
    --checkpoint-dir "$RUN_ROOT/generation-smoke/checkpoints" \
    --distributed-backend ddp --epochs 10 --batch-size 2 --total-batch-size 2 \
    --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
    --atom-temperature 1 --atom-top-k 0 --atom-top-p 0.92 \
    --coeff-temperature 1 --coeff-top-k 0 --coeff-top-p 0.92 \
    --generation-smoke-test --fid-batch-size 2 --fid-every 0 \
    --save-step-freq 0 --sample-grid-every 0 --wandb-mode disabled \
    2>&1 | tee "$RUN_ROOT/logs/scalar-generation-smoke.log"
  touch "$RUN_ROOT/.generation_smoke_complete"
  status "$active_phase" complete "all eight scalar depths generated and decoded"
fi

active_phase=scalar_training_smoke
if [[ ! -f "$RUN_ROOT/.training_smoke_complete" ]]; then
  status "$active_phase" starting "one optimizer step on cached atom/coefficient targets"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    "${common_args[@]}" \
    --output "$RUN_ROOT/training-smoke" \
    --checkpoint-dir "$RUN_ROOT/training-smoke/checkpoints" \
    --distributed-backend ddp --epochs 10 --batch-size 32 --total-batch-size 32 \
    --max-optimizer-steps 1 --smoke-test \
    --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
    --fid-every 0 --save-step-freq 0 --sample-grid-every 0 \
    --wandb-mode disabled \
    2>&1 | tee "$RUN_ROOT/logs/scalar-training-smoke.log"
  touch "$RUN_ROOT/.training_smoke_complete"
  status "$active_phase" complete "one scalar-stream optimizer step passed"
fi

active_phase=scalar_stage2
status "$active_phase" starting "10/300 epochs from scratch; 50k FID at epoch 10"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  "${common_args[@]}" \
  --cache-rfid-preflight "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  --output "$OUTPUT" --checkpoint-dir "$OUTPUT/checkpoints" \
  --distributed-backend ddp --epochs 10 --batch-size 32 --total-batch-size 256 \
  --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
  --atom-temperature 1 --atom-top-k 0 --atom-top-p 0.92 \
  --coeff-temperature 1 --coeff-top-k 0 --coeff-top-p 0.92 \
  --fid-num-samples 50000 --fid-batch-size 250 --fid-every 10 \
  --metric-backend original-rqvae --fid-reference-stats "$FID_STATS" \
  --save-ckpt-freq 10 --keep-best-checkpoints 1 \
  --model-only-best-checkpoints --save-step-freq 500 \
  --sample-grid-every 500 --sample-grid-size 64 \
  --sample-grid-batch-size 8 --sample-grid-samples-per-class 8 \
  --sample-grid-seed 0 --sample-grid-compare-original --resume \
  --wandb-mode "$WANDB_MODE" --wandb-entity helloimlixin-rutgers \
  --wandb-project laser --wandb-id "$WANDB_ID" \
  --wandb-name "lsun-church-scalar-sparse-ar-10of300-20260822" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
