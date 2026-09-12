#!/usr/bin/env bash
# Fresh 300-epoch LSUN Church stage-2 run with the exact compound chain:
#   a1 -> c1 -> a2 -> c2 -> a3 -> c3 -> a4 -> c4
# Coefficients are local categorical decisions, while each completed pair is
# one cached RQ-Transformer depth event consumed by every later event.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/tmp/laser-b300-venv/bin/python}"
STAGE1="$ROOT/outputs/lsun-church-a16384-k4-fair-stage1-b256-20260815-064515/church256-rqvae-laser-8x8-a16384-k4-fair/15082026_064530/best_rfid_slot1_model.pt"
BASELINE_ROOT="$ROOT/outputs/lsun-church-a16384-k4-compound-v5-stage2-20260816-052724"
DATA="$BASELINE_ROOT/data/lsun"
TOKEN_CACHE="$BASELINE_ROOT/token_cache/lsun_church_train_a16384k4_compound_pairs.pt"
CONTINUOUS_RFID="$ROOT/outputs/lsun-church-a16384-k4-cache-rfid-preflight/continuous.json"
QUANTIZED_RFID="$ROOT/outputs/lsun-church-a16384-k4-cache-rfid-preflight/quantized.json"
FID_STATS="$ROOT/third_party/rq-vae-transformer/assets/fid_stats/lsun_256_church.npz"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-church-a16384-k4-compound-full-pair-ar-e300-20260822}"
OUTPUT="$RUN_ROOT/stage2"
WANDB_ID="${WANDB_RUN_ID:-lsunchurcha16384k4compoundfullpairar-e300-20260822}"
WANDB_MODE="${WANDB_MODE:-online}"
STATUS="$RUN_ROOT/status.tsv"

for required in "$PYTHON_BIN" "$STAGE1" "$TOKEN_CACHE" \
  "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  "$DATA/church/church_outdoor_train_lmdb/data.mdb" "$FID_STATS"; do
  [[ -e "$required" ]] || {
    echo "missing required input: $required" >&2
    exit 1
  }
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
  --cache-rfid-preflight "$CONTINUOUS_RFID" "$QUANTIZED_RFID"
  --num-atoms 16384 --sparsity-level 4 --coeff-vocab-size 2048
  --coeff-max 3 --coeff-scale 6.4
  --compound-tokens --compound-pair-autoregressive
  --compound-micro-transformer-layers 2
  --compound-depth-specific-coeff-heads
  --compound-distribution-geometry --geometry-top-k 4
  --atom-loss-weight 1.5 --geometry-loss-weight 0.05
  --geometry-start-epoch 2 --geometry-warmup-epochs 3
  --coeff-target-mode hard --coeff-target-temperature 0.5
)

active_phase=causality_tests
if [[ ! -f "$RUN_ROOT/.causality_tests_complete" ]]; then
  status "$active_phase" starting "pair causality and cached/parallel parity"
  "$PYTHON_BIN" - <<'PY' 2>&1 | tee "$RUN_ROOT/logs/causality-tests.log"
import importlib.util
from pathlib import Path

path = Path("tests/test_compound_pair_autoregressive.py")
spec = importlib.util.spec_from_file_location("compound_pair_ar_tests", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tests = sorted(
    (name, value) for name, value in vars(module).items()
    if name.startswith("test_") and callable(value)
)
for name, test in tests:
    test()
    print(f"PASS {name}", flush=True)
print(f"{len(tests)} causality tests passed", flush=True)
PY
  touch "$RUN_ROOT/.causality_tests_complete"
  status "$active_phase" complete "five exact-chain tests passed"
fi

active_phase=generation_smoke
if [[ ! -f "$RUN_ROOT/.generation_smoke_complete" ]]; then
  status "$active_phase" starting "sample/decode all 256 compound events"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    "${common_args[@]}" \
    --output "$RUN_ROOT/generation-smoke" \
    --checkpoint-dir "$RUN_ROOT/generation-smoke/checkpoints" \
    --distributed-backend ddp --epochs 300 --batch-size 2 --total-batch-size 2 \
    --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
    --atom-temperature 1 --atom-top-k 0 --atom-top-p 0.92 \
    --coeff-temperature 1 --coeff-top-k 0 --coeff-top-p 0.92 \
    --generation-smoke-test --fid-batch-size 2 --fid-every 0 \
    --save-step-freq 0 --sample-grid-every 0 --wandb-mode disabled \
    2>&1 | tee "$RUN_ROOT/logs/generation-smoke.log"
  touch "$RUN_ROOT/.generation_smoke_complete"
  status "$active_phase" complete "sampled coefficients consumed by later events"
fi

active_phase=training_smoke
if [[ ! -f "$RUN_ROOT/.training_smoke_complete" ]]; then
  status "$active_phase" starting "one optimizer step at production microbatch"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    "${common_args[@]}" \
    --output "$RUN_ROOT/training-smoke" \
    --checkpoint-dir "$RUN_ROOT/training-smoke/checkpoints" \
    --distributed-backend ddp --epochs 300 --batch-size 64 --total-batch-size 64 \
    --max-optimizer-steps 1 --smoke-test \
    --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
    --fid-every 0 --save-step-freq 0 --sample-grid-every 0 \
    --wandb-mode disabled \
    2>&1 | tee "$RUN_ROOT/logs/training-smoke.log"
  touch "$RUN_ROOT/.training_smoke_complete"
  status "$active_phase" complete "one full-pair optimizer step passed"
fi

active_phase=stage2
status "$active_phase" starting "fresh exact-chain compound run; 300 epochs"
set +e
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  "${common_args[@]}" \
  --output "$OUTPUT" --checkpoint-dir "$OUTPUT/checkpoints" \
  --distributed-backend ddp --epochs 300 --batch-size 64 --total-batch-size 256 \
  --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
  --atom-temperature 1 --atom-top-k 0 --atom-top-p 0.92 \
  --coeff-temperature 1 --coeff-top-k 0 --coeff-top-p 0.92 \
  --fid-num-samples 50000 --fid-batch-size 250 --fid-every 10 \
  --metric-backend original-rqvae --fid-reference-stats "$FID_STATS" \
  --save-ckpt-freq 10 --keep-best-checkpoints 3 \
  --model-only-best-checkpoints --save-step-freq 1000 \
  --sample-grid-every 1000 --sample-grid-size 64 \
  --sample-grid-batch-size 8 --sample-grid-samples-per-class 8 \
  --sample-grid-seed 0 --resume \
  --wandb-mode "$WANDB_MODE" --wandb-entity helloimlixin-rutgers \
  --wandb-project laser --wandb-id "$WANDB_ID" \
  --wandb-name "lsun-church-compound-full-pair-ar-e300-20260822" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
code="${PIPESTATUS[0]}"
set -e
if (( code == 0 )); then
  status "$active_phase" complete "300 epochs and final FID complete"
else
  status "$active_phase" failed "stage2 exit=$code"
  exit "$code"
fi
trap - EXIT
