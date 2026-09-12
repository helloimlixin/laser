#!/usr/bin/env bash
# Paired continuous/quantized reconstruction rFID on 50k cached Bedroom rows.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/tmp/laser-b300-venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-bedroom-a16384-k4-official-stage2-20260818-231500}"
CHECKPOINT="$ROOT/outputs/lsun-bedroom-a16384-k4-imagenet5.08-ft1e-20260818-084230/bedroom256-rqvae-laser-8x8-a16384-k4-finetune/18082026_084335/best_rfid_slot1_model.pt"
TOKEN_CACHE="$RUN_ROOT/token_cache/lsun_bedroom_train_a16384k4_compound_pairs.pt"
OUTPUT_DIR="$RUN_ROOT/token_cache"
LOG_DIR="$RUN_ROOT/logs/cache-rfid-preflight"
CONTINUOUS_RFID="$OUTPUT_DIR/rfid_lsun_bedroom_train50000_continuous.json"
QUANTIZED_RFID="$OUTPUT_DIR/rfid_lsun_bedroom_train50000_quantized.json"
STATUS_FILE="$RUN_ROOT/status.tsv"

mkdir -p "$LOG_DIR"
for required in "$PYTHON_BIN" "$CHECKPOINT" "$TOKEN_CACHE" \
  "$RUN_ROOT/data/lsun/bedroom/bedroom_train_lmdb/data.mdb"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "cache_rfid" "$1" "$2" >> "$STATUS_FILE"
}

run_mode() {
  local mode="$1"
  local output="$2"
  status starting "$mode reconstruction rFID on 50000 row-aligned train images"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/scripts/evaluate_upstream_laser_rfid.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$RUN_ROOT/data/lsun" \
    --token-cache "$TOKEN_CACHE" \
    --cache-coeff-mode "$mode" \
    --output "$output" \
    --dataset lsun_bedroom \
    --num-images 50000 \
    --num-atoms 16384 \
    --sparsity-level 4 \
    --coeff-vocab-size 2048 \
    --batch-size 256 \
    --backend native \
    --wandb-mode disabled \
    2>&1 | tee "$LOG_DIR/${mode}.log"
  status complete "$mode result=$output"
}

run_mode continuous "$CONTINUOUS_RFID"
run_mode quantized "$QUANTIZED_RFID"

"$PYTHON_BIN" - "$CONTINUOUS_RFID" "$QUANTIZED_RFID" <<'PY'
import json
import math
import sys

continuous = json.load(open(sys.argv[1]))
quantized = json.load(open(sys.argv[2]))
for mode, payload in (("continuous", continuous), ("quantized", quantized)):
    assert payload["dataset"] == "lsun_bedroom", payload
    assert payload["num_images"] == 50_000, payload
    assert payload["cache_coeff_mode"] == mode, payload
    assert math.isfinite(float(payload["rfid"])), payload
print(
    "Bedroom cache reconstruction rFID: "
    f"continuous={continuous['rfid']:.6f}, "
    f"quantized={quantized['rfid']:.6f}, "
    f"delta={quantized['rfid'] - continuous['rfid']:+.6f}"
)
PY

# Resume the 100-epoch Stage-2 schedule and publish both diagnostics to W&B.
exec env RUN_ROOT="$RUN_ROOT" STAMP=20260818-231500 \
  WANDB_RUN_ID=lsunbedrooma16384k4s2-20260818231500 \
  bash "$ROOT/scripts/run_lsun_bedroom_a16384_k4_official_stage2.sh"
