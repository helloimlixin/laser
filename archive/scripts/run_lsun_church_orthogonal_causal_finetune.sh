#!/usr/bin/env bash
# Corrected 10-epoch LSUN Church orthogonal fine-tune. Retains the complete
# pair-hard causal-prefix pathway and evaluates 50k-sample FID at epoch 10.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/tmp/laser-b300-venv/bin/python}"
STAGE1="$ROOT/outputs/lsun-church-a16384-k4-fair-stage1-b256-20260815-064515/church256-rqvae-laser-8x8-a16384-k4-fair/15082026_064530/best_rfid_slot1_model.pt"
STAGE2_INIT="$ROOT/outputs/lsun-church-a16384-k4-compound-v7-pairhard-ft15-seed0-20260818/checkpoints/best_fid_17.6033_epoch_015.pt"
BASELINE_ROOT="$ROOT/outputs/lsun-church-a16384-k4-compound-v5-stage2-20260816-052724"
DATA="$BASELINE_ROOT/data/lsun"
FID_STATS="$ROOT/third_party/rq-vae-transformer/assets/fid_stats/lsun_256_church.npz"
ORTHOGONAL_ROOT="$ROOT/outputs/lsun-church-a16384-k4-orthogonal-stage2-20260822"
ORTHOGONAL_CACHE="$ORTHOGONAL_ROOT/token_cache/lsun_church_train_a16384k4_orthogonal_compound.pt"
CONTINUOUS_RFID="$ORTHOGONAL_ROOT/token_cache/rfid_lsun_church_train126227_orthogonal_continuous.json"
QUANTIZED_RFID="$ORTHOGONAL_ROOT/token_cache/rfid_lsun_church_train126227_orthogonal_quantized.json"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-church-a16384-k4-orthogonal-causal-stage2-20260822}"
OUTPUT="$RUN_ROOT/stage2"
WANDB_ID="${WANDB_RUN_ID:-lsunchurcha16384k4orthogonalcausal-20260822}"
WANDB_MODE="${WANDB_MODE:-online}"
STATUS="$RUN_ROOT/status.tsv"

for required in "$PYTHON_BIN" "$STAGE1" "$STAGE2_INIT" \
  "$ORTHOGONAL_CACHE" "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
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
  --model-preset lsun-church-350m --token-cache "$ORTHOGONAL_CACHE"
  --orthogonal-compound-tokens --causal-prefix-state
  --causal-prefix-loss-weight 0.5 --coeff-target-mode hard
  --num-atoms 16384 --sparsity-level 4 --coeff-vocab-size 2048
  --coeff-max 3 --coeff-scale 6.4
  --compound-micro-transformer-layers 2
  --compound-depth-specific-coeff-heads --compound-distribution-geometry
  --geometry-top-k 4 --atom-loss-weight 1.5
  --geometry-loss-weight 0.05 --geometry-start-epoch 0
  --geometry-warmup-epochs 0
)

active_phase=orthogonal_causal_smoke
if [[ ! -f "$RUN_ROOT/.smoke_complete" ]]; then
  status "$active_phase" starting "full pair-hard transfer plus one optimizer step"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    "${common_args[@]}" --init-stage2-checkpoint "$STAGE2_INIT" \
    --output "$RUN_ROOT/smoke" --checkpoint-dir "$RUN_ROOT/smoke/checkpoints" \
    --distributed-backend ddp --epochs 10 --batch-size 32 --total-batch-size 32 \
    --max-optimizer-steps 1 --smoke-test --lr 0.00005 \
    --lr-schedule cosine --lr-schedule-epochs 10 --min-lr 0.00001 \
    --fid-every 0 --save-step-freq 0 --sample-grid-every 0 \
    --wandb-mode disabled \
    2>&1 | tee "$RUN_ROOT/logs/orthogonal-causal-smoke.log"
  touch "$RUN_ROOT/.smoke_complete"
  status "$active_phase" complete "one optimizer step passed"
fi

active_phase=orthogonal_causal_stage2
status "$active_phase" starting "10 epochs; exact prefix; 50k FID at epoch 10"
init_args=()
if [[ ! -f "$OUTPUT/checkpoints/last.pt" ]]; then
  init_args=(--init-stage2-checkpoint "$STAGE2_INIT")
fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  "${common_args[@]}" \
  --cache-rfid-preflight "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  --output "$OUTPUT" --checkpoint-dir "$OUTPUT/checkpoints" \
  "${init_args[@]}" \
  --distributed-backend ddp --epochs 10 --batch-size 32 --total-batch-size 256 \
  --lr 0.00005 --lr-schedule cosine --lr-schedule-epochs 10 --min-lr 0.00001 \
  --atom-temperature 1 --atom-top-k 0 --atom-top-p 0.92 \
  --coeff-temperature 1 --coeff-top-k 0 --coeff-top-p 0.92 \
  --fid-num-samples 50000 --fid-batch-size 250 --fid-every 10 \
  --metric-backend original-rqvae --fid-reference-stats "$FID_STATS" \
  --save-ckpt-freq 10 --keep-best-checkpoints 1 \
  --model-only-best-checkpoints --save-step-freq 500 \
  --sample-grid-every 500 --sample-grid-size 64 \
  --sample-grid-batch-size 8 --sample-grid-samples-per-class 8 \
  --sample-grid-seed 0 --resume \
  --wandb-mode "$WANDB_MODE" --wandb-entity helloimlixin-rutgers \
  --wandb-project laser --wandb-id "$WANDB_ID" \
  --wandb-name "lsun-church-orthogonal-causal-prefix-ft10-20260822" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
