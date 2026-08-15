#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="ffhq-a2048-k2-rqvae-strict-20260720-145706"
DATA_ROOT="${FFHQ_ROOT:-/workspace/Projects/data/ffhq}"
RUN_ROOT="$ROOT/outputs/$SOURCE_RUN"
CHECKPOINT="$RUN_ROOT/stage1_checkpoint/best_rfid_slot1_model.pt"
TOKEN_CACHE="$RUN_ROOT/token_cache/ffhq_full_compound_pairs_v2048_p995.pt"
CONTINUOUS_CACHE_RFID="$RUN_ROOT/token_cache/rfid_ffhq_full_continuous_v2048_p995.json"
QUANTIZED_CACHE_RFID="$RUN_ROOT/token_cache/rfid_ffhq_full_quantized_v2048_p995.json"
RFID_LOG_DIR="$RUN_ROOT/token_cache/reference_matched_logs"
STAGE2_OUT="$RUN_ROOT/stage2-levelwise-var-350M-gbs1024-v2048-p995-hard-crps"
DIAGNOSTIC_DIR="$STAGE2_OUT/diagnostics/cache_reconstruction"
WANDB_ID="ffhq-levelwise-var-a2048k2-v2048-p995-hard-crps-20260812"
WANDB_NAME="ffhq-a2048-k2-levelwise-var-350M-gbs1024-v2048-p995-hard-crps"
EXPECTED_FFHQ_IMAGES=70000

for required in "$CHECKPOINT" "$DATA_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

image_count="$(find "$DATA_ROOT" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)"
if [[ "$image_count" -ne "$EXPECTED_FFHQ_IMAGES" ]]; then
  echo "Expected ${EXPECTED_FFHQ_IMAGES} FFHQ images; found $image_count below $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/token_cache" "$RFID_LOG_DIR" "$DIAGNOSTIC_DIR" "$ROOT/.cache/wandb" \
  "$ROOT/.local/share/wandb" "$ROOT/wandb"
echo "$$" > "$RUN_ROOT/levelwise_var_pipeline.pid"

export WANDB_MODE=online
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

cache_valid=0
if [[ -f "$TOKEN_CACHE" ]]; then
  cache_valid="$(python -c 'import math,sys,torch; p=torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True); m=p.get("meta", {}); ok=(tuple(p["atoms"].shape)==(70000,8,8,2) and tuple(p["coeffs"].shape)==(70000,8,8,2) and tuple(p["labels"].shape)==(70000,) and m.get("dataset")=="ffhq" and m.get("num_atoms")==2048 and m.get("coeff_vocab_size")==2048 and m.get("shape")==[8,8,2] and math.isclose(float(m.get("auto_coeff_scales_percentile", -1)),99.5) and bool(torch.isfinite(p["coeffs"]).all())); print(int(ok))' "$TOKEN_CACHE")"
fi
if [[ "$cache_valid" -ne 1 ]]; then
  echo "=== Stage 1.5: full FFHQ sparse-pair cache from best-rFID Stage 1 ==="
  torchrun --standalone --nproc_per_node=2 \
    "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_ROOT" \
    --output "$TOKEN_CACHE" \
    --dataset ffhq \
    --batch-size 64 \
    --num-workers 8 \
    --num-atoms 2048 \
    --coeff-vocab-size 2048 \
    --coeff-max 3 \
    --auto-coeff-scales-percentile 99.5 \
    --verify-samples 64 \
    --compound
fi
python -c 'import sys,torch; p=torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True); m=p["meta"]; assert tuple(p["atoms"].shape)==(70000,8,8,2) and tuple(p["coeffs"].shape)==(70000,8,8,2) and m["stage1_checkpoint"].endswith("best_rfid_slot1_model.pt"), m; print("validated full FFHQ sparse-pair cache:", sys.argv[1])' "$TOKEN_CACHE"

run_cache_rfid_preflight() {
  local coefficient_mode="$1"
  local output_json="$2"
  local output_log="$RFID_LOG_DIR/${coefficient_mode}.log"
  local result_valid=0

  if [[ -f "$output_json" && "$output_json" -nt "$CHECKPOINT" && "$output_json" -nt "$TOKEN_CACHE" ]]; then
    result_valid="$(python -c 'import json,math,sys; from pathlib import Path; p=json.load(open(sys.argv[1])); ok=(p.get("dataset")=="ffhq" and p.get("num_images")==70000 and p.get("cache_coeff_mode")==sys.argv[2] and Path(p.get("token_cache", "")).resolve()==Path(sys.argv[3]).resolve() and math.isfinite(float(p.get("rfid", float("nan"))))); print(int(ok))' "$output_json" "$coefficient_mode" "$TOKEN_CACHE")"
  fi
  if [[ "$result_valid" -ne 1 ]]; then
    echo "=== Stage 1.6: full FFHQ ${coefficient_mode}-cache reconstruction rFID ==="
    torchrun --standalone --nproc_per_node=2 \
      "$ROOT/scripts/evaluate_upstream_laser_rfid.py" \
      --checkpoint "$CHECKPOINT" \
      --data "$DATA_ROOT" \
      --token-cache "$TOKEN_CACHE" \
      --cache-coeff-mode "$coefficient_mode" \
      --output "$output_json" \
      --dataset ffhq \
      --num-images "$EXPECTED_FFHQ_IMAGES" \
      --num-atoms 2048 \
      --coeff-vocab-size 2048 \
      --batch-size 64 \
      --backend torchmetrics \
      2>&1 | tee "$output_log"
  fi
}

# These two gates separate cache/order fidelity from the Stage-2 coefficient
# tokenizer. Training only begins after both full-dataset diagnostics complete.
run_cache_rfid_preflight continuous "$CONTINUOUS_CACHE_RFID"
run_cache_rfid_preflight quantized "$QUANTIZED_CACHE_RFID"
python -c 'import json,sys; c=json.load(open(sys.argv[1])); q=json.load(open(sys.argv[2])); print(f"validated cache reconstruction rFID: continuous={c['"'"'rfid'"'"']:.6f}, quantized={q['"'"'rfid'"'"']:.6f}, quantization_delta={q['"'"'rfid'"'"']-c['"'"'rfid'"'"']:+.6f}")' "$CONTINUOUS_CACHE_RFID" "$QUANTIZED_CACHE_RFID"

# Put the exact cache rows decoded by both paths and both 70k rFIDs on the
# Stage-2 run before optimization. The trainer resumes this W&B run below.
echo "=== Stage 1.7: aligned 8x8 cache reconstruction diagnostics ==="
python "$ROOT/scripts/log_laser_cache_reconstruction_diagnostics.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_ROOT" \
  --dataset ffhq \
  --output-dir "$DIAGNOSTIC_DIR" \
  --cache-spec reference_v2048_p995 "$TOKEN_CACHE" \
    "$CONTINUOUS_CACHE_RFID" "$QUANTIZED_CACHE_RFID" \
  --grid-size 64 \
  --decode-batch-size 8 \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id "$WANDB_ID" \
  --wandb-name "$WANDB_NAME"

echo "=== Stage 2: FFHQ 350M VAR-style next-sparsity-level prior ==="
echo "$$" > "$STAGE2_OUT/launcher.pid"
exec torchrun --standalone --nproc_per_node=2 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_ROOT" \
  --dataset ffhq \
  --model-preset ffhq-350m \
  --token-cache "$TOKEN_CACHE" \
  --cache-rfid-preflight "$CONTINUOUS_CACHE_RFID" "$QUANTIZED_CACHE_RFID" \
  --output "$STAGE2_OUT" \
  --distributed-backend ddp \
  --epochs 200 \
  --batch-size 128 \
  --total-batch-size 1024 \
  --num-atoms 2048 \
  --sparsity-level 2 \
  --coeff-vocab-size 2048 \
  --coeff-max 3 \
  --compound-tokens \
  --levelwise-var \
  --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads \
  --coeff-target-mode hard \
  --coeff-crps-weight 1.0 \
  --atom-loss-weight 1.5 \
  --lr 0.00032 \
  --lr-schedule warmup-linear \
  --lr-schedule-epochs 200 \
  --min-lr 0.0000032 \
  --warmup-epochs 4 \
  --warmup-start-ratio 0.005 \
  --atom-temperature 1.0 \
  --atom-top-k 250 \
  --atom-top-p 1.0 \
  --coeff-temperature 1.0 \
  --coeff-top-k 250 \
  --coeff-top-p 1.0 \
  --fid-num-samples 50000 \
  --fid-batch-size 64 \
  --fid-every 50 \
  --save-ckpt-freq 10 \
  --save-step-freq 250 \
  --sample-grid-every 1000 \
  --sample-grid-size 64 \
  --sample-grid-batch-size 64 \
  --sample-grid-sweep \
  --sample-grid-on-start \
  --upload-checkpoints \
  --resume \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id "$WANDB_ID" \
  --wandb-name "$WANDB_NAME"
