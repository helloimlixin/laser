#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="ffhq-a2048-k2-rqvae-strict-20260720-145706"
DATA_ROOT="${FFHQ_ROOT:-/home/xl598/Projects/data/ffhq}"
RUN_ROOT="$ROOT/outputs/$SOURCE_RUN"
CHECKPOINT="$RUN_ROOT/stage1_checkpoint/best_rfid_slot1_model.pt"
SOURCE_CACHE="$RUN_ROOT/token_cache/ffhq_full_compound_pairs.pt"
TOKEN_CACHE="$RUN_ROOT/token_cache/ffhq_full_compound_pairs_lloyd512.pt"
TOKEN_REPORT="${TOKEN_CACHE%.pt}.validation.json"
RFID_RESULT="$RUN_ROOT/token_cache/rfid_ffhq_full_quantized_lloyd512.json"
STAGE2_OUT="$RUN_ROOT/stage2-compound-v6-lloyd512-hard-rqtransformer-350M"
WANDB_ID="ffhq-compound-rqt350-a2048k2-lloyd512-hard-20260806"
WANDB_NAME="ffhq-a2048-k2-compound-v6-lloyd512-hard-rqtransformer-350M"
EXPECTED_FFHQ_IMAGES=70000

for required in "$CHECKPOINT" "$SOURCE_CACHE" "$DATA_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

image_count="$(find "$DATA_ROOT" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)"
if [[ "$image_count" -ne "$EXPECTED_FFHQ_IMAGES" ]]; then
  echo "Expected ${EXPECTED_FFHQ_IMAGES} FFHQ images; found $image_count" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/token_cache" "$STAGE2_OUT/diagnostics" "$ROOT/.cache/wandb" \
  "$ROOT/.local/share/wandb" "$ROOT/wandb"
echo "$$" > "$RUN_ROOT/lloyd512_pipeline.pid"

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
if [[ -f "$TOKEN_CACHE" && -f "$TOKEN_REPORT" ]]; then
  cache_valid="$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("passed") is True and p.get("items") == 70000 and p.get("num_bins") == 512))' "$TOKEN_REPORT")"
fi
if [[ "$cache_valid" -ne 1 ]]; then
  echo "=== Quantizer: shared hybrid Lloyd-Max, 512 bins ==="
  python "$ROOT/scripts/tools/requantize_compound_cache.py" \
    --input "$SOURCE_CACHE" \
    --output "$TOKEN_CACHE" \
    --num-bins 512 \
    --fit-samples 2000000 \
    --iterations 60 \
    --quantile-blend 0.5 \
    --seed 42
fi
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and p["items"] == 70000 and p["num_bins"] == 512, p; print("validated Lloyd-512 FFHQ cache:", sys.argv[1])' "$TOKEN_REPORT"

echo "=== Diagnostic: matched 8x8 quantized reconstruction grid ==="
python "$ROOT/scripts/log_compound_cache_reconstruction.py" \
  --checkpoint "$CHECKPOINT" \
  --token-cache "$TOKEN_CACHE" \
  --data "$DATA_ROOT" \
  --output "$STAGE2_OUT/diagnostics/reconstruction_8x8.png" \
  --original-output "$STAGE2_OUT/diagnostics/original_8x8.png" \
  --num-images 64 \
  --nrow 8 \
  --batch-size 8 \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id "$WANDB_ID" \
  --wandb-name "$WANDB_NAME"

rfid_valid=0
if [[ -f "$RFID_RESULT" ]]; then
  rfid_valid="$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("dataset") == "ffhq" and p.get("num_images") == 70000 and p.get("token_cache") == sys.argv[2] and isinstance(p.get("rfid"), (int, float))))' "$RFID_RESULT" "$TOKEN_CACHE")"
fi
if [[ "$rfid_valid" -ne 1 ]]; then
  echo "=== Diagnostic: full 70,000-image quantized reconstruction FID ==="
  torchrun --standalone --nproc_per_node=2 \
    "$ROOT/scripts/evaluate_upstream_laser_rfid.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_ROOT" \
    --token-cache "$TOKEN_CACHE" \
    --output "$RFID_RESULT" \
    --dataset ffhq \
    --num-images 70000 \
    --num-atoms 2048 \
    --coeff-vocab-size 512 \
    --batch-size 8 \
    --backend torchmetrics \
    --wandb-mode online \
    --wandb-entity helloimlixin-rutgers \
    --wandb-project laser \
    --wandb-id "$WANDB_ID" \
    --wandb-name "$WANDB_NAME"
fi
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["dataset"] == "ffhq" and p["num_images"] == 70000 and isinstance(p["rfid"], (int, float)), p; print("validated full quantized FFHQ rFID: {:.6f}".format(p["rfid"]))' "$RFID_RESULT"

echo "=== Stage 2: fresh official FFHQ 350M run, Lloyd-512 hard coefficient targets ==="
echo "$$" > "$STAGE2_OUT/launcher.pid"
exec torchrun --standalone --nproc_per_node=2 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_ROOT" \
  --dataset ffhq \
  --model-preset ffhq-350m \
  --token-cache "$TOKEN_CACHE" \
  --output "$STAGE2_OUT" \
  --distributed-backend ddp \
  --epochs 200 \
  --batch-size 8 \
  --total-batch-size 128 \
  --num-atoms 2048 \
  --coeff-vocab-size 512 \
  --coeff-max 3 \
  --coeff-target-mode hard \
  --compound-tokens \
  --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads \
  --compound-distribution-geometry \
  --geometry-top-k 4 \
  --atom-loss-weight 1.5 \
  --geometry-loss-weight 0.05 \
  --geometry-start-epoch 2 \
  --geometry-warmup-epochs 3 \
  --lr 0.0005 \
  --lr-schedule constant \
  --atom-temperature 1.0 \
  --atom-top-k 250 \
  --atom-top-p 1.0 \
  --coeff-temperature 1.0 \
  --coeff-top-k 250 \
  --coeff-top-p 1.0 \
  --fid-num-samples 50000 \
  --fid-batch-size 8 \
  --fid-every 50 \
  --save-ckpt-freq 10 \
  --save-step-freq 250 \
  --sample-grid-every 1000 \
  --sample-grid-size 64 \
  --sample-grid-batch-size 8 \
  --sample-grid-sweep \
  --sample-grid-on-start \
  --upload-checkpoints \
  --upload-token-cache \
  --resume \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id "$WANDB_ID" \
  --wandb-name "$WANDB_NAME"
