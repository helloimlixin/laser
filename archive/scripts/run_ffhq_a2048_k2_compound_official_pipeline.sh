#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="ffhq-a2048-k2-rqvae-strict-20260720-145706"
DATA_ROOT="${FFHQ_ROOT:-/home/xl598/Projects/data/ffhq}"
RUN_ROOT="$ROOT/outputs/$SOURCE_RUN"
CHECKPOINT="$RUN_ROOT/stage1_checkpoint/best_rfid_slot1_model.pt"
RFID_RESULT="$RUN_ROOT/stage1_checkpoint/rfid_ffhq_50000.json"
TOKEN_CACHE="$RUN_ROOT/token_cache/ffhq_full_compound_pairs.pt"
TOKEN_REPORT="${TOKEN_CACHE%.pt}.validation.json"
STAGE2_OUT="$RUN_ROOT/stage2-compound-v5b-official-rqtransformer-350M"
EXPECTED_FFHQ_IMAGES=70000

for required in "$CHECKPOINT" "$DATA_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

image_count="$(find "$DATA_ROOT" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)"
if [[ "$image_count" -ne "$EXPECTED_FFHQ_IMAGES" ]]; then
  echo "Expected a complete ${EXPECTED_FFHQ_IMAGES}-image FFHQ corpus; found $image_count below $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/token_cache" "$STAGE2_OUT" "$ROOT/.cache/wandb" \
  "$ROOT/.local/share/wandb" "$ROOT/wandb"
echo "$$" > "$RUN_ROOT/pipeline.pid"

export WANDB_MODE=online
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

rfid_valid=0
if [[ -f "$RFID_RESULT" ]]; then
  rfid_valid="$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("dataset") == "ffhq" and p.get("num_images") == 50000 and isinstance(p.get("rfid"), (int, float))))' "$RFID_RESULT")"
fi
if [[ "$rfid_valid" -ne 1 ]]; then
  echo "=== Stage 1: FFHQ 50,000-image reconstruction FID ==="
  torchrun --standalone --nproc_per_node=2 \
    "$ROOT/scripts/evaluate_upstream_laser_rfid.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_ROOT" \
    --output "$RFID_RESULT" \
    --dataset ffhq \
    --num-images 50000 \
    --num-atoms 2048 \
    --coeff-vocab-size 1024 \
    --batch-size 8 \
    --backend torchmetrics
fi
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["dataset"] == "ffhq" and p["num_images"] == 50000 and isinstance(p["rfid"], (int, float)), p; print("validated 50k FFHQ rFID: {:.6f}".format(p["rfid"]))' "$RFID_RESULT"

cache_valid=0
if [[ -f "$TOKEN_CACHE" && -f "$TOKEN_REPORT" ]]; then
  cache_valid="$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("passed") is True and p.get("items") == 70000))' "$TOKEN_REPORT")"
fi
if [[ "$cache_valid" -ne 1 ]]; then
  echo "=== Stage 1.5: full 70,000-image FFHQ compound-token cache ==="
  torchrun --standalone --nproc_per_node=2 \
    "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_ROOT" \
    --output "$TOKEN_CACHE" \
    --dataset ffhq \
    --batch-size 64 \
    --num-workers 8 \
    --num-atoms 2048 \
    --coeff-vocab-size 1024 \
    --coeff-max 3 \
    --auto-coeff-scales-percentile 100 \
    --verify-samples 64 \
    --compound
fi
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and p["items"] == 70000 and p["compound_sequence_length"] == 128, p; print("validated full FFHQ compound cache:", sys.argv[1])' "$TOKEN_REPORT"

echo "=== Stage 2: official FFHQ 350M RQ-Transformer settings with compound events ==="
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
  --coeff-vocab-size 1024 \
  --coeff-max 3 \
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
  --wandb-id ffhq-compound-rqt350-a2048k2-20260805 \
  --wandb-name ffhq-a2048-k2-compound-v5b-official-rqtransformer-350M
