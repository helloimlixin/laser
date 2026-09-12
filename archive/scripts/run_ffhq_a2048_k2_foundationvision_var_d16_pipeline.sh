#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="ffhq-a2048-k2-rqvae-strict-20260720-145706"
RUN_ROOT="$ROOT/outputs/$SOURCE_RUN"
DATA_ROOT="${FFHQ_ROOT:-/workspace/Projects/data/ffhq}"
CHECKPOINT="$RUN_ROOT/stage1_checkpoint/best_rfid_slot1_model.pt"
TOKEN_CACHE="$RUN_ROOT/token_cache/ffhq_full_compound_pairs_v2048_p995.pt"
CONTINUOUS_RFID="$RUN_ROOT/token_cache/rfid_ffhq_full_continuous_v2048_p995.json"
QUANTIZED_RFID="$RUN_ROOT/token_cache/rfid_ffhq_full_quantized_v2048_p995.json"
FV_VAR_ROOT="$ROOT/third_party/FoundationVision_VAR"
FV_VAR_COMMIT="78b95394fc5896192e3a003e4b295f8ea743c48f"
OUTPUT="$RUN_ROOT/stage2-foundationvision-var-d16-v2048-p995"
DIAGNOSTICS="$OUTPUT/diagnostics/cache_reconstruction"
WANDB_ID="ffhq-fv-var-d16-a2048k2-v2048-p995-20260812"
WANDB_NAME="ffhq-a2048-k2-foundationvision-var-d16-v2048-p995"

for required in \
  "$CHECKPOINT" "$TOKEN_CACHE" "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  "$DATA_ROOT" "$FV_VAR_ROOT/models/var.py"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

actual_commit="$(git -C "$FV_VAR_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$FV_VAR_COMMIT" ]]; then
  echo "FoundationVision VAR revision mismatch: $actual_commit != $FV_VAR_COMMIT" >&2
  exit 1
fi

python -c 'import json,math,sys,torch; from pathlib import Path; c=torch.load(sys.argv[1],map_location="cpu",weights_only=True,mmap=True); m=c["meta"]; assert tuple(c["atoms"].shape)==(70000,8,8,2) and tuple(c["coeffs"].shape)==(70000,8,8,2); assert m["num_atoms"]==2048 and m["coeff_vocab_size"]==2048 and m["shape"]==[8,8,2] and math.isclose(float(m["auto_coeff_scales_percentile"]),99.5); [(_ for _ in ()).throw(AssertionError(p)) if (j:=json.load(open(p)))["num_images"] != 70000 or Path(j["token_cache"]).resolve()!=Path(sys.argv[1]).resolve() else None for p in sys.argv[2:]]; print("validated reference-matched full FFHQ cache and rFID gates")' \
  "$TOKEN_CACHE" "$CONTINUOUS_RFID" "$QUANTIZED_RFID"

mkdir -p "$OUTPUT" "$DIAGNOSTICS" "$ROOT/.cache/wandb" \
  "$ROOT/.local/share/wandb" "$ROOT/wandb"
echo "$$" > "$OUTPUT/pipeline.pid"

export WANDB_MODE=online
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=8
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

echo "=== Preflight: aligned 8x8 LASER cache reconstructions and 70k rFIDs ==="
python "$ROOT/scripts/log_laser_cache_reconstruction_diagnostics.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_ROOT" \
  --dataset ffhq \
  --output-dir "$DIAGNOSTICS" \
  --cache-spec reference_v2048_p995 "$TOKEN_CACHE" \
    "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  --grid-size 64 \
  --decode-batch-size 8 \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id "$WANDB_ID" \
  --wandb-name "$WANDB_NAME"

echo "=== Stage 2: official FoundationVision VAR-d16 over two LASER levels ==="
exec torchrun --standalone --nproc_per_node=2 \
  "$ROOT/scripts/train_foundationvision_var_laser_stage2.py" \
  --checkpoint "$CHECKPOINT" \
  --token-cache "$TOKEN_CACHE" \
  --data "$DATA_ROOT" \
  --cache-rfid-preflight "$CONTINUOUS_RFID" "$QUANTIZED_RFID" \
  --output "$OUTPUT" \
  --depth 16 \
  --epochs 200 \
  --batch-size 192 \
  --accumulation 2 \
  --base-lr 0.0001 \
  --weight-decay 0.05 \
  --warmup-epochs 4 \
  --end-lr-ratio 0.1 \
  --grad-clip 2 \
  --save-step-freq 250 \
  --save-epoch-freq 10 \
  --sample-grid-every 500 \
  --no-sample-grid-on-start \
  --fid-every 50 \
  --fid-num-samples 50000 \
  --fid-batch-size 64 \
  --resume \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id "$WANDB_ID" \
  --wandb-name "$WANDB_NAME" \
  --wandb-mode online
