#!/usr/bin/env bash
# ImageNet LASER 8x8x4 pipeline:
#   Stage 1: x3h5cl0h-compatible a16384 recipe with k=4.
#   Stage 2: v8dup 1.4B recipe with FFHQ-style compound events.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
STAGE1_CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-laser-8x8-a16384-k4.yaml"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATA_ROOT="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/imagenet-a16384-k4-compound-official-$STAMP}"
STAGE1_OUT="$RUN_ROOT/stage1"
STAGE2_OUT="$RUN_ROOT/stage2"
TOKEN_CACHE="$RUN_ROOT/token_cache/imagenet_train_a16384k4_compound_pairs.pt"
TOKEN_REPORT="${TOKEN_CACHE%.pt}.validation.json"
STAGE1_WANDB_ID="${STAGE1_WANDB_ID:-imga16384k4s1-${STAMP//-/}}"
STAGE2_WANDB_ID="${STAGE2_WANDB_ID:-imga16384k4cmp-${STAMP//-/}}"
NPROC="${NPROC:-4}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-32}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-64}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
TASK_TMPDIR="${TASK_TMPDIR:-/tmp/laser_imagenet_a16384k4_$STAMP}"

for required in "$PYTHON_BIN" "$DATA_ROOT/train" "$DATA_ROOT/val" \
  "$THIRD_PARTY/main_stage1.py" "$STAGE1_CONFIG" \
  "$ROOT/vgg_lpips/vgg.pth" "$ROOT/vgg_lpips/vgg16-397923af.pth"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done
if (( STAGE1_BATCH_SIZE * NPROC != 128 )); then
  echo "Stage-1 batch * world size must equal the reference total batch 128" >&2
  exit 1
fi
if (( 2048 % (STAGE2_BATCH_SIZE * NPROC) != 0 )); then
  echo "Stage-2 batch * world size must divide the reference total batch 2048" >&2
  exit 1
fi

train_classes="$(find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d | wc -l)"
val_classes="$(find "$DATA_ROOT/val" -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [[ "$train_classes" -ne 1000 || "$val_classes" -ne 1000 ]]; then
  echo "Expected 1,000 ImageNet classes; found train=$train_classes val=$val_classes" >&2
  exit 1
fi

mkdir -p "$STAGE1_OUT" "$STAGE2_OUT/checkpoints" "$RUN_ROOT/token_cache" \
  "$RUN_ROOT/logs" "$ROOT/.cache/wandb" "$ROOT/.local/share/wandb" \
  "$ROOT/wandb" "$TASK_TMPDIR"
printf '%s\n' "$$" > "$RUN_ROOT/pipeline.pid"
if [[ ! -s "$RUN_ROOT/status.tsv" ]]; then
  printf 'time_utc\tphase\tstate\tdetail\n' > "$RUN_ROOT/status.tsv"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export LASER_VGG_LPIPS_DIR="$ROOT/vgg_lpips"
export LASER_VGG16_WEIGHTS="$ROOT/vgg_lpips/vgg16-397923af.pth"
export TMPDIR="$TASK_TMPDIR"
export TMP="$TASK_TMPDIR"
export TEMP="$TASK_TMPDIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" \
    >> "$RUN_ROOT/status.tsv"
}

active_phase=driver
on_exit() {
  exit_code="$?"
  if (( exit_code == 0 )); then
    status "$active_phase" complete "pipeline exit=0"
  else
    status "$active_phase" failed "pipeline exit=$exit_code"
  fi
}
trap on_exit EXIT

cat > "$RUN_ROOT/run.info" <<EOF
run_root=$RUN_ROOT
data_root=$DATA_ROOT
latent_code_shape=8,8,4
stage1_reference=helloimlixin-rutgers/laser/x3h5cl0h-a16384-k2-20260719-014434
stage2_reference=helloimlixin-rutgers/laser/v8dup0731113220
compound_reference=helloimlixin-rutgers/laser/ffhqcmp0804205803
stage1_wandb_id=$STAGE1_WANDB_ID
stage2_wandb_id=$STAGE2_WANDB_ID
world_size=$NPROC
stage1_microbatch=$STAGE1_BATCH_SIZE
stage1_effective_batch=128
stage2_microbatch=$STAGE2_BATCH_SIZE
stage2_effective_batch=2048
checkpoint_policy=fixed W&B run files: last plus metric-best three
stage1_entrypoint=$THIRD_PARTY/main_stage1.py
stage1_encoder_decoder=$THIRD_PARTY/rqvae/models/rqvae/modules.py
restart=RUN_ROOT=$RUN_ROOT STAMP=$STAMP STAGE1_WANDB_ID=$STAGE1_WANDB_ID STAGE2_WANDB_ID=$STAGE2_WANDB_ID $0
EOF

active_phase=stage1
latest_stage1_file() {
  find "$STAGE1_OUT" -type f -name "$1" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

select_stage1_best() {
  "$PYTHON_BIN" - "$STAGE1_OUT" <<'PY'
import json
import sys
from pathlib import Path

policies = sorted(
    Path(sys.argv[1]).rglob('checkpoint_policy.json'),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if policies:
    policy = json.loads(policies[0].read_text())
    best = policy.get('best', [])
    if best:
        selected = min(best, key=lambda item: float(item['rfid']))
        print(policies[0].parent / selected['path'])
PY
}

STAGE1_BEST="$(select_stage1_best)"
if [[ ! -f "$STAGE1_OUT/.phase_complete" ]]; then
  stage1_resume=()
  stage1_last="$(latest_stage1_file last_model.pt)"
  if [[ -f "$stage1_last" ]]; then
    stage1_resume=(--load-path "$stage1_last" --resume)
  fi
  status "$active_phase" starting "third-party RQ-VAE encoder/decoder; a16384 k4; f32; effective batch 128; rFID top 3"
  export WANDB_ENTITY="helloimlixin-rutgers"
  export WANDB_PROJECT="laser"
  export WANDB_RUN_ID="$STAGE1_WANDB_ID"
  export WANDB_NAME="imagenet-a16384-k4-third-party-rqvae-stage1-$STAMP"
  export WANDB_RUN_GROUP="imagenet-a16384-k4-compound-$STAMP"
  export WANDB_TAGS="stage1,imagenet,laser,rqvae,third_party_encoder_decoder,a16384,k4,8x8x4,f32,effective-batch128,rfid-top3,wandb-files-overwrite"
  export WANDB_CHECKPOINT_UPLOAD=1
  (
    cd "$THIRD_PARTY"
    torchrun --standalone --nproc_per_node="$NPROC" main_stage1.py \
      --model-config "$STAGE1_CONFIG" \
      --result-path "$STAGE1_OUT" \
      --seed 0 \
      "${stage1_resume[@]}" \
      "dataset.root=$DATA_ROOT" \
      "experiment.batch_size=$STAGE1_BATCH_SIZE" \
      experiment.total_batch_size=128
  ) 2>&1 | tee -a "$RUN_ROOT/logs/stage1.log"
  STAGE1_BEST="$(select_stage1_best)"
  if [[ ! -f "$STAGE1_BEST" ]]; then
    echo "Stage 1 completed without an rFID-ranked checkpoint" >&2
    exit 1
  fi
  touch "$STAGE1_OUT/.phase_complete"
  status "$active_phase" complete "checkpoint=$STAGE1_BEST"
fi
unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_TAGS WANDB_CHECKPOINT_UPLOAD
STAGE1_BEST="$(select_stage1_best)"
if [[ ! -f "$STAGE1_BEST" ]]; then
  echo "Stage-1 completion marker exists but best rFID checkpoint is missing" >&2
  exit 1
fi
printf '%s\n' "$STAGE1_BEST" > "$RUN_ROOT/stage1_checkpoint.txt"

active_phase=token_cache
cache_valid=0
if [[ -f "$TOKEN_CACHE" && -f "$TOKEN_REPORT" ]]; then
  cache_valid="$("$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("passed") is True and p.get("items") == 1281167 and p.get("compound_sequence_length") == 256))' "$TOKEN_REPORT")"
fi
if [[ "$cache_valid" -ne 1 ]]; then
  status "$active_phase" starting "full ImageNet train cache; 8x8x4 compound events"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$STAGE1_BEST" --data "$DATA_ROOT" --output "$TOKEN_CACHE" \
    --dataset imagenet --batch-size "$CACHE_BATCH_SIZE" --num-workers "$NUM_WORKERS" \
    --num-atoms 16384 --sparsity-level 4 --coeff-vocab-size 2048 \
    --coeff-max 20 --coeff-scale 6.4 --verify-samples 256 --compound \
    2>&1 | tee -a "$RUN_ROOT/logs/token_cache.log"
fi
"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and p["items"] == 1281167 and p["compound_sequence_length"] == 256, p' "$TOKEN_REPORT"
status "$active_phase" complete "cache=$TOKEN_CACHE"

active_phase=stage2
status "$active_phase" starting "ImageNet 1.4B compound RQ-Transformer; best 3 by FID"
torchrun --standalone --nproc_per_node="$NPROC" \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1_BEST" --data "$DATA_ROOT" --dataset imagenet \
  --model-preset imagenet-1400m --token-cache "$TOKEN_CACHE" \
  --output "$STAGE2_OUT" --checkpoint-dir "$STAGE2_OUT/checkpoints" \
  --distributed-backend ddp --epochs 100 --batch-size "$STAGE2_BATCH_SIZE" \
  --total-batch-size 2048 --num-atoms 16384 --sparsity-level 4 \
  --coeff-vocab-size 2048 --coeff-max 20 --coeff-scale 6.4 \
  --compound-tokens --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads --compound-distribution-geometry \
  --geometry-top-k 4 --atom-loss-weight 1.5 --geometry-loss-weight 0.05 \
  --geometry-start-epoch 2 --geometry-warmup-epochs 3 \
  --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 100 --min-lr 0 \
  --atom-temperature 1.0 --atom-top-p 0.92 \
  --coeff-temperature 1.0 --coeff-top-p 0.92 \
  --fid-num-samples 50000 --fid-batch-size "$FID_BATCH_SIZE" --fid-every 5 \
  --save-ckpt-freq 5 --save-step-freq 500 \
  --sample-grid-every 500 --sample-grid-size 64 \
  --sample-grid-batch-size "$FID_BATCH_SIZE" \
  --upload-checkpoints --checkpoint-upload-mode files --resume \
  --wandb-mode online --wandb-entity helloimlixin-rutgers --wandb-project laser \
  --wandb-id "$STAGE2_WANDB_ID" \
  --wandb-name "imagenet-a16384-k4-compound-official-rqtransformer-1400M-$STAMP" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
status "$active_phase" complete "stage2 exit=0"
