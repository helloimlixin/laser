#!/usr/bin/env bash
# Native third-party RQ-VAE -> rFID -> LASER cache -> RQ-Transformer pipeline.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_ROOT="${FFHQ_ROOT:-/home/xl598/Projects/data/ffhq}"
CONFIG="$THIRD_PARTY/configs/ffhq/stage1/ffhq256-rqvae-laser-8x8-a2048-k4.yaml"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/ffhq-a2048-k4-rqvae-strict-$STAMP}"
STAGE1_ROOT="$RUN_ROOT/stage1"
RFID_RESULT="$RUN_ROOT/stage1_checkpoint/rfid_ffhq_70000.json"
TOKEN_CACHE="$RUN_ROOT/token_cache/ffhq_full_a2048_k4_compound.pt"
TOKEN_REPORT="${TOKEN_CACHE%.pt}.validation.json"
STAGE2_OUT="$RUN_ROOT/stage2-compound-official-rqtransformer-350M"
EXPECTED_FFHQ_IMAGES=70000

PIPELINE_CUDA_VISIBLE_DEVICES="${PIPELINE_CUDA_VISIBLE_DEVICES:-1}"
NPROC="${NPROC:-1}"
STAGE1_MICROBATCH="${STAGE1_MICROBATCH:-8}"
STAGE2_MICROBATCH="${STAGE2_MICROBATCH:-2}"
STAGE2_FID_BATCH="${STAGE2_FID_BATCH:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TASK_TMPDIR="${TASK_TMPDIR:-/tmp/laser_ffhq_k4_$STAMP}"

for required in "$PYTHON_BIN" "$DATA_ROOT" "$CONFIG"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done
if (( 128 % (STAGE1_MICROBATCH * NPROC) != 0 )); then
  echo "Stage-1 microbatch * world size must divide the reference total batch 128" >&2
  exit 1
fi
if (( 128 % (STAGE2_MICROBATCH * NPROC) != 0 )); then
  echo "Stage-2 microbatch * world size must divide the official total batch 128" >&2
  exit 1
fi
STAGE1_ACCUMULATE=$((128 / (STAGE1_MICROBATCH * NPROC)))

mkdir -p "$STAGE1_ROOT" "$RUN_ROOT/stage1_checkpoint" "$RUN_ROOT/token_cache" \
  "$STAGE2_OUT" "$RUN_ROOT/logs" "$ROOT/.cache/wandb" \
  "$ROOT/.local/share/wandb" "$ROOT/wandb" "$TASK_TMPDIR"
printf '%s\n' "$$" > "$RUN_ROOT/pipeline.pid"
if [[ ! -s "$RUN_ROOT/status.tsv" ]]; then
  printf 'time_utc\tphase\tstate\tdetail\n' > "$RUN_ROOT/status.tsv"
fi

export CUDA_VISIBLE_DEVICES="$PIPELINE_CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME=allow
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export TMPDIR="$TASK_TMPDIR"
export TMP="$TASK_TMPDIR"
export TEMP="$TASK_TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" \
    >> "$RUN_ROOT/status.tsv"
}

latest_checkpoint() {
  find "$1" -type f -name "$2" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print; }'
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

"$PYTHON_BIN" - "$DATA_ROOT" "$EXPECTED_FFHQ_IMAGES" <<'PY'
from pathlib import Path
import sys
root, expected = Path(sys.argv[1]), int(sys.argv[2])
files = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
stems = [p.stem for p in files]
assert len(files) == expected, f'expected {expected} FFHQ images, found {len(files)}'
assert len(set(stems)) == expected, 'FFHQ contains duplicate canonical image stems'
assert set(stems) == {f'{index:05d}' for index in range(expected)}, 'FFHQ canonical stems are incomplete'
print(f'validated complete FFHQ corpus: {len(files):,} unique images')
PY

printf '%s\n' \
  "run_root=$RUN_ROOT" \
  "source_model_run=helloimlixin-rutgers/laser/ffhq-a2048-k2-rqvae-strict-20260720-145706" \
  "stage1_implementation=$THIRD_PARTY/main_stage1.py" \
  "stage1_config=$CONFIG" \
  "data_root=$DATA_ROOT" \
  "cuda_visible_devices=$PIPELINE_CUDA_VISIBLE_DEVICES" \
  "world_size=$NPROC" \
  "stage1_microbatch=$STAGE1_MICROBATCH" \
  "stage1_accumulation=$STAGE1_ACCUMULATE" \
  "stage1_effective_batch=128" \
  "stage2_microbatch=$STAGE2_MICROBATCH" \
  "stage2_effective_batch=128" \
  "restart=RUN_ROOT=$RUN_ROOT STAMP=$STAMP PIPELINE_CUDA_VISIBLE_DEVICES=$PIPELINE_CUDA_VISIBLE_DEVICES NPROC=$NPROC $0" \
  > "$RUN_ROOT/run.info"

active_phase=stage1
stage1_checkpoint="$(latest_checkpoint "$STAGE1_ROOT" last_model.pt)"
if [[ ! -f "$STAGE1_ROOT/.phase_complete" ]]; then
  stage1_resume=()
  if [[ -n "$stage1_checkpoint" ]]; then
    stage1_resume=(--load-path "$stage1_checkpoint" --resume)
  fi
  status "$active_phase" starting \
    "native rq-vae; a2048 k4; microbatch=$STAGE1_MICROBATCH accumulation=$STAGE1_ACCUMULATE"
  export WANDB_ENTITY=helloimlixin-rutgers
  export WANDB_PROJECT=laser
  export WANDB_RUN_ID="ffhqa2048k4rqvaestrict$STAMP"
  export WANDB_NAME="ffhq-a2048-k4-rqvae-strict-$STAMP"
  export WANDB_RUN_GROUP="ffhq-a2048-k4-rqvae-full-$STAMP"
  export WANDB_TAGS="stage1,ffhq,rqvae,third_party_encoder_decoder,dictionary,a2048,k4,8x8x4,effective_batch128"
  (
    cd "$THIRD_PARTY"
    if (( NPROC == 1 )); then
      "$PYTHON_BIN" main_stage1.py \
        --model-config "$CONFIG" \
        --result-path "$STAGE1_ROOT" \
        --seed 0 \
        "${stage1_resume[@]}" \
        "dataset.root=$DATA_ROOT" \
        "experiment.batch_size=$STAGE1_MICROBATCH" \
        experiment.total_batch_size=128
    else
      torchrun --standalone --nproc_per_node="$NPROC" main_stage1.py \
        --model-config "$CONFIG" \
        --result-path "$STAGE1_ROOT" \
        --seed 0 \
        "${stage1_resume[@]}" \
        "dataset.root=$DATA_ROOT" \
        "experiment.batch_size=$STAGE1_MICROBATCH" \
        experiment.total_batch_size=128
    fi
  ) 2>&1 | tee -a "$RUN_ROOT/logs/stage1.log"
  stage1_checkpoint="$(latest_checkpoint "$STAGE1_ROOT" last_model.pt)"
  if [[ -z "$stage1_checkpoint" ]]; then
    echo "Native stage 1 completed without last_model.pt" >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$stage1_checkpoint" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from scripts.train_official_rqtransformer_laser_stage2 import load_stage1_checkpoint
payload = load_stage1_checkpoint(Path(sys.argv[1]))
assert payload['epoch'] == 150, payload['epoch']
PY
  touch "$STAGE1_ROOT/.phase_complete"
  status "$active_phase" complete "checkpoint=$stage1_checkpoint"
fi
if [[ -z "$stage1_checkpoint" ]]; then
  echo "Stage-1 completion marker exists but last_model.pt is missing" >&2
  exit 1
fi

best_checkpoint="$("$PYTHON_BIN" - "$STAGE1_ROOT" <<'PY'
from pathlib import Path
import json, sys
policies = sorted(Path(sys.argv[1]).rglob('checkpoint_policy.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if policies:
    policy = json.loads(policies[0].read_text())
    if policy.get('best'):
        print(policies[0].parent / min(policy['best'], key=lambda item: item['rfid'])['path'])
PY
)"
if [[ -f "$best_checkpoint" ]]; then
  stage1_checkpoint="$best_checkpoint"
fi
printf '%s\n' "$stage1_checkpoint" > "$RUN_ROOT/stage1_checkpoint/selected_checkpoint.txt"

unset WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_TAGS

active_phase=rfid
rfid_valid=0
if [[ -f "$RFID_RESULT" ]]; then
  rfid_valid="$("$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("dataset") == "ffhq" and p.get("num_images") == 70000 and isinstance(p.get("rfid"), (int, float))))' "$RFID_RESULT")"
fi
if [[ "$rfid_valid" -ne 1 ]]; then
  status "$active_phase" starting "native full-70000 reconstruction FID"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/evaluate_upstream_laser_rfid.py" \
    --checkpoint "$stage1_checkpoint" \
    --data "$DATA_ROOT" \
    --output "$RFID_RESULT" \
    --dataset ffhq \
    --num-images 70000 \
    --num-atoms 2048 \
    --sparsity-level 4 \
    --coeff-vocab-size 1024 \
    --batch-size 4 \
    --backend native \
    --wandb-entity helloimlixin-rutgers \
    --wandb-project laser \
    --wandb-id "ffhqa2048k4rqvaerfid$STAMP" \
    --wandb-name "ffhq-a2048-k4-rqvae-rfid-$STAMP" \
    --wandb-mode "$WANDB_MODE" \
    2>&1 | tee -a "$RUN_ROOT/logs/rfid.log"
fi
"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["dataset"] == "ffhq" and p["num_images"] == 70000 and isinstance(p["rfid"], (int, float)), p' "$RFID_RESULT"
status "$active_phase" complete "result=$RFID_RESULT"

active_phase=token_cache
cache_valid=0
if [[ -f "$TOKEN_CACHE" && -f "$TOKEN_REPORT" ]]; then
  cache_valid="$("$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("passed") is True and p.get("items") == 70000 and p.get("compound_sequence_length") == 256))' "$TOKEN_REPORT")"
fi
if [[ "$cache_valid" -ne 1 ]]; then
  status "$active_phase" starting "full-70000 compound a2048/k4 cache"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$stage1_checkpoint" \
    --data "$DATA_ROOT" \
    --output "$TOKEN_CACHE" \
    --dataset ffhq \
    --batch-size 4 \
    --num-workers "$NUM_WORKERS" \
    --num-atoms 2048 \
    --sparsity-level 4 \
    --coeff-vocab-size 1024 \
    --coeff-max 3 \
    --auto-coeff-scales-percentile 100 \
    --verify-samples 64 \
    --compound \
    2>&1 | tee -a "$RUN_ROOT/logs/token_cache.log"
fi
"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and p["items"] == 70000 and p["compound_sequence_length"] == 256, p' "$TOKEN_REPORT"
status "$active_phase" complete "cache=$TOKEN_CACHE"

active_phase=stage2
status "$active_phase" starting "official FFHQ RQ-Transformer 350M; 8x8x4 compound events"
printf '%s\n' "$$" > "$STAGE2_OUT/launcher.pid"
exec torchrun --standalone --nproc_per_node="$NPROC" \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$stage1_checkpoint" \
  --data "$DATA_ROOT" \
  --dataset ffhq \
  --model-preset ffhq-350m \
  --token-cache "$TOKEN_CACHE" \
  --output "$STAGE2_OUT" \
  --distributed-backend ddp \
  --epochs 200 \
  --batch-size "$STAGE2_MICROBATCH" \
  --total-batch-size 128 \
  --num-atoms 2048 \
  --sparsity-level 4 \
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
  --fid-batch-size "$STAGE2_FID_BATCH" \
  --fid-every 5 \
  --save-ckpt-freq 10 \
  --save-step-freq 250 \
  --sample-grid-every 1000 \
  --sample-grid-size 64 \
  --sample-grid-batch-size "$STAGE2_FID_BATCH" \
  --sample-grid-sweep \
  --sample-grid-on-start \
  --upload-token-cache \
  --resume \
  --wandb-entity helloimlixin-rutgers \
  --wandb-project laser \
  --wandb-id "ffhqcompoundrqt350a2048k4$STAMP" \
  --wandb-name "ffhq-a2048-k4-compound-official-rqtransformer-350M-$STAMP" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
