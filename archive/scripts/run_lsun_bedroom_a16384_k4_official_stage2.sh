#!/usr/bin/env bash
# KakaoBrain LSUN-Bedroom 600M RQ-Transformer schedule for the LASER K=4 tokenizer.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
OFFICIAL_CONFIG_REL="configs/lsun-bedroom/stage2/bedroom256-rqtransformer-8x8x4-600M.yaml"
OFFICIAL_CONFIG="$THIRD_PARTY/$OFFICIAL_CONFIG_REL"
STAGE1_RUN="$ROOT/outputs/lsun-bedroom-a16384-k4-imagenet5.08-ft1e-20260818-084230"
STAGE1_RESULT="$STAGE1_RUN/bedroom256-rqvae-laser-8x8-a16384-k4-finetune/18082026_084335"
DEFAULT_CHECKPOINT="$STAGE1_RESULT/best_rfid_slot1_model.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
PYTHON_BIN="${PYTHON_BIN:-/tmp/laser-b300-venv/bin/python}"
if [[ -e /tmp/laser-lsun-bedroom/bedroom_train_lmdb/data.mdb ]]; then
  DEFAULT_DATA_ROOT="/tmp/laser-lsun-bedroom"
else
  DEFAULT_DATA_ROOT="/workspace/Projects/data/lsun"
fi
DATA_ROOT="${LSUN_ROOT:-$DEFAULT_DATA_ROOT}"
FID_REFERENCE_STATS="$THIRD_PARTY/assets/fid_stats/lsun_256_bedroom.npz"
FID_REFERENCE_SHA256="7782fdfdcca1582330bd35882182c8c09a887ecc16f34c43c11141e73450fb51"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-bedroom-a16384-k4-official-stage2-$STAMP}"
DATA_VIEW="$RUN_ROOT/data/lsun"
TOKEN_CACHE="$RUN_ROOT/token_cache/lsun_bedroom_train_a16384k4_compound_pairs.pt"
TOKEN_REPORT="${TOKEN_CACHE%.pt}.validation.json"
CONTINUOUS_CACHE_RFID="$RUN_ROOT/token_cache/rfid_lsun_bedroom_train50000_continuous.json"
QUANTIZED_CACHE_RFID="$RUN_ROOT/token_cache/rfid_lsun_bedroom_train50000_quantized.json"
STAGE2_OUT="$RUN_ROOT/stage2"
CHECKPOINT_DIR="$STAGE2_OUT/checkpoints"
SMOKE_OUT="$RUN_ROOT/stage2-smoke"
NPROC="${NPROC:-1}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-256}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
STAGE2_TOTAL_BATCH_SIZE="${STAGE2_TOTAL_BATCH_SIZE:-2048}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-250}"
WANDB_RUN_ID="${WANDB_RUN_ID:-lsunbedrooma16384k4s2-${STAMP//-/}}"
WANDB_NAME="${WANDB_NAME:-lsun-bedroom-a16384-k4-official-rqtransformer-600M-$STAMP}"
PREFLIGHT_ONLY=0
if [[ "${1:-}" == "--preflight" ]]; then
  PREFLIGHT_ONLY=1
elif (( $# > 0 )); then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

for required in "$PYTHON_BIN" "$CHECKPOINT" "$OFFICIAL_CONFIG" \
  "$DATA_ROOT/bedroom_train_lmdb/data.mdb" \
  "$DATA_ROOT/bedroom_val_lmdb/data.mdb" "$FID_REFERENCE_STATS"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done
if (( NPROC != 1 )); then
  echo "This launch is calibrated for the single available B300" >&2
  exit 1
fi
if (( STAGE2_TOTAL_BATCH_SIZE % (STAGE2_BATCH_SIZE * NPROC) != 0 )); then
  echo "Stage-2 total batch must divide batch_size * world_size" >&2
  exit 1
fi
if (( FID_BATCH_SIZE <= 0 )); then
  echo "FID batch size must be positive" >&2
  exit 1
fi
if [[ "$(sha256sum "$FID_REFERENCE_STATS" | cut -d' ' -f1)" != "$FID_REFERENCE_SHA256" ]]; then
  echo "LSUN Bedroom FID reference checksum mismatch" >&2
  exit 1
fi
if ! git -C "$THIRD_PARTY" diff --quiet upstream/main -- "$OFFICIAL_CONFIG_REL"; then
  echo "Official LSUN Bedroom stage-2 reference config differs from upstream" >&2
  exit 1
fi

mkdir -p "$DATA_VIEW/bedroom" "$RUN_ROOT/token_cache" "$RUN_ROOT/logs" \
  "$STAGE2_OUT" "$CHECKPOINT_DIR" "$ROOT/.cache/wandb" \
  "$ROOT/.local/share/wandb" "$ROOT/wandb"
ln -sfn "$DATA_ROOT/bedroom_train_lmdb" \
  "$DATA_VIEW/bedroom/bedroom_train_lmdb"
ln -sfn "$DATA_ROOT/bedroom_val_lmdb" \
  "$DATA_VIEW/bedroom/bedroom_val_lmdb"

LASER_STAGE2_ROOT="$ROOT" LASER_STAGE2_CONFIG="$OFFICIAL_CONFIG" \
LASER_STAGE2_CHECKPOINT="$CHECKPOINT" LASER_LSUN_DATA_VIEW="$DATA_VIEW" \
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from scripts.train_official_rqtransformer_laser_stage2 import (
    LaserAux,
    build_model,
    load_stage1_checkpoint,
    source_image_dataset,
    val_image_transform,
)
from src.rqvae_metrics import load_reference_statistics

config = OmegaConf.load(os.environ["LASER_STAGE2_CONFIG"])
assert config.dataset.type == "LSUN-bedroom"
assert int(config.dataset.vocab_size) == 16_384
assert list(config.arch.block_size) == [8, 8, 4]
assert int(config.arch.embed_dim) == 1_280
assert int(config.arch.input_embed_dim) == 256
assert bool(config.arch.shared_tok_emb)
assert bool(config.arch.shared_cls_emb)
assert bool(config.arch.input_emb_vqvae)
assert bool(config.arch.head_emb_vqvae)
assert bool(config.arch.cumsum_depth_ctx)
assert int(config.arch.body.n_layer) == 26
assert int(config.arch.body.block.n_head) == 20
assert int(config.arch.head.n_layer) == 4
assert int(config.arch.head.block.n_head) == 20
assert int(config.arch.vocab_size_cond) == 1
assert str(config.loss.type) == "soft_target_cross_entropy"
assert bool(config.loss.stochastic_codes)
assert float(config.loss.temp) == 0.5
assert str(config.optimizer.type).lower() == "adamw"
assert float(config.optimizer.init_lr) == 5e-4
assert float(config.optimizer.weight_decay) == 1e-4
assert list(config.optimizer.betas) == [0.9, 0.95]
assert int(config.experiment.batch_size) == 32
assert int(config.experiment.total_batch_size) == 2_048
assert int(config.experiment.epochs) == 100
assert int(config.experiment.save_ckpt_freq) == 10
assert int(config.experiment.test_freq) == 50
assert int(config.experiment.sample.top_k) == 250
assert float(config.experiment.sample.top_p) == 1.0

checkpoint_path = Path(os.environ["LASER_STAGE2_CHECKPOINT"])
checkpoint = load_stage1_checkpoint(checkpoint_path)
assert int(checkpoint["epoch"]) == 1
assert int(checkpoint["global_step"]) == 23_696

dataset = source_image_dataset(
    "lsun_bedroom",
    Path(os.environ["LASER_LSUN_DATA_VIEW"]),
    val_image_transform(),
)
assert len(dataset) == 3_033_042, len(dataset)
image, label = dataset[0]
assert tuple(image.shape) == (3, 256, 256)
assert label == 0

aux = LaserAux(
    checkpoint_path,
    num_atoms=16_384,
    coeff_vocab_size=2_048,
    coeff_max=3.0,
    coeff_scale=6.4,
    attn_resolutions=(8,),
    sparsity_level=4,
)
assert tuple(aux.dictionary.shape) == (256, 16_384)
model = build_model(
    16_384 + 2_048,
    16_384,
    compound=True,
    coeff_vocab_size=2_048,
    sparsity_level=4,
    model_preset="lsun-bedroom-600m",
)
assert tuple(model.block_size) == (8, 8, 4)
assert int(model.config.embed_dim) == 1_280
assert int(model.config.body.n_layer) == 26
assert int(model.config.body.block.n_head) == 20
assert int(model.config.head.n_layer) == 4
assert int(model.config.head.block.n_head) == 20
load_reference_statistics(
    Path(os.environ["LASER_STAGE2_ROOT"])
    / "third_party/rq-vae-transformer/assets/fid_stats/lsun_256_bedroom.npz"
)
policy = json.loads(
    (checkpoint_path.parent / "checkpoint_policy.json").read_text()
)
assert abs(float(policy["best"][0]["rfid"]) - 0.9679) < 5e-5, policy
print("LSUN Bedroom official 600M Stage-2 preflight passed")
PY
if (( PREFLIGHT_ONLY )); then
  exit 0
fi

if [[ ! -s "$RUN_ROOT/status.tsv" ]]; then
  printf 'time_utc\tphase\tstate\tdetail\n' > "$RUN_ROOT/status.tsv"
fi
printf '%s\n' "$$" > "$RUN_ROOT/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" \
    >> "$RUN_ROOT/status.tsv"
}

active_phase=driver
on_exit() {
  exit_code="$?"
  if (( exit_code != 0 )); then
    status "$active_phase" failed "pipeline exit=$exit_code"
  fi
}
trap on_exit EXIT

cat > "$RUN_ROOT/run.info" <<EOF
run_root=$RUN_ROOT
data_root=$DATA_ROOT
data_view=$DATA_VIEW
stage1_checkpoint=$CHECKPOINT
stage1_rfid=0.9679
stage1_wandb_run=helloimlixin-rutgers/laser/lsunbedrooma16384k4-imagenet508-ft1e-20260818084230
official_stage2_config=$OFFICIAL_CONFIG
official_stage2_config_url=https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-bedroom/stage2/bedroom256-rqtransformer-8x8x4-600M.yaml
token_cache=$TOKEN_CACHE
continuous_cache_rfid=$CONTINUOUS_CACHE_RFID
quantized_cache_rfid=$QUANTIZED_CACHE_RFID
stage2_output=$STAGE2_OUT
wandb_run_id=$WANDB_RUN_ID
world_size=$NPROC
microbatch_per_gpu=$STAGE2_BATCH_SIZE
gradient_accumulation=$((STAGE2_TOTAL_BATCH_SIZE / (STAGE2_BATCH_SIZE * NPROC)))
effective_batch_size=$STAGE2_TOTAL_BATCH_SIZE
epochs=100
precision=bf16_amp
model=official LSUN Bedroom RQ-Transformer 600M geometry with LASER compound adapter
body=26x1280x20heads
depth_head=4x1280x20heads
block_size=8x8x4
conditioning=unconditional
compound_factorization=p(atom_d|history) p(coeff_d|atom_d,history)
num_atoms=16384
sparsity_level=4
coeff_vocab_size=2048
coeff_max=3
coeff_scale_calibration=per-depth percentile100
optimizer=AdamW betas0.9,0.95 weight_decay0.0001 clip1.0
learning_rate=0.0005
lr_schedule=cosine epochs100 min_lr0
stochastic_codes=true
soft_target_temperature=0.5
fid_backend=original-rqvae
fid_reference_stats=$FID_REFERENCE_STATS
fid_reference_sha256=$FID_REFERENCE_SHA256
fid_num_samples=50000
fid_batch_size=$FID_BATCH_SIZE
fid_every_epochs=10
sampling=temperature1 top_k250 top_p1 for atom and coefficient heads; 8x8 (64-image) preview every 5000 optimizer steps
save_checkpoint_every_epochs=10
save_recovery_every_optimizer_steps=500
restart=RUN_ROOT=$RUN_ROOT STAMP=$STAMP WANDB_RUN_ID=$WANDB_RUN_ID $0
EOF

active_phase=stage2_smoke
if [[ ! -f "$RUN_ROOT/.stage2_smoke_complete" ]]; then
  status "$active_phase" starting "one real image batch and optimizer step at production microbatch"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    --checkpoint "$CHECKPOINT" --data "$DATA_VIEW" --dataset lsun_bedroom \
    --model-preset lsun-bedroom-600m \
    --output "$SMOKE_OUT" --checkpoint-dir "$SMOKE_OUT/checkpoints" \
    --distributed-backend ddp --epochs 100 --batch-size "$STAGE2_BATCH_SIZE" \
    --total-batch-size "$STAGE2_TOTAL_BATCH_SIZE" --num-atoms 16384 \
    --sparsity-level 4 --coeff-vocab-size 2048 --coeff-max 3 --coeff-scale 6.4 \
    --compound-tokens --coeff-target-mode soft --coeff-target-temperature 0.5 \
    --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 100 --min-lr 0 \
    --fid-every 0 --sample-grid-every 0 --save-step-freq 0 \
    --max-optimizer-steps 1 --smoke-test --no-resume --wandb-mode disabled \
    2>&1 | tee -a "$RUN_ROOT/logs/stage2-smoke.log"
  touch "$RUN_ROOT/.stage2_smoke_complete"
fi
status "$active_phase" complete "production-shape optimizer step passed"

active_phase=token_cache
cache_valid=0
if [[ -f "$TOKEN_CACHE" && -f "$TOKEN_REPORT" ]]; then
  cache_valid="$("$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("passed") is True and p.get("items") == 3033042 and p.get("compound_sequence_length") == 256))' "$TOKEN_REPORT")"
fi
if [[ "$cache_valid" -ne 1 ]]; then
  status "$active_phase" starting "full 3033042-image LSUN Bedroom 8x8x4 compound cache"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$CHECKPOINT" --data "$DATA_VIEW" --output "$TOKEN_CACHE" \
    --dataset lsun_bedroom --batch-size "$CACHE_BATCH_SIZE" \
    --num-workers "$CACHE_NUM_WORKERS" --num-atoms 16384 --sparsity-level 4 \
    --coeff-vocab-size 2048 --coeff-max 3 --coeff-scale 6.4 \
    --auto-coeff-scales-percentile 100 --verify-samples 256 --compound \
    2>&1 | tee -a "$RUN_ROOT/logs/token_cache.log"
fi
"$PYTHON_BIN" - "$TOKEN_REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["passed"] is True, report
assert report["items"] == 3_033_042, report
assert report["compound_sequence_length"] == 256, report
assert report["atom_exact_fraction"] == 1.0, report
assert report["coeff_finite"] is True, report
assert report["duplicate_atom_within_support_fraction"] == 0.0, report
print("Validated full LSUN Bedroom compound token cache")
PY
status "$active_phase" complete "cache=$TOKEN_CACHE"

active_phase=stage2
status "$active_phase" starting "official Bedroom 600M compound RQ-Transformer; 100 epochs"
CACHE_RFID_ARGS=()
if [[ -f "$CONTINUOUS_CACHE_RFID" && -f "$QUANTIZED_CACHE_RFID" ]]; then
  CACHE_RFID_ARGS=(
    --cache-rfid-preflight "$CONTINUOUS_CACHE_RFID" "$QUANTIZED_CACHE_RFID"
  )
fi
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$CHECKPOINT" --data "$DATA_VIEW" --dataset lsun_bedroom \
  --model-preset lsun-bedroom-600m --token-cache "$TOKEN_CACHE" \
  "${CACHE_RFID_ARGS[@]}" \
  --output "$STAGE2_OUT" --checkpoint-dir "$CHECKPOINT_DIR" \
  --distributed-backend ddp --epochs 100 --batch-size "$STAGE2_BATCH_SIZE" \
  --total-batch-size "$STAGE2_TOTAL_BATCH_SIZE" --num-atoms 16384 \
  --sparsity-level 4 --coeff-vocab-size 2048 --coeff-max 3 --coeff-scale 6.4 \
  --compound-tokens --coeff-target-mode soft --coeff-target-temperature 0.5 \
  --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 100 --min-lr 0 \
  --atom-temperature 1.0 --atom-top-k 250 --atom-top-p 1.0 \
  --coeff-temperature 1.0 --coeff-top-k 250 --coeff-top-p 1.0 \
  --fid-num-samples 50000 --fid-batch-size "$FID_BATCH_SIZE" --fid-every 10 \
  --metric-backend original-rqvae --fid-reference-stats "$FID_REFERENCE_STATS" \
  --save-ckpt-freq 10 --save-step-freq 500 \
  --sample-grid-every 5000 --sample-grid-size 64 \
  --sample-grid-batch-size 8 --sample-grid-samples-per-class 8 --resume \
  --wandb-mode online --wandb-entity "$WANDB_ENTITY" --wandb-project "$WANDB_PROJECT" \
  --wandb-id "$WANDB_RUN_ID" --wandb-name "$WANDB_NAME" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
status "$active_phase" complete "stage2 exit=0"
trap - EXIT
