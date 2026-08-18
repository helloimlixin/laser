#!/usr/bin/env bash
# Official LSUN-Church 350M RQ-Transformer schedule with LASER compound events.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
OFFICIAL_CONFIG_REL="configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml"
OFFICIAL_CONFIG="$THIRD_PARTY/$OFFICIAL_CONFIG_REL"
STAGE1_RUN="$ROOT/outputs/lsun-church-a16384-k4-fair-stage1-b256-20260815-064515"
DEFAULT_CHECKPOINT="$STAGE1_RUN/church256-rqvae-laser-8x8-a16384-k4-fair/15082026_064530/best_rfid_slot1_model.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/tmp/laser-h100-venv/bin/python}"
DATA_ROOT="${LSUN_ROOT:-/workspace/Projects/data/lsun}"
FID_REFERENCE_STATS="$THIRD_PARTY/assets/fid_stats/lsun_256_church.npz"
FID_REFERENCE_SHA256="809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-church-a16384-k4-compound-stage2-$STAMP}"
DATA_VIEW="$RUN_ROOT/data/lsun"
TOKEN_CACHE="$RUN_ROOT/token_cache/lsun_church_train_a16384k4_compound_pairs.pt"
TOKEN_REPORT="${TOKEN_CACHE%.pt}.validation.json"
DEFAULT_CACHE_SOURCE="$ROOT/outputs/lsun-church-a16384-k4-compound-stage2-20260815-084408/token_cache/lsun_church_train_a16384k4_compound_pairs.pt"
CACHE_SOURCE="${CACHE_SOURCE:-$DEFAULT_CACHE_SOURCE}"
STAGE2_OUT="$RUN_ROOT/stage2"
CHECKPOINT_DIR="$STAGE2_OUT/checkpoints"
SMOKE_OUT="$RUN_ROOT/stage2-smoke"
NPROC="${NPROC:-4}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-64}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-8}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
STAGE2_TOTAL_BATCH_SIZE="${STAGE2_TOTAL_BATCH_SIZE:-256}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-250}"
SAMPLE_GRID_BATCH_SIZE="${SAMPLE_GRID_BATCH_SIZE:-64}"
WANDB_RUN_ID="${WANDB_RUN_ID:-lsunchurcha16384k4cmp-${STAMP//-/}}"
WANDB_NAME="${WANDB_NAME:-lsun-church-a16384-k4-compound-official-rqtransformer-350M-$STAMP}"
PREFLIGHT_ONLY=0
if [[ "${1:-}" == "--preflight" ]]; then
  PREFLIGHT_ONLY=1
elif (( $# > 0 )); then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

for required in "$PYTHON_BIN" "$CHECKPOINT" "$OFFICIAL_CONFIG" \
  "$DATA_ROOT/church_outdoor_train_lmdb/data.mdb" \
  "$DATA_ROOT/church_outdoor_val_lmdb/data.mdb" "$FID_REFERENCE_STATS"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done
if (( STAGE2_TOTAL_BATCH_SIZE % (STAGE2_BATCH_SIZE * NPROC) != 0 )); then
  echo "Stage-2 total batch must divide batch_size * world_size" >&2
  exit 1
fi
if (( FID_BATCH_SIZE <= 0 )); then
  echo "FID batch size must be positive" >&2
  exit 1
fi
if (( SAMPLE_GRID_BATCH_SIZE <= 0 || SAMPLE_GRID_BATCH_SIZE > 64 )); then
  echo "Sample-grid batch size must be in [1, 64]" >&2
  exit 1
fi
if [[ "$(sha256sum "$FID_REFERENCE_STATS" | cut -d' ' -f1)" != "$FID_REFERENCE_SHA256" ]]; then
  echo "LSUN Church FID reference checksum mismatch" >&2
  exit 1
fi
if ! git -C "$THIRD_PARTY" diff --quiet upstream/main -- "$OFFICIAL_CONFIG_REL"; then
  echo "Official LSUN Church stage-2 reference config differs from upstream" >&2
  exit 1
fi

mkdir -p "$DATA_VIEW/church" "$RUN_ROOT/token_cache" "$RUN_ROOT/logs" \
  "$STAGE2_OUT" "$CHECKPOINT_DIR" "$ROOT/.cache/wandb" \
  "$ROOT/.local/share/wandb" "$ROOT/wandb"
ln -sfn "$DATA_ROOT/church_outdoor_train_lmdb" \
  "$DATA_VIEW/church/church_outdoor_train_lmdb"
ln -sfn "$DATA_ROOT/church_outdoor_val_lmdb" \
  "$DATA_VIEW/church/church_outdoor_val_lmdb"

LASER_STAGE2_ROOT="$ROOT" LASER_STAGE2_CONFIG="$OFFICIAL_CONFIG" \
LASER_STAGE2_CHECKPOINT="$CHECKPOINT" LASER_LSUN_DATA_VIEW="$DATA_VIEW" \
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

from scripts.train_official_rqtransformer_laser_stage2 import (
    LaserAux,
    source_image_dataset,
    val_image_transform,
)
from src.rqvae_metrics import load_reference_statistics

config = OmegaConf.load(os.environ["LASER_STAGE2_CONFIG"])
assert list(config.arch.block_size) == [8, 8, 4]
assert int(config.arch.embed_dim) == 1024
assert int(config.arch.input_embed_dim) == 256
assert int(config.arch.body.n_layer) == 24
assert int(config.arch.body.block.n_head) == 16
assert int(config.arch.head.n_layer) == 4
assert int(config.arch.head.block.n_head) == 16
assert int(config.arch.vocab_size_cond) == 1
assert str(config.loss.type) == "soft_target_cross_entropy"
assert bool(config.loss.stochastic_codes)
assert float(config.loss.temp) == 0.5
assert str(config.optimizer.type).lower() == "adamw"
assert float(config.optimizer.init_lr) == 5e-4
assert float(config.optimizer.weight_decay) == 1e-4
assert list(config.optimizer.betas) == [0.9, 0.95]
assert int(config.experiment.epochs) == 300
assert int(config.experiment.total_batch_size) == 256
assert int(config.experiment.save_ckpt_freq) == 10
assert int(config.experiment.test_freq) == 5
assert int(config.experiment.sample.top_k) == 250
assert float(config.experiment.sample.top_p) == 1.0

dataset = source_image_dataset(
    "lsun_church",
    Path(os.environ["LASER_LSUN_DATA_VIEW"]),
    val_image_transform(),
)
assert len(dataset) == 126_227, len(dataset)
image, label = dataset[0]
assert tuple(image.shape) == (3, 256, 256)
assert label == 0

aux = LaserAux(
    Path(os.environ["LASER_STAGE2_CHECKPOINT"]),
    num_atoms=16_384,
    coeff_vocab_size=2_048,
    coeff_max=3.0,
    coeff_scale=6.4,
    attn_resolutions=(8,),
    sparsity_level=4,
)
assert tuple(aux.dictionary.shape) == (256, 16_384)
load_reference_statistics(
    Path(os.environ["LASER_STAGE2_ROOT"])
    / "third_party/rq-vae-transformer/assets/fid_stats/lsun_256_church.npz"
)
print("LSUN Church compound Stage-2 preflight passed")
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
stage1_wandb_run=helloimlixin-rutgers/laser/lsunchurcha16384k4s1fair-b256-20260815064515
official_stage2_config=$OFFICIAL_CONFIG
compound_reference=helloimlixin-rutgers/laser/ffhqcmp0804205803
token_cache=$TOKEN_CACHE
token_cache_source=$CACHE_SOURCE
stage2_output=$STAGE2_OUT
wandb_run_id=$WANDB_RUN_ID
world_size=$NPROC
microbatch_per_gpu=$STAGE2_BATCH_SIZE
gradient_accumulation=$((STAGE2_TOTAL_BATCH_SIZE / (STAGE2_BATCH_SIZE * NPROC)))
effective_batch_size=$STAGE2_TOTAL_BATCH_SIZE
epochs=300
precision=bf16_amp
model=official LSUN Church RQ-Transformer 350M geometry
body=24x1024x16heads
depth_head=4x1024x16heads
conditioning=unconditional
compound_scheme=ffhqcmp0804205803 compound-v5 causal-atom-depth physical-soft-targets micro2 depth-heads distribution-geometry
within_site_factorization=p(atom_d|prior_atoms,prior_sites) p(coeff_d|atom_d,prior_atoms,prior_sites)
spatial_history=full completed atom-coefficient pairs
num_atoms=16384
sparsity_level=4
coeff_vocab_size=2048
coeff_max=3
coeff_scale_calibration=per-depth percentile100
compound_sequence_length=256
optimizer=AdamW betas0.9,0.95 weight_decay0.0001 clip1.0
learning_rate=0.0005
lr_schedule=cosine epochs300 min_lr0
stochastic_codes=true
soft_target_temperature=0.5
atom_loss_weight=1.5
geometry_loss_weight=0.05
geometry_start_epoch=2
geometry_warmup_epochs=3
fid_backend=original-rqvae
fid_reference_stats=$FID_REFERENCE_STATS
fid_reference_sha256=$FID_REFERENCE_SHA256
fid_num_samples=50000
fid_batch_size_per_gpu=$FID_BATCH_SIZE
fid_every_epochs=5
sample_grid_batch_size=$SAMPLE_GRID_BATCH_SIZE
sampling=temperature1 top_k250 top_p1 on both factorized atom and coefficient heads
sampling_reference=KakaoBrain LSUN Church Stage-2 config sample.top_k=250 sample.top_p=1.0
save_checkpoint_every_epochs=1
wandb_checkpoint_retention=last-per-epoch plus top3-lowest-fid
restart=RUN_ROOT=$RUN_ROOT STAMP=$STAMP WANDB_RUN_ID=$WANDB_RUN_ID $0
EOF

active_phase=token_cache
cache_valid=0
if [[ ! -f "$TOKEN_CACHE" && -f "$CACHE_SOURCE" \
      && -f "${CACHE_SOURCE%.pt}.validation.json" ]]; then
  # The cache contains continuous final-OMP coefficients, so the corrected
  # causal factorization and physical soft targets do not require re-encoding
  # the 126,227 source images.  Hard-link the already built and validated cache
  # into this run; fall back to a copy if the paths are on different devices.
  ln "$CACHE_SOURCE" "$TOKEN_CACHE" 2>/dev/null || cp --reflink=auto "$CACHE_SOURCE" "$TOKEN_CACHE"
  cp "${CACHE_SOURCE%.pt}.validation.json" "$TOKEN_REPORT"
  status "$active_phase" imported "validated continuous OMP cache source=$CACHE_SOURCE"
fi
if [[ -f "$TOKEN_CACHE" && -f "$TOKEN_REPORT" ]]; then
  cache_valid="$("$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(int(p.get("passed") is True and p.get("items") == 126227 and p.get("compound_sequence_length") == 256))' "$TOKEN_REPORT")"
fi
if [[ "$cache_valid" -ne 1 ]]; then
  status "$active_phase" starting "full 126227-image LSUN Church 8x8x4 compound cache"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/tools/build_official_imagenet_token_cache.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA_VIEW" \
    --output "$TOKEN_CACHE" \
    --dataset lsun_church \
    --batch-size "$CACHE_BATCH_SIZE" \
    --num-workers "$CACHE_NUM_WORKERS" \
    --num-atoms 16384 \
    --sparsity-level 4 \
    --coeff-vocab-size 2048 \
    --coeff-max 3 \
    --coeff-scale 6.4 \
    --auto-coeff-scales-percentile 100 \
    --verify-samples 256 \
    --compound \
    2>&1 | tee -a "$RUN_ROOT/logs/token_cache.log"
fi
"$PYTHON_BIN" - "$TOKEN_REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["passed"] is True, report
assert report["items"] == 126_227, report
assert report["compound_sequence_length"] == 256, report
assert report["atom_exact_fraction"] == 1.0, report
assert report["coeff_finite"] is True, report
assert report["duplicate_atom_within_support_fraction"] == 0.0, report
print("Validated full LSUN Church compound token cache")
PY
status "$active_phase" complete "cache=$TOKEN_CACHE"

active_phase=stage2_smoke
if [[ ! -f "$RUN_ROOT/.stage2_smoke_complete" ]]; then
  status "$active_phase" starting "one real batch/optimizer step at production microbatch"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
    --checkpoint "$CHECKPOINT" --data "$DATA_VIEW" --dataset lsun_church \
    --model-preset lsun-church-350m --token-cache "$TOKEN_CACHE" \
    --output "$SMOKE_OUT" --checkpoint-dir "$SMOKE_OUT/checkpoints" \
    --distributed-backend ddp --epochs 300 --batch-size "$STAGE2_BATCH_SIZE" \
    --total-batch-size "$STAGE2_TOTAL_BATCH_SIZE" --num-atoms 16384 \
    --sparsity-level 4 --coeff-vocab-size 2048 --coeff-max 3 --coeff-scale 6.4 \
    --compound-tokens --compound-micro-transformer-layers 2 \
    --compound-depth-specific-coeff-heads --compound-distribution-geometry \
    --geometry-top-k 4 --atom-loss-weight 1.5 --geometry-loss-weight 0.05 \
    --geometry-start-epoch 2 --geometry-warmup-epochs 3 \
    --coeff-target-mode soft --coeff-target-temperature 0.5 \
    --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
    --fid-every 0 --sample-grid-every 0 --save-step-freq 0 \
    --max-optimizer-steps 1 --smoke-test --no-resume --wandb-mode disabled \
    2>&1 | tee -a "$RUN_ROOT/logs/stage2-smoke.log"
  touch "$RUN_ROOT/.stage2_smoke_complete"
fi
status "$active_phase" complete "production-shape optimizer step passed"

active_phase=stage2
status "$active_phase" starting "official Church 350M compound RQ-Transformer; 300 epochs"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$CHECKPOINT" --data "$DATA_VIEW" --dataset lsun_church \
  --model-preset lsun-church-350m --token-cache "$TOKEN_CACHE" \
  --output "$STAGE2_OUT" --checkpoint-dir "$CHECKPOINT_DIR" \
  --distributed-backend ddp --epochs 300 --batch-size "$STAGE2_BATCH_SIZE" \
  --total-batch-size "$STAGE2_TOTAL_BATCH_SIZE" --num-atoms 16384 \
  --sparsity-level 4 --coeff-vocab-size 2048 --coeff-max 3 --coeff-scale 6.4 \
  --compound-tokens --compound-micro-transformer-layers 2 \
  --compound-depth-specific-coeff-heads --compound-distribution-geometry \
  --geometry-top-k 4 --atom-loss-weight 1.5 --geometry-loss-weight 0.05 \
  --geometry-start-epoch 2 --geometry-warmup-epochs 3 \
  --coeff-target-mode soft --coeff-target-temperature 0.5 \
  --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 300 --min-lr 0 \
  --atom-temperature 1.0 --atom-top-k 250 --atom-top-p 1.0 \
  --coeff-temperature 1.0 --coeff-top-k 250 --coeff-top-p 1.0 \
  --fid-num-samples 50000 --fid-batch-size "$FID_BATCH_SIZE" --fid-every 5 \
  --metric-backend original-rqvae --fid-reference-stats "$FID_REFERENCE_STATS" \
  --save-ckpt-freq 1 --save-step-freq 0 \
  --sample-grid-every 500 --sample-grid-size 64 \
  --sample-grid-samples-per-class 8 --sample-grid-batch-size "$SAMPLE_GRID_BATCH_SIZE" \
  --upload-checkpoints --checkpoint-upload-mode files --upload-token-cache --resume \
  --wandb-mode online --wandb-entity "$WANDB_ENTITY" --wandb-project "$WANDB_PROJECT" \
  --wandb-id "$WANDB_RUN_ID" --wandb-name "$WANDB_NAME" \
  2>&1 | tee -a "$RUN_ROOT/logs/stage2.log"
status "$active_phase" complete "stage2 exit=0"
trap - EXIT
