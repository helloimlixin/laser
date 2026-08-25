#!/usr/bin/env bash
# Sequential, resumable ImageNet Stage-1 calibration: RQ first, then LASER.
# Both members use the same codec, GAN, data, seed, global batch, optimizer,
# cosine-to-zero schedule, validation metric, and physical GPU allocation.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
MEMBER_LAUNCHER="$ROOT/scripts/run_imagenet_a16384_k4_fair_stage1.sh"
RQ_CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-8x8x4-paired-cosine.yaml"
LASER_CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-laser-8x8-a16384-k4-paired-cosine.yaml"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
PAIR_ROOT="${PAIR_ROOT:-$ROOT/outputs/imagenet-rq-vs-laser-paired-cosine-$STAMP}"
NPROC="${NPROC:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-0}"
LEARNING_RATE="${LEARNING_RATE:-4.0e-5}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-0.0}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet-rq-vs-laser-paired-cosine-$STAMP}"

PREFLIGHT_ONLY=0
if [[ "${1:-}" == "--preflight" ]]; then
  PREFLIGHT_ONLY=1
elif (( $# > 0 )); then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

for required in "$MEMBER_LAUNCHER" "$RQ_CONFIG" "$LASER_CONFIG"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing paired-run input: $required" >&2
    exit 1
  fi
done

# Fail closed if anything outside the explicitly exchanged bottleneck differs.
LASER_PAIR_RQ_CONFIG="$RQ_CONFIG" LASER_PAIR_LASER_CONFIG="$LASER_CONFIG" \
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"${PYTHON_BIN:-/workspace/tmp/laser-eval-cc3m-venv/bin/python}" - <<'PY'
import os

from omegaconf import OmegaConf

rq = OmegaConf.to_container(
    OmegaConf.load(os.environ["LASER_PAIR_RQ_CONFIG"]), resolve=True
)
laser = OmegaConf.to_container(
    OmegaConf.load(os.environ["LASER_PAIR_LASER_CONFIG"]), resolve=True
)

for section in ("dataset", "optimizer", "experiment", "gan"):
    assert rq[section] == laser[section], f"paired section differs: {section}"
for key in ("type", "code_hier", "ddconfig", "checkpointing"):
    assert rq["arch"][key] == laser["arch"][key], f"paired arch field differs: {key}"

rq_hparams = dict(rq["arch"]["hparams"])
laser_hparams = dict(laser["arch"]["hparams"])
assert rq_hparams.pop("bottleneck_type") == "rq"
assert laser_hparams.pop("bottleneck_type") == "laser"
laser_only = {
    "sparsity_level",
    "commitment_cost",
    "progressive_loss",
    "dict_learning_rate",
    "patch_based",
    "patch_size",
    "patch_stride",
    "data_init_from_first_batch",
    "dead_atom_revival",
    "dead_atom_revival_interval",
    "dead_atom_revival_max_fraction",
    "dead_atom_revival_noise",
    "dead_atom_revival_patience",
    "omp_compute_precision",
}
extras = set(laser_hparams) - set(rq_hparams)
assert extras == laser_only, (extras, laser_only)
for key in laser_only:
    laser_hparams.pop(key)
assert rq_hparams == laser_hparams, "shared bottleneck-independent hparams differ"

assert rq["experiment"]["epochs"] == 10
assert rq["experiment"]["total_batch_size"] == 128
assert rq["optimizer"]["warmup"]["start_from_zero"] is True
assert float(rq["optimizer"]["warmup"]["min_lr"]) == 0.0
assert float(rq["gan"]["disc"]["optimizer"]["warmup"]["min_lr"]) == 0.0
assert rq["experiment"]["rfid_backend"] == "original-rqvae"
PY

common_env=(
  "NPROC=$NPROC"
  "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
  "EPOCHS=$EPOCHS"
  "SEED=$SEED"
  "LEARNING_RATE=$LEARNING_RATE"
  "MIN_LEARNING_RATE=$MIN_LEARNING_RATE"
  "DISCRIMINATOR_LEARNING_RATE=$LEARNING_RATE"
  "DISCRIMINATOR_MIN_LEARNING_RATE=$MIN_LEARNING_RATE"
  "WANDB_RUN_GROUP=$WANDB_RUN_GROUP"
  "STAMP=$STAMP"
)

run_member() {
  local member="$1"
  local config="$2"
  local run_id="$3"
  local run_name="$4"
  local member_root="$PAIR_ROOT/$member"
  local args=()
  if (( PREFLIGHT_ONLY )); then
    args=(--preflight)
  fi
  env \
    "${common_env[@]}" \
    "BOTTLENECK_TYPE=$member" \
    "CONFIG=$config" \
    "RUN_ROOT=$member_root" \
    "WANDB_RUN_ID=$run_id" \
    "WANDB_NAME=$run_name" \
    "$MEMBER_LAUNCHER" "${args[@]}"
}

if (( PREFLIGHT_ONLY )); then
  run_member \
    rq "$RQ_CONFIG" \
    "imgrqvslaser-rq-${STAMP//-/}" \
    "imagenet-rq-paired-cosine-$STAMP"
  run_member \
    laser "$LASER_CONFIG" \
    "imgrqvslaser-laser-${STAMP//-/}" \
    "imagenet-laser-paired-cosine-$STAMP"
  echo "Paired RQ/LASER preflight passed"
  exit 0
fi

mkdir -p "$PAIR_ROOT"
printf '%s\n' "$$" > "$PAIR_ROOT/launcher.pid"
if [[ ! -f "$PAIR_ROOT/status.tsv" ]]; then
  printf 'time_utc\tmember\tstate\tdetail\n' > "$PAIR_ROOT/status.tsv"
fi
status() {
  printf '%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" \
    >> "$PAIR_ROOT/status.tsv"
}

{
  printf 'pair_root=%s\n' "$PAIR_ROOT"
  printf 'rq_config=%s\n' "$RQ_CONFIG"
  printf 'laser_config=%s\n' "$LASER_CONFIG"
  printf 'world_size=%s\n' "$NPROC"
  printf 'local_batch_size=%s\n' "$((TOTAL_BATCH_SIZE / NPROC))"
  printf 'global_batch_size=%s\n' "$TOTAL_BATCH_SIZE"
  printf 'epochs=%s\n' "$EPOCHS"
  printf 'seed=%s\n' "$SEED"
  printf 'initial_lr=%s\n' "$LEARNING_RATE"
  printf 'minimum_lr=%s\n' "$MIN_LEARNING_RATE"
  printf 'member_order=rq,laser\n'
  printf 'wandb_group=%s\n' "$WANDB_RUN_GROUP"
} > "$PAIR_ROOT/pair.info"

status rq starting "clean paired control"
run_member \
  rq "$RQ_CONFIG" \
  "imgrqvslaser-rq-${STAMP//-/}" \
  "imagenet-rq-paired-cosine-$STAMP"
status rq complete

status laser starting "clean paired treatment"
run_member \
  laser "$LASER_CONFIG" \
  "imgrqvslaser-laser-${STAMP//-/}" \
  "imagenet-laser-paired-cosine-$STAMP"
status laser complete
status pair complete
