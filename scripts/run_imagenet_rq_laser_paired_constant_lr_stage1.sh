#!/usr/bin/env bash
# Sequential, resumable ImageNet Stage-1 calibration: RQ first, then LASER.
# Both members use the published warmup-then-constant learning-rate schedule
# and consume both GPUs with the same global batch, seed, metric, and codec.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
MEMBER_LAUNCHER="$ROOT/scripts/run_imagenet_a16384_k4_fair_stage1.sh"
RQ_CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-8x8x4-paired-constant-lr.yaml"
LASER_CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-laser-8x8-a16384-k4-paired-constant-lr.yaml"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
PAIR_ROOT="${PAIR_ROOT:-$ROOT/outputs/imagenet-rq-vs-laser-paired-constant-lr-$STAMP}"
NPROC="${NPROC:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-0}"
LEARNING_RATE="${LEARNING_RATE:-4.0e-5}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-$LEARNING_RATE}"
RECOVERY_CKPT_FREQ_STEPS="${RECOVERY_CKPT_FREQ_STEPS:-2000}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet-rq-vs-laser-paired-constant-lr-$STAMP}"

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
if [[ "$NPROC" != "2" ]]; then
  echo "This calibration must use both B300s (NPROC=2)" >&2
  exit 1
fi
if [[ "$TOTAL_BATCH_SIZE" != "128" || "$EPOCHS" != "10" || "$SEED" != "0" ]]; then
  echo "Calibration requires global batch 128, 10 epochs, and seed 0" >&2
  exit 1
fi
if [[ "$LEARNING_RATE" != "4.0e-5" || "$MIN_LEARNING_RATE" != "$LEARNING_RATE" ]]; then
  echo "Calibration requires warmup followed by a constant 4.0e-5 learning rate" >&2
  exit 1
fi

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

assert rq["arch"]["checkpointing"] is True
assert rq["experiment"]["batch_size"] == 64
assert rq["experiment"]["total_batch_size"] == 128
assert rq["experiment"]["epochs"] == 10
assert rq["experiment"]["precision"] == "float32"
assert rq["experiment"]["amp"] is False
assert rq["experiment"]["recovery_ckpt_freq_steps"] == 2000
assert rq["experiment"]["rfid_backend"] == "original-rqvae"
assert float(rq["optimizer"]["init_lr"]) == 4.0e-5
assert float(rq["optimizer"]["warmup"]["epoch"]) == 0.5
assert rq["optimizer"]["warmup"]["start_from_zero"] is True
assert float(rq["optimizer"]["warmup"]["min_lr"]) == 4.0e-5
assert float(rq["gan"]["disc"]["optimizer"]["init_lr"]) == 4.0e-5
assert rq["gan"]["disc"]["optimizer"]["warmup"]["start_from_zero"] is True
assert float(rq["gan"]["disc"]["optimizer"]["warmup"]["min_lr"]) == 4.0e-5
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
  "RECOVERY_CKPT_FREQ_STEPS=$RECOVERY_CKPT_FREQ_STEPS"
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
    "imgrqvslaser-rq-constantlr-${STAMP//-/}" \
    "imagenet-rq-paired-constant-lr-$STAMP"
  run_member \
    laser "$LASER_CONFIG" \
    "imgrqvslaser-laser-constantlr-${STAMP//-/}" \
    "imagenet-laser-paired-constant-lr-$STAMP"
  echo "Paired RQ/LASER constant-LR preflight passed"
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
on_exit() {
  local exit_code="$?"
  if (( exit_code != 0 )); then
    status pair failed "exit=$exit_code"
  fi
}
trap on_exit EXIT

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
  printf 'lr_schedule=0.5-epoch-warmup-then-constant\n'
  printf 'activation_checkpointing=true\n'
  printf 'recovery_ckpt_freq_steps=%s\n' "$RECOVERY_CKPT_FREQ_STEPS"
  printf 'member_order=rq,laser\n'
  printf 'wandb_group=%s\n' "$WANDB_RUN_GROUP"
} > "$PAIR_ROOT/pair.info"

status rq starting "paper-faithful constant-LR control on both GPUs"
run_member \
  rq "$RQ_CONFIG" \
  "imgrqvslaser-rq-constantlr-${STAMP//-/}" \
  "imagenet-rq-paired-constant-lr-$STAMP"
status rq complete

status laser starting "matched constant-LR treatment on both GPUs"
run_member \
  laser "$LASER_CONFIG" \
  "imgrqvslaser-laser-constantlr-${STAMP//-/}" \
  "imagenet-laser-paired-constant-lr-$STAMP"
status laser complete
status pair complete
trap - EXIT
