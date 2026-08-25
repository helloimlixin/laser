#!/usr/bin/env bash
# From-scratch ImageNet LASER run that alternates direct-backbone and LASER
# reconstruction steps while updating the dictionary from every batch.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
MEMBER_LAUNCHER="$ROOT/scripts/run_imagenet_a16384_k4_fair_stage1.sh"
CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-laser-8x8-a16384-k4-backbone-alternating.yaml"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/imagenet-laser-backbone-alternating-$STAMP}"
NPROC="${NPROC:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-0}"
LEARNING_RATE="${LEARNING_RATE:-4.0e-5}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-0.0}"
RECOVERY_CKPT_FREQ_STEPS="${RECOVERY_CKPT_FREQ_STEPS:-500}"
WANDB_RUN_ID="${WANDB_RUN_ID:-imglaser-backbone-alt-${STAMP//-/}}"
WANDB_NAME="${WANDB_NAME:-imagenet-laser-backbone-alternating-$STAMP}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet-laser-backbone-alternating-$STAMP}"

PREFLIGHT_ONLY=0
if [[ "${1:-}" == "--preflight" ]]; then
  PREFLIGHT_ONLY=1
elif (( $# > 0 )); then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

for required in "$MEMBER_LAUNCHER" "$CONFIG"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing backbone-alternating input: $required" >&2
    exit 1
  fi
done

LASER_BACKBONE_ALT_CONFIG="$CONFIG" \
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"${PYTHON_BIN:-/workspace/tmp/laser-eval-cc3m-venv/bin/python}" - <<'PY'
import os

from omegaconf import OmegaConf

config = OmegaConf.load(os.environ['LASER_BACKBONE_ALT_CONFIG'])
assert str(config.arch.hparams.bottleneck_type).lower() == 'laser'
assert str(config.arch.hparams.dictionary_update_mode).lower() == 'alternating_residual'
assert config.arch.hparams.dict_learning_rate is None
assert bool(config.experiment.train_bypass_alternating)
assert bool(config.experiment.compute_bypass_rfid)
assert int(config.experiment.total_batch_size) == 128
assert int(config.experiment.epochs) == 10
assert float(config.optimizer.warmup.min_lr) == 0.0
assert float(config.gan.disc.optimizer.warmup.min_lr) == 0.0
PY

launcher_env=(
  "CONFIG=$CONFIG"
  "BOTTLENECK_TYPE=laser"
  "DICTIONARY_UPDATE_MODE=alternating_residual"
  "NPROC=$NPROC"
  "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
  "EPOCHS=$EPOCHS"
  "SEED=$SEED"
  "LEARNING_RATE=$LEARNING_RATE"
  "MIN_LEARNING_RATE=$MIN_LEARNING_RATE"
  "DISCRIMINATOR_LEARNING_RATE=$LEARNING_RATE"
  "DISCRIMINATOR_MIN_LEARNING_RATE=$MIN_LEARNING_RATE"
  "RECOVERY_CKPT_FREQ_STEPS=$RECOVERY_CKPT_FREQ_STEPS"
  "RUN_ROOT=$RUN_ROOT"
  "STAMP=$STAMP"
  "WANDB_RUN_ID=$WANDB_RUN_ID"
  "WANDB_NAME=$WANDB_NAME"
  "WANDB_RUN_GROUP=$WANDB_RUN_GROUP"
  "RUN_VARIANT_TAG=backbone-alternating"
)

if (( PREFLIGHT_ONLY )); then
  env "${launcher_env[@]}" "$MEMBER_LAUNCHER" --preflight
  echo "LASER backbone-alternating preflight passed"
  exit 0
fi

mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/experiment.info" <<EOF
run_root=$RUN_ROOT
config=$CONFIG
world_size=$NPROC
global_batch_size=$TOTAL_BATCH_SIZE
epochs=$EPOCHS
seed=$SEED
schedule=half-epoch-linear-warmup-then-cosine-to-zero
training_paths=even-global-step-bypass,odd-global-step-laser
dictionary_observation=every-batch
dictionary_update=alternating_residual-every-batch
checkpoint_primary=valid/rfid-top3
checkpoint_backbone=valid/bypass_rfid-top1
wandb_run_id=$WANDB_RUN_ID
wandb_group=$WANDB_RUN_GROUP
EOF

exec env "${launcher_env[@]}" "$MEMBER_LAUNCHER"
