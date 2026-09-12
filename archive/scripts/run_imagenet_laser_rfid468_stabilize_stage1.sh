#!/usr/bin/env bash
# One-epoch stabilization branch from the best 4.6819-rFID LASER checkpoint.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
MEMBER_LAUNCHER="$ROOT/scripts/run_imagenet_a16384_k4_fair_stage1.sh"
CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-laser-8x8-a16384-k4-rfid468-stabilize.yaml"
REFERENCE_CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-laser-8x8-a16384-k4-paired-alternating.yaml"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT/outputs/imagenet-rq-vs-laser-paired-alternating-20260823-051002/laser/in256-rqvae-laser-8x8-a16384-k4-paired-alternating/23082026_054325/best_rfid_slot1_model.pt}"
EXPECTED_INIT_SHA256="61551803fb322bea61476a530508e66a66afe3458e6601e3a57efbd688fe5f61"
SOURCE_RUN="${SOURCE_RUN:-helloimlixin-rutgers/laser/imgrqvslaser-laser-20260823051002}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/imagenet-laser-rfid468-stabilize-$STAMP}"
NPROC="${NPROC:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-0}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-6}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-0.0}"
DISCRIMINATOR_LEARNING_RATE="${DISCRIMINATOR_LEARNING_RATE:-1.0e-7}"
DISCRIMINATOR_MIN_LEARNING_RATE="${DISCRIMINATOR_MIN_LEARNING_RATE:-0.0}"
RECOVERY_CKPT_FREQ_STEPS="${RECOVERY_CKPT_FREQ_STEPS:-500}"
WANDB_RUN_ID="${WANDB_RUN_ID:-imglaser-rfid468-stabilize-${STAMP//-/}}"
WANDB_NAME="${WANDB_NAME:-imagenet-laser-rfid468-stabilize-$STAMP}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet-laser-rfid468-stabilize-$STAMP}"

PREFLIGHT_ONLY=0
if [[ "${1:-}" == "--preflight" ]]; then
  PREFLIGHT_ONLY=1
elif (( $# > 0 )); then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

for required in "$MEMBER_LAUNCHER" "$CONFIG" "$REFERENCE_CONFIG" "$INIT_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing stabilization input: $required" >&2
    exit 1
  fi
done
actual_init_sha256="$(sha256sum "$INIT_CHECKPOINT" | cut -d' ' -f1)"
if [[ "$actual_init_sha256" != "$EXPECTED_INIT_SHA256" ]]; then
  echo "4.6819-rFID initializer checksum mismatch" >&2
  exit 1
fi

LASER_STABILIZE_CONFIG="$CONFIG" LASER_STABILIZE_REFERENCE="$REFERENCE_CONFIG" \
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"${PYTHON_BIN:-/workspace/tmp/laser-eval-cc3m-venv/bin/python}" - <<'PY'
import os

from omegaconf import OmegaConf

config = OmegaConf.to_container(
    OmegaConf.load(os.environ['LASER_STABILIZE_CONFIG']), resolve=True
)
reference = OmegaConf.to_container(
    OmegaConf.load(os.environ['LASER_STABILIZE_REFERENCE']), resolve=True
)
assert config['dataset'] == reference['dataset']
assert config['arch'] == reference['arch']
assert config['gan']['disc']['arch'] == reference['gan']['disc']['arch']
assert config['gan']['loss'] == reference['gan']['loss']

main_opt = dict(config['optimizer'])
reference_main_opt = dict(reference['optimizer'])
assert float(main_opt.pop('init_lr')) == 1.0e-6
assert float(reference_main_opt.pop('init_lr')) == 4.0e-5
assert main_opt == reference_main_opt

disc_opt = dict(config['gan']['disc']['optimizer'])
reference_disc_opt = dict(reference['gan']['disc']['optimizer'])
assert float(disc_opt.pop('init_lr')) == 1.0e-7
assert float(reference_disc_opt.pop('init_lr')) == 4.0e-5
assert disc_opt == reference_disc_opt

experiment = dict(config['experiment'])
reference_experiment = dict(reference['experiment'])
assert experiment.pop('epochs') == 1
assert reference_experiment.pop('epochs') == 10
assert experiment.pop('recovery_ckpt_freq_steps') == 500
assert reference_experiment.pop('recovery_ckpt_freq_steps') == 250
assert experiment == reference_experiment
PY

launcher_env=(
  "CONFIG=$CONFIG"
  "BOTTLENECK_TYPE=laser"
  "DICTIONARY_UPDATE_MODE=alternating_residual"
  "INIT_CHECKPOINT=$INIT_CHECKPOINT"
  "SOURCE_RUN=$SOURCE_RUN"
  "NPROC=$NPROC"
  "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
  "EPOCHS=$EPOCHS"
  "SEED=$SEED"
  "LEARNING_RATE=$LEARNING_RATE"
  "MIN_LEARNING_RATE=$MIN_LEARNING_RATE"
  "DISCRIMINATOR_LEARNING_RATE=$DISCRIMINATOR_LEARNING_RATE"
  "DISCRIMINATOR_MIN_LEARNING_RATE=$DISCRIMINATOR_MIN_LEARNING_RATE"
  "RECOVERY_CKPT_FREQ_STEPS=$RECOVERY_CKPT_FREQ_STEPS"
  "RUN_ROOT=$RUN_ROOT"
  "STAMP=$STAMP"
  "WANDB_RUN_ID=$WANDB_RUN_ID"
  "WANDB_NAME=$WANDB_NAME"
  "WANDB_RUN_GROUP=$WANDB_RUN_GROUP"
  "RUN_VARIANT_TAG=rfid468-stabilize"
)

if (( PREFLIGHT_ONLY )); then
  env "${launcher_env[@]}" "$MEMBER_LAUNCHER" --preflight
  echo "LASER rFID-4.6819 stabilization preflight passed"
  exit 0
fi

mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/experiment.info" <<EOF
run_root=$RUN_ROOT
config=$CONFIG
initializer=$INIT_CHECKPOINT
initializer_sha256=$actual_init_sha256
initializer_epoch=8
initializer_full_rfid=4.681939125061035
initializer_bypass_rfid=4.5367
source_run=$SOURCE_RUN
world_size=$NPROC
global_batch_size=$TOTAL_BATCH_SIZE
epochs=$EPOCHS
main_peak_lr=$LEARNING_RATE
discriminator_peak_lr=$DISCRIMINATOR_LEARNING_RATE
schedule=half-epoch-linear-warmup-then-cosine-to-zero
optimizer_policy=fresh-branch-optimizers;preserve-model-and-discriminator
training_path=full-laser
checkpoint_primary=valid/rfid-top3
checkpoint_backbone=valid/bypass_rfid-top1
wandb_run_id=$WANDB_RUN_ID
wandb_group=$WANDB_RUN_GROUP
EOF

exec env "${launcher_env[@]}" "$MEMBER_LAUNCHER"
