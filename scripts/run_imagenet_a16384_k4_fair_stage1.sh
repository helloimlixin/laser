#!/usr/bin/env bash
# Clean, apples-to-apples ImageNet Stage-1 comparison against RQ-VAE 8x8x4.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-laser-8x8-a16384-k4-fair.yaml"
DEFAULT_PYTHON="/workspace/tmp/laser-eval-cc3m-venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
DATA_ROOT="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/imagenet-a16384-k4-fair-stage1-$STAMP}"
NPROC="${NPROC:-2}"
TOTAL_BATCH_SIZE=128
WANDB_RUN_ID="${WANDB_RUN_ID:-imga16384k4s1fair-${STAMP//-/}}"
WANDB_NAME="${WANDB_NAME:-imagenet-a16384-k4-fair-stage1-$STAMP}"
PREFLIGHT_ONLY=0
if [[ "${1:-}" == "--preflight" ]]; then
  PREFLIGHT_ONLY=1
elif (( $# > 0 )); then
  echo "Usage: $0 [--preflight]" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is not executable: $PYTHON_BIN" >&2
  exit 1
fi
for required in "$CONFIG" "$DATA_ROOT/train" "$DATA_ROOT/val" \
  "$ROOT/vgg_lpips/vgg.pth" "$ROOT/vgg_lpips/vgg16-397923af.pth"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done
if (( TOTAL_BATCH_SIZE % NPROC != 0 )); then
  echo "Effective batch $TOTAL_BATCH_SIZE must be divisible by NPROC=$NPROC" >&2
  exit 1
fi
LOCAL_BATCH_SIZE=$((TOTAL_BATCH_SIZE / NPROC))

# These components must remain exactly upstream for the comparison to isolate
# the sparse bottleneck and its matched cumulative-depth objective.
if ! git -C "$THIRD_PARTY" diff --quiet upstream/main -- \
  rqvae/models/rqvae/modules.py \
  rqvae/losses/vqgan/discriminator.py \
  rqvae/losses/vqgan/gan_loss.py \
  configs/imagenet256/stage1/in256-rqvae-8x8x4.yaml; then
  echo "Encoder/decoder, discriminator, GAN loss, or reference config differs from upstream" >&2
  exit 1
fi
if [[ "$(md5sum "$ROOT/vgg_lpips/vgg.pth" | cut -d' ' -f1)" != d507d7349b931f0638a25a48a722f98a ]]; then
  echo "LPIPS checkpoint checksum mismatch" >&2
  exit 1
fi
if [[ "$(sha256sum "$ROOT/vgg_lpips/vgg16-397923af.pth" | cut -d' ' -f1)" != 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0 ]]; then
  echo "VGG-16 checkpoint checksum mismatch" >&2
  exit 1
fi
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import lmdb
import rqvae.img_datasets
import rqvae.models
import rqvae.optimizer
import rqvae.trainers
import rqvae.utils.setup
import src.rqvae_metrics
PY
if (( PREFLIGHT_ONLY )); then
  echo "Fair Stage-1 preflight passed"
  exit 0
fi

mkdir -p "$RUN_ROOT/logs" "$ROOT/.cache/wandb" "$ROOT/.local/share/wandb" "$ROOT/wandb"
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"
printf 'time_utc\tphase\tstate\tdetail\n' > "$RUN_ROOT/status.tsv"

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" \
    >> "$RUN_ROOT/status.tsv"
}

on_exit() {
  exit_code="$?"
  if (( exit_code != 0 )); then
    status stage1 failed "exit=$exit_code"
  fi
}
trap on_exit EXIT

cat > "$RUN_ROOT/run.info" <<EOF
run_root=$RUN_ROOT
data_root=$DATA_ROOT
config=$CONFIG
wandb_run_id=$WANDB_RUN_ID
python=$PYTHON_BIN
world_size=$NPROC
local_batch_size=$LOCAL_BATCH_SIZE
effective_batch_size=$TOTAL_BATCH_SIZE
epochs=10
seed=0
precision=float32
encoder_decoder=unmodified upstream RQ-VAE
discriminator=unmodified upstream PatchGAN
commitment=depth-averaged cumulative OMP reconstruction
commitment_cost=1.0
latent_loss_weight=0.25
effective_commitment_weight=0.25
checkpoint_format=model+main_optimizer+main_scheduler+discriminator+discriminator_optimizer+discriminator_scheduler
rfid_backend=original-rqvae
EOF

export PYTHONUNBUFFERED=1
export PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
export WANDB_RUN_ID
export WANDB_NAME
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet-a16384-k4-fair-$STAMP}"
export WANDB_TAGS="stage1,imagenet,laser,rqvae,fair-comparison,a16384,k4,8x8x4,f32,effective-batch128,progressive-commitment,full-gan-checkpoint,original-rqvae-rfid"
export WANDB_CHECKPOINT_UPLOAD=1
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export LASER_VGG_LPIPS_DIR="$ROOT/vgg_lpips"
export LASER_VGG16_WEIGHTS="$ROOT/vgg_lpips/vgg16-397923af.pth"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

resume_args=()
last_checkpoint="$(find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "$last_checkpoint" && -f "$last_checkpoint" ]]; then
  resume_args=(--load-path "$last_checkpoint" --resume)
  status stage1 resuming "checkpoint=$last_checkpoint"
else
  status stage1 starting "clean seed=0 run"
fi

(
  cd "$THIRD_PARTY"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    main_stage1.py \
    --model-config "$CONFIG" \
    --result-path "$RUN_ROOT" \
    --seed 0 \
    "${resume_args[@]}" \
    "dataset.root=$DATA_ROOT" \
    "experiment.batch_size=$LOCAL_BATCH_SIZE" \
    experiment.total_batch_size="$TOTAL_BATCH_SIZE" \
    experiment.rfid_backend=original-rqvae
) 2>&1 | tee -a "$RUN_ROOT/logs/stage1.log"

completed_checkpoint="$(find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$completed_checkpoint" || ! -f "$completed_checkpoint" ]]; then
  echo "Stage 1 exited without a last_model.pt checkpoint" >&2
  exit 1
fi
status stage1 complete "checkpoint=$completed_checkpoint"
trap - EXIT
