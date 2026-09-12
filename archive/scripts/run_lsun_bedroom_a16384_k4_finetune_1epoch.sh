#!/usr/bin/env bash
# One-epoch LSUN Bedroom Stage-1 fine-tune from the best (5.084 rFID)
# ImageNet Stage-1 LASER checkpoint, matched to the upstream 8x8x4 recipe.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
CONFIG="$THIRD_PARTY/configs/lsun-bedroom/stage1/bedroom256-rqvae-laser-8x8-a16384-k4-finetune.yaml"
REFERENCE_CONFIG="$THIRD_PARTY/configs/lsun-bedroom/stage1/bedroom256-rqvae-8x8x4.yaml"
DEFAULT_PYTHON="/tmp/laser-b300-venv/bin/python"
DEFAULT_INIT_CHECKPOINT="$ROOT/outputs/imagenet-a16384-k4-fair-stage1-source-behavior-20260815-125037/in256-rqvae-laser-8x8-a16384-k4-fair/15082026_125202/best_rfid_slot1_model.pt"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
DEFAULT_DATA_ROOT="/workspace/Projects/data/lsun"
# Prefer the byte-verified local staging copy when this host has one. Random
# LMDB reads from the workspace mount are otherwise dominated by page latency.
if [[ -e /tmp/laser-lsun-bedroom/bedroom_train_lmdb/data.mdb ]]; then
  DEFAULT_DATA_ROOT="/tmp/laser-lsun-bedroom"
fi
DATA_ROOT="${LSUN_ROOT:-$DEFAULT_DATA_ROOT}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$DEFAULT_INIT_CHECKPOINT}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-bedroom-a16384-k4-imagenet5.08-ft1e-$STAMP}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
WANDB_RUN_ID="${WANDB_RUN_ID:-lsunbedrooma16384k4-imagenet508-ft1e-${STAMP//-/}}"
WANDB_NAME="${WANDB_NAME:-lsun-bedroom-a16384-k4-imagenet5.08-ft1e-$STAMP}"
SOURCE_RUN="${SOURCE_RUN:-helloimlixin-rutgers/laser/imga16384k4s1fair-source-behavior-20260815125037}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is not executable: $PYTHON_BIN" >&2
  exit 1
fi
for required in "$CONFIG" "$REFERENCE_CONFIG" "$INIT_CHECKPOINT" \
  "$DATA_ROOT/bedroom_train_lmdb/data.mdb" \
  "$DATA_ROOT/bedroom_val_lmdb/data.mdb" \
  "$ROOT/vgg_lpips/vgg.pth" "$ROOT/vgg_lpips/vgg16-397923af.pth"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

# Guard the requested upstream architecture and objective against drift.
if ! git -C "$THIRD_PARTY" diff --quiet upstream/main -- \
  rqvae/models/rqvae/modules.py \
  rqvae/losses/vqgan/discriminator.py \
  rqvae/losses/vqgan/gan_loss.py \
  configs/lsun-bedroom/stage1/bedroom256-rqvae-8x8x4.yaml; then
  echo "Encoder/decoder, discriminator, GAN loss, or Bedroom reference config differs from upstream" >&2
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
if [[ "$(sha256sum "$INIT_CHECKPOINT" | cut -d' ' -f1)" != 32ffb287a70de3f8a072ad305037227fca45788cee6fe43f96474da39c05a77e ]]; then
  echo "ImageNet 5.084-rFID initializer checksum mismatch" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/data/lsun/bedroom" \
  "$ROOT/.cache/wandb" "$ROOT/.local/share/wandb" "$ROOT/wandb"
ln -sfn "$DATA_ROOT/bedroom_train_lmdb" \
  "$RUN_ROOT/data/lsun/bedroom/bedroom_train_lmdb"
ln -sfn "$DATA_ROOT/bedroom_val_lmdb" \
  "$RUN_ROOT/data/lsun/bedroom/bedroom_val_lmdb"
DATA_VIEW="$RUN_ROOT/data/lsun"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
export WANDB_RUN_ID
export WANDB_NAME
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-lsun-bedroom-a16384-k4-imagenet508-ft1e-$STAMP}"
export WANDB_TAGS="stage1,lsun-bedroom,laser,rqvae,a16384,k4,8x8x4,float32,effective-batch128,imagenet-rfid5.084-finetune,one-epoch,online"
export WANDB_CHECKPOINT_UPLOAD=1
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export LASER_VGG_LPIPS_DIR="$ROOT/vgg_lpips"
export LASER_VGG16_WEIGHTS="$ROOT/vgg_lpips/vgg16-397923af.pth"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# Build the LMDB key cache before the training loader forks workers.
LASER_LSUN_DATA_VIEW="$DATA_VIEW" "$PYTHON_BIN" - <<'PY'
import os
from rqvae.img_datasets.lsun import LSUNClass

dataset = LSUNClass(os.environ["LASER_LSUN_DATA_VIEW"], category_name="bedroom")
assert len(dataset) == 3_033_042, len(dataset)
image, target = dataset[0]
assert image.mode == "RGB" and target == 0
print(f"validated LSUN Bedroom train LMDB: {len(dataset)} images")
PY

cat > "$RUN_ROOT/run.info" <<EOF
run_root=$RUN_ROOT
data_root=$DATA_ROOT
data_view=$DATA_VIEW
config=$CONFIG
reference_config=$REFERENCE_CONFIG
initializer=$INIT_CHECKPOINT
initializer_sha256=32ffb287a70de3f8a072ad305037227fca45788cee6fe43f96474da39c05a77e
initializer_imagenet_rfid=5.084157943725586
source_run=$SOURCE_RUN
wandb_run_id=$WANDB_RUN_ID
python=$PYTHON_BIN
gpu_type=NVIDIA B300 SXM6 AC
world_size=1
local_batch_size=$TOTAL_BATCH_SIZE
effective_batch_size=$TOTAL_BATCH_SIZE
epochs=1
seed=0
precision=float32
learning_rate=4.0e-5
dictionary_learning_rate=4.0e-5
discriminator_learning_rate=4.0e-5
EOF

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

resume_args=()
init_args=(--load-path "$INIT_CHECKPOINT")
last_checkpoint="$(find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "$last_checkpoint" && -f "$last_checkpoint" ]]; then
  init_args=(--load-path "$last_checkpoint")
  resume_args=(--resume)
  status stage1 resuming "checkpoint=$last_checkpoint"
else
  status stage1 starting "ImageNet rFID 5.084 fine-tune; seed=0"
fi

(
  cd "$THIRD_PARTY"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
    main_stage1.py \
    --model-config "$CONFIG" \
    --result-path "$RUN_ROOT" \
    --seed 0 \
    "${init_args[@]}" \
    "${resume_args[@]}" \
    "dataset.root=$DATA_VIEW" \
    "experiment.batch_size=$TOTAL_BATCH_SIZE" \
    "experiment.total_batch_size=$TOTAL_BATCH_SIZE" \
    "experiment.source_run=$SOURCE_RUN"
) 2>&1 | tee -a "$RUN_ROOT/logs/stage1.log"

completed_checkpoint="$(find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$completed_checkpoint" || ! -f "$completed_checkpoint" ]]; then
  echo "Stage 1 exited without a last_model.pt checkpoint" >&2
  exit 1
fi
status stage1 complete "checkpoint=$completed_checkpoint"
trap - EXIT
