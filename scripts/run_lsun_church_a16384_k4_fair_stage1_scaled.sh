#!/usr/bin/env bash
# Four-H100 LSUN-Church Stage-1 fine-tune, matched to upstream RQ-VAE 8x8x4.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
CONFIG="$THIRD_PARTY/configs/lsun-church/stage1/church256-rqvae-laser-8x8-a16384-k4-fair.yaml"
REFERENCE_CONFIG="$THIRD_PARTY/configs/lsun-church/stage1/church256-rqvae-8x8x4.yaml"
DEFAULT_PYTHON="/workspace/tmp/laser-h100-venv/bin/python"
DEFAULT_INIT_CHECKPOINT="$ROOT/outputs/imagenet-a16384-k4-compound-thirdparty-20260811-181448/stage1/in256-rqvae-laser-8x8-a16384-k4/12082026_040704/last_model.pt"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
DATA_ROOT="${LSUN_ROOT:-/workspace/Projects/data/lsun}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$DEFAULT_INIT_CHECKPOINT}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-church-a16384-k4-fair-stage1-b256-$STAMP}"
NPROC="${NPROC:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
BASE_TOTAL_BATCH_SIZE="${BASE_TOTAL_BATCH_SIZE:-128}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-4.0e-5}"
LEARNING_RATE="${LEARNING_RATE:-8.0e-5}"
DICTIONARY_LEARNING_RATE="${DICTIONARY_LEARNING_RATE:-$LEARNING_RATE}"
SOURCE_RUN="${SOURCE_RUN:-helloimlixin-rutgers/laser/imga16384k4s1-20260811181448}"
WANDB_RUN_ID="${WANDB_RUN_ID:-lsunchurcha16384k4s1fair-b256-${STAMP//-/}}"
WANDB_NAME="${WANDB_NAME:-lsun-church-a16384-k4-fair-stage1-b256-$STAMP}"
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
for required in "$CONFIG" "$REFERENCE_CONFIG" "$INIT_CHECKPOINT" \
  "$DATA_ROOT/church_outdoor_train_lmdb/data.mdb" \
  "$DATA_ROOT/church_outdoor_val_lmdb/data.mdb" \
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
if (( LOCAL_BATCH_SIZE < 1 )); then
  echo "Local batch size must be positive" >&2
  exit 1
fi

# These components must remain exactly upstream for a comparison that isolates
# only the RQ versus LASER bottleneck.
if ! git -C "$THIRD_PARTY" diff --quiet upstream/main -- \
  rqvae/models/rqvae/modules.py \
  rqvae/losses/vqgan/discriminator.py \
  rqvae/losses/vqgan/gan_loss.py \
  configs/lsun-church/stage1/church256-rqvae-8x8x4.yaml; then
  echo "Encoder/decoder, discriminator, GAN loss, or Church reference config differs from upstream" >&2
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

LASER_PREFLIGHT_ROOT="$ROOT" LASER_PREFLIGHT_CONFIG="$CONFIG" \
LASER_PREFLIGHT_REFERENCE="$REFERENCE_CONFIG" \
LASER_PREFLIGHT_CHECKPOINT="$INIT_CHECKPOINT" \
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import inspect
import os
from pathlib import Path

import lmdb
import torch
from omegaconf import OmegaConf
from rqvae.losses.vqgan.discriminator import NLayerDiscriminator
from rqvae.models.rqvae.modules import Decoder, Encoder
from src.models.dictionary_learner import DictionaryLearning

root = Path(os.environ["LASER_PREFLIGHT_ROOT"]).resolve()
third_party = root / "third_party" / "rq-vae-transformer"
config = OmegaConf.load(os.environ["LASER_PREFLIGHT_CONFIG"])
reference = OmegaConf.load(os.environ["LASER_PREFLIGHT_REFERENCE"])

assert Path(inspect.getfile(Encoder)).resolve() == (
    third_party / "rqvae" / "models" / "rqvae" / "modules.py"
).resolve()
assert Path(inspect.getfile(Decoder)).resolve() == (
    third_party / "rqvae" / "models" / "rqvae" / "modules.py"
).resolve()
assert Path(inspect.getfile(NLayerDiscriminator)).resolve() == (
    third_party / "rqvae" / "losses" / "vqgan" / "discriminator.py"
).resolve()

assert OmegaConf.to_container(config.arch.ddconfig, resolve=True) == OmegaConf.to_container(
    reference.arch.ddconfig, resolve=True
)
assert OmegaConf.to_container(config.gan.disc.arch, resolve=True) == OmegaConf.to_container(
    reference.gan.disc.arch, resolve=True
)
assert OmegaConf.to_container(config.gan.loss, resolve=True) == OmegaConf.to_container(
    reference.gan.loss, resolve=True
)
for key in ("latent_shape", "code_shape"):
    assert list(config.arch.hparams[key]) == list(reference.arch.hparams[key])
for key in ("embed_dim", "n_embed"):
    assert int(config.arch.hparams[key]) == int(reference.arch.hparams[key])
assert int(config.experiment.epochs) == int(reference.experiment.epochs) == 1
assert int(config.optimizer.warmup.epoch) == int(reference.optimizer.warmup.epoch) == 0
assert bool(config.arch.hparams.progressive_loss)
assert float(config.arch.hparams.commitment_cost) == 1.0
assert float(config.arch.hparams.latent_loss_weight) == float(
    reference.arch.hparams.latent_loss_weight
) == 0.25

bottleneck = DictionaryLearning(
    num_embeddings=2,
    embedding_dim=2,
    sparsity_level=2,
    commitment_cost=1.0,
    progressive_loss=True,
)
with torch.no_grad():
    bottleneck.dictionary.copy_(torch.eye(2))
z = torch.tensor([[[[2.0]], [[1.0]]]], requires_grad=True)
bottleneck(z)
objective = bottleneck._last_bottleneck_objective_for_backward
expected = (
    bottleneck._last_dictionary_loss_for_backward
    + bottleneck._last_commitment_loss
)
assert torch.allclose(objective.detach(), expected.detach())

# Check the trusted local copy of the requested W&B ImageNet initializer without
# materializing its full optimizer state. Model/discriminator shapes are
# validated again by the actual load in main_stage1.py.
checkpoint_path = Path(os.environ["LASER_PREFLIGHT_CHECKPOINT"])
assert checkpoint_path.stat().st_size > 1_000_000_000
PY
if (( PREFLIGHT_ONLY )); then
  echo "Fair LSUN-Church Stage-1 preflight passed"
  exit 0
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/data/lsun/church" \
  "$ROOT/.cache/wandb" "$ROOT/.local/share/wandb" "$ROOT/wandb"
# Preserve the upstream loader and give it the category-nested layout it
# expects, while leaving the verified source LMDBs untouched.
ln -sfn "$DATA_ROOT/church_outdoor_train_lmdb" \
  "$RUN_ROOT/data/lsun/church/church_outdoor_train_lmdb"
ln -sfn "$DATA_ROOT/church_outdoor_val_lmdb" \
  "$RUN_ROOT/data/lsun/church/church_outdoor_val_lmdb"
DATA_VIEW="$RUN_ROOT/data/lsun"

# Build and validate the upstream LMDB key cache once before four DDP ranks
# open the dataset concurrently.
LASER_LSUN_DATA_VIEW="$DATA_VIEW" \
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import os

from rqvae.img_datasets.lsun import LSUNClass

dataset = LSUNClass(os.environ["LASER_LSUN_DATA_VIEW"], category_name="church")
assert len(dataset) == 126_227, len(dataset)
image, target = dataset[0]
assert image.mode == "RGB"
assert target == 0
print(f"validated LSUN-Church train LMDB: {len(dataset)} images")
PY

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
data_view=$DATA_VIEW
config=$CONFIG
reference_config=$REFERENCE_CONFIG
initializer=$INIT_CHECKPOINT
source_run=$SOURCE_RUN
wandb_run_id=$WANDB_RUN_ID
python=$PYTHON_BIN
gpu_type=NVIDIA H100 80GB HBM3
world_size=$NPROC
local_batch_size=$LOCAL_BATCH_SIZE
effective_batch_size=$TOTAL_BATCH_SIZE
upstream_effective_batch_size=$BASE_TOTAL_BATCH_SIZE
batch_scale=$(awk -v actual="$TOTAL_BATCH_SIZE" -v base="$BASE_TOTAL_BATCH_SIZE" 'BEGIN { printf "%.6g", actual/base }')
epochs=1
seed=0
precision=float32
encoder_decoder=unmodified upstream RQ-VAE
discriminator=unmodified upstream PatchGAN
latent_shape=8x8x256
code_shape=8x8x4
dictionary_size=16384
sparsity=4
commitment=depth-averaged cumulative OMP reconstruction
commitment_cost=1.0
latent_loss_weight=0.25
effective_dictionary_loss_weight=0.25
effective_commitment_weight=0.25
upstream_learning_rate=$BASE_LEARNING_RATE
main_learning_rate=$LEARNING_RATE
dictionary_learning_rate=$DICTIONARY_LEARNING_RATE
discriminator_learning_rate=$LEARNING_RATE
learning_rate_scaling=linear_with_effective_batch
checkpoint_format=model+main_optimizer+main_scheduler+discriminator+discriminator_optimizer+discriminator_scheduler
rfid_backend=original-rqvae
wandb_mode=online
nccl_nvls_enable=${NCCL_NVLS_ENABLE:-0}
EOF

export PYTHONUNBUFFERED=1
export PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
export WANDB_RUN_ID
export WANDB_NAME
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-lsun-church-a16384-k4-fair-scaled-b256-$STAMP}"
export WANDB_TAGS="stage1,lsun-church,laser,rqvae,fair-comparison,a16384,k4,8x8x4,f32,effective-batch${TOTAL_BATCH_SIZE},linear-lr-scale,imagenet-finetune,online"
export WANDB_CHECKPOINT_UPLOAD=1
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export LASER_VGG_LPIPS_DIR="$ROOT/vgg_lpips"
export LASER_VGG16_WEIGHTS="$ROOT/vgg_lpips/vgg16-397923af.pth"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# This H100 host exposes NVLink but its driver rejects NCCL NVLS setup with
# CUDA error 401. Ring/tree collectives retain exact DDP semantics.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

resume_args=()
init_args=(--load-path "$INIT_CHECKPOINT")
last_checkpoint="$(find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "$last_checkpoint" && -f "$last_checkpoint" ]]; then
  init_args=(--load-path "$last_checkpoint")
  resume_args=(--resume)
  status stage1 resuming "checkpoint=$last_checkpoint"
else
  status stage1 starting "ImageNet fine-tune; seed=0"
fi

(
  cd "$THIRD_PARTY"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    main_stage1.py \
    --model-config "$CONFIG" \
    --result-path "$RUN_ROOT" \
    --seed 0 \
    "${init_args[@]}" \
    "${resume_args[@]}" \
    "dataset.root=$DATA_VIEW" \
    "experiment.batch_size=$LOCAL_BATCH_SIZE" \
    experiment.total_batch_size="$TOTAL_BATCH_SIZE" \
    experiment.source_run="$SOURCE_RUN" \
    optimizer.init_lr="$LEARNING_RATE" \
    optimizer.warmup.min_lr="$LEARNING_RATE" \
    arch.hparams.dict_learning_rate="$DICTIONARY_LEARNING_RATE" \
    gan.disc.optimizer.init_lr="$LEARNING_RATE" \
    gan.disc.optimizer.warmup.min_lr="$LEARNING_RATE" \
    experiment.rfid_backend=original-rqvae
) 2>&1 | tee -a "$RUN_ROOT/logs/stage1.log"

completed_checkpoint="$(find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$completed_checkpoint" || ! -f "$completed_checkpoint" ]]; then
  echo "Stage 1 exited without a last_model.pt checkpoint" >&2
  exit 1
fi
status stage1 complete "checkpoint=$completed_checkpoint"
trap - EXIT
