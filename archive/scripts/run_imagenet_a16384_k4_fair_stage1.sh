#!/usr/bin/env bash
# Clean, apples-to-apples ImageNet Stage-1 comparison against RQ-VAE 8x8x4.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party/rq-vae-transformer"
DEFAULT_CONFIG="$THIRD_PARTY/configs/imagenet256/stage1/in256-rqvae-laser-8x8-a16384-k4-fair.yaml"
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
BOTTLENECK_TYPE="${BOTTLENECK_TYPE:-laser}"
DEFAULT_PYTHON="/workspace/tmp/laser-eval-cc3m-venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
DATA_ROOT="${IMAGENET_ROOT:-/workspace/Projects/data/imagenet}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/imagenet-a16384-k4-fair-stage1-$STAMP}"
# The upstream entrypoint runs from $THIRD_PARTY. Resolve caller-provided paths
# now so relative paths keep referring to the caller's working directory.
CONFIG="$(realpath -m "$CONFIG")"
DATA_ROOT="$(realpath -m "$DATA_ROOT")"
RUN_ROOT="$(realpath -m "$RUN_ROOT")"
NPROC="${NPROC:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-0}"
LEARNING_RATE="${LEARNING_RATE:-4.0e-5}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-$LEARNING_RATE}"
DISCRIMINATOR_LEARNING_RATE="${DISCRIMINATOR_LEARNING_RATE:-$LEARNING_RATE}"
DISCRIMINATOR_MIN_LEARNING_RATE="${DISCRIMINATOR_MIN_LEARNING_RATE:-$MIN_LEARNING_RATE}"
DICTIONARY_UPDATE_MODE="${DICTIONARY_UPDATE_MODE:-gradient}"
if [[ "$DICTIONARY_UPDATE_MODE" == "alternating_residual" ]]; then
  DICTIONARY_LEARNING_RATE="${DICTIONARY_LEARNING_RATE:-not-applicable}"
else
  DICTIONARY_LEARNING_RATE="${DICTIONARY_LEARNING_RATE:-$LEARNING_RATE}"
fi
SOURCE_RUN="${SOURCE_RUN:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
REBASE_LEGACY_EPOCH_BOUNDARY="${REBASE_LEGACY_EPOCH_BOUNDARY:-0}"
PRECISION="${PRECISION:-float32}"
REBASE_PRECISION="${REBASE_PRECISION:-0}"
COMPUTE_RFID="${COMPUTE_RFID:-true}"
CHECKPOINT_UPLOAD="${WANDB_CHECKPOINT_UPLOAD:-1}"
RECOVERY_CKPT_FREQ_STEPS="${RECOVERY_CKPT_FREQ_STEPS:-250}"
RUN_VARIANT_TAG="${RUN_VARIANT_TAG:-paired-control}"
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
if [[ "$BOTTLENECK_TYPE" != "laser" && "$BOTTLENECK_TYPE" != "rq" ]]; then
  echo "BOTTLENECK_TYPE must be laser or rq" >&2
  exit 1
fi
if [[ "$DICTIONARY_UPDATE_MODE" != "gradient" && \
      "$DICTIONARY_UPDATE_MODE" != "alternating_residual" ]]; then
  echo "DICTIONARY_UPDATE_MODE must be gradient or alternating_residual" >&2
  exit 1
fi
if (( EPOCHS < 1 )); then
  echo "EPOCHS must be positive" >&2
  exit 1
fi
if [[ ! "$RECOVERY_CKPT_FREQ_STEPS" =~ ^[0-9]+$ ]]; then
  echo "RECOVERY_CKPT_FREQ_STEPS must be a non-negative integer" >&2
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
if (( LOCAL_BATCH_SIZE < 1 )); then
  echo "Local batch size must be positive" >&2
  exit 1
fi
if [[ "$PRECISION" != "float32" && "$PRECISION" != "bfloat16" ]]; then
  echo "PRECISION must be float32 or bfloat16" >&2
  exit 1
fi
if [[ "$REBASE_PRECISION" != "0" && "$REBASE_PRECISION" != "1" ]]; then
  echo "REBASE_PRECISION must be 0 or 1" >&2
  exit 1
fi
if [[ -n "$INIT_CHECKPOINT" ]]; then
  INIT_CHECKPOINT="$(realpath -m "$INIT_CHECKPOINT")"
  if [[ ! -f "$INIT_CHECKPOINT" ]]; then
    echo "Requested initializer checkpoint does not exist: $INIT_CHECKPOINT" >&2
    exit 1
  fi
fi
if [[ -n "$INIT_CHECKPOINT" && -n "$RESUME_CHECKPOINT" ]]; then
  echo "INIT_CHECKPOINT and RESUME_CHECKPOINT are mutually exclusive" >&2
  exit 1
fi

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
LASER_PREFLIGHT_ROOT="$ROOT" LASER_PREFLIGHT_CONFIG="$CONFIG" \
LASER_PREFLIGHT_BOTTLENECK_TYPE="$BOTTLENECK_TYPE" \
LASER_PREFLIGHT_DICTIONARY_UPDATE_MODE="$DICTIONARY_UPDATE_MODE" \
PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import inspect
import os
from pathlib import Path

import lmdb
import torch
from omegaconf import OmegaConf
import rqvae.img_datasets
import rqvae.models
import rqvae.optimizer
import rqvae.trainers
import rqvae.utils.setup
from rqvae.losses.vqgan.discriminator import NLayerDiscriminator
from rqvae.models.rqvae.modules import Decoder, Encoder
from src.models.dictionary_learner import DictionaryLearning
import src.rqvae_metrics

root = Path(os.environ["LASER_PREFLIGHT_ROOT"]).resolve()
third_party = root / "third_party" / "rq-vae-transformer"
config = OmegaConf.load(os.environ["LASER_PREFLIGHT_CONFIG"])
bottleneck_type = os.environ["LASER_PREFLIGHT_BOTTLENECK_TYPE"]
dictionary_update_mode = os.environ["LASER_PREFLIGHT_DICTIONARY_UPDATE_MODE"]
reference = OmegaConf.load(
    third_party / "configs" / "imagenet256" / "stage1" / "in256-rqvae-8x8x4.yaml"
)

expected_modules = (third_party / "rqvae" / "models" / "rqvae" / "modules.py").resolve()
expected_discriminator = (
    third_party / "rqvae" / "losses" / "vqgan" / "discriminator.py"
).resolve()
assert Path(inspect.getfile(Encoder)).resolve() == expected_modules
assert Path(inspect.getfile(Decoder)).resolve() == expected_modules
assert Path(inspect.getfile(NLayerDiscriminator)).resolve() == expected_discriminator

# Architecture and GAN behavior must stay identical to the original RQ-VAE
# recipe. Only the bottleneck implementation is intentionally exchanged.
assert OmegaConf.to_container(config.arch.ddconfig, resolve=True) == OmegaConf.to_container(
    reference.arch.ddconfig, resolve=True
)
assert OmegaConf.to_container(config.gan.disc.arch, resolve=True) == OmegaConf.to_container(
    reference.gan.disc.arch, resolve=True
)
assert OmegaConf.to_container(config.gan.loss, resolve=True) == OmegaConf.to_container(
    reference.gan.loss, resolve=True
)
assert list(config.arch.hparams.latent_shape) == list(reference.arch.hparams.latent_shape)
assert list(config.arch.hparams.code_shape) == list(reference.arch.hparams.code_shape)
assert int(config.arch.hparams.embed_dim) == int(reference.arch.hparams.embed_dim)
assert int(config.arch.hparams.n_embed) == int(reference.arch.hparams.n_embed)
assert str(config.arch.hparams.bottleneck_type).lower() == bottleneck_type
assert float(config.arch.hparams.latent_loss_weight) == float(
    reference.arch.hparams.latent_loss_weight
) == 0.25

# Match the upstream RQ objective: average cumulative-depth distortion first,
# then apply the single latent_loss_weight=0.25 multiplier. The trainable
# dictionary term receives the same outer weight as encoder commitment.
if bottleneck_type == "laser":
    assert bool(config.arch.hparams.progressive_loss)
    assert float(config.arch.hparams.commitment_cost) == 1.0
    bottleneck = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=2,
        commitment_cost=1.0,
        progressive_loss=True,
        dictionary_update_mode=dictionary_update_mode,
    )
    with torch.no_grad():
        bottleneck.dictionary.copy_(torch.eye(2))
    z = torch.tensor([[[[2.0]], [[1.0]]]], requires_grad=True)
    bottleneck(z)
    dictionary_loss = bottleneck._last_dictionary_loss_for_backward
    commitment_loss = bottleneck._last_commitment_loss
    objective = bottleneck._last_bottleneck_objective_for_backward
    if dictionary_update_mode == "alternating_residual":
        assert torch.allclose(objective.detach(), commitment_loss)
    else:
        assert torch.allclose(
            objective.detach(), dictionary_loss.detach() + commitment_loss
        )
    weighted_dictionary = float(config.arch.hparams.latent_loss_weight) * dictionary_loss
    weighted_commitment = float(config.arch.hparams.latent_loss_weight) * commitment_loss
    if dictionary_update_mode == "alternating_residual":
        assert torch.allclose(
            float(config.arch.hparams.latent_loss_weight) * objective,
            weighted_commitment,
        )
    else:
        assert torch.allclose(
            float(config.arch.hparams.latent_loss_weight) * objective,
            weighted_dictionary + weighted_commitment,
        )
    configured_mode = str(
        config.arch.hparams.get("dictionary_update_mode", "gradient")
    ).lower()
    assert configured_mode == dictionary_update_mode
    assert bottleneck.dictionary.requires_grad == (dictionary_update_mode == "gradient")
PY
if (( PREFLIGHT_ONLY )); then
  echo "Fair Stage-1 preflight passed for $BOTTLENECK_TYPE"
  exit 0
fi

mkdir -p "$RUN_ROOT/logs" "$ROOT/.cache/wandb" "$ROOT/.local/share/wandb" "$ROOT/wandb"
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"
if [[ ! -f "$RUN_ROOT/status.tsv" ]]; then
  printf 'time_utc\tphase\tstate\tdetail\n' > "$RUN_ROOT/status.tsv"
fi

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

if [[ "$BOTTLENECK_TYPE" == "laser" ]]; then
  if [[ "$DICTIONARY_UPDATE_MODE" == "alternating_residual" ]]; then
    bottleneck_objective="depth-averaged cumulative OMP commitment; dictionary fit is an alternating block step"
    dictionary_update="fixed-code residual least-squares with safeguarded relaxation"
  else
    bottleneck_objective="depth-averaged cumulative OMP dictionary+commitment"
    dictionary_update="adam-tangent-projected-normalized"
  fi
else
  bottleneck_objective="depth-averaged cumulative RQ commitment"
  dictionary_update="ema-shared-codebook"
fi

cat > "$RUN_ROOT/run.info" <<EOF
run_root=$RUN_ROOT
data_root=$DATA_ROOT
config=$CONFIG
wandb_run_id=$WANDB_RUN_ID
python=$PYTHON_BIN
world_size=$NPROC
local_batch_size=$LOCAL_BATCH_SIZE
effective_batch_size=$TOTAL_BATCH_SIZE
epochs=$EPOCHS
seed=$SEED
precision=$PRECISION
bottleneck_type=$BOTTLENECK_TYPE
encoder_decoder=unmodified upstream RQ-VAE
discriminator=unmodified upstream PatchGAN
bottleneck_objective=$bottleneck_objective
dictionary_update=$dictionary_update
dictionary_update_mode=$DICTIONARY_UPDATE_MODE
latent_loss_weight=0.25
main_learning_rate=$LEARNING_RATE
minimum_learning_rate=$MIN_LEARNING_RATE
dictionary_learning_rate=$DICTIONARY_LEARNING_RATE
discriminator_learning_rate=$DISCRIMINATOR_LEARNING_RATE
discriminator_minimum_learning_rate=$DISCRIMINATOR_MIN_LEARNING_RATE
source_run=$SOURCE_RUN
initializer_checkpoint=$INIT_CHECKPOINT
requested_resume_checkpoint=$RESUME_CHECKPOINT
rebase_legacy_epoch_boundary=$REBASE_LEGACY_EPOCH_BOUNDARY
rebase_precision=$REBASE_PRECISION
nccl_nvls_enable=${NCCL_NVLS_ENABLE:-default}
checkpoint_format=v5-model+optimizers+schedulers+rank-rng+loader-replay+partial-epoch-accumulator+compatibility-signature+lineage
recovery_checkpoint_frequency_steps=$RECOVERY_CKPT_FREQ_STEPS
training_variant=$RUN_VARIANT_TAG
rfid_backend=original-rqvae
compute_rfid=$COMPUTE_RFID
EOF

export PYTHONUNBUFFERED=1
export PYTHONPATH="$THIRD_PARTY:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-helloimlixin-rutgers}"
export WANDB_PROJECT="${WANDB_PROJECT:-laser}"
export WANDB_RUN_ID
export WANDB_NAME
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-imagenet-a16384-k4-fair-$STAMP}"
precision_tag="f32"
if [[ "$PRECISION" == "bfloat16" ]]; then
  precision_tag="bf16-mixed-fp32-omp"
fi
if [[ "$BOTTLENECK_TYPE" == "laser" ]]; then
  bottleneck_tags="laser,a16384,k4,progressive-rq-weighted-dictionary,${DICTIONARY_UPDATE_MODE}"
else
  bottleneck_tags="rq-control,a16384,d4,ema-codebook"
fi
schedule_tag="cosine-decay"
if [[ "$MIN_LEARNING_RATE" == "$LEARNING_RATE" && \
      "$DISCRIMINATOR_MIN_LEARNING_RATE" == "$DISCRIMINATOR_LEARNING_RATE" ]]; then
  schedule_tag="constant-lr-after-warmup"
fi
export WANDB_TAGS="stage1,imagenet,rqvae,${RUN_VARIANT_TAG},${schedule_tag},${bottleneck_tags},8x8x4,${precision_tag},effective-batch${TOTAL_BATCH_SIZE},full-gan-checkpoint,original-rqvae-rfid"
export WANDB_CHECKPOINT_UPLOAD="$CHECKPOINT_UPLOAD"
export WANDB_CACHE_DIR="$ROOT/.cache/wandb"
export WANDB_DATA_DIR="$ROOT/.local/share/wandb"
export WANDB_DIR="$ROOT/wandb"
export XDG_CACHE_HOME="$ROOT/.cache"
export LASER_VGG_LPIPS_DIR="$ROOT/vgg_lpips"
export LASER_VGG16_WEIGHTS="$ROOT/vgg_lpips/vgg16-397923af.pth"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

resume_args=()
source_run_args=()
model_args=()
if [[ "$BOTTLENECK_TYPE" == "laser" ]]; then
  model_args=("arch.hparams.dictionary_update_mode=$DICTIONARY_UPDATE_MODE")
  if [[ "$DICTIONARY_UPDATE_MODE" == "gradient" ]]; then
    model_args+=("arch.hparams.dict_learning_rate=$DICTIONARY_LEARNING_RATE")
  fi
fi
if [[ -n "$SOURCE_RUN" ]]; then
  source_run_args=("experiment.source_run=$SOURCE_RUN")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  last_checkpoint="$(realpath -m "$RESUME_CHECKPOINT")"
  if [[ ! -f "$last_checkpoint" ]]; then
    echo "Requested resume checkpoint does not exist: $last_checkpoint" >&2
    exit 1
  fi
else
  last_checkpoint="$(find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [[ -n "$last_checkpoint" && -f "$last_checkpoint" ]]; then
  resume_args=(--load-path "$last_checkpoint" --resume)
  if [[ "$REBASE_LEGACY_EPOCH_BOUNDARY" == "1" ]]; then
    resume_args+=(--rebase-legacy-epoch-boundary)
  elif [[ "$REBASE_LEGACY_EPOCH_BOUNDARY" != "0" ]]; then
    echo "REBASE_LEGACY_EPOCH_BOUNDARY must be 0 or 1" >&2
    exit 1
  fi
  if [[ "$REBASE_PRECISION" == "1" ]]; then
    resume_args+=(--rebase-precision)
  fi
  status stage1 resuming "checkpoint=$last_checkpoint"
elif [[ -n "$INIT_CHECKPOINT" ]]; then
  if [[ "$REBASE_LEGACY_EPOCH_BOUNDARY" != "0" || "$REBASE_PRECISION" != "0" ]]; then
    echo "Rebase flags apply only to an exact resume, not an initializer branch" >&2
    exit 1
  fi
  resume_args=(--load-path "$INIT_CHECKPOINT")
  status stage1 starting "initializer=$INIT_CHECKPOINT; fresh optimizer schedule"
else
  if [[ "$REBASE_LEGACY_EPOCH_BOUNDARY" != "0" ]]; then
    echo "Cannot rebase a legacy lineage without a resume checkpoint" >&2
    exit 1
  fi
  if [[ "$REBASE_PRECISION" != "0" ]]; then
    echo "Cannot rebase precision without a resume checkpoint" >&2
    exit 1
  fi
  status stage1 starting "clean seed=$SEED run"
fi

(
  cd "$THIRD_PARTY"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    main_stage1.py \
    --model-config "$CONFIG" \
    --result-path "$RUN_ROOT" \
    --seed "$SEED" \
    --precision "$PRECISION" \
    "${resume_args[@]}" \
    "${source_run_args[@]}" \
    "${model_args[@]}" \
    "dataset.root=$DATA_ROOT" \
    "experiment.batch_size=$LOCAL_BATCH_SIZE" \
    experiment.total_batch_size="$TOTAL_BATCH_SIZE" \
    experiment.epochs="$EPOCHS" \
    experiment.compute_rfid="$COMPUTE_RFID" \
    experiment.recovery_ckpt_freq_steps="$RECOVERY_CKPT_FREQ_STEPS" \
    optimizer.init_lr="$LEARNING_RATE" \
    optimizer.warmup.min_lr="$MIN_LEARNING_RATE" \
    gan.disc.optimizer.init_lr="$DISCRIMINATOR_LEARNING_RATE" \
    gan.disc.optimizer.warmup.min_lr="$DISCRIMINATOR_MIN_LEARNING_RATE" \
    experiment.rfid_backend=original-rqvae
) 2>&1 | tee -a "$RUN_ROOT/logs/stage1.log"

completed_checkpoint="$(find "$RUN_ROOT" -type f -name last_model.pt -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "$completed_checkpoint" || ! -f "$completed_checkpoint" ]]; then
  echo "Stage 1 exited without a last_model.pt checkpoint" >&2
  exit 1
fi
status stage1 complete "checkpoint=$completed_checkpoint"
trap - EXIT
