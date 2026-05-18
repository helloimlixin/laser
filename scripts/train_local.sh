#!/usr/bin/env bash
# Local / lab launcher: vision patch p8 VQGAN-style stage1 + full-grid GPT stage2
# (same Hydra overrides as scripts/launch_vision_patch_gpt_fullgrid_sweep.sh).
#
# Usage:
#   export FFHQ_DIR=/path/to/ffhq          # if CASE=ffhq (default)
#   export CELEBAHQ_DIR=/path/to/celebahq_packed_256  # if CASE=celebahq
#   bash scripts/train_local.sh
#
# Optional env:
#   CASE=ffhq|celebahq   (default ffhq)
#   CUDA_VISIBLE_DEVICES=0,1
#   GPUS=2               (must match visible GPU count; used as train.devices / train_ar.devices)
#   PYTHON_BIN=python3
#   STAGE1_EPOCHS=10 STAGE2_EPOCHS=10
#   RUN_ROOT=/path/to/run_dir   (default: <repo>/runs/lab_patch_gpt_<case>_<timestamp>)
#   WANDB_PROJECT=laser-debugging
#   WANDB_GROUP=my-group
#   STAGE1_ONLY=1        (train stage1 only; skip cache + stage2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CASE="${CASE:-ffhq}"
GPUS="${GPUS:-2}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE1_ONLY="${STAGE1_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

PROJECT="${WANDB_PROJECT:-laser-debugging}"
STAMP="$(date +%Y%m%d_%H%M%S)"
WANDB_GROUP="${WANDB_GROUP:-lab-vision-patch-p8-fullgrid-${CASE}-s1-${STAGE1_EPOCHS}-s2-${STAGE2_EPOCHS}-${STAMP}}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/lab_patch_gpt_${CASE}_${STAMP}}"

FFHQ_DIR="${FFHQ_DIR:-$HOME/datasets/ffhq}"
CELEBAHQ_DIR="${CELEBAHQ_DIR:-$HOME/datasets/celebahq_packed_256}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$CASE" == "ffhq" ]]; then
  DATA_CONFIG="ffhq"
  DATA_DIR="$FFHQ_DIR"
elif [[ "$CASE" == "celebahq" ]]; then
  DATA_CONFIG="celebahq"
  DATA_DIR="$CELEBAHQ_DIR"
else
  echo "ERROR: CASE must be ffhq or celebahq, got: $CASE" >&2
  exit 2
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: dataset directory missing for CASE=$CASE: $DATA_DIR" >&2
  echo "Set FFHQ_DIR or CELEBAHQ_DIR to your local dataset root." >&2
  exit 2
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: need Python >= 3.10 (got $PYTHON_BIN)." >&2
  exit 2
fi

if ! "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= int('$GPUS')" 2>/dev/null; then
  echo "ERROR: need CUDA and at least GPUS=$GPUS visible devices (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)." >&2
  exit 2
fi

TRAIN_STRATEGY="ddp"
if [[ "${GPUS}" -le 1 ]]; then
  TRAIN_STRATEGY="auto"
fi

STAGE1_DIR="$RUN_ROOT/stage1"
STAGE2_DIR="$RUN_ROOT/stage2"
TOKEN_CACHE="$RUN_ROOT/token_cache.pt"
mkdir -p "$STAGE1_DIR/wandb" "$STAGE2_DIR/wandb"

echo "Repo:       $REPO_ROOT"
echo "Run root:   $RUN_ROOT"
echo "Case:       $CASE  data_dir=$DATA_DIR"
echo "GPUs:       $GPUS  strategy=$TRAIN_STRATEGY  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Epochs:     stage1=$STAGE1_EPOCHS  stage2=$STAGE2_EPOCHS"
echo "W&B group:  $WANDB_GROUP  project=$PROJECT"
echo ""

STAGE1_ARGS=(
  "output_dir=${STAGE1_DIR}"
  "model=laser"
  "data=${DATA_CONFIG}"
  "data.data_dir=${DATA_DIR}"
  "data.image_size=256"
  "data.batch_size=4"
  "data.eval_batch_size=4"
  "data.num_workers=4"
  "seed=42"
  "train.max_epochs=${STAGE1_EPOCHS}"
  "train.max_steps=-1"
  "train.limit_train_batches=1.0"
  "train.limit_val_batches=1.0"
  "train.limit_test_batches=1.0"
  "train.log_every_n_steps=50"
  "train.run_test_after_fit=false"
  "train.devices=${GPUS}"
  "train.strategy=${TRAIN_STRATEGY}"
  "train.precision=bf16-mixed"
  "train.accelerator=gpu"
  "train.gradient_clip_val=1.0"
  "train.val_check_interval=1.0"
  "train.warmup_steps=1000"
  "train.min_lr_ratio=0.05"
  "train.learning_rate=5.0e-5"
  "model.compute_fid=false"
  "model.out_tanh=true"
  "model.log_images_every_n_steps=0"
  "model.enable_val_latent_visuals=true"
  "model.backbone=vqgan"
  "model.num_downsamples=2"
  "model.channel_multipliers=[1,1,2]"
  "model.num_hiddens=160"
  "model.num_residual_blocks=3"
  "model.num_residual_hiddens=80"
  "model.backbone_latent_channels=160"
  "model.embedding_dim=32"
  "model.patch_based=true"
  "model.patch_size=8"
  "model.patch_stride=8"
  "model.patch_reconstruction=tile"
  "model.num_embeddings=65536"
  "model.sparsity_level=16"
  "model.attn_resolutions=[]"
  "model.use_mid_attention=true"
  "model.decoder_extra_residual_layers=2"
  "model.bottleneck_loss_weight=0.75"
  "model.commitment_cost=1.0"
  "model.dict_learning_rate=1.0e-4"
  "model.coef_max=16.0"
  "model.bounded_omp_refine_steps=16"
  "model.sparsity_reg_weight=0.0"
  "model.recon_mse_weight=0.5"
  "model.recon_l1_weight=0.5"
  "model.recon_edge_weight=0.0"
  "model.perceptual_weight=0.10"
  "model.perceptual_start_step=0"
  "model.perceptual_warmup_steps=1000"
  "wandb.project=${PROJECT}"
  "wandb.group=${WANDB_GROUP}"
  "wandb.name=${CASE}-stage1-autoencoder"
  "wandb.tags=[lab,laser,${CASE},stage1,autoencoder,vqgan-patch-p8]"
  "wandb.append_timestamp=false"
  "wandb.save_dir=${STAGE1_DIR}/wandb"
)

echo "=== Stage 1: autoencoder ($CASE) ==="
"$PYTHON_BIN" train_stage1_autoencoder.py "${STAGE1_ARGS[@]}"

if [[ "$STAGE1_ONLY" == "1" || "$STAGE1_ONLY" == "true" ]]; then
  echo "=== STAGE1_ONLY set; skipping cache + stage 2 ==="
  exit 0
fi

CKPT="$(find "$STAGE1_DIR" -path '*/final.ckpt' -type f 2>/dev/null | sort | tail -1)"
if [[ -z "$CKPT" ]]; then
  CKPT="$(find "$STAGE1_DIR" -path '*/last.ckpt' -type f 2>/dev/null | sort | tail -1)"
fi
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no stage-1 checkpoint under $STAGE1_DIR" >&2
  exit 1
fi
echo "Using checkpoint: $CKPT"

echo "=== Token cache extraction ($CASE) ==="
"$PYTHON_BIN" cache.py \
  --stage1-checkpoint "$CKPT" \
  --output-path "$TOKEN_CACHE" \
  --dataset "$CASE" \
  --data-dir "$DATA_DIR" \
  --image-size 256 \
  --batch-size 8 \
  --num-workers 4 \
  --seed 42 \
  --max-items 0 \
  --model-type laser \
  --coeff-bins 512 \
  --coeff-max 16.0

STAGE2_ARGS=(
  "token_cache_path=${TOKEN_CACHE}"
  "output_dir=${STAGE2_DIR}"
  "seed=42"
  "ar.max_steps=-1"
  "train_ar.max_epochs=${STAGE2_EPOCHS}"
  "train_ar.batch_size=2"
  "train_ar.max_items=0"
  "train_ar.limit_train_batches=1.0"
  "train_ar.limit_val_batches=1.0"
  "train_ar.limit_test_batches=1.0"
  "train_ar.log_every_n_steps=50"
  "train_ar.sample_every_n_epochs=10"
  "train_ar.sample_log_to_wandb=true"
  "train_ar.sample_num_images=4"
  "train_ar.generation_metric_num_samples=0"
  "train_ar.compute_generation_fid=false"
  "train_ar.compute_audio_generation_metrics=false"
  "train_ar.run_test_after_fit=false"
  "train_ar.save_final_samples_after_fit=false"
  "train_ar.sample_temperature=0.9"
  "train_ar.sample_top_k=128"
  "train_ar.devices=${GPUS}"
  "train_ar.strategy=${TRAIN_STRATEGY}"
  "train_ar.precision=bf16-mixed"
  "train_ar.accelerator=gpu"
  "data.dataset=${CASE}"
  "data.data_dir=${DATA_DIR}"
  "data.image_size=256"
  "data.num_workers=4"
  "ar.type=gpt"
  "ar.window_sites=16"
  "ar.n_global_spatial_tokens=8"
  "ar.d_model=512"
  "ar.n_heads=8"
  "ar.n_layers=10"
  "ar.d_ff=2048"
  "ar.learning_rate=3.0e-4"
  "ar.warmup_steps=1000"
  "ar.min_lr_ratio=0.05"
  "ar.coeff_loss_type=auto"
  "ar.coeff_loss_weight=1.0"
  "train_ar.crop_h_sites=0"
  "train_ar.crop_w_sites=0"
  "wandb.project=${PROJECT}"
  "wandb.group=${WANDB_GROUP}"
  "wandb.name=${CASE}-stage2-transformer"
  "wandb.tags=[lab,laser,${CASE},stage2,transformer,gpt-fullgrid]"
  "wandb.append_timestamp=false"
  "wandb.save_dir=${STAGE2_DIR}/wandb"
)

echo "=== Stage 2: GPT prior ($CASE) ==="
"$PYTHON_BIN" train_stage2_prior.py "${STAGE2_ARGS[@]}"

echo "Done. Outputs under: $RUN_ROOT"
