#!/bin/bash
# Sweep: CelebA-HQ 256x256 with VQGAN-level encoder/decoder capacity.
#
# Based on the promising CIFAR-10 run f70yglfn (src_pf_p4s4_k16_d4k_stage2),
# scaled up for 256x256 images with increased encoder/decoder capacity.
#
# Full 3-stage pipeline: stage 1 (autoencoder) + token extraction + stage 2 (AR prior).
# Dictionary atom animation enabled (enable_val_latent_visuals=true).
# This launcher is configured to produce an 8x8 latent grid from 256x256 inputs.
#
# Axes explored:
#   - Patch layout: p4s4 (tile), p8s8 (tile), p4s2 (hann), p8s4 (hann), non-patched
#   - Dictionary size: K=1024, 4096, 8192, 16384
#   - Sparsity: k=8, 16, 24, 32
#   - Embedding dim: d=4, 8, 16
#   - Coefficient bound: coef_max=3, 5, 8
#   - Encoder/decoder capacity: moderate-by-default so 256x256 stage-1 fits on
#     current L40S nodes without sharding; larger settings can still be passed
#     explicitly through env overrides.
#
# Legacy usage:
#   ./scripts/legacy/sweeps/sweep_celebahq256_vqgan_cap.sh
#   CASE_FILTER=p4s4_k16_d4k ./scripts/legacy/sweeps/sweep_celebahq256_vqgan_cap.sh
#   DRY_RUN=1 ./scripts/legacy/sweeps/sweep_celebahq256_vqgan_cap.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# -- Cluster ----------------------------------------------------------------
if [[ "${PARTITION:-auto}" == "auto" ]]; then
  gpu_idle=$(sinfo -p gpu -h -o '%A' 2>/dev/null | awk -F/ '{print $2}' || echo 0)
  gpu_rh_idle=$(sinfo -p gpu-redhat -h -o '%A' 2>/dev/null | awk -F/ '{print $2}' || echo 0)
  cgpu_idle=$(sinfo -p cgpu -h -o '%A' 2>/dev/null | awk -F/ '{print $2}' || echo 0)
  if (( gpu_rh_idle >= gpu_idle && gpu_rh_idle >= cgpu_idle )); then
    PARTITION="gpu-redhat"
  elif (( gpu_idle >= cgpu_idle )); then
    PARTITION="gpu"
  else
    PARTITION="cgpu"
  fi
  echo "[Auto] selected partition=$PARTITION (gpu=$gpu_idle, gpu-redhat=$gpu_rh_idle, cgpu=$cgpu_idle idle)"
fi
PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
NODES="${NODES:-1}"
GPUS="${GPUS:-2}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-320000}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-}"
TRAIN_AR_STRATEGY="${TRAIN_AR_STRATEGY:-}"
if (( NODES > 1 )); then
  TASKS_PER_NODE="$GPUS"
  TOTAL_TASKS="$((NODES * GPUS))"
  CPUS_PER_TASK="$(((CPUS + GPUS - 1) / GPUS))"
else
  TASKS_PER_NODE="1"
  TOTAL_TASKS="$NODES"
  CPUS_PER_TASK="$CPUS"
fi
if [[ -z "$TRAIN_STRATEGY" ]]; then
  if (( NODES > 1 || GPUS > 1 )); then
    TRAIN_STRATEGY="ddp"
  else
    TRAIN_STRATEGY="auto"
  fi
fi
if [[ -z "$TRAIN_AR_STRATEGY" ]]; then
  if (( NODES > 1 || GPUS > 1 )); then
    TRAIN_AR_STRATEGY="ddp"
  else
    TRAIN_AR_STRATEGY="auto"
  fi
fi

# -- Data -------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DATASET="${DATASET:-celeba}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# -- Stage 1 defaults -------------------------------------------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE1_LR="${STAGE1_LR:-1.5e-4}"
DICT_LR="${DICT_LR:-2.5e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.0}"
NUM_DOWNSAMPLES="${NUM_DOWNSAMPLES:-5}"
EXPECTED_LATENT_SIDE="${EXPECTED_LATENT_SIDE:-8}"
STAGE1_VARIATIONAL_COEFFS="${STAGE1_VARIATIONAL_COEFFS:-false}"
STAGE1_VARIATIONAL_COEFF_KL_WEIGHT="${STAGE1_VARIATIONAL_COEFF_KL_WEIGHT:-0.0}"
STAGE1_VARIATIONAL_COEFF_PRIOR_STD="${STAGE1_VARIATIONAL_COEFF_PRIOR_STD:-0.25}"
STAGE1_VARIATIONAL_COEFF_MIN_STD="${STAGE1_VARIATIONAL_COEFF_MIN_STD:-0.01}"
# Strengthen global context with one extra 32x32 attention stage while keeping
# the expensive 64x64 path disabled.
ATTN_RESOLUTIONS="${ATTN_RESOLUTIONS:-[32,16,8]}"
CHANNEL_MULTIPLIERS="${CHANNEL_MULTIPLIERS:-}"
BACKBONE_LATENT_CHANNELS="${BACKBONE_LATENT_CHANNELS:-}"
MAX_CH_MULT="${MAX_CH_MULT:-2}"
DECODER_EXTRA_RESIDUAL_LAYERS="${DECODER_EXTRA_RESIDUAL_LAYERS:-1}"
USE_MID_ATTENTION="${USE_MID_ATTENTION:-true}"
DEFAULT_NUM_HIDDENS="${DEFAULT_NUM_HIDDENS:-128}"
DEFAULT_NUM_RES_BLOCKS="${DEFAULT_NUM_RES_BLOCKS:-2}"
DEFAULT_NUM_RES_HIDDENS="${DEFAULT_NUM_RES_HIDDENS:-64}"
S1_LOG_IMAGES_EVERY="${S1_LOG_IMAGES_EVERY:-100}"
S1_DIAG_LOG_INTERVAL="${S1_DIAG_LOG_INTERVAL:-100}"
S1_ENABLE_VAL_LATENT_VISUALS="${S1_ENABLE_VAL_LATENT_VISUALS:-true}"
S1_COMPUTE_FID="${S1_COMPUTE_FID:-false}"
S1_FID_FEATURE="${S1_FID_FEATURE:-2048}"
S1_VAL_CHECK_INTERVAL="${S1_VAL_CHECK_INTERVAL:-0.5}"
S1_LOG_EVERY_N_STEPS="${S1_LOG_EVERY_N_STEPS:-25}"

# -- Stage 2 defaults -------------------------------------------------------
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
STAGE2_LR="${STAGE2_LR:-3e-4}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-2000}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.01}"
COEFF_BINS="${COEFF_BINS:-256}"
COEFF_QUANTIZATION="${COEFF_QUANTIZATION:-uniform}"
COEFF_MU="${COEFF_MU:-0.0}"
SAMPLE_TEMP="${SAMPLE_TEMP:-0.5}"
# AR model matches the proven CIFAR-10 config from f70yglfn
AR_D_MODEL="${AR_D_MODEL:-512}"
AR_N_HEADS="${AR_N_HEADS:-8}"
AR_N_LAYERS="${AR_N_LAYERS:-12}"
AR_D_FF="${AR_D_FF:-1024}"
AR_AUTOREGRESSIVE_COEFFS="${AR_AUTOREGRESSIVE_COEFFS:-true}"
AR_COEFF_LOSS_TYPE="${AR_COEFF_LOSS_TYPE:-gt_atom_recon_mse}"
AR_COEFF_LOSS_WEIGHT="${AR_COEFF_LOSS_WEIGHT:-0.1}"
AR_SAMPLE_COEFF_MODE="${AR_SAMPLE_COEFF_MODE:-mean}"
AR_SAMPLE_COEFF_TEMPERATURE="${AR_SAMPLE_COEFF_TEMPERATURE:-null}"
S2_SAMPLE_EVERY_N_EPOCHS="${S2_SAMPLE_EVERY_N_EPOCHS:-1}"
S2_LOG_RECON_EVERY_N_STEPS="${S2_LOG_RECON_EVERY_N_STEPS:-200}"
S2_SAMPLE_NUM_IMAGES="${S2_SAMPLE_NUM_IMAGES:-16}"
S2_LOG_EVERY_N_STEPS="${S2_LOG_EVERY_N_STEPS:-25}"

# -- Output / logging -------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_vqgan_cap_sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_PREFIX="${RUN_PREFIX:-chq256_vc}"
JOB_PREFIX="${JOB_PREFIX:-chq256vc}"

CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"
OVERRIDE_NUM_EMBEDDINGS="${OVERRIDE_NUM_EMBEDDINGS:-}"
OVERRIDE_SPARSITY_LEVEL="${OVERRIDE_SPARSITY_LEVEL:-}"
OVERRIDE_EMBEDDING_DIM="${OVERRIDE_EMBEDDING_DIM:-}"

# -- Container (Amarel) -----------------------------------------------------
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_src_py311}"

# -- Pre-flight --------------------------------------------------------------
echo "=== Login node GPU check ==="
nvidia-smi 2>/dev/null || echo "(no GPUs on login node -- compute nodes will have them)"
echo ""

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  echo "Expected packed npy at: $DATA_DIR/celeba_${IMAGE_SIZE}x${IMAGE_SIZE}_rgb_uint8.npy" >&2
  exit 1
fi

downsample_factor=$((1 << NUM_DOWNSAMPLES))
if (( IMAGE_SIZE % downsample_factor != 0 )); then
  echo "IMAGE_SIZE=$IMAGE_SIZE is not divisible by 2^NUM_DOWNSAMPLES=$downsample_factor" >&2
  exit 1
fi
latent_side=$((IMAGE_SIZE / downsample_factor))
if (( latent_side != EXPECTED_LATENT_SIDE )); then
  echo "Expected ${EXPECTED_LATENT_SIDE}x${EXPECTED_LATENT_SIDE} latent grid, got ${latent_side}x${latent_side}." >&2
  echo "Adjust NUM_DOWNSAMPLES or EXPECTED_LATENT_SIDE before launching." >&2
  exit 1
fi
if [[ -z "$CHANNEL_MULTIPLIERS" ]]; then
  case "$NUM_DOWNSAMPLES" in
    0) CHANNEL_MULTIPLIERS='[1]' ;;
    1) CHANNEL_MULTIPLIERS='[1,2]' ;;
    2) CHANNEL_MULTIPLIERS='[1,2,4]' ;;
    3) CHANNEL_MULTIPLIERS='[1,2,2,4]' ;;
    4) CHANNEL_MULTIPLIERS='[1,1,2,3,4]' ;;
    5) CHANNEL_MULTIPLIERS='[1,1,2,3,4,4]' ;;
    *)
      echo "No default channel schedule for NUM_DOWNSAMPLES=$NUM_DOWNSAMPLES" >&2
      echo "Set CHANNEL_MULTIPLIERS explicitly." >&2
      exit 1
      ;;
  esac
fi
if [[ -z "$BACKBONE_LATENT_CHANNELS" ]]; then
  if (( latent_side <= 8 )); then
    BACKBONE_LATENT_CHANNELS="256"
  else
    BACKBONE_LATENT_CHANNELS="128"
  fi
fi
echo "[Sweep] image_size=$IMAGE_SIZE -> latent=${latent_side}x${latent_side} via num_downsamples=$NUM_DOWNSAMPLES"
echo "[Sweep] attention resolutions=$ATTN_RESOLUTIONS, channel_multipliers=$CHANNEL_MULTIPLIERS, backbone_latent_channels=$BACKBONE_LATENT_CHANNELS, max_ch_mult=$MAX_CH_MULT, mid_attention=$USE_MID_ATTENTION"
echo "[Sweep] stage1 coeffs: variational=$STAGE1_VARIATIONAL_COEFFS kl_weight=$STAGE1_VARIATIONAL_COEFF_KL_WEIGHT prior_std=$STAGE1_VARIATIONAL_COEFF_PRIOR_STD min_std=$STAGE1_VARIATIONAL_COEFF_MIN_STD"
echo "[Sweep] stage1 visuals: image_every=$S1_LOG_IMAGES_EVERY diag_every=$S1_DIAG_LOG_INTERVAL val_visuals=$S1_ENABLE_VAL_LATENT_VISUALS fid=$S1_COMPUTE_FID"
echo "[Sweep] token cache: coeff_bins=$COEFF_BINS quantization=$COEFF_QUANTIZATION mu=$COEFF_MU"
echo "[Sweep] stage2 coeffs: autoregressive=$AR_AUTOREGRESSIVE_COEFFS loss=$AR_COEFF_LOSS_TYPE weight=$AR_COEFF_LOSS_WEIGHT sample_mode=$AR_SAMPLE_COEFF_MODE sample_temp=$AR_SAMPLE_COEFF_TEMPERATURE"
echo "[Sweep] stage2 visuals: sample_every=$S2_SAMPLE_EVERY_N_EPOCHS recon_every=$S2_LOG_RECON_EVERY_N_STEPS sample_images=$S2_SAMPLE_NUM_IMAGES"
echo "[Sweep] cluster: partition=$PARTITION nodes=$NODES gpus_per_node=$GPUS tasks_per_node=$TASKS_PER_NODE total_tasks=$TOTAL_TASKS cpus_per_task=$CPUS_PER_TASK"

mkdir -p "$OUT_ROOT"

# -- Cases: name | patch_based | num_embeddings | sparsity | embedding_dim | patch_size | patch_stride | patch_recon | coef_max | num_hiddens | num_res_blocks | num_res_hiddens
#
# Use "-" to inherit defaults (coef_max=3, h=128, res=2, rh=64).
#
cases=(
  # ===== Anchor: direct scale-up of f70yglfn =====
  "p4s4_k16_d4k|true|4096|16|4|4|4|tile|-|-|-|-"

  # ===== Dictionary size =====
  "p4s4_k16_d8k|true|8192|16|4|4|4|tile|-|-|-|-"
  "p4s4_k16_d16k|true|16384|16|4|4|4|tile|-|-|-|-"
  "p4s4_k24_d4k|true|4096|24|4|4|4|tile|-|-|-|-"
  "p4s4_k24_d8k|true|8192|24|4|4|4|tile|-|-|-|-"
  "p4s4_k24_d16k|true|16384|24|4|4|4|tile|-|-|-|-"

  # ===== Sparsity extremes =====
  "p4s4_k8_d4k|true|4096|8|4|4|4|tile|-|-|-|-"
  "p4s4_k32_d4k|true|4096|32|4|4|4|tile|-|-|-|-"
  "p4s4_k32_d8k|true|8192|32|4|4|4|tile|-|-|-|-"

  # ===== Embedding dim =====
  "p4s4_k16_d4k_e8|true|4096|16|8|4|4|tile|-|-|-|-"
  "p4s4_k16_d4k_e16|true|4096|16|16|4|4|tile|-|-|-|-"

  # ===== Larger non-overlapping patches =====
  "p2s2_k8_d4k|true|4096|8|4|2|2|tile|-|-|-|-"
  "p2s2_k16_d4k|true|4096|16|4|2|2|tile|-|-|-|-"
  "p2s2_k24_d4k|true|4096|24|4|2|2|tile|-|-|-|-"
  "p2s2_k16_d8k|true|8192|16|4|2|2|tile|-|-|-|-"
  "p2s2_k16_d4k_e8|true|4096|16|8|2|2|tile|-|-|-|-"
  "p8s8_k16_d4k|true|4096|16|4|8|8|tile|-|-|-|-"
  "p8s8_k24_d4k|true|4096|24|4|8|8|tile|-|-|-|-"
  "p8s8_k32_d8k|true|8192|32|4|8|8|tile|-|-|-|-"

  # ===== Overlapping patches with Hann stitching =====
  "p4s2h_k16_d4k|true|4096|16|4|4|2|hann|-|-|-|-"
  "p4s2h_k24_d4k|true|4096|24|4|4|2|hann|-|-|-|-"
  "p8s4h_k16_d4k|true|4096|16|4|8|4|hann|-|-|-|-"
  "p8s4h_k24_d4k|true|4096|24|4|8|4|hann|-|-|-|-"

  # ===== Coefficient bound =====
  "p4s4_k16_d4k_cm5|true|4096|16|4|4|4|tile|5|-|-|-"
  "p4s4_k16_d4k_cm8|true|4096|16|4|4|4|tile|8|-|-|-"
  "p4s4_k24_d4k_cm5|true|4096|24|4|4|4|tile|5|-|-|-"

  # ===== Larger capacity overrides (h=192, res=3, rh=96) =====
  "p4s4_k16_d4k_med|true|4096|16|4|4|4|tile|-|192|3|96"
  "p4s4_k24_d4k_med|true|4096|24|4|4|4|tile|-|192|3|96"

  # ===== Non-patched baselines =====
  "nopatch_k8_d1k|false|1024|8|16|4|4|tile|-|-|-|-"
  "nopatch_k16_d4k|false|4096|16|16|4|4|tile|-|-|-|-"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name patch_based num_embeddings sparsity embedding_dim \
    patch_size patch_stride patch_recon \
    case_coef_max case_hiddens case_res_blocks case_res_hiddens <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  # Resolve per-case overrides
  if [[ -n "$OVERRIDE_NUM_EMBEDDINGS" ]]; then
    num_embeddings="$OVERRIDE_NUM_EMBEDDINGS"
  fi
  if [[ -n "$OVERRIDE_SPARSITY_LEVEL" ]]; then
    sparsity="$OVERRIDE_SPARSITY_LEVEL"
  fi
  if [[ -n "$OVERRIDE_EMBEDDING_DIM" ]]; then
    embedding_dim="$OVERRIDE_EMBEDDING_DIM"
  fi
  coef_max="${case_coef_max}"; [[ "$coef_max" == "-" ]] && coef_max="3"
  num_hiddens="${case_hiddens}"; [[ "$num_hiddens" == "-" ]] && num_hiddens="$DEFAULT_NUM_HIDDENS"
  num_res_blocks="${case_res_blocks}"; [[ "$num_res_blocks" == "-" ]] && num_res_blocks="$DEFAULT_NUM_RES_BLOCKS"
  num_res_hiddens="${case_res_hiddens}"; [[ "$num_res_hiddens" == "-" ]] && num_res_hiddens="$DEFAULT_NUM_RES_HIDDENS"

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  run_dir="$OUT_ROOT/$run_name"

  echo "[Sweep] $case_name  patch=$patch_based  K=$num_embeddings  k=$sparsity  d=$embedding_dim  p=${patch_size}s${patch_stride}($patch_recon)  cm=$coef_max  h=$num_hiddens  res=$num_res_blocks"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry run -- skipped)"
    submitted=$((submitted + 1))
    continue
  fi

  mkdir -p "$run_dir"

  RUNNER="$run_dir/run_${case_name}.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/bin/bash
set -euo pipefail

echo "=== GPU inventory ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "Warning: nvidia-smi not found in runner environment" >&2
fi
echo ""

if (( $NODES <= 1 )); then
  unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODELIST 2>/dev/null || true
fi

export PYTHONUSERBASE="$PYDEPS"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="\$(command -v python3 || command -v python || true)"
if [[ -z "\$PYTHON_BIN" ]]; then
  echo "ERROR: neither python3 nor python is available in runner environment" >&2
  exit 127
fi

"\$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true

cd "$PROJECT_DIR"

STAGE1_DIR="$run_dir/stage1"
STAGE2_DIR="$run_dir/stage2"
TOKEN_CACHE="$run_dir/token_cache_q${COEFF_BINS}_cm${coef_max}.pt"
TOKEN_CACHE_READY="\${TOKEN_CACHE}.ready"
TOKEN_CACHE_FAILED="\${TOKEN_CACHE}.failed"
mkdir -p "\$STAGE1_DIR" "\$STAGE2_DIR"

echo ""
echo "========================================"
echo "STAGE 1: Autoencoder (coef_max=$coef_max)"
echo "  $run_name  patch=$patch_based  K=$num_embeddings  k=$sparsity  d=$embedding_dim"
echo "  hiddens=$num_hiddens  res_blocks=$num_res_blocks  res_hiddens=$num_res_hiddens"
echo "  recon=$patch_recon  image_size=${IMAGE_SIZE}  batch_size=$BATCH_SIZE  lr=$STAGE1_LR"
echo "========================================"

"\$PYTHON_BIN" train_stage1_autoencoder.py \\
  seed=42 \\
  output_dir="\$STAGE1_DIR" \\
  model=laser \\
  model.backbone=vqgan \\
  model.num_downsamples=$NUM_DOWNSAMPLES \\
  model.attn_resolutions=$ATTN_RESOLUTIONS \\
  model.channel_multipliers=$CHANNEL_MULTIPLIERS \\
  model.backbone_latent_channels=$BACKBONE_LATENT_CHANNELS \\
  model.max_ch_mult=$MAX_CH_MULT \\
  model.decoder_extra_residual_layers=$DECODER_EXTRA_RESIDUAL_LAYERS \\
  model.use_mid_attention=$USE_MID_ATTENTION \\
  model.num_hiddens=$num_hiddens \\
  model.num_residual_blocks=$num_res_blocks \\
  model.num_residual_hiddens=$num_res_hiddens \\
  model.num_embeddings=$num_embeddings \\
  model.embedding_dim=$embedding_dim \\
  model.sparsity_level=$sparsity \\
  model.patch_based=$patch_based \\
  model.patch_size=$patch_size \\
  model.patch_stride=$patch_stride \\
  model.patch_reconstruction=$patch_recon \\
  model.coef_max=$coef_max \\
  model.dict_learning_rate=$DICT_LR \\
  model.variational_coeffs=$STAGE1_VARIATIONAL_COEFFS \\
  model.variational_coeff_kl_weight=$STAGE1_VARIATIONAL_COEFF_KL_WEIGHT \\
  model.variational_coeff_prior_std=$STAGE1_VARIATIONAL_COEFF_PRIOR_STD \\
  model.variational_coeff_min_std=$STAGE1_VARIATIONAL_COEFF_MIN_STD \\
  model.perceptual_weight=$PERCEPTUAL_WEIGHT \\
  model.log_images_every_n_steps=$S1_LOG_IMAGES_EVERY \\
  model.diag_log_interval=$S1_DIAG_LOG_INTERVAL \\
  model.enable_val_latent_visuals=$S1_ENABLE_VAL_LATENT_VISUALS \\
  model.compute_fid=$S1_COMPUTE_FID \\
  model.fid_feature=$S1_FID_FEATURE \\
  data.dataset=$DATASET \\
  data.data_dir="$DATA_DIR" \\
  data.image_size=$IMAGE_SIZE \\
  data.batch_size=$BATCH_SIZE \\
  data.num_workers=$NUM_WORKERS \\
  train.learning_rate=$STAGE1_LR \\
  train.warmup_steps=$WARMUP_STEPS \\
  train.min_lr_ratio=$MIN_LR_RATIO \\
  train.max_epochs=$STAGE1_EPOCHS \\
  train.accelerator=gpu \\
  train.num_nodes=$NODES \\
  train.devices=$GPUS \\
  train.strategy=$TRAIN_STRATEGY \\
  train.precision=bf16-mixed \\
  train.gradient_clip_val=1.0 \\
  train.log_every_n_steps=$S1_LOG_EVERY_N_STEPS \\
  train.val_check_interval=$S1_VAL_CHECK_INTERVAL \\
  wandb.project="$WANDB_PROJECT" \\
  wandb.name="${run_name}_stage1"

STAGE1_CKPT="\$(find "\$STAGE1_DIR" -name '*.ckpt' -path '*/checkpoints/*' | sort | tail -1)"
if [[ -z "\$STAGE1_CKPT" ]]; then
  echo "ERROR: No stage-1 checkpoint found in \$STAGE1_DIR" >&2
  exit 1
fi
echo "Using stage-1 checkpoint: \$STAGE1_CKPT"

echo ""
echo "========================================"
echo "TOKEN EXTRACTION (bins=$COEFF_BINS, coef_max=$coef_max)"
echo "========================================"
run_token_extraction() {
  local output_path="\$1"
  "\$PYTHON_BIN" cache.py \\
    --stage1-checkpoint "\$STAGE1_CKPT" \\
    --output-path "\$output_path" \\
    --dataset "$DATASET" \\
    --data-dir "$DATA_DIR" \\
    --image-size $IMAGE_SIZE \\
    --batch-size $BATCH_SIZE \\
    --num-workers $NUM_WORKERS \\
    --seed 42 \\
    --coeff-max $coef_max \\
    --coeff-bins $COEFF_BINS \\
    --coeff-quantization $COEFF_QUANTIZATION \\
    --coeff-mu $COEFF_MU
}

if (( $NODES > 1 )); then
  if [[ "\${SLURM_PROCID:-0}" == "0" ]]; then
    rm -f "\$TOKEN_CACHE" "\$TOKEN_CACHE_READY" "\$TOKEN_CACHE_FAILED"
    tmp_token_cache="\${TOKEN_CACHE}.tmp"
    rm -f "\$tmp_token_cache"
    if run_token_extraction "\$tmp_token_cache"; then
      mv "\$tmp_token_cache" "\$TOKEN_CACHE"
      touch "\$TOKEN_CACHE_READY"
    else
      rm -f "\$tmp_token_cache"
      touch "\$TOKEN_CACHE_FAILED"
      exit 1
    fi
  else
    while [[ ! -f "\$TOKEN_CACHE_READY" ]]; do
      if [[ -f "\$TOKEN_CACHE_FAILED" ]]; then
        echo "ERROR: rank-zero token extraction failed for $run_name" >&2
        exit 1
      fi
      sleep 5
    done
  fi
else
  run_token_extraction "\$TOKEN_CACHE"
fi

echo ""
echo "========================================"
echo "STAGE 2: AR Prior (${AR_N_LAYERS}L, d=$AR_D_MODEL, ${STAGE2_EPOCHS}ep)"
echo "========================================"

"\$PYTHON_BIN" train_stage2_prior.py \\
  token_cache_path="\$TOKEN_CACHE" \\
  output_dir="\$STAGE2_DIR" \\
  seed=42 \\
  ar.type=sparse_spatial_depth \\
  ar.d_model=$AR_D_MODEL \\
  ar.n_heads=$AR_N_HEADS \\
  ar.n_layers=$AR_N_LAYERS \\
  ar.d_ff=$AR_D_FF \\
  ar.dropout=0.1 \\
  ar.learning_rate=$STAGE2_LR \\
  ar.warmup_steps=$STAGE2_WARMUP_STEPS \\
  ar.min_lr_ratio=$STAGE2_MIN_LR_RATIO \\
  ar.autoregressive_coeffs=$AR_AUTOREGRESSIVE_COEFFS \\
  ar.coeff_loss_type=$AR_COEFF_LOSS_TYPE \\
  ar.coeff_loss_weight=$AR_COEFF_LOSS_WEIGHT \\
  ar.sample_coeff_temperature=$AR_SAMPLE_COEFF_TEMPERATURE \\
  ar.sample_coeff_mode=$AR_SAMPLE_COEFF_MODE \\
  train_ar.batch_size=$STAGE2_BATCH_SIZE \\
  train_ar.max_epochs=$STAGE2_EPOCHS \\
  train_ar.accelerator=gpu \\
  train_ar.num_nodes=$NODES \\
  train_ar.devices=$GPUS \\
  train_ar.strategy=$TRAIN_AR_STRATEGY \\
  train_ar.precision=bf16-mixed \\
  train_ar.gradient_clip_val=1.0 \\
  train_ar.log_every_n_steps=$S2_LOG_EVERY_N_STEPS \\
  train_ar.sample_every_n_epochs=$S2_SAMPLE_EVERY_N_EPOCHS \\
  train_ar.sample_log_to_wandb=true \\
  train_ar.log_recon_every_n_steps=$S2_LOG_RECON_EVERY_N_STEPS \\
  train_ar.sample_num_images=$S2_SAMPLE_NUM_IMAGES \\
  train_ar.sample_temperature=$SAMPLE_TEMP \\
  train_ar.sample_coeff_temperature=$AR_SAMPLE_COEFF_TEMPERATURE \\
  train_ar.sample_coeff_mode=$AR_SAMPLE_COEFF_MODE \\
  data.num_workers=$NUM_WORKERS \\
  wandb.project="$WANDB_PROJECT" \\
  wandb.name="${run_name}_stage2"

echo ""
echo "========================================"
echo "DONE: $run_name"
echo "========================================"
RUNNER_EOF
  chmod +x "$RUNNER"

  SBATCH_ARGS=(
    --partition="$PARTITION"
    --job-name="$job_name"
    --nodes="$NODES"
    --ntasks="$TOTAL_TASKS"
    --ntasks-per-node="$TASKS_PER_NODE"
    --cpus-per-task="$CPUS_PER_TASK"
    --gres="gpu:${GPUS}"
    --mem="${MEM_MB}"
    --time="$TIME_LIMIT"
    --chdir="$PROJECT_DIR"
    --output="$run_dir/${run_name}_%j.out"
    --error="$run_dir/${run_name}_%j.err"
    --requeue
  )
  if [[ -n "$EXCLUDE_NODES" ]]; then
    SBATCH_ARGS+=(--exclude="$EXCLUDE_NODES")
  fi

  BATCH_WRAPPER="$run_dir/sbatch_${case_name}.sh"
  cat > "$BATCH_WRAPPER" <<SBATCH_EOF
#!/bin/bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
  fi
fi

CONTAINER_BIN=""
for candidate in singularity apptainer; do
  if command -v "\$candidate" >/dev/null 2>&1; then
    CONTAINER_BIN="\$candidate"
    break
  fi
done

if [[ -z "\$CONTAINER_BIN" ]]; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
  module load apptainer 2>/dev/null || true
  for candidate in singularity apptainer; do
    if command -v "\$candidate" >/dev/null 2>&1; then
      CONTAINER_BIN="\$candidate"
      break
    fi
  done
fi

echo "=== GPU inventory ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "Warning: nvidia-smi not found in batch wrapper" >&2
fi
echo ""

if (( $NODES > 1 )); then
  SRUN_CMD=(
    srun
    --nodes="$NODES"
    --ntasks="$TOTAL_TASKS"
    --ntasks-per-node="$TASKS_PER_NODE"
    --cpus-per-task="$CPUS_PER_TASK"
    --gpus-per-task=1
    --gpu-bind=single:1
  )
else
  SRUN_CMD=()
fi

if [[ -n "\$CONTAINER_BIN" ]]; then
  "\${SRUN_CMD[@]}" "\$CONTAINER_BIN" exec --nv \
    --bind "$PROJECT_DIR" \
    --bind "/scratch/$USER" \
    --bind "$DATA_DIR" \
    --bind "$run_dir" \
    --bind /dev/shm \
    "$IMAGE" \
    bash "$RUNNER"
else
  echo "Warning: no container runtime found; running bare" >&2
  "\${SRUN_CMD[@]}" bash "$RUNNER"
fi
SBATCH_EOF
  chmod +x "$BATCH_WRAPPER"

  sbatch "${SBATCH_ARGS[@]}" "$BATCH_WRAPPER"

  submitted=$((submitted + 1))
done

echo ""
if ((submitted == 0)); then
  echo "No cases matched CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi
echo "[Sweep] submitted $submitted jobs to $OUT_ROOT"
