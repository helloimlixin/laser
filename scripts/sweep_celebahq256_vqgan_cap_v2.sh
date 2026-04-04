#!/bin/bash
# Sweep v2: CelebA-HQ 256x256 — extended exploration axes.
#
# Complements v1 (which covers p4s4/p8s8 tile, K=4096/8192, k=16/24/32, d=4/8).
# This sweep adds:
#   - Overlapping patches with Hann stitching (p4s2, p8s4)
#   - Different coef_max values (5, 8)
#   - Lower and higher sparsity extremes (k=8, k=32 on p4s4)
#   - Larger dictionary (K=16384)
#   - Larger embedding dim (d=16)
#   - Medium capacity comparison (h=192, res=3, rh=96)
#   - Non-patched baseline
#
# Dictionary atom animation is enabled (enable_val_latent_visuals=true).
#
# Usage:
#   ./scripts/sweep_celebahq256_vqgan_cap_v2.sh
#   CASE_FILTER=p4s2h_k16_d4k ./scripts/sweep_celebahq256_vqgan_cap_v2.sh
#   DRY_RUN=1 ./scripts/sweep_celebahq256_vqgan_cap_v2.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
GPUS="${GPUS:-2}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-320000}"

# -- Data -------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DATASET="${DATASET:-celeba}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# -- Stage 1 defaults -------------------------------------------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE1_LR="${STAGE1_LR:-1.5e-4}"
DICT_LR="${DICT_LR:-2.5e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.5}"

# -- Stage 2 defaults -------------------------------------------------------
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
STAGE2_LR="${STAGE2_LR:-3e-4}"
COEFF_BINS="${COEFF_BINS:-256}"
SAMPLE_TEMP="${SAMPLE_TEMP:-0.5}"
AR_D_MODEL="${AR_D_MODEL:-512}"
AR_N_HEADS="${AR_N_HEADS:-8}"
AR_N_LAYERS="${AR_N_LAYERS:-12}"
AR_D_FF="${AR_D_FF:-1024}"

# -- Output / logging -------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_vqgan_cap_v2_sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_PREFIX="${RUN_PREFIX:-chq256_v2}"
JOB_PREFIX="${JOB_PREFIX:-chq256v2}"

CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

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

mkdir -p "$OUT_ROOT"

# -- Cases: name | patch_based | num_embeddings | sparsity | embedding_dim | patch_size | patch_stride | patch_recon | coef_max | num_hiddens | num_res_blocks | num_res_hiddens
#
# Extended format: last 4 fields allow per-case overrides for coef_max and capacity.
# Use "-" to inherit the default.
#
cases=(
  # ===== Overlapping patches with Hann stitching =====
  "p4s2h_k16_d4k|true|4096|16|4|4|2|hann|-|-|-|-"
  "p4s2h_k24_d4k|true|4096|24|4|4|2|hann|-|-|-|-"
  "p8s4h_k16_d4k|true|4096|16|4|8|4|hann|-|-|-|-"
  "p8s4h_k24_d4k|true|4096|24|4|8|4|hann|-|-|-|-"

  # ===== Higher coef_max (wider coefficient range) =====
  "p4s4_k16_d4k_cm5|true|4096|16|4|4|4|tile|5|-|-|-"
  "p4s4_k16_d4k_cm8|true|4096|16|4|4|4|tile|8|-|-|-"
  "p4s4_k24_d4k_cm5|true|4096|24|4|4|4|tile|5|-|-|-"

  # ===== Sparsity extremes =====
  "p4s4_k8_d4k|true|4096|8|4|4|4|tile|-|-|-|-"
  "p4s4_k32_d4k|true|4096|32|4|4|4|tile|-|-|-|-"
  "p4s4_k32_d8k|true|8192|32|4|4|4|tile|-|-|-|-"

  # ===== Larger dictionary =====
  "p4s4_k16_d16k|true|16384|16|4|4|4|tile|-|-|-|-"
  "p4s4_k24_d16k|true|16384|24|4|4|4|tile|-|-|-|-"

  # ===== Larger embedding dim =====
  "p4s4_k16_d4k_e16|true|4096|16|16|4|4|tile|-|-|-|-"

  # ===== Medium capacity comparison (h=192, res=3, rh=96) =====
  "p4s4_k16_d4k_med|true|4096|16|4|4|4|tile|-|192|3|96"
  "p4s4_k24_d4k_med|true|4096|24|4|4|4|tile|-|192|3|96"

  # ===== Non-patched baseline =====
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

  # Resolve per-case overrides (use defaults if "-")
  coef_max="${case_coef_max}"
  [[ "$coef_max" == "-" ]] && coef_max="3"
  num_hiddens="${case_hiddens}"
  [[ "$num_hiddens" == "-" ]] && num_hiddens="256"
  num_res_blocks="${case_res_blocks}"
  [[ "$num_res_blocks" == "-" ]] && num_res_blocks="4"
  num_res_hiddens="${case_res_hiddens}"
  [[ "$num_res_hiddens" == "-" ]] && num_res_hiddens="128"

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
nvidia-smi
echo ""

unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODELIST 2>/dev/null || true

export PYTHONUSERBASE="$PYDEPS"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"

pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true

cd "$PROJECT_DIR"

STAGE1_DIR="$run_dir/stage1"
STAGE2_DIR="$run_dir/stage2"
TOKEN_CACHE="$run_dir/token_cache_q${COEFF_BINS}_cm${coef_max}.pt"
mkdir -p "\$STAGE1_DIR" "\$STAGE2_DIR"

echo ""
echo "========================================"
echo "STAGE 1: Autoencoder (coef_max=$coef_max)"
echo "  $run_name  patch=$patch_based  K=$num_embeddings  k=$sparsity  d=$embedding_dim"
echo "  hiddens=$num_hiddens  res_blocks=$num_res_blocks  res_hiddens=$num_res_hiddens"
echo "  recon=$patch_recon  image_size=${IMAGE_SIZE}  batch_size=$BATCH_SIZE  lr=$STAGE1_LR"
echo "========================================"

python train.py \\
  seed=42 \\
  output_dir="\$STAGE1_DIR" \\
  model=laser \\
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
  model.perceptual_weight=$PERCEPTUAL_WEIGHT \\
  model.log_images_every_n_steps=200 \\
  model.enable_val_latent_visuals=true \\
  model.compute_fid=false \\
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
  train.devices=auto \\
  train.strategy=auto \\
  train.precision=bf16-mixed \\
  train.gradient_clip_val=1.0 \\
  train.log_every_n_steps=50 \\
  train.val_check_interval=1.0 \\
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

python extract_token_cache.py \\
  --stage1-checkpoint "\$STAGE1_CKPT" \\
  --output-path "\$TOKEN_CACHE" \\
  --dataset "$DATASET" \\
  --data-dir "$DATA_DIR" \\
  --image-size $IMAGE_SIZE \\
  --batch-size $BATCH_SIZE \\
  --num-workers $NUM_WORKERS \\
  --seed 42 \\
  --coeff-max $coef_max \\
  --coeff-bins $COEFF_BINS \\
  --coeff-quantization uniform

echo ""
echo "========================================"
echo "STAGE 2: AR Prior (${AR_N_LAYERS}L, d=$AR_D_MODEL, ${STAGE2_EPOCHS}ep)"
echo "========================================"

python train_ar.py \\
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
  ar.warmup_steps=2000 \\
  ar.min_lr_ratio=0.01 \\
  ar.coeff_loss_type=gt_atom_recon_mse \\
  ar.coeff_loss_weight=0.1 \\
  train_ar.batch_size=$STAGE2_BATCH_SIZE \\
  train_ar.max_epochs=$STAGE2_EPOCHS \\
  train_ar.accelerator=gpu \\
  train_ar.devices=auto \\
  train_ar.strategy=auto \\
  train_ar.precision=bf16-mixed \\
  train_ar.gradient_clip_val=1.0 \\
  train_ar.log_every_n_steps=50 \\
  train_ar.sample_every_n_epochs=5 \\
  train_ar.sample_log_to_wandb=true \\
  train_ar.log_recon_every_n_steps=500 \\
  train_ar.sample_num_images=8 \\
  train_ar.sample_temperature=$SAMPLE_TEMP \\
  train_ar.sample_coeff_mode=mean \\
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
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task="$CPUS"
    --gres="gpu:${GPUS}"
    --mem="${MEM_MB}"
    --time="$TIME_LIMIT"
    --chdir="$PROJECT_DIR"
    --output="$run_dir/${run_name}_%j.out"
    --error="$run_dir/${run_name}_%j.err"
    --requeue
  )

  sbatch "${SBATCH_ARGS[@]}" --wrap='#!/bin/bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
  fi
fi

if ! command -v singularity >/dev/null 2>&1; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
fi

echo "=== GPU inventory ==="
nvidia-smi
echo ""

if command -v singularity >/dev/null 2>&1; then
  singularity exec --nv \
    --bind "'"$PROJECT_DIR"'" \
    --bind "/scratch/'"$USER"'" \
    --bind "'"$DATA_DIR"'" \
    --bind "'"$run_dir"'" \
    --bind /dev/shm \
    "'"$IMAGE"'" \
    bash "'"$RUNNER"'"
else
  echo "Warning: singularity not found; running bare" >&2
  bash "'"$RUNNER"'"
fi
'

  submitted=$((submitted + 1))
done

echo ""
if ((submitted == 0)); then
  echo "No cases matched CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi
echo "[Sweep] submitted $submitted jobs to $OUT_ROOT"
