#!/bin/bash
# Sweep: Full proto-matched pipeline (stage 1 + extraction + stage 2).
#
# Retrains stage 1 with coef_max=3 (matching proto.py 351lokx0), then
# extracts quantized tokens, then trains a 12-layer AR prior for 100 epochs.
#
# Usage:
#   ./scripts/sweep_proto_matched_full.sh
#   CASE_FILTER=fast_r ./scripts/sweep_proto_matched_full.sh
#   DRY_RUN=1 ./scripts/sweep_proto_matched_full.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -- Cluster ----------------------------------------------------------------
PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-96000}"

# -- Data -------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
DATASET="${DATASET:-celeba}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# -- Stage 1 (proto-matched) -----------------------------------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE1_LR="${STAGE1_LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"
COEF_MAX="${COEF_MAX:-3}"

# -- Stage 2 (proto-matched) -----------------------------------------------
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
STAGE2_LR="${STAGE2_LR:-1e-3}"
COEFF_BINS="${COEFF_BINS:-256}"
SAMPLE_TEMP="${SAMPLE_TEMP:-0.5}"

# -- Output / logging ------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/src_proto_full_sweep}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_PREFIX="${RUN_PREFIX:-src_pf}"
JOB_PREFIX="${JOB_PREFIX:-srcpf}"

CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

# -- Container (Amarel) ----------------------------------------------------
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_src_py311}"

# -- Pre-flight -------------------------------------------------------------
echo "=== Login node GPU check ==="
nvidia-smi 2>/dev/null || echo "(no GPUs on login node -- compute nodes will have them)"
echo ""

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# -- Cases: name | patch_based | num_embeddings | sparsity | embedding_dim | patch_size | patch_stride | patch_recon
#
# All use coef_max=3 for stage 1 training AND token extraction.
# Stage 2 uses proto-matched: 12L, d=512, 100ep, temp=0.5, gt_atom_recon_mse.
#
cases=(
  # Non-patched baseline (exact proto.py match)
  "fast_r|false|1024|8|16|4|4|tile"

  # Patched p4s4 (proven k=16 and k=24 from v3)
  "p4s4_k16_d4k|true|4096|16|4|4|4|tile"
  "p4s4_k24_d4k|true|4096|24|4|4|4|tile"

  # Patched p8s8
  "p8s8_k24_d4k|true|4096|24|4|8|8|tile"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name patch_based num_embeddings sparsity embedding_dim \
    patch_size patch_stride patch_recon <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  run_dir="$OUT_ROOT/$run_name"

  echo "[Sweep] $case_name  patch=$patch_based  K=$num_embeddings  k=$sparsity  coef_max=$COEF_MAX"

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

pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib 2>/dev/null || true

cd "$PROJECT_DIR"

STAGE1_DIR="$run_dir/stage1"
STAGE2_DIR="$run_dir/stage2"
TOKEN_CACHE="$run_dir/token_cache_q${COEFF_BINS}_cm${COEF_MAX}.pt"
mkdir -p "\$STAGE1_DIR" "\$STAGE2_DIR"

echo ""
echo "========================================"
echo "STAGE 1: Autoencoder (coef_max=$COEF_MAX)"
echo "  $run_name  patch=$patch_based  K=$num_embeddings  k=$sparsity"
echo "========================================"

python train.py \\
  seed=42 \\
  output_dir="\$STAGE1_DIR" \\
  model=laser \\
  model.num_embeddings=$num_embeddings \\
  model.embedding_dim=$embedding_dim \\
  model.sparsity_level=$sparsity \\
  model.patch_based=$patch_based \\
  model.patch_size=$patch_size \\
  model.patch_stride=$patch_stride \\
  model.patch_reconstruction=$patch_recon \\
  model.coef_max=$COEF_MAX \\
  model.log_images_every_n_steps=200 \\
  model.enable_val_latent_visuals=true \\
  model.compute_fid=false \\
  model.perceptual_weight=0.0 \\
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
  train.devices=$GPUS \\
  train.strategy=auto \\
  train.precision=bf16-mixed \\
  train.gradient_clip_val=1.0 \\
  train.log_every_n_steps=50 \\
  train.val_check_interval=0.5 \\
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
echo "TOKEN EXTRACTION (bins=$COEFF_BINS, coef_max=$COEF_MAX)"
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
  --coeff-max $COEF_MAX \\
  --coeff-bins $COEFF_BINS \\
  --coeff-quantization uniform

echo ""
echo "========================================"
echo "STAGE 2: Proto-matched AR (12L, 100ep, temp=$SAMPLE_TEMP)"
echo "========================================"

python train_ar.py \\
  token_cache_path="\$TOKEN_CACHE" \\
  output_dir="\$STAGE2_DIR" \\
  seed=42 \\
  ar.type=sparse_spatial_depth \\
  ar.d_model=512 \\
  ar.n_heads=8 \\
  ar.n_layers=12 \\
  ar.d_ff=1024 \\
  ar.dropout=0.1 \\
  ar.learning_rate=$STAGE2_LR \\
  ar.warmup_steps=1000 \\
  ar.min_lr_ratio=0.01 \\
  ar.coeff_loss_type=gt_atom_recon_mse \\
  ar.coeff_loss_weight=0.1 \\
  train_ar.batch_size=$STAGE2_BATCH_SIZE \\
  train_ar.max_epochs=$STAGE2_EPOCHS \\
  train_ar.accelerator=gpu \\
  train_ar.devices=$GPUS \\
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
