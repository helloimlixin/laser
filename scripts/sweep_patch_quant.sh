#!/bin/bash
# Sweep: patched x quantized on the maintained src/ pipeline.
#
# Each SLURM job runs the three-stage Lightning pipeline:
#   1. train.py        — stage-1 autoencoder
#   2. extract_token_cache.py — sparse-token extraction
#   3. train_ar.py     — stage-2 autoregressive prior
#
# Configurations (4 jobs):
#   fast_q   — non-patched, quantized
#   fast_r   — non-patched, real-valued
#   patch_q  — patched, quantized
#   patch_r  — patched, real-valued
#
# Usage:
#   ./scripts/sweep_patch_quant.sh                             # CelebA 128
#   IMAGE_SIZE=256 DATA_DIR=/scratch/$USER/datasets/celebahq_packed_256 \
#       BATCH_SIZE=16 STAGE1_LR=1.4e-4 ./scripts/sweep_patch_quant.sh
#   CASE_FILTER=fast_q,patch_r ./scripts/sweep_patch_quant.sh  # subset
#   DRY_RUN=1 ./scripts/sweep_patch_quant.sh                   # preview

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Cluster ──────────────────────────────────────────────────────────────────
PARTITION="${PARTITION:-auto}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM_MB="${MEM_MB:-96000}"

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
DATASET="${DATASET:-celeba}"

# ── Training ─────────────────────────────────────────────────────────────────
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${BATCH_SIZE}}"
STAGE1_LR="${STAGE1_LR:-1e-3}"
STAGE2_LR="${STAGE2_LR:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ── Output / logging ────────────────────────────────────────────────────────
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/src_pq_sweep_${STAGE1_EPOCHS}ep}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_PREFIX="${RUN_PREFIX:-src_pq_${STAGE1_EPOCHS}ep}"
JOB_PREFIX="${JOB_PREFIX:-srcpq}"

CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

# ── Container (Amarel) ──────────────────────────────────────────────────────
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_src_py311}"

# ── Pre-flight ───────────────────────────────────────────────────────────────
echo "=== Login node GPU check ==="
nvidia-smi 2>/dev/null || echo "(no GPUs on login node — compute nodes will have them)"
echo ""

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

# ── Partition auto-selection ─────────────────────────────────────────────────
auto_select_partition() {
  if [[ -n "${PARTITION:-}" && "$PARTITION" != "auto" ]]; then
    return 0
  fi
  local best="" best_idle=-1
  while read -r part avail tl nodes _rest; do
    [[ -z "${part:-}" || "$part" == "PARTITION" ]] && continue
    part="${part%\*}"
    case "$part" in gpu|gpu-redhat|cgpu) ;; *) continue ;; esac
    local idle
    idle=$(echo "$nodes" | grep -oP '\d+(?=/\d+/\d+$)' 2>/dev/null || echo "0")
    idle="${idle:-0}"
    if (( idle > best_idle )); then
      best="$part"; best_idle=$idle
    fi
  done < <(sinfo -s 2>/dev/null)
  PARTITION="${best:-gpu-redhat}"
  echo "[Sweep] auto-selected PARTITION=$PARTITION (idle=$best_idle)"
}
auto_select_partition

mkdir -p "$OUT_ROOT"

# ── Cases: name | patch_based | quantize | num_embeddings | sparsity | embedding_dim | patch_size | patch_stride | patch_recon | coef_max | coeff_bins
#
# patch_dim = embedding_dim * patch_size^2  (non-patched: patch_dim = embedding_dim)
# density   = sparsity / patch_dim          (higher = better recon, longer sequences)
#
# Goal: find configs with sparsity=8 that maintain good reconstruction
# by shrinking patch_dim and/or growing the dictionary.
#
#                                                    patch  sparsity/
#  name                  patch  quant  K    k   C  ps  ps  recon  patch_dim
cases=(
  # ── Baseline: non-patched, emb=16, k=8 (density=50%), coef_max=10 for stability ──
  "fast_r|false|false|1024|8|16|4|2|hann|10.0|0"

  # ══════════════════════════════════════════════════════════
  # Overlapping patches (stride < size, hann blending)
  # ══════════════════════════════════════════════════════════

  # patch_dim = C * ps^2    density = k / patch_dim
  # 128x128 → 32x32 latent (2 downsamples)

  # ── 4x4 s2, big dict: dim=64, k=8 → 12.5% ──
  "p4s2_k8_d4k_r|true|false|4096|8|4|4|2|hann|10.0|0"
  # ── 4x4 s2, huge dict: dim=64, k=8 → 12.5% ──
  "p4s2_k8_d8k_r|true|false|8192|8|4|4|2|hann|10.0|0"
  # ── 8x8 s4: dim=256, k=8 → 3%, 16k dict ──
  "p8s4_k8_d16k_r|true|false|16384|8|4|8|4|hann|10.0|0"
  # ── 8x8 s4, k=16: dim=256, 16%, 8k dict ──
  "p8s4_k16_d8k_r|true|false|8192|16|4|8|4|hann|10.0|0"
  # ── 8x8 s4, emb=8: dim=512, k=16, 16k dict ──
  "p8s4_e8_k16_d16k_r|true|false|16384|16|8|8|4|hann|10.0|0"
  # ── 16x16 s8: dim=1024, k=16, 16k dict, ~3x3 grid ──
  "p16s8_k16_d16k_r|true|false|16384|16|4|16|8|hann|10.0|0"

  # ══════════════════════════════════════════════════════════
  # Non-overlapping patches (stride = size, tile stitching)
  # Fewer patches → shorter sequences for generation.
  # ══════════════════════════════════════════════════════════

  # ── 4x4 s4: dim=64, k=8, 8k dict ──
  "np4_k8_d8k_r|true|false|8192|8|4|4|4|tile|10.0|0"
  # ── 8x8 s8: dim=256, k=8, 16k dict — 4x4=16 patches, 128 tokens ──
  "np8_k8_d16k_r|true|false|16384|8|4|8|8|tile|10.0|0"
  # ── 8x8 s8, k=16, 8k dict — 256 tokens ──
  "np8_k16_d8k_r|true|false|8192|16|4|8|8|tile|10.0|0"
  # ── 8x8 s8, emb=8: dim=512, k=16, 16k dict — 256 tokens ──
  "np8_e8_k16_d16k_r|true|false|16384|16|8|8|8|tile|10.0|0"
  # ── 16x16 s16: dim=1024, k=16, 16k dict — 2x2=4 patches, 64 tokens ──
  "np16_k16_d16k_r|true|false|16384|16|4|16|16|tile|10.0|0"
  # ── 16x16 s16, emb=8: dim=2048, k=16, 16k dict — 64 tokens ──
  "np16_e8_k16_d16k_r|true|false|16384|16|8|16|16|tile|10.0|0"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name patch_based quantize num_embeddings sparsity embedding_dim \
    patch_size patch_stride patch_recon coef_max coeff_bins <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  run_dir="$OUT_ROOT/$run_name"

  echo "[Sweep] $case_name  patch=$patch_based  quantize=$quantize  img=${IMAGE_SIZE}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry run — skipped)"
    submitted=$((submitted + 1))
    continue
  fi

  mkdir -p "$run_dir"

  # ── Build the runner script that executes inside the container ────────────
  RUNNER="$run_dir/run_${case_name}.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/bin/bash
set -euo pipefail

echo "=== GPU inventory ==="
nvidia-smi
echo ""

# Unset SLURM variables that confuse Lightning's DDP auto-detection.
# We run a single singularity process that internally spawns DDP workers.
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODELIST 2>/dev/null || true

export PYTHONUSERBASE="$PYDEPS"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$PROJECT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"

# Install missing deps into the user site (idempotent)
pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib 2>/dev/null || true

cd "$PROJECT_DIR"

STAGE1_DIR="$run_dir/stage1"
STAGE2_DIR="$run_dir/stage2"
TOKEN_CACHE="$run_dir/token_cache.pt"
mkdir -p "\$STAGE1_DIR" "\$STAGE2_DIR"

echo ""
echo "========================================"
echo "STAGE 1: Autoencoder Training"
echo "  run=$run_name  patch=$patch_based  quantize=$quantize"
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
  ${coef_max:+model.coef_max=$coef_max} \\
  model.log_images_every_n_steps=200 \\
  model.enable_val_latent_visuals=true \\
  data.dataset=$DATASET \\
  data.data_dir="$DATA_DIR" \\
  data.image_size=$IMAGE_SIZE \\
  data.batch_size=$BATCH_SIZE \\
  data.num_workers=$NUM_WORKERS \\
  train.learning_rate=$STAGE1_LR \\
  train.max_epochs=$STAGE1_EPOCHS \\
  train.accelerator=gpu \\
  model.compute_fid=false \\
  model.perceptual_weight=0.0 \\
  train.devices=$GPUS \\
  train.strategy=auto \\
  train.precision=bf16-mixed \\
  train.gradient_clip_val=1.0 \\
  train.log_every_n_steps=50 \\
  train.val_check_interval=0.5 \\
  wandb.project="$WANDB_PROJECT" \\
  wandb.name="${run_name}_stage1"

# Find best stage-1 checkpoint
STAGE1_CKPT="\$(find "\$STAGE1_DIR" -name '*.ckpt' -path '*/checkpoints/*' | sort | tail -1)"
if [[ -z "\$STAGE1_CKPT" ]]; then
  echo "ERROR: No stage-1 checkpoint found in \$STAGE1_DIR" >&2
  exit 1
fi
echo "Using stage-1 checkpoint: \$STAGE1_CKPT"

echo ""
echo "========================================"
echo "TOKEN EXTRACTION"
echo "========================================"

EXTRACT_ARGS=(
  --stage1-checkpoint "\$STAGE1_CKPT"
  --output-path "\$TOKEN_CACHE"
  --dataset "$DATASET"
  --data-dir "$DATA_DIR"
  --image-size $IMAGE_SIZE
  --batch-size $BATCH_SIZE
  --num-workers $NUM_WORKERS
  --seed 42
  --coeff-max auto
)
if [[ "$quantize" == "true" ]]; then
  EXTRACT_ARGS+=(--coeff-bins ${coeff_bins:-256})
else
  # Real-valued: still need a token cache with atoms, but skip binning.
  # Use 0 bins to signal real-valued mode — extract_token_cache stores
  # atoms_flat + coeffs_flat when coeff_bins=0.
  EXTRACT_ARGS+=(--coeff-bins 0)
fi
python extract_token_cache.py "\${EXTRACT_ARGS[@]}"

echo ""
echo "========================================"
echo "STAGE 2: Autoregressive Prior Training"
echo "========================================"

python train_ar.py \\
  token_cache_path="\$TOKEN_CACHE" \\
  output_dir="\$STAGE2_DIR" \\
  seed=42 \\
  ar.type=sparse_spatial_depth \\
  ar.d_model=256 \\
  ar.n_heads=4 \\
  ar.n_layers=6 \\
  ar.d_ff=512 \\
  ar.dropout=0.1 \\
  ar.learning_rate=$STAGE2_LR \\
  ar.warmup_steps=500 \\
  ar.coeff_loss_type=auto \\
  train_ar.batch_size=$STAGE2_BATCH_SIZE \\
  train_ar.max_epochs=$STAGE2_EPOCHS \\
  train_ar.accelerator=gpu \\
  train_ar.devices=1 \\
  train_ar.strategy=auto \\
  train_ar.precision=bf16-mixed \\
  train_ar.gradient_clip_val=1.0 \\
  train_ar.log_every_n_steps=50 \\
  train_ar.sample_every_n_epochs=1 \\
  train_ar.sample_log_to_wandb=true \\
  train_ar.log_recon_every_n_steps=200 \\
  train_ar.sample_num_images=8 \\
  data.num_workers=$NUM_WORKERS \\
  wandb.project="$WANDB_PROJECT" \\
  wandb.name="${run_name}_stage2"

echo ""
echo "========================================"
echo "DONE: $run_name"
echo "========================================"
RUNNER_EOF
  chmod +x "$RUNNER"

  # ── Submit via sbatch ────────────────────────────────────────────────────
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

  # The wrap script initialises the module system, loads singularity,
  # runs nvidia-smi for diagnostics, then launches the runner inside
  # the container.
  sbatch "${SBATCH_ARGS[@]}" --wrap='#!/bin/bash
set -euo pipefail

# ── Initialise module system (needed on Amarel compute nodes) ──
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
  fi
fi

# ── Load singularity ──
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
