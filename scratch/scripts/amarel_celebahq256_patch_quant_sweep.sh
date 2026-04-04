#!/bin/bash
# Sweep: patched x quantized on CelebA-HQ 256x256, 10 epochs each stage.
#
# Launches 4 SLURM jobs:
#   1. fast_q   — non-patched, quantized
#   2. fast_r   — non-patched, real-valued (non-quantized)
#   3. patch_q  — patched, quantized
#   4. patch_r  — patched, real-valued (non-quantized)
#
# Usage:
#   ./amarel_celebahq256_patch_quant_sweep.sh
#   CASE_FILTER=fast_q,patch_r ./amarel_celebahq256_patch_quant_sweep.sh
#   DRY_RUN=1 ./amarel_celebahq256_patch_quant_sweep.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Cluster / resources ──────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_pq_sweep_10ep}"
PARTITION="${PARTITION:-auto}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

# ── Training knobs (adjusted for 256x256) ────────────────────────────────────
ENTRYPOINT="${ENTRYPOINT:-laser.py}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
# Halve batch size vs 128x128 to fit 4x larger images in GPU memory.
BATCH_SIZE="${BATCH_SIZE:-16}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-12}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-4}"
TOKEN_SUBSET="${TOKEN_SUBSET:-98304}"

# ── Architecture (match celebahq256_20ep_matrix_sweep defaults) ──────────────
FAST_AE_NUM_DOWNSAMPLES="${FAST_AE_NUM_DOWNSAMPLES:-4}"
PATCH_AE_NUM_DOWNSAMPLES="${PATCH_AE_NUM_DOWNSAMPLES:-2}"
NUM_HIDDENS="${NUM_HIDDENS:-64}"
NUM_RES_HIDDENS="${NUM_RES_HIDDENS:-32}"
NUM_RES_LAYERS="${NUM_RES_LAYERS:-1}"
PATCH_SIZE="${PATCH_SIZE:-8}"
PATCH_STRIDE="${PATCH_STRIDE:-4}"
PATCH_RECONSTRUCTION="${PATCH_RECONSTRUCTION:-hann}"

# Scale LRs down for smaller effective batch (half batch size).
# proto.py applies sqrt(world_size) scaling internally, so these are the
# per-GPU base rates.  The 128 sweep uses 2e-4 / 1e-3 at bs=32; halving
# the batch to 16 means ~0.7x scale → round to 1.4e-4 / 7e-4.
STAGE1_LR="${STAGE1_LR:-1.4e-4}"
STAGE2_LR="${STAGE2_LR:-7e-4}"

# ── Logging ──────────────────────────────────────────────────────────────────
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
RUN_PREFIX="${RUN_PREFIX:-chq256_pq_10ep}"
JOB_PREFIX="${JOB_PREFIX:-chq256-pq10}"
CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

# ── Pre-flight ───────────────────────────────────────────────────────────────
echo "=== Login node GPU check ==="
nvidia-smi 2>/dev/null || echo "(no GPUs on login node — expected; compute nodes will have them)"
echo ""

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  echo "Expected packed npy at: $DATA_DIR/celeba_256x256_rgb_uint8.npy" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# nvidia-smi runs automatically at the start of each SLURM job via
# launch_proto_rqsd_multinode_slurm.sh (srun nvidia-smi on all nodes).

# ── Cases: family | name | quantize | embedding_dim | num_atoms | sparsity | n_bins | coef_max
cases=(
  "fast|fast_q|true|16|1024|8|256|3.0"
  "fast|fast_r|false|16|1024|8|256|3.0"
  "patch|patch_q|true|16|4096|16|512|8.0"
  "patch|patch_r|false|16|4096|16|256|8.0"
)

CASE_FILTER="${CASE_FILTER// /}"
submitted=0

for case_spec in "${cases[@]}"; do
  IFS='|' read -r family case_name quantize_sparse_coeffs embedding_dim num_atoms sparsity_level n_bins coef_max <<< "$case_spec"

  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}_${case_name}"
  job_name="${JOB_PREFIX}-${case_name}"
  out_dir="$OUT_ROOT/$run_name"

  # Select launcher and family-specific defaults
  launcher="$ROOT_DIR/scripts/fast100.sh"
  patch_based="false"
  ae_num_downsamples="$FAST_AE_NUM_DOWNSAMPLES"
  local_patch_size="4"
  local_patch_stride="2"
  local_patch_reconstruction="center_crop"
  if [[ "$family" == "patch" ]]; then
    launcher="$ROOT_DIR/scripts/patch100.sh"
    patch_based="true"
    ae_num_downsamples="$PATCH_AE_NUM_DOWNSAMPLES"
    local_patch_size="$PATCH_SIZE"
    local_patch_stride="$PATCH_STRIDE"
    local_patch_reconstruction="$PATCH_RECONSTRUCTION"
  fi

  echo "[Sweep] $case_name  patch=$patch_based  quantize=$quantize_sparse_coeffs  atoms=$num_atoms  k=$sparsity_level  img=${IMAGE_SIZE}x${IMAGE_SIZE}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (dry run — skipped)"
    submitted=$((submitted + 1))
    continue
  fi

  submit_output="$(
    DATA_DIR="$DATA_DIR" \
    OUT_DIR="$out_dir" \
    PARTITION="$PARTITION" \
    NODES="$NODES" \
    GPUS_PER_NODE="$GPUS_PER_NODE" \
    CPUS_PER_TASK="$CPUS_PER_TASK" \
    MEM_MB="$MEM_MB" \
    TIME_LIMIT="$TIME_LIMIT" \
    ENTRYPOINT="$ENTRYPOINT" \
    IMAGE_SIZE="$IMAGE_SIZE" \
    STAGE1_EPOCHS="$STAGE1_EPOCHS" \
    STAGE2_EPOCHS="$STAGE2_EPOCHS" \
    STAGE1_LR="$STAGE1_LR" \
    STAGE2_LR="$STAGE2_LR" \
    BATCH_SIZE="$BATCH_SIZE" \
    STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    TOKEN_NUM_WORKERS="$TOKEN_NUM_WORKERS" \
    TOKEN_SUBSET="$TOKEN_SUBSET" \
    AE_NUM_DOWNSAMPLES="$ae_num_downsamples" \
    NUM_HIDDENS="$NUM_HIDDENS" \
    NUM_RES_HIDDENS="$NUM_RES_HIDDENS" \
    NUM_RES_LAYERS="$NUM_RES_LAYERS" \
    EMBEDDING_DIM="$embedding_dim" \
    NUM_ATOMS="$num_atoms" \
    SPARSITY_LEVEL="$sparsity_level" \
    QUANTIZE_SPARSE_COEFFS="$quantize_sparse_coeffs" \
    N_BINS="$n_bins" \
    COEF_MAX="$coef_max" \
    COEF_QUANTIZATION="uniform" \
    COEF_MU="0.0" \
    PATCH_BASED="$patch_based" \
    PATCH_SIZE="$local_patch_size" \
    PATCH_STRIDE="$local_patch_stride" \
    PATCH_RECONSTRUCTION="$local_patch_reconstruction" \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="$run_name" \
    LOG_PREFIX="$run_name" \
    JOB_NAME="$job_name" \
    STAGE2_SAMPLE_EVERY_STEPS="500" \
    STAGE2_SAMPLE_IMAGE_SIZE="$IMAGE_SIZE" \
    RFID_NUM_SAMPLES="0" \
    STAGE1_AUTO_RESUME_FROM_LATEST="false" \
    "$launcher"
  )"
  printf '%s\n' "$submit_output"
  submitted=$((submitted + 1))
done

echo ""
if ((submitted == 0)); then
  echo "No cases matched CASE_FILTER=$CASE_FILTER" >&2
  exit 1
fi
echo "[Sweep] submitted $submitted jobs to $OUT_ROOT"
