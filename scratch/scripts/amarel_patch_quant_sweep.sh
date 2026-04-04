#!/bin/bash
# Sweep: patched x quantized, 10 epochs each stage, Amarel cluster.
#
# Launches 4 SLURM jobs:
#   1. fast_q   — non-patched, quantized
#   2. fast_r   — non-patched, real-valued (non-quantized)
#   3. patch_q  — patched, quantized
#   4. patch_r  — patched, real-valued (non-quantized)
#
# Usage:
#   ./amarel_patch_quant_sweep.sh
#   CASE_FILTER=fast_q,patch_r ./amarel_patch_quant_sweep.sh   # subset
#   DRY_RUN=1 ./amarel_patch_quant_sweep.sh                    # preview only

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Cluster / resources ──────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/patch_quant_sweep_10ep}"
PARTITION="${PARTITION:-auto}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

# ── Training knobs ───────────────────────────────────────────────────────────
ENTRYPOINT="${ENTRYPOINT:-laser.py}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-4}"
TOKEN_SUBSET="${TOKEN_SUBSET:-98304}"

# ── Logging ──────────────────────────────────────────────────────────────────
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
RUN_PREFIX="${RUN_PREFIX:-pq_sweep_10ep}"
JOB_PREFIX="${JOB_PREFIX:-pq10}"
CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

# ── Pre-flight: nvidia-smi on login node (informational) ────────────────────
echo "=== Login node GPU check ==="
nvidia-smi 2>/dev/null || echo "(no GPUs on login node — expected; compute nodes will have them)"
echo ""

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# nvidia-smi runs automatically at the start of each SLURM job via
# launch_proto_rqsd_multinode_slurm.sh (srun nvidia-smi on all nodes).

# ── Cases: family | name | quantize | embedding_dim | num_atoms | sparsity | n_bins | coef_max
cases=(
  "fast|fast_q|true|16|1024|8|2048|20.0"
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
  ae_num_downsamples="4"
  patch_size="4"
  patch_stride="2"
  patch_reconstruction="center_crop"
  if [[ "$family" == "patch" ]]; then
    launcher="$ROOT_DIR/scripts/patch100.sh"
    patch_based="true"
    ae_num_downsamples="2"
    patch_size="8"
    patch_stride="4"
    patch_reconstruction="hann"
  fi

  echo "[Sweep] $case_name  patch=$patch_based  quantize=$quantize_sparse_coeffs  atoms=$num_atoms  k=$sparsity_level"

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
    BATCH_SIZE="$BATCH_SIZE" \
    STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    TOKEN_NUM_WORKERS="$TOKEN_NUM_WORKERS" \
    TOKEN_SUBSET="$TOKEN_SUBSET" \
    AE_NUM_DOWNSAMPLES="$ae_num_downsamples" \
    EMBEDDING_DIM="$embedding_dim" \
    NUM_ATOMS="$num_atoms" \
    SPARSITY_LEVEL="$sparsity_level" \
    QUANTIZE_SPARSE_COEFFS="$quantize_sparse_coeffs" \
    N_BINS="$n_bins" \
    COEF_MAX="$coef_max" \
    COEF_QUANTIZATION="uniform" \
    COEF_MU="0.0" \
    PATCH_BASED="$patch_based" \
    PATCH_SIZE="$patch_size" \
    PATCH_STRIDE="$patch_stride" \
    PATCH_RECONSTRUCTION="$patch_reconstruction" \
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
