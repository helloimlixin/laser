#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/celebahq256_patch_coeff_compare}"
PARTITION="${PARTITION:-gpu-redhat}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-128000}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
ENTRYPOINT="${ENTRYPOINT:-proto.py}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
AE_NUM_DOWNSAMPLES="${AE_NUM_DOWNSAMPLES:-4}"
NUM_HIDDENS="${NUM_HIDDENS:-128}"
NUM_RES_HIDDENS="${NUM_RES_HIDDENS:-64}"
NUM_RES_LAYERS="${NUM_RES_LAYERS:-2}"
PATCH_SIZE="${PATCH_SIZE:-4}"
PATCH_STRIDE="${PATCH_STRIDE:-2}"
BATCH_SIZE="${BATCH_SIZE:-6}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-6}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-dl}"
NUM_ATOMS_LIST="${NUM_ATOMS_LIST:-4096 6144}"
SPARSITY_LEVEL_LIST="${SPARSITY_LEVEL_LIST:-16 24}"
QUANTIZED_N_BINS="${QUANTIZED_N_BINS:-512}"
QUANTIZED_COEF_MAX="${QUANTIZED_COEF_MAX:-4.0}"
QUANTIZED_COEF_QUANTIZATION="${QUANTIZED_COEF_QUANTIZATION:-uniform}"
QUANTIZED_COEF_MU="${QUANTIZED_COEF_MU:-0.0}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

read -r -a num_atoms_list <<< "$NUM_ATOMS_LIST"
read -r -a sparsity_level_list <<< "$SPARSITY_LEVEL_LIST"

if ((${#num_atoms_list[@]} == 0)); then
  echo "NUM_ATOMS_LIST must contain at least one value." >&2
  exit 1
fi
if ((${#sparsity_level_list[@]} == 0)); then
  echo "SPARSITY_LEVEL_LIST must contain at least one value." >&2
  exit 1
fi

for coeff_mode in real quantized; do
  if [[ "$coeff_mode" == "real" ]]; then
    quantize_sparse_coeffs="false"
    n_bins="256"
    coef_max="3.0"
    coef_quantization="uniform"
    coef_mu="0.0"
    mode_tag="rv"
  else
    quantize_sparse_coeffs="true"
    n_bins="$QUANTIZED_N_BINS"
    coef_max="$QUANTIZED_COEF_MAX"
    coef_quantization="$QUANTIZED_COEF_QUANTIZATION"
    coef_mu="$QUANTIZED_COEF_MU"
    coef_max_tag="${coef_max//./p}"
    mode_tag="qb${n_bins}_c${coef_max_tag}"
  fi

  for num_atoms in "${num_atoms_list[@]}"; do
    for sparsity_level in "${sparsity_level_list[@]}"; do
      run_name="celebahq256_cmp10_${mode_tag}_p${PATCH_SIZE}s${PATCH_STRIDE}_a${num_atoms}_k${sparsity_level}"
      job_name="chq256-cmp-${mode_tag}-a${num_atoms}-k${sparsity_level}"
      out_dir="$OUT_ROOT/$run_name"

      echo "[Sweep] submitting $run_name"
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
        STAGE2_SAMPLE_IMAGE_SIZE="$IMAGE_SIZE" \
        AE_NUM_DOWNSAMPLES="$AE_NUM_DOWNSAMPLES" \
        NUM_HIDDENS="$NUM_HIDDENS" \
        NUM_RES_HIDDENS="$NUM_RES_HIDDENS" \
        NUM_RES_LAYERS="$NUM_RES_LAYERS" \
        PATCH_SIZE="$PATCH_SIZE" \
        PATCH_STRIDE="$PATCH_STRIDE" \
        BATCH_SIZE="$BATCH_SIZE" \
        STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE" \
        QUANTIZE_SPARSE_COEFFS="$quantize_sparse_coeffs" \
        N_BINS="$n_bins" \
        COEF_MAX="$coef_max" \
        COEF_QUANTIZATION="$coef_quantization" \
        COEF_MU="$coef_mu" \
        WANDB_MODE="$WANDB_MODE" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_NAME="$run_name" \
        LOG_PREFIX="$run_name" \
        JOB_NAME="$job_name" \
        NUM_ATOMS="$num_atoms" \
        SPARSITY_LEVEL="$sparsity_level" \
        "$ROOT_DIR/scripts/patch100.sh"
      )"
      printf '%s\n' "$submit_output"
    done
  done
done
