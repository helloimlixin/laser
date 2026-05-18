#!/bin/bash
# Canonical maintained sweep submitter.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$PROJECT_DIR/scripts/run.sh"
PROFILE_SH="$PROJECT_DIR/scripts/profile.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "Missing runner: $RUNNER" >&2
  exit 1
fi
if [[ ! -f "$PROFILE_SH" ]]; then
  echo "Missing profile file: $PROFILE_SH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$PROFILE_SH"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-36:00:00}"
GPUS="${GPUS:-3}"
CPUS_PER_TASK="${CPUS_PER_TASK:-24}"
MEM_MB="${MEM_MB:-240000}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu043}"

IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-/scratch/$USER/runs}"

debug_profile_load
CASE_FILTER="${CASE_FILTER:-}"

case_enabled() {
  local name="$1"
  if [[ -z "$CASE_FILTER" ]]; then
    return 0
  fi
  [[ ",$CASE_FILTER," == *",$name,"* ]]
}

debug_profile_cases

submitted=0
run_family="$(debug_profile_run_family)"
job_family="$(debug_profile_job_family)"

for dict_size in $DICT_SIZES; do
  dict_label="$(debug_profile_dict_label "$dict_size")"
  out_root="$OUT_ROOT_BASE/${run_family}-${dict_label}_${STAMP}"
  mkdir -p "$out_root"

  for case_def in "${DEBUG_CASES[@]}"; do
    IFS="|" read -r case_name patch_based patch_size patch_stride patch_reconstruction sparsity_level embedding_dim <<<"$case_def"
    if ! case_enabled "$case_name"; then
      continue
    fi

    case_label="$(debug_profile_case_label "$case_name")"
    run_name="$(debug_profile_slug "$dict_size" "$case_name")"
    run_dir="$out_root/$run_name"
    mkdir -p "$run_dir"
    run_script="$run_dir/run.sh"
    sbatch_script="$run_dir/sbatch.sh"
    log_base="$run_dir/$run_name"

    cat > "$run_script" <<EOF
#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="/scratch/$USER/.pydeps/laser_src_py311"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export WANDB_PROJECT="$WANDB_PROJECT"
export CASE_NAME="$run_name"
export CASE_LABEL="$case_label"
export DICT_LABEL="$dict_label"
export RUN_FAMILY="$run_family"
export EXP_SLUG="$run_name"
export WANDB_GROUP="$run_name"
export WANDB_STAGE1_NAME="${run_name}-s1"
export WANDB_STAGE2_NAME="${run_name}-s2"
export RUN_DIR="$run_dir"
export DATA_DIR="$DATA_DIR"
export NUM_DOWNSAMPLES="$NUM_DOWNSAMPLES"
export NUM_HIDDENS="$NUM_HIDDENS"
export NUM_RESIDUAL_BLOCKS="$NUM_RESIDUAL_BLOCKS"
export NUM_RESIDUAL_HIDDENS="$NUM_RESIDUAL_HIDDENS"
export ATTN_RESOLUTIONS='$ATTN_RESOLUTIONS'
export CHANNEL_MULTIPLIERS='$CHANNEL_MULTIPLIERS'
export BACKBONE_LATENT_CHANNELS="$BACKBONE_LATENT_CHANNELS"
export MAX_CH_MULT="$MAX_CH_MULT"
export DECODER_EXTRA_RESIDUAL_LAYERS="$DECODER_EXTRA_RESIDUAL_LAYERS"
export USE_MID_ATTENTION="$USE_MID_ATTENTION"
export NUM_EMBEDDINGS="$dict_size"
export SPARSITY_LEVEL="$sparsity_level"
export EMBEDDING_DIM="$embedding_dim"
export PATCH_BASED="$patch_based"
export PATCH_SIZE="$patch_size"
export PATCH_STRIDE="$patch_stride"
export PATCH_RECONSTRUCTION="$patch_reconstruction"
export COEFF_MAX="$COEFF_MAX"
export COEFF_BINS="$COEFF_BINS"
export COEFF_QUANTIZATION="$COEFF_QUANTIZATION"
export COEFF_MU="$COEFF_MU"
export VARIATIONAL_COEFFS="$VARIATIONAL_COEFFS"
export VARIATIONAL_COEFF_KL_WEIGHT="$VARIATIONAL_COEFF_KL_WEIGHT"
export VARIATIONAL_COEFF_PRIOR_STD="$VARIATIONAL_COEFF_PRIOR_STD"
export VARIATIONAL_COEFF_MIN_STD="$VARIATIONAL_COEFF_MIN_STD"
export STAGE1_LR="$STAGE1_LR"
export DICT_LR="$DICT_LR"
export STAGE1_EPOCHS="$STAGE1_EPOCHS"
export WARMUP_STEPS="$WARMUP_STEPS"
export MIN_LR_RATIO="$MIN_LR_RATIO"
export PERCEPTUAL_WEIGHT="$PERCEPTUAL_WEIGHT"
export S1_LOG_IMAGES_EVERY="$S1_LOG_IMAGES_EVERY"
export S1_DIAG_LOG_INTERVAL="$S1_DIAG_LOG_INTERVAL"
export S1_ENABLE_VAL_LATENT_VISUALS="$S1_ENABLE_VAL_LATENT_VISUALS"
export S1_COMPUTE_FID="$S1_COMPUTE_FID"
export S1_VAL_CHECK_INTERVAL="$S1_VAL_CHECK_INTERVAL"
export STAGE2_LR="$STAGE2_LR"
export STAGE2_EPOCHS="$STAGE2_EPOCHS"
export STAGE2_WARMUP_STEPS="$STAGE2_WARMUP_STEPS"
export STAGE2_MIN_LR_RATIO="$STAGE2_MIN_LR_RATIO"
export AR_D_MODEL="$AR_D_MODEL"
export AR_N_HEADS="$AR_N_HEADS"
export AR_N_LAYERS="$AR_N_LAYERS"
export AR_D_FF="$AR_D_FF"
export AR_AUTOREGRESSIVE_COEFFS="$AR_AUTOREGRESSIVE_COEFFS"
export AR_COEFF_LOSS_TYPE="$AR_COEFF_LOSS_TYPE"
export AR_COEFF_LOSS_WEIGHT="$AR_COEFF_LOSS_WEIGHT"
export AR_SAMPLE_COEFF_TEMPERATURE="$AR_SAMPLE_COEFF_TEMPERATURE"
export AR_SAMPLE_COEFF_MODE="$AR_SAMPLE_COEFF_MODE"
export S2_LOG_EVERY_N_STEPS="$S2_LOG_EVERY_N_STEPS"
export S2_SAMPLE_EVERY_N_EPOCHS="$S2_SAMPLE_EVERY_N_EPOCHS"
export S2_LOG_RECON_EVERY_N_STEPS="$S2_LOG_RECON_EVERY_N_STEPS"
export S2_SAMPLE_NUM_IMAGES="$S2_SAMPLE_NUM_IMAGES"
export SAMPLE_TEMP="$SAMPLE_TEMP"
export BATCH_SIZE="$BATCH_SIZE"
export STAGE2_BATCH_SIZE="$STAGE2_BATCH_SIZE"
export NUM_WORKERS="$NUM_WORKERS"
export DEVICES="$GPUS"
export TRAIN_STRATEGY="ddp"
export TRAIN_AR_STRATEGY="ddp"
export INSTALL_DEPS="true"

cd "$PROJECT_DIR"
bash "$RUNNER" all
EOF
    chmod +x "$run_script"

    cat > "$sbatch_script" <<EOF
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

if [[ -n "\$CONTAINER_BIN" ]]; then
  "\$CONTAINER_BIN" exec --nv \
    --bind "$PROJECT_DIR" \
    --bind "/scratch/$USER" \
    --bind "$DATA_DIR" \
    --bind "$run_dir" \
    --bind /dev/shm \
    "$IMAGE" \
    bash "$run_script"
else
  bash "$run_script"
fi
EOF
    chmod +x "$sbatch_script"

    sbatch_args=(
      --partition="$PARTITION"
      --job-name="${job_family}-${dict_label}-${case_label}"
      --nodes=1
      --ntasks=1
      --cpus-per-task="$CPUS_PER_TASK"
      --gres="gpu:${GPUS}"
      --mem="$MEM_MB"
      --time="$TIME_LIMIT"
      --chdir="$PROJECT_DIR"
      --output="${log_base}_%j.out"
      --error="${log_base}_%j.err"
    )
    if [[ -n "$EXCLUDE_NODES" ]]; then
      sbatch_args+=(--exclude="$EXCLUDE_NODES")
    fi
    sbatch_args+=("$sbatch_script")
    sbatch "${sbatch_args[@]}"

    submitted=$((submitted + 1))
  done
done

if (( submitted == 0 )); then
  echo "No cases submitted." >&2
  exit 1
fi

echo "[Sweep] submitted $submitted jobs via $RUNNER"
