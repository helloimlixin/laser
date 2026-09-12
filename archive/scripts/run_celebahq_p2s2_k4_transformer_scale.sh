#!/usr/bin/env bash
# Stage-2-only capacity sweep for a CelebA-HQ p2s2/k4/a4096 quantized sparse
# cache that was built from an adversarially tuned stage-1 checkpoint.
#
# Do not default to an old cache here: pass CACHE=... and confirm that it came
# from adversarial stage-1 training with ADVERSARIAL_CACHE_CONFIRMED=true.
set -euo pipefail

cd /home/xl598/Projects/laser

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
VARIANT="${VARIANT:-rq_paper_350m}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba_hq}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_GROUP="${WANDB_GROUP:-celebahq_p2s2_k4_transformer_scale_${STAMP}}"
RUN_ROOT="${RUN_ROOT:-/home/xl598/Projects/laser/runs/celebahq_p2s2_k4_transformer_scale_${STAMP}}"
CACHE="${CACHE:-}"
ADVERSARIAL_CACHE_CONFIRMED="${ADVERSARIAL_CACHE_CONFIRMED:-false}"
ALLOW_UNVERIFIED_CACHE="${ALLOW_UNVERIFIED_CACHE:-false}"

DEVICES="${DEVICES:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ -z "${STRATEGY+x}" ]]; then
  if [[ "${DEVICES}" == "1" ]]; then
    STRATEGY="auto"
  else
    STRATEGY="ddp"
  fi
fi

STAGE2_EPOCHS="${STAGE2_EPOCHS:-200}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:--1}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-4}"
STAGE2_SAMPLE_EVERY_N_EPOCHS="${STAGE2_SAMPLE_EVERY_N_EPOCHS:-10}"
STAGE2_SAMPLE_NUM_IMAGES="${STAGE2_SAMPLE_NUM_IMAGES:-8}"
STAGE2_SAMPLE_TEMPERATURE="${STAGE2_SAMPLE_TEMPERATURE:-0.7}"
STAGE2_SAMPLE_TOP_K="${STAGE2_SAMPLE_TOP_K:-250}"
STAGE2_COMPUTE_GENERATION_FID="${STAGE2_COMPUTE_GENERATION_FID:-true}"
STAGE2_GENERATION_METRIC_NUM_SAMPLES="${STAGE2_GENERATION_METRIC_NUM_SAMPLES:-256}"
STAGE2_SAVE_FINAL_SAMPLES_AFTER_FIT="${STAGE2_SAVE_FINAL_SAMPLES_AFTER_FIT:-true}"
STAGE2_RUN_TEST_AFTER_FIT="${STAGE2_RUN_TEST_AFTER_FIT:-false}"
PRECISION="${PRECISION:-bf16-mixed}"

export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/home/xl598/Projects/laser
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -z "${CACHE}" ]]; then
  {
    echo "CACHE is required for stage-2-only transformer scaling."
    echo "Build it from adversarial stage 1 with scripts/run_celebahq_p2s2_k4_quant_pipeline.sh, then pass CACHE=/path/token_cache_q256.pt."
  } >&2
  exit 1
fi
if [[ ! -f "${CACHE}" ]]; then
  echo "Token cache not found: ${CACHE}" >&2
  exit 1
fi
if ! truthy "${ALLOW_UNVERIFIED_CACHE}" && ! truthy "${ADVERSARIAL_CACHE_CONFIRMED}"; then
  {
    echo "Refusing stage-2-only training on an unverified cache:"
    echo "  CACHE=${CACHE}"
    echo "Set ADVERSARIAL_CACHE_CONFIRMED=true only when this cache was produced from adversarial stage-1 training."
  } >&2
  exit 1
fi

run_one() {
  local variant="$1"
  local ar_type="sparse_spatial_depth"
  local d_model="768"
  local n_heads="12"
  local n_layers="16"
  local d_ff="3072"
  local batch_size="8"
  local lr="2.0e-4"
  local warmup="1500"
  local min_lr_ratio="0.05"
  local dropout="0.1"
  local window_sites="0"
  local global_tokens="8"
  local coeff_huber_delta="0.25"
  local sample_top_k="${STAGE2_SAMPLE_TOP_K}"
  local sample_temp="${STAGE2_SAMPLE_TEMPERATURE}"

  case "${variant}" in
    rq_paper_350m|rq_large)
      # RQ-Transformer/VQGAN-scale factorization. This matches the common
      # 1024-dim, 16-head, 24-layer scale used by the face-domain paper configs,
      # adapted to our spatial-depth sparse-token prior.
      ar_type="sparse_spatial_depth"
      d_model="${RQ_D_MODEL:-1024}"
      n_heads="${RQ_N_HEADS:-16}"
      n_layers="${RQ_N_LAYERS:-24}"
      d_ff="${RQ_D_FF:-4096}"
      batch_size="${RQ_BATCH_SIZE:-4}"
      lr="${RQ_LR:-2.0e-4}"
      warmup="${RQ_WARMUP_STEPS:-2000}"
      coeff_huber_delta="${RQ_COEFF_HUBER_DELTA:-0.25}"
      ;;
    vqgan_gpt)
      # VQGAN-style flattened causal GPT over the whole 8*8*8 token stream.
      ar_type="gpt"
      d_model="${GPT_D_MODEL:-1024}"
      n_heads="${GPT_N_HEADS:-16}"
      n_layers="${GPT_N_LAYERS:-24}"
      d_ff="${GPT_D_FF:-4096}"
      batch_size="${GPT_BATCH_SIZE:-2}"
      lr="${GPT_LR:-2.0e-4}"
      warmup="${GPT_WARMUP_STEPS:-2000}"
      window_sites="${GPT_WINDOW_SITES:-0}"
      sample_top_k="${GPT_SAMPLE_TOP_K:-250}"
      ;;
    baseline_long)
      # Same capacity as de06l508/local baseline, but without the 50k-step cap.
      ar_type="sparse_spatial_depth"
      d_model="${BASELINE_D_MODEL:-512}"
      n_heads="${BASELINE_N_HEADS:-8}"
      n_layers="${BASELINE_N_LAYERS:-8}"
      d_ff="${BASELINE_D_FF:-2048}"
      batch_size="${BASELINE_BATCH_SIZE:-8}"
      lr="${BASELINE_LR:-5.0e-4}"
      warmup="${BASELINE_WARMUP_STEPS:-167}"
      coeff_huber_delta="${BASELINE_COEFF_HUBER_DELTA:-0.25}"
      ;;
    *)
      echo "Unknown VARIANT=${variant}; expected rq_paper_350m, rq_large, vqgan_gpt, baseline_long, or all" >&2
      exit 2
      ;;
  esac

  local out_dir="${RUN_ROOT}/${variant}/stage2"
  local log_dir="${RUN_ROOT}/${variant}/logs"
  mkdir -p "${out_dir}" "${log_dir}" "${out_dir}/wandb"

  echo "[$(date --iso-8601=seconds)] starting ${variant}"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "CACHE=${CACHE}"
  echo "AR=${ar_type} d_model=${d_model} heads=${n_heads} layers=${n_layers} d_ff=${d_ff}"
  echo "batch_size=${batch_size} epochs=${STAGE2_EPOCHS} max_steps=${STAGE2_MAX_STEPS} lr=${lr} warmup=${warmup}"

  "${PYTHON_BIN}" train_stage2_prior.py \
    output_dir="${out_dir}" \
    hydra.run.dir="${out_dir}/hydra" \
    token_cache_path="${CACHE}" \
    seed=42 \
    ar.type="${ar_type}" \
    ar.window_sites="${window_sites}" \
    ar.n_global_spatial_tokens="${global_tokens}" \
    ar.d_model="${d_model}" \
    ar.n_heads="${n_heads}" \
    ar.n_layers="${n_layers}" \
    ar.d_ff="${d_ff}" \
    ar.dropout="${dropout}" \
    ar.learning_rate="${lr}" \
    ar.warmup_steps="${warmup}" \
    ar.max_steps="${STAGE2_MAX_STEPS}" \
    ar.min_lr_ratio="${min_lr_ratio}" \
    ar.coeff_loss_type=auto \
    ar.coeff_loss_weight=1.0 \
    ar.coeff_huber_delta="${coeff_huber_delta}" \
    train_ar.accelerator=gpu \
    train_ar.devices="${DEVICES}" \
    train_ar.strategy="${STRATEGY}" \
    train_ar.precision="${PRECISION}" \
    train_ar.max_epochs="${STAGE2_EPOCHS}" \
    train_ar.batch_size="${batch_size}" \
    train_ar.gradient_clip_val=1.0 \
    train_ar.log_every_n_steps=20 \
    train_ar.val_check_interval=1.0 \
    train_ar.sample_every_n_epochs="${STAGE2_SAMPLE_EVERY_N_EPOCHS}" \
    train_ar.sample_num_images="${STAGE2_SAMPLE_NUM_IMAGES}" \
    train_ar.sample_temperature="${sample_temp}" \
    train_ar.sample_top_k="${sample_top_k}" \
    train_ar.sample_log_to_wandb=true \
    train_ar.compute_generation_fid="${STAGE2_COMPUTE_GENERATION_FID}" \
    train_ar.generation_metric_num_samples="${STAGE2_GENERATION_METRIC_NUM_SAMPLES}" \
    train_ar.run_test_after_fit="${STAGE2_RUN_TEST_AFTER_FIT}" \
    train_ar.save_final_samples_after_fit="${STAGE2_SAVE_FINAL_SAMPLES_AFTER_FIT}" \
    data.dataset=celebahq \
    data.data_dir="${DATA_DIR}" \
    data.image_size=256 \
    data.num_workers="${STAGE2_NUM_WORKERS}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.name="celebahq_s2_p2s2_k4_a4096_${variant}_${STAMP}" \
    wandb.tags="[train,laser,celebahq,stage2,transformer,${variant},p2s2,k4,a4096,q256]" \
    wandb.append_timestamp=false \
    wandb.save_dir="${out_dir}/wandb"

  echo "[$(date --iso-8601=seconds)] finished ${variant}: ${out_dir}"
}

mkdir -p "${RUN_ROOT}"

case "${VARIANT}" in
  all)
    run_one rq_paper_350m
    run_one vqgan_gpt
    ;;
  *)
    run_one "${VARIANT}"
    ;;
esac
