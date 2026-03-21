#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
DATA_DIR="${DATA_DIR:-/cache/home/xl598/Projects/data/celeba}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/runs/laser_var_celeba128_quantized}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER/.pydeps/laser_var_py311}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-180}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
STAGE1_LR="${STAGE1_LR:-2e-4}"
STAGE2_LR="${STAGE2_LR:-1e-3}"
STAGE2_RQ_ATOM_LOSS_WEIGHT="${STAGE2_RQ_ATOM_LOSS_WEIGHT:-1.0}"
STAGE2_RQ_COEFF_LOSS_WEIGHT="${STAGE2_RQ_COEFF_LOSS_WEIGHT:-1.0}"
STAGE2_COEFF_LOSS_WEIGHT="${STAGE2_COEFF_LOSS_WEIGHT:-0.1}"
STAGE2_COEFF_HUBER_DELTA="${STAGE2_COEFF_HUBER_DELTA:-1.0}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-500}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.01}"
STAGE2_WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-0.01}"
COEFF_DEPTH_WEIGHTING="${COEFF_DEPTH_WEIGHTING:-none}"
COEFF_FOCAL_GAMMA="${COEFF_FOCAL_GAMMA:-0.0}"

BATCH_SIZE="${BATCH_SIZE:-32}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOKEN_NUM_WORKERS="${TOKEN_NUM_WORKERS:-0}"
TOKEN_SUBSET="${TOKEN_SUBSET:-0}"
REBUILD_TOKEN_CACHE="${REBUILD_TOKEN_CACHE:-false}"

QUANTIZE_SPARSE_COEFFS="${QUANTIZE_SPARSE_COEFFS:-true}"
NUM_HIDDENS="${NUM_HIDDENS:-128}"
AE_NUM_DOWNSAMPLES="${AE_NUM_DOWNSAMPLES:-4}"
NUM_RES_LAYERS="${NUM_RES_LAYERS:-2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
NUM_ATOMS="${NUM_ATOMS:-1024}"
SPARSITY_LEVEL="${SPARSITY_LEVEL:-8}"
N_BINS="${N_BINS:-256}"
COEF_MAX="${COEF_MAX:-3.0}"
COEF_QUANTIZATION="${COEF_QUANTIZATION:-uniform}"
COEF_MU="${COEF_MU:-0.0}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
PATCH_BASED="${PATCH_BASED:-false}"
PATCH_SIZE="${PATCH_SIZE:-4}"
PATCH_STRIDE="${PATCH_STRIDE:-2}"
PATCH_RECONSTRUCTION="${PATCH_RECONSTRUCTION:-center_crop}"
if [[ -z "${STAGE2_COEFF_LOSS_TYPE+x}" ]]; then
  if [[ "${PATCH_BASED}" == "1" || "${PATCH_BASED}" == "true" || "${PATCH_BASED}" == "TRUE" || "${PATCH_BASED}" == "yes" || "${PATCH_BASED}" == "YES" ]]; then
    STAGE2_COEFF_LOSS_TYPE="mse"
  elif [[ "${QUANTIZE_SPARSE_COEFFS}" == "0" || "${QUANTIZE_SPARSE_COEFFS}" == "false" || "${QUANTIZE_SPARSE_COEFFS}" == "FALSE" || "${QUANTIZE_SPARSE_COEFFS}" == "no" || "${QUANTIZE_SPARSE_COEFFS}" == "NO" ]]; then
    STAGE2_COEFF_LOSS_TYPE="mse"
  else
    STAGE2_COEFF_LOSS_TYPE="gt_atom_recon_mse"
  fi
fi

VAR_D_MODEL="${VAR_D_MODEL:-512}"
VAR_HEADS="${VAR_HEADS:-8}"
VAR_LAYERS="${VAR_LAYERS:-12}"
VAR_FF="${VAR_FF:-1024}"
VAR_DROPOUT="${VAR_DROPOUT:-0.1}"
VAR_GLOBAL_TOKENS="${VAR_GLOBAL_TOKENS:-16}"
STAGE2_SAMPLE_EVERY_STEPS="${STAGE2_SAMPLE_EVERY_STEPS:-2000}"
STAGE2_SAMPLE_BATCH_SIZE="${STAGE2_SAMPLE_BATCH_SIZE:-32}"
STAGE2_SAMPLE_CANDIDATE_FACTOR="${STAGE2_SAMPLE_CANDIDATE_FACTOR:-4}"
STAGE2_SAMPLE_TEMPERATURE="${STAGE2_SAMPLE_TEMPERATURE:-0.9}"
STAGE2_SAMPLE_TOP_K="${STAGE2_SAMPLE_TOP_K:-256}"
STAGE2_SAMPLE_TOP_P="${STAGE2_SAMPLE_TOP_P:-0.95}"
STAGE2_SAMPLE_TEMPERATURE_END="${STAGE2_SAMPLE_TEMPERATURE_END:-1.0}"
STAGE2_SAMPLE_IMAGE_SIZE="${STAGE2_SAMPLE_IMAGE_SIZE:-128}"
STAGE2_AMP="${STAGE2_AMP:-true}"
STAGE2_AMP_DTYPE="${STAGE2_AMP_DTYPE:-auto}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-laser-scratch}"
WANDB_NAME="${WANDB_NAME:-laser_var_${SLURM_NNODES}n_${SLURM_NTASKS}g_s1${STAGE1_EPOCHS}_s2${STAGE2_EPOCHS}}"

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    set +u
    source /etc/profile.d/modules.sh
    set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u
    source /usr/share/Modules/init/bash
    set -u
  fi
fi
if ! command -v singularity >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  fi
fi
command -v singularity >/dev/null 2>&1 || { echo singularity_not_found >&2; exit 1; }

USER_NAME="$(id -un)"
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="$((29000 + SLURM_JOB_ID % 1000))"

mkdir -p "$OUT_DIR" "$PYTHONUSERBASE_DIR" "$(dirname "$WANDB_API_KEY_FILE")"
if [[ -f "$WANDB_API_KEY_FILE" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "$WANDB_API_KEY_FILE")"
fi
export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR MASTER_PORT

if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$ROOT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$OUT_DIR" "$IMAGE" python3 -c "import scipy, wandb, torch_fidelity" >/dev/null 2>&1; then
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" singularity exec --bind "$ROOT_DIR" --bind "/scratch/$USER_NAME" --bind "$DATA_DIR" --bind "$OUT_DIR" "$IMAGE" python3 -m pip install --user scipy wandb torch-fidelity
fi

RUNNER="$OUT_DIR/slurm_laser_var_${SLURM_JOB_ID}.sh"
cat > "$RUNNER" <<EOF_RUNNER
#!/bin/bash
set -euo pipefail
export RANK="\$SLURM_PROCID"
export WORLD_SIZE="\$SLURM_NTASKS"
export LOCAL_RANK="\$SLURM_LOCALID"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE\${PYTHONPATH:+:\$PYTHONPATH}"
STAGE1_LOG="$OUT_DIR/var_stage1_bootstrap_${SLURM_JOB_ID}.log"
RUN_DIR_FILE="$OUT_DIR/var_run_dir_${SLURM_JOB_ID}.txt"
python3 "$ROOT_DIR/proto.py" \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --image_size 128 \
  --out_dir "$OUT_DIR" \
  --dist_timeout_minutes "$DIST_TIMEOUT_MINUTES" \
  --stage1_epochs "$STAGE1_EPOCHS" \
  --stage2_epochs 0 \
  --stage1_lr "$STAGE1_LR" \
  --num_workers "$NUM_WORKERS" \
  --token_num_workers "$TOKEN_NUM_WORKERS" \
  --batch_size "$BATCH_SIZE" \
  --ae_num_downsamples "$AE_NUM_DOWNSAMPLES" \
  --embedding_dim "$EMBEDDING_DIM" \
  --num_atoms "$NUM_ATOMS" \
  --sparsity_level "$SPARSITY_LEVEL" \
  --n_bins "$N_BINS" \
  --coef_max "$COEF_MAX" \
  --quantize_sparse_coeffs "$QUANTIZE_SPARSE_COEFFS" \
  --coef_quantization "$COEF_QUANTIZATION" \
  --coef_mu "$COEF_MU" \
  --commitment_cost "$COMMITMENT_COST" \
  --patch_size "$PATCH_SIZE" \
  --patch_stride "$PATCH_STRIDE" \
  --patch_reconstruction "$PATCH_RECONSTRUCTION" \
  --rfid_num_samples 0 \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "${WANDB_NAME}_stage1" \
  > >(if [[ "\$RANK" == "0" ]]; then tee "\$STAGE1_LOG"; else cat >/dev/null; fi) \
  2> >(if [[ "\$RANK" == "0" ]]; then tee -a "\$STAGE1_LOG" >&2; else cat >/dev/null; fi)
if [[ "\$RANK" == "0" ]]; then
  python3 - "\$STAGE1_LOG" "\$RUN_DIR_FILE" <<'PY'
import re
import sys
from pathlib import Path
log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
text = log_path.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"Outputs saved to:\s*(.+)", text)
if not matches:
    raise SystemExit("Could not determine run_dir from stage1 log")
run_dir = matches[-1].strip()
out_path.write_text(run_dir + "\n", encoding="utf-8")
print("[launcher] run_dir={}".format(run_dir))
PY
fi
while [[ ! -f "\$RUN_DIR_FILE" ]]; do sleep 1; done
RUN_DIR="\$(tr -d '\\r\\n' < "\$RUN_DIR_FILE")"
VAR_ARGS=()
if [[ "$REBUILD_TOKEN_CACHE" == "true" ]]; then
  VAR_ARGS+=(--rebuild_token_cache)
fi
if [[ "$PATCH_BASED" == "true" ]]; then
  VAR_ARGS+=(--patch_based)
fi
exec python3 "$ROOT_DIR/var_stage2.py" \
  --run_dir "\$RUN_DIR" \
  --dataset celeba \
  --data_dir "$DATA_DIR" \
  --image_size 128 \
  --dist_timeout_minutes "$DIST_TIMEOUT_MINUTES" \
  --stage2_epochs "$STAGE2_EPOCHS" \
  --stage2_lr "$STAGE2_LR" \
  --stage2_rq_atom_loss_weight "$STAGE2_RQ_ATOM_LOSS_WEIGHT" \
  --stage2_rq_coeff_loss_weight "$STAGE2_RQ_COEFF_LOSS_WEIGHT" \
  --stage2_coeff_loss_weight "$STAGE2_COEFF_LOSS_WEIGHT" \
  --stage2_coeff_loss_type "$STAGE2_COEFF_LOSS_TYPE" \
  --stage2_coeff_huber_delta "$STAGE2_COEFF_HUBER_DELTA" \
  --stage2_warmup_steps "$STAGE2_WARMUP_STEPS" \
  --stage2_min_lr_ratio "$STAGE2_MIN_LR_RATIO" \
  --stage2_weight_decay "$STAGE2_WEIGHT_DECAY" \
  --coeff_depth_weighting "$COEFF_DEPTH_WEIGHTING" \
  --coeff_focal_gamma "$COEFF_FOCAL_GAMMA" \
  --num_workers "$NUM_WORKERS" \
  --token_num_workers "$TOKEN_NUM_WORKERS" \
  --batch_size "$BATCH_SIZE" \
  --stage2_batch_size "$STAGE2_BATCH_SIZE" \
  --token_subset "$TOKEN_SUBSET" \
  --num_hiddens "$NUM_HIDDENS" \
  --ae_num_downsamples "$AE_NUM_DOWNSAMPLES" \
  --num_res_layers "$NUM_RES_LAYERS" \
  --embedding_dim "$EMBEDDING_DIM" \
  --num_atoms "$NUM_ATOMS" \
  --sparsity_level "$SPARSITY_LEVEL" \
  --n_bins "$N_BINS" \
  --coef_max "$COEF_MAX" \
  --quantize_sparse_coeffs "$QUANTIZE_SPARSE_COEFFS" \
  --coef_quantization "$COEF_QUANTIZATION" \
  --coef_mu "$COEF_MU" \
  --commitment_cost "$COMMITMENT_COST" \
  --patch_size "$PATCH_SIZE" \
  --patch_stride "$PATCH_STRIDE" \
  --patch_reconstruction "$PATCH_RECONSTRUCTION" \
  --var_d_model "$VAR_D_MODEL" \
  --var_heads "$VAR_HEADS" \
  --var_layers "$VAR_LAYERS" \
  --var_ff "$VAR_FF" \
  --var_dropout "$VAR_DROPOUT" \
  --var_global_tokens "$VAR_GLOBAL_TOKENS" \
  --stage2_sample_every_steps "$STAGE2_SAMPLE_EVERY_STEPS" \
  --stage2_sample_batch_size "$STAGE2_SAMPLE_BATCH_SIZE" \
  --stage2_sample_candidate_factor "$STAGE2_SAMPLE_CANDIDATE_FACTOR" \
  --stage2_sample_temperature "$STAGE2_SAMPLE_TEMPERATURE" \
  --stage2_sample_temperature_end "$STAGE2_SAMPLE_TEMPERATURE_END" \
  --stage2_sample_top_k "$STAGE2_SAMPLE_TOP_K" \
  --stage2_sample_top_p "$STAGE2_SAMPLE_TOP_P" \
  --stage2_sample_image_size "$STAGE2_SAMPLE_IMAGE_SIZE" \
  --stage2_amp "$STAGE2_AMP" \
  --stage2_amp_dtype "$STAGE2_AMP_DTYPE" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "${WANDB_NAME}_var" \
  "\${VAR_ARGS[@]}"
EOF_RUNNER
chmod +x "$RUNNER"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 nvidia-smi
srun --ntasks-per-node="$SLURM_NTASKS_PER_NODE" --gpus-per-task=1 singularity exec --nv \
  --bind "$ROOT_DIR" \
  --bind "/scratch/$USER_NAME" \
  --bind "$DATA_DIR" \
  --bind "$OUT_DIR" \
  "$IMAGE" \
  bash "$RUNNER"
