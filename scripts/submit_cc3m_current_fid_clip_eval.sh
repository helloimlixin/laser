#!/bin/bash
set -euo pipefail

#SBATCH --partition=gpu-redhat
#SBATCH --job-name=cc3m-metrics
#SBATCH --time=04:00:00

PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/Projects/laser}"
SELF="${SELF:-$PROJECT_DIR/scripts/submit_cc3m_current_fid_clip_eval.sh}"
RUN_ROOT="${RUN_ROOT:-/scratch/$USER/runs/laser/cc3m-official-rqt650m-compound-a16384-k2-20260808_124922}"
EVAL_DIR="${EVAL_DIR:?Set EVAL_DIR to the pinned evaluation directory}"
STAGE1="${STAGE1:-/scratch/$USER/runs/imagenet_x3h5cl0h_strict_bottleneck_sweep/imagenet-x3h5cl0h-strict-bottleneck-sweep-20260719_014434/k2-a16384/in256-rqvae-laser-8x8-a16384-k2/19072026_024726/best_rfid_slot3_model.pt}"
STAGE2="${STAGE2:?Set STAGE2 to the pinned checkpoint hardlink}"
DATA_ROOT="${CC3M_ROOT:-/scratch/$USER/Projects/data/cc3m}"
EXPECTED_EPOCH="${EXPECTED_EPOCH:?Set EXPECTED_EPOCH}"
EXPECTED_STEP="${EXPECTED_STEP:?Set EXPECTED_STEP}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-2048}"
SOURCE_RUN="${SOURCE_RUN:-helloimlixin-rutgers/laser/cc3mcmp0808124922}"
WANDB_EVAL_RUN_ID="${WANDB_EVAL_RUN_ID:-cc3mcmp0808124922-eval-s${EXPECTED_STEP}}"
DEFAULT_IMAGE_CACHE="/home/$USER/.apptainer/cache/oci-tmp/1241b86b11c283b8df64b2b73d365766bf4a172a8cc3c4e339a74c2d4b3e0083"
if [[ -f "$DEFAULT_IMAGE_CACHE" ]]; then
  IMAGE="${IMAGE:-$DEFAULT_IMAGE_CACHE}"
else
  IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
fi
PYDEPS="${PYDEPS:-/scratch/$USER/.pydeps/laser_stage2_py311}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-/scratch/$USER/.cache/clip}"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-/mnt/scratch/$USER/laser/$WANDB_EVAL_RUN_ID}"

load_container_runtime() {
  if ! command -v module >/dev/null 2>&1; then
    set +u
    [[ -f /usr/share/lmod/lmod/init/bash ]] && source /usr/share/lmod/lmod/init/bash
    set -u
  fi
  command -v singularity >/dev/null 2>&1 || module load singularity 2>/dev/null || true
  command -v singularity >/dev/null 2>&1 || {
    echo "singularity is unavailable" >&2
    return 2
  }
}

run_node() {
  local master_addr="${1:?master address required}"
  local master_port="${2:?master port required}"
  local node_rank="${SLURM_PROCID:?SLURM_PROCID required}"

  load_container_runtime
  mkdir -p "$EVAL_DIR" "$LOCAL_SCRATCH_ROOT" "$CLIP_CACHE_DIR"
  export APPTAINERENV_PYTHONUSERBASE="$PYDEPS"
  export APPTAINERENV_PYTHONPATH="$PROJECT_DIR"
  export APPTAINERENV_PROJECT_DIR="$PROJECT_DIR"
  export APPTAINERENV_WANDB_DIR="$LOCAL_SCRATCH_ROOT/wandb/run"
  export APPTAINERENV_WANDB_CACHE_DIR="$LOCAL_SCRATCH_ROOT/wandb/cache"
  export APPTAINERENV_WANDB_DATA_DIR="$LOCAL_SCRATCH_ROOT/wandb/data"
  export APPTAINERENV_WANDB_ARTIFACT_DIR="$LOCAL_SCRATCH_ROOT/wandb/artifacts"
  export APPTAINERENV_XDG_CACHE_HOME="$LOCAL_SCRATCH_ROOT/cache"
  export APPTAINERENV_TORCH_HOME="/scratch/$USER/.cache/torch"
  export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export APPTAINERENV_OMP_NUM_THREADS="$CPUS_PER_TASK"
  mkdir -p "$LOCAL_SCRATCH_ROOT/wandb/run" "$LOCAL_SCRATCH_ROOT/wandb/cache" \
    "$LOCAL_SCRATCH_ROOT/wandb/data" "$LOCAL_SCRATCH_ROOT/wandb/artifacts" \
    "$LOCAL_SCRATCH_ROOT/cache" "/scratch/$USER/.cache/torch"

  singularity exec --nv --bind /scratch --bind /mnt/scratch --bind /dev/shm \
    "$IMAGE" \
    python -m torch.distributed.run \
      --nnodes="$NODES" \
      --nproc_per_node="$GPUS_PER_NODE" \
      --node_rank="$node_rank" \
      --rdzv_id="${SLURM_JOB_ID}-${EXPECTED_STEP}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${master_addr}:${master_port}" \
      "$PROJECT_DIR/scripts/evaluate_cc3m_checkpoint_fid_clip.py" \
        --stage1 "$STAGE1" \
        --stage2 "$STAGE2" \
        --data "$DATA_ROOT" \
        --output "$EVAL_DIR" \
        --clip-cache-dir "$CLIP_CACHE_DIR" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size "$EVAL_BATCH_SIZE" \
        --num-workers 2 \
        --expected-epoch "$EXPECTED_EPOCH" \
        --expected-step "$EXPECTED_STEP" \
        --source-run "$SOURCE_RUN" \
        --wandb-eval-run-id "$WANDB_EVAL_RUN_ID"
}

run_allocation() {
  local node_count="${SLURM_NNODES:?SLURM_NNODES required}"
  local master_addr
  local master_port=$((24000 + SLURM_JOB_ID % 16000))
  master_addr="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
  echo "evaluation_allocation job=$SLURM_JOB_ID nodes=$node_count gpus_per_node=$GPUS_PER_NODE master=$master_addr:$master_port"
  srun --nodes="$node_count" --ntasks="$node_count" --ntasks-per-node=1 \
    --cpus-per-task="$CPUS_PER_TASK" --gres="gpu:$GPUS_PER_NODE" \
    --kill-on-bad-exit=1 \
    bash "$SELF" --node "$master_addr" "$master_port"
}

case "${1:-}" in
  --node)
    shift
    run_node "$@"
    ;;
  *)
    run_allocation
    ;;
esac
