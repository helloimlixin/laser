#!/bin/bash
# Submit a raw-waveform VCTK VQ-VAE stage-1 reconstruction sweep.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$SNAPSHOT_ROOT/laser_vctk_waveform_vqvae_$STAMP}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/vctk_waveform_vqvae_$STAMP}"
DATA_DIR="${DATA_DIR:-/scratch/$USER/datasets/VCTK-Corpus-0.92}"
PARTITION="${PARTITION:-gpu-redhat}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-96000}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
WANDB_MODE="${WANDB_MODE:-online}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-32}"
LEARNING_RATE="${LEARNING_RATE:-4.5e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-750}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-25}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1.0}"
CASE_FILTER="${CASE_FILTER:-}"
ENABLE_CODEBOOK_VISUALS="${ENABLE_CODEBOOK_VISUALS:-true}"
CODEBOOK_VISUAL_MAX_VECTORS="${CODEBOOK_VISUAL_MAX_VECTORS:-1024}"
SPEAKER_CONDITIONING="${SPEAKER_CONDITIONING:-false}"
SPEAKER_EMBEDDING_DIM="${SPEAKER_EMBEDDING_DIM:-64}"
SPEAKER_CONVERSION_LOG="${SPEAKER_CONVERSION_LOG:-true}"

mkdir -p "$SNAPSHOT_ROOT" "$OUT_ROOT"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "VCTK data directory not found: $DATA_DIR" >&2
  exit 1
fi

if [[ -e "$SNAPSHOT_DIR" ]]; then
  echo "Snapshot already exists: $SNAPSHOT_DIR" >&2
  exit 1
fi

EXCLUDES=(
  --exclude=.git
  --exclude=__pycache__
  --exclude=.pytest_cache
  --exclude=.mypy_cache
  --exclude=.ruff_cache
  --exclude=.tmp
  --exclude='.tmp_*'
  --exclude=cluster_logs
  --exclude=/wandb
  --exclude=runs
  --exclude='source_snapshot_*'
  --exclude='pre_variation_snapshot_*'
  --exclude='*.out'
  --exclude='*.err'
  --exclude='*.pyc'
  --exclude='*.pyo'
  --exclude='*.swp'
)

if command -v rsync >/dev/null 2>&1; then
  rsync -a "${EXCLUDES[@]}" "$PROJECT_DIR/" "$SNAPSHOT_DIR/"
else
  mkdir -p "$SNAPSHOT_DIR"
  tar -C "$PROJECT_DIR" \
    --exclude=.git \
    --exclude=__pycache__ \
    --exclude=.pytest_cache \
    --exclude=./wandb \
    --exclude=runs \
    -cf - . | tar -C "$SNAPSHOT_DIR" -xf -
fi

echo "Snapshot: $SNAPSHOT_DIR"
echo "Output root: $OUT_ROOT"

CASES=(
  "ds64_k512_d64_originalish|[4,4,4]|512|64|128|3|64"
  "ds64_k1024_d64_restart|[4,4,4]|1024|64|128|3|64"
  "ds64_k1024_d96_restart|[4,4,4]|1024|96|160|3|80"
  "ds64_k2048_d64_restart|[4,4,4]|2048|64|128|3|64"
  "ds32_k1024_d64_restart|[4,4,2]|1024|64|128|3|64"
)

submitted=0
for case_def in "${CASES[@]}"; do
  IFS="|" read -r case_name rates num_embeddings embedding_dim num_hiddens residual_blocks residual_hiddens <<<"$case_def"
  if [[ -n "$CASE_FILTER" && ",$CASE_FILTER," != *",$case_name,"* ]]; then
    continue
  fi
  run_dir="$OUT_ROOT/$case_name"
  mkdir -p "$run_dir"
  run_script="$run_dir/run.sh"
  sbatch_script="$run_dir/sbatch.sh"
  log_base="$run_dir/${case_name}"

  cat > "$run_script" <<EOF
#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="$run_dir/pydeps"
export PATH="\$PYTHONUSERBASE/bin:\$PATH"
export PYTHONPATH="$SNAPSHOT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export WANDB_MODE="$WANDB_MODE"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="\${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TMPDIR="/tmp/laser_\${SLURM_JOB_ID:-\$\$}"
export TEMP="\$TMPDIR"
export TMP="\$TMPDIR"

mkdir -p "\$TMPDIR" "$run_dir/wandb"

python -m pip install --user --quiet \
  scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \
  torch-fidelity matplotlib lpips soundfile pystoi

cd "$SNAPSHOT_DIR"
python train_stage1_autoencoder.py \
  seed=42 \
  output_dir="$run_dir" \
  model=vqvae_audio_waveform \
  data=vctk_waveform \
  data.data_dir="$DATA_DIR" \
  data.batch_size="$BATCH_SIZE" \
  data.num_workers="$NUM_WORKERS" \
  model.audio_downsample_rates="$rates" \
  model.num_embeddings="$num_embeddings" \
  model.embedding_dim="$embedding_dim" \
  model.num_hiddens="$num_hiddens" \
  model.num_residual_blocks="$residual_blocks" \
  model.num_residual_hiddens="$residual_hiddens" \
  model.decay=0.95 \
  model.codebook_init=true \
  model.dead_code_threshold=1.0 \
  model.audio_waveform_l1_weight=1.0 \
  model.audio_multires_stft_loss_weight=1.0 \
  model.audio_multires_stft_fft_sizes=[512,1024,2048] \
  model.compute_fid=false \
  model.enable_codebook_visuals="$ENABLE_CODEBOOK_VISUALS" \
  model.codebook_visual_max_vectors="$CODEBOOK_VISUAL_MAX_VECTORS" \
  model.speaker_conditioning="$SPEAKER_CONDITIONING" \
  model.speaker_embedding_dim="$SPEAKER_EMBEDDING_DIM" \
  model.speaker_conversion_log="$SPEAKER_CONVERSION_LOG" \
  train.learning_rate="$LEARNING_RATE" \
  train.warmup_steps="$WARMUP_STEPS" \
  train.min_lr_ratio="$MIN_LR_RATIO" \
  train.max_epochs="$STAGE1_EPOCHS" \
  train.accelerator=gpu \
  train.devices=1 \
  train.strategy=auto \
  train.precision="$PRECISION" \
  train.gradient_clip_val=1.0 \
  train.deterministic=false \
  train.log_every_n_steps="$LOG_EVERY_N_STEPS" \
  train.val_check_interval="$VAL_CHECK_INTERVAL" \
  train.run_test_after_fit=false \
  wandb.project="$WANDB_PROJECT" \
  wandb.name="vctk-waveform-vqvae-${case_name}" \
  wandb.group="vctk-waveform-vqvae-$STAMP" \
  wandb.tags="[vctk,waveform,vqvae,stage1,${case_name}]" \
  wandb.append_timestamp=false \
  wandb.save_dir="$run_dir/wandb"
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
    --bind "$SNAPSHOT_DIR" \
    --bind "/scratch/$USER" \
    --bind "/cache/home/$USER" \
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

  sbatch \
    --partition="$PARTITION" \
    --job-name="vctk-wav-${case_name}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --gres=gpu:1 \
    --mem="$MEM_MB" \
    --time="$TIME_LIMIT" \
    --chdir="$SNAPSHOT_DIR" \
    --output="${log_base}_%j.out" \
    --error="${log_base}_%j.err" \
    "$sbatch_script"

  submitted=$((submitted + 1))
done

echo "Submitted $submitted VCTK waveform VQ-VAE jobs."
