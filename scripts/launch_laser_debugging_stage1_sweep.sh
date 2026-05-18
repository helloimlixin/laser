#!/usr/bin/env bash
# Launch stage-1-only LASER debugging sweep across vision + audio datasets.
#
# Vision: celeba, celebahq, ffhq, coco × {laser-f32, laser-f16, vqvae-f16}
# Audio:  vctk (spectrogram), maestro (waveform) × {laser, vqvae}
# Target W&B project: laser-debugging
# Submits via scripts/submit_multimodal_sweep.py (SLURM sbatch per job).
#
# Authored May 2026 to validate the A2/A3/A4 stage-1 changes; uses --stage1-only
# (no token cache / stage 2). Total: 12 image + 4 audio = 16 SLURM jobs.

set -euo pipefail

PROJECT="${PROJECT:-laser-debugging}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"  # required >0 by submit_multimodal_sweep.py; unused with --stage1-only
PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
IMAGE_MEM_MB="${IMAGE_MEM_MB:-240000}"
AUDIO_MEM_MB="${AUDIO_MEM_MB:-128000}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-/scratch/$USER/runs/laser_debugging_stage1}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/scratch/$USER/submission_snapshots}"
COCO_DIR="${COCO_DIR:-/scratch/$USER/data/coco}"
FFHQ_DIR="${FFHQ_DIR:-/scratch/$USER/datasets/ffhq}"
FFHQ_TRAIN_CROP_SIZE="${FFHQ_TRAIN_CROP_SIZE:-192}"
VCTK_DIR="${VCTK_DIR:-/scratch/$USER/datasets/VCTK-Corpus-0.92}"
MAESTRO_DIR="${MAESTRO_DIR:-/scratch/$USER/datasets/maestro/maestro-v3.0.0}"
CELEBA_DIR="${CELEBA_DIR:-/scratch/$USER/datasets/celeba_packed_128}"
CELEBAHQ_DIR="${CELEBAHQ_DIR:-/scratch/$USER/datasets/celebahq_packed_256}"
PYTHON_SUBMIT="${PYTHON_SUBMIT:-python3}"
DRY_RUN="${DRY_RUN:-0}"

LASER_BACKBONE_LATENT_CHANNELS="${LASER_BACKBONE_LATENT_CHANNELS:-512}"
LASER_EMBEDDING_DIM="${LASER_EMBEDDING_DIM:-512}"
LASER_SPARSITY_LEVEL="${LASER_SPARSITY_LEVEL:-16}"
LASER_COEF_MAX="${LASER_COEF_MAX:-16.0}"
LASER_COMMITMENT_COST="${LASER_COMMITMENT_COST:-0.05}"
LASER_RECON_MSE_WEIGHT="${LASER_RECON_MSE_WEIGHT:-0.25}"
LASER_RECON_L1_WEIGHT="${LASER_RECON_L1_WEIGHT:-1.0}"
LASER_RECON_EDGE_WEIGHT="${LASER_RECON_EDGE_WEIGHT:-0.25}"

# ---------- preflight ----------
if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  module load python/3.8.2 2>/dev/null || module load python 2>/dev/null || true
  hash -r 2>/dev/null || true
fi
if ! "$PYTHON_SUBMIT" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "ERROR: submit_multimodal_sweep.py requires Python >= 3.8; set PYTHON_SUBMIT." >&2
  exit 2
fi

for d in "$CELEBA_DIR" "$CELEBAHQ_DIR" "$FFHQ_DIR" "$COCO_DIR" "$VCTK_DIR" "$MAESTRO_DIR"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: dataset directory not found: $d" >&2
    exit 1
  fi
done

# ---------- shared args ----------
COMMON_ARGS=(
  --full-training
  --stage1-only
  --stage1-epochs "$STAGE1_EPOCHS"
  --stage2-epochs "$STAGE2_EPOCHS"
  --partition "$PARTITION"
  --time-limit "$TIME_LIMIT"
  --gpus "$GPUS"
  --cpus-per-task "$CPUS_PER_TASK"
  --project "$PROJECT"
  --run-root-base "$RUN_ROOT_BASE"
  --snapshot-root "$SNAPSHOT_ROOT"
  --coco-dir "$COCO_DIR"
  --ffhq-dir "$FFHQ_DIR"
  --vctk-dir "$VCTK_DIR"
  --maestro-dir "$MAESTRO_DIR"
)
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

COMMON_STAGE1=(
  --stage1-override train.run_test_after_fit=false
  --stage1-override train.gradient_clip_val=1.0
  --stage1-override model.out_tanh=true
)
# compute_fid is only meaningful for 3-channel image runs (the Inception net
# wants RGB and the audio path skips val_rfid.update()). Apply per-modality.
IMAGE_STAGE1=(
  --stage1-override model.compute_fid=true
)
AUDIO_STAGE1=(
  --stage1-override model.compute_fid=false
)

# ---------- per-dataset image knobs (mirrors launch_image_debugging_sweep.sh, +ffhq) ----------
image_num_workers() {
  case "$1" in
    celebahq|ffhq) printf "0" ;;
    *) printf "4" ;;
  esac
}

attn_f32_for() {
  case "$1" in
    coco) printf "[16]" ;;
    celebahq|ffhq) printf "[8]" ;;
    celeba) printf "[4]" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

laser_batch_for() {
  case "$1" in
    coco) printf "1" ;;
    celebahq) printf "4" ;;
    ffhq) printf "12" ;;
    celeba) printf "8" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

laser_eval_batch_for() {
  case "$1" in
    ffhq) printf "6" ;;
    *) laser_batch_for "$1" ;;
  esac
}

vqvae_batch_for() {
  case "$1" in
    coco) printf "2" ;;
    celebahq) printf "8" ;;
    ffhq) printf "24" ;;
    celeba) printf "16" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

vqvae_eval_batch_for() {
  case "$1" in
    ffhq) printf "12" ;;
    *) vqvae_batch_for "$1" ;;
  esac
}

train_crop_size_for() {
  case "$1" in
    ffhq) printf "%s" "$FFHQ_TRAIN_CROP_SIZE" ;;
    *) printf "0" ;;
  esac
}

stage1_warmup_for() {
  case "$1" in
    ffhq) printf "250" ;;
    *) printf "0" ;;
  esac
}

stage1_min_lr_ratio_for() {
  case "$1" in
    ffhq) printf "0.05" ;;
    *) printf "0.01" ;;
  esac
}

laser_lr_for() {
  case "$1" in
    ffhq) printf "7.0e-5" ;;
    *) printf "4.0e-5" ;;
  esac
}

vqvae_lr_for() {
  case "$1" in
    ffhq) printf "1.75e-4" ;;
    *) printf "1.0e-4" ;;
  esac
}

# ---------- vision submissions ----------
submit_laser_f32() {
  local dataset="$1"
  local workers batch eval_batch train_crop attn lr warmup min_lr_ratio
  workers="$(image_num_workers "$dataset")"
  batch="$(laser_batch_for "$dataset")"
  eval_batch="$(laser_eval_batch_for "$dataset")"
  train_crop="$(train_crop_size_for "$dataset")"
  attn="$(attn_f32_for "$dataset")"
  lr="$(laser_lr_for "$dataset")"
  warmup="$(stage1_warmup_for "$dataset")"
  min_lr_ratio="$(stage1_min_lr_ratio_for "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --mem-mb "$IMAGE_MEM_MB" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "stage1-${dataset}-laser-f32-k16384-z${LASER_EMBEDDING_DIM}-e${STAGE1_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${IMAGE_STAGE1[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override data.eval_batch_size="$eval_batch" \
    --stage1-override data.train_crop_size="$train_crop" \
    --stage1-override train.warmup_steps="$warmup" \
    --stage1-override train.min_lr_ratio="$min_lr_ratio" \
    --stage1-override model.backbone=vqgan \
    --stage1-override model.num_downsamples=5 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4,4] \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_residual_blocks=2 \
    --stage1-override model.num_residual_hiddens=64 \
    --stage1-override model.backbone_latent_channels="$LASER_BACKBONE_LATENT_CHANNELS" \
    --stage1-override model.embedding_dim="$LASER_EMBEDDING_DIM" \
    --stage1-override model.num_embeddings=16384 \
    --stage1-override model.sparsity_level="$LASER_SPARSITY_LEVEL" \
    --stage1-override "model.attn_resolutions=${attn}" \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.decoder_extra_residual_layers=1 \
    --stage1-override model.bottleneck_loss_weight=0.25 \
    --stage1-override model.commitment_cost="$LASER_COMMITMENT_COST" \
    --stage1-override model.coef_max="$LASER_COEF_MAX" \
    --stage1-override model.recon_mse_weight="$LASER_RECON_MSE_WEIGHT" \
    --stage1-override model.recon_l1_weight="$LASER_RECON_L1_WEIGHT" \
    --stage1-override model.recon_edge_weight="$LASER_RECON_EDGE_WEIGHT" \
    --stage1-override model.perceptual_weight=0.0 \
    --stage1-override train.learning_rate="$lr" \
    --stage1-override train.precision=bf16-mixed
}

submit_laser_f16() {
  local dataset="$1"
  local workers batch eval_batch train_crop lr warmup min_lr_ratio
  workers="$(image_num_workers "$dataset")"
  batch="$(laser_batch_for "$dataset")"
  eval_batch="$(laser_eval_batch_for "$dataset")"
  train_crop="$(train_crop_size_for "$dataset")"
  lr="$(laser_lr_for "$dataset")"
  warmup="$(stage1_warmup_for "$dataset")"
  min_lr_ratio="$(stage1_min_lr_ratio_for "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --mem-mb "$IMAGE_MEM_MB" \
    --cases "$dataset" \
    --model-family laser \
    --run-label "stage1-${dataset}-laser-f16-k16384-z${LASER_EMBEDDING_DIM}-e${STAGE1_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${IMAGE_STAGE1[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override data.eval_batch_size="$eval_batch" \
    --stage1-override data.train_crop_size="$train_crop" \
    --stage1-override train.warmup_steps="$warmup" \
    --stage1-override train.min_lr_ratio="$min_lr_ratio" \
    --stage1-override model.backbone=vqgan \
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.channel_multipliers=[1,1,2,2,4] \
    --stage1-override model.num_hiddens=128 \
    --stage1-override model.num_residual_blocks=2 \
    --stage1-override model.num_residual_hiddens=64 \
    --stage1-override model.backbone_latent_channels="$LASER_BACKBONE_LATENT_CHANNELS" \
    --stage1-override model.embedding_dim="$LASER_EMBEDDING_DIM" \
    --stage1-override model.num_embeddings=16384 \
    --stage1-override model.sparsity_level="$LASER_SPARSITY_LEVEL" \
    --stage1-override model.attn_resolutions=[] \
    --stage1-override model.use_mid_attention=true \
    --stage1-override model.decoder_extra_residual_layers=1 \
    --stage1-override model.bottleneck_loss_weight=0.25 \
    --stage1-override model.commitment_cost="$LASER_COMMITMENT_COST" \
    --stage1-override model.coef_max="$LASER_COEF_MAX" \
    --stage1-override model.recon_mse_weight="$LASER_RECON_MSE_WEIGHT" \
    --stage1-override model.recon_l1_weight="$LASER_RECON_L1_WEIGHT" \
    --stage1-override model.recon_edge_weight="$LASER_RECON_EDGE_WEIGHT" \
    --stage1-override model.perceptual_weight=0.0 \
    --stage1-override train.learning_rate="$lr" \
    --stage1-override train.precision=bf16-mixed
}

submit_vqvae_f16() {
  local dataset="$1"
  local workers batch eval_batch train_crop lr warmup min_lr_ratio
  workers="$(image_num_workers "$dataset")"
  batch="$(vqvae_batch_for "$dataset")"
  eval_batch="$(vqvae_eval_batch_for "$dataset")"
  train_crop="$(train_crop_size_for "$dataset")"
  lr="$(vqvae_lr_for "$dataset")"
  warmup="$(stage1_warmup_for "$dataset")"
  min_lr_ratio="$(stage1_min_lr_ratio_for "$dataset")"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --mem-mb "$IMAGE_MEM_MB" \
    --cases "$dataset" \
    --model-family vqvae \
    --run-label "stage1-${dataset}-vqvae-f16-k8192-z256-e${STAGE1_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${IMAGE_STAGE1[@]}" \
    --stage1-override data.num_workers="$workers" \
    --stage1-override data.batch_size="$batch" \
    --stage1-override data.eval_batch_size="$eval_batch" \
    --stage1-override data.train_crop_size="$train_crop" \
    --stage1-override train.warmup_steps="$warmup" \
    --stage1-override train.min_lr_ratio="$min_lr_ratio" \
    --stage1-override model.num_downsamples=4 \
    --stage1-override model.num_hiddens=192 \
    --stage1-override model.num_residual_blocks=3 \
    --stage1-override model.num_residual_hiddens=96 \
    --stage1-override model.num_embeddings=8192 \
    --stage1-override model.embedding_dim=256 \
    --stage1-override model.commitment_cost=0.25 \
    --stage1-override model.decay=0.99 \
    --stage1-override model.codebook_init=true \
    --stage1-override model.dead_code_threshold=1.0 \
    --stage1-override model.perceptual_weight=0.0 \
    --stage1-override train.learning_rate="$lr" \
    --stage1-override train.precision=32
}

# ---------- audio submissions ----------
# Force the *waveform* path for both vctk and maestro to match the project's
# established audio sweeps (launch_vctk_waveform_{laser,vqvae}_sweep.sh):
#   model=<family>_audio_waveform  -> wires AudioEncoder/AudioDecoder (1D convs)
#   data=<dataset>_waveform        -> dataset returns raw waveform tensors
# Without these overrides, submit_multimodal_sweep.py defaults VCTK to the
# spectrogram path, which silently runs a 2D image-style encoder on log-mag
# STFTs — not what the maintained audio code is tuned for.
submit_audio() {
  local dataset="$1"     # vctk | maestro
  local family="$2"      # laser | vqvae
  local label_family="${family}"

  "$PYTHON_SUBMIT" scripts/submit_multimodal_sweep.py \
    "${COMMON_ARGS[@]}" \
    --mem-mb "$AUDIO_MEM_MB" \
    --exclude-nodes gpu018 \
    --cases "$dataset" \
    --model-family "$family" \
    --run-label "stage1-${dataset}-${label_family}-waveform-e${STAGE1_EPOCHS}" \
    "${COMMON_STAGE1[@]}" \
    "${AUDIO_STAGE1[@]}" \
    --stage1-override model="${family}_audio_waveform" \
    --stage1-override data="${dataset}_waveform" \
    --stage1-override data.num_workers=4 \
    --stage1-override data.batch_size=8 \
    --stage1-override train.learning_rate=3.0e-4 \
    --stage1-override train.precision=bf16-mixed
}

# ---------- dispatch ----------
echo "=== laser-debugging stage-1 sweep ==="
echo "Project:        $PROJECT"
echo "Partition:      $PARTITION  (time $TIME_LIMIT, gpus $GPUS)"
echo "Stage 1 epochs: $STAGE1_EPOCHS"
echo "Run root:       $RUN_ROOT_BASE"
echo "Dry run:        $DRY_RUN"
echo

# Optional comma-separated allowlist via DATASETS env var (default = all 6).
WANTED_DATASETS="${DATASETS:-celeba,celebahq,ffhq,coco,vctk,maestro}"
_wanted() {
  local target="$1"
  case ",${WANTED_DATASETS}," in
    *",${target},"*) return 0 ;;
    *) return 1 ;;
  esac
}

for dataset in celeba celebahq ffhq coco; do
  _wanted "$dataset" || continue
  echo "--- vision: $dataset ---"
  submit_laser_f32 "$dataset"
  submit_laser_f16 "$dataset"
  submit_vqvae_f16 "$dataset"
done

for dataset in vctk maestro; do
  _wanted "$dataset" || continue
  echo "--- audio: $dataset ---"
  submit_audio "$dataset" laser
  submit_audio "$dataset" vqvae
done

echo
echo "=== submission complete ==="
echo "Inspect:  squeue -u \$USER -h | head -30"
