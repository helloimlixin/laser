#!/usr/bin/env bash
# Stage-1-only VCTK audio-GAN ablation sweep.
#
# The discriminator only affects the stage-1 codec, so every arm is stage-1-only
# (clean stage-1 + adversarial leg, NO token cache / stage-2) and is judged by
# stage-1 reconstruction quality: val/audio_pesq, val/audio_snr_db, val/audio_stoi,
# val/audio_multires_stft_spectral_convergence.
#
# Design = baseline + the proper HiFi-GAN/DAC objective + a leave-one-out ablation
# of each new component + a capacity arm. Compare in W&B by the "vctkgan-<stamp>"
# run-tag prefix.
#   baseline    crippled old GAN (hinge, adaptive-weight, no FM/mel/STFT, adv 0.03)
#   proper      lsgan + FM(2) + mel(15) + STFT critics + adv(1.0), no adaptive weight
#   noFM        proper but feature-matching off            -> isolates feature matching
#   noMel       proper but multi-scale mel loss off        -> isolates mel loss
#   noSTFT      proper but DAC complex-STFT critics off    -> isolates the STFT critic
#   proper-ds128 proper + 2x latent capacity (downsample 256->128) -> isolates capacity
#
# All arms: ds256 codec (rates [8,8,4]) p2s2 k4 a8192 unless noted, stage1=50 /
# adv=25 epochs, 2 GPUs. Set DRY_RUN=1 to print the sbatch commands without submitting.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A fixed stamp shared by all arms so they sort together in W&B. Passed in by the
# caller (date is unavailable to deterministic re-runs); default to a literal.
STAMP="${SWEEP_STAMP:-0613}"
DRY="${DRY_RUN:-0}"

run_arm() {
  local name="$1"; shift
  echo ""
  echo "############ ARM: ${name} ############"
  env "$@" \
    DRY_RUN="${DRY}" \
    STAGE1_ONLY=1 \
    RUN_TAG="vctkgan-${STAMP}-${name}" \
    bash "${HERE}/launch_power_vctk_sweep.sh"
}

# 1. Baseline: the previously-crippled GAN (what the historical VCTK runs used).
run_arm baseline \
  VCTK_DISC_LOSS=hinge \
  VCTK_USE_ADAPTIVE_DISC_WEIGHT=true \
  VCTK_AUDIO_FM_WEIGHT=0 \
  VCTK_AUDIO_MEL_WEIGHT=0 \
  VCTK_AUDIO_DISC_STFT_FFT_SIZES="[]" \
  VCTK_ADVERSARIAL_WEIGHT=0.03

# 2. Proper objective (launcher defaults already give lsgan/no-adaptive/fm2/mel15/STFT).
run_arm proper \
  VCTK_ADVERSARIAL_WEIGHT=1.0

# 3-5. Leave-one-out ablations from the proper objective.
run_arm noFM \
  VCTK_ADVERSARIAL_WEIGHT=1.0 \
  VCTK_AUDIO_FM_WEIGHT=0

run_arm noMel \
  VCTK_ADVERSARIAL_WEIGHT=1.0 \
  VCTK_AUDIO_MEL_WEIGHT=0

run_arm noSTFT \
  VCTK_ADVERSARIAL_WEIGHT=1.0 \
  VCTK_AUDIO_DISC_STFT_FFT_SIZES="[]"

# 6. Proper + 2x latent capacity (downsample 256 -> 128).
run_arm proper-ds128 \
  VCTK_ADVERSARIAL_WEIGHT=1.0 \
  VCTK_DOWNSAMPLE_RATES="[8,4,4]"

echo ""
echo "=== VCTK GAN ablation sweep submission complete (stamp=${STAMP}) ==="
