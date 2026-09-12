#!/bin/bash
# Sweep launcher for the FIXED ImageNet-site stage-1 recipe (post online_ksvd fix).
# Fans out one sbatch job per grid cell via submit_imagenet_site_stage1.sh.
#
# Grid (8 cells):
#   dictionary_update_mode in {gradient, online_ksvd}   # the fix makes both work
#   commitment_cost        in {0.25, 0.5}               # encoder-pinning strength
#   sparsity_level         in {16, 24}                  # per-site capacity
#                                                       # NB: tokens = 256 sites x k
#                                                       # so k=24 -> 6144 stage-2 tokens
# Each run: 3-GPU DDP, 150k steps (full), automatic optimization.
#
# Usage:
#   DRY_RUN=1 ./scripts/launch_imagenet_site_sweep.sh   # preview every job, submit nothing
#   ./scripts/launch_imagenet_site_sweep.sh             # submit all 8
#   MODES="gradient" CCS="0.25" KS="16" ./scripts/launch_imagenet_site_sweep.sh  # subset

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="$HERE/submit_imagenet_site_stage1.sh"

MODES="${MODES:-gradient online_ksvd}"
CCS="${CCS:-0.25 0.5}"
KS="${KS:-16 24}"
KSVD_LR="${KSVD_LR:-0.08}"
TAG="${TAG:-150k}"
export MAX_STEPS="${MAX_STEPS:-150000}"
export GPUS="${GPUS:-3}"
DRY_RUN="${DRY_RUN:-0}"

count=0
for mode in $MODES; do
  for cc in $CCS; do
    for k in $KS; do
      count=$((count + 1))
      ov="model.dictionary_update_mode=${mode} model.commitment_cost=${cc} model.sparsity_level=${k}"
      [[ "$mode" == "online_ksvd" ]] && ov="$ov model.dictionary_ksvd_lr=${KSVD_LR}"
      ccslug="${cc/./p}"   # 0.25 -> 0p25
      run_name="imagenet-site-ds4-a8192-${mode}-cc${ccslug}-k${k}-${TAG}"
      echo "=========================================================="
      echo "[$count] $run_name"
      echo "    overrides: $ov"
      EXTRA_OVERRIDES="$ov" RUN_NAME="$run_name" DRY_RUN="$DRY_RUN" \
        bash "$SUBMIT"
    done
  done
done

echo "=========================================================="
if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run) previewed $count jobs; nothing submitted."
else
  echo "submitted $count jobs."
fi
