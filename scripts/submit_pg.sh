#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/pg.sh"

submitted=0
array_spec="$(pg_array_spec)"

IFS=',' read -r -a cases <<< "$CASES"
for tag in "${cases[@]}"; do
  tag="${tag// /}"
  [[ -n "$tag" ]] || continue

  case "$tag" in
    p4s4)
      patch=4
      stride=4
      ;;
    p8s8)
      patch=8
      stride=8
      ;;
    *)
      echo "unknown_case $tag" >&2
      exit 1
      ;;
  esac

  case_root="$RUN_ROOT/$tag"
  common_export="ALL,ROOT_DIR=$ROOT_DIR,PARTITION=$PARTITION,RUN_ROOT=$case_root,DATA_DIR=$DATA_DIR,IMG=$IMG,TAG=$tag,PATCH=$patch,STRIDE=$stride,ATOMS=$ATOMS,K=$K,EMB=$EMB,NHID=$NHID,RBLK=$RBLK,RHID=$RHID,CMAX=$CMAX,S1_EPOCHS=$S1_EPOCHS,S1_BSZ=$S1_BSZ,S1_WORKERS=$S1_WORKERS,S1_LR=$S1_LR,S2_EPOCHS=$S2_EPOCHS,S2_BSZ=$S2_BSZ,S2_WORKERS=$S2_WORKERS,S2_LR=$S2_LR,D_MODEL=$D_MODEL,HEADS=$HEADS,LAYERS=$LAYERS,D_FF=$D_FF,WIN=$WIN,BINS_LIST=$BINS_LIST,WANDB_PROJECT=$WANDB_PROJECT"

  s1_id="$(sbatch --parsable --partition="$PARTITION" --job-name="${tag}-s1" --export="$common_export" "$ROOT_DIR/scripts/job_ps1.sbatch")"
  s2_id="$(sbatch --parsable --partition="$PARTITION" --job-name="${tag}-s2" --dependency="afterok:${s1_id}" --array="$array_spec" --export="$common_export" "$ROOT_DIR/scripts/job_ps2.sbatch")"

  echo "tag=$tag"
  echo "stage1_job=$s1_id"
  echo "stage2_job=$s2_id"
  echo "run_root=$case_root"
  echo "bins=${BINS_LIST//,/ }"
  echo ""
  submitted=$((submitted + 1))
done

if ((submitted == 0)); then
  echo "no_cases_submitted" >&2
  exit 1
fi
