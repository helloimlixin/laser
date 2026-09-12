#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
IMAGENET_DIR="${IMAGENET_DIR:-/workspace/Projects/data/imagenet}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-${REPO_ROOT}/outputs/imagenet-rqvae-no-perceptual-k2-${STAMP}}"
VARIANTS="${VARIANTS:-256 512 1024}"
GPU_INDEX="${GPU_INDEX:-0}"
GPU_MEMORY_START_THRESHOLD_MIB="${GPU_MEMORY_START_THRESHOLD_MIB:-4096}"
GPU_UTIL_START_THRESHOLD_PCT="${GPU_UTIL_START_THRESHOLD_PCT:-10}"
GPU_IDLE_CONFIRMATIONS="${GPU_IDLE_CONFIRMATIONS:-3}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

mkdir -p "${QUEUE_ROOT}"
cd "${REPO_ROOT}"
printf '%s\n' "$$" > "${QUEUE_ROOT}/queue.pid"
printf '%s\n' "${QUEUE_ROOT}" > "${REPO_ROOT}/outputs/imagenet-rqvae-no-perceptual-k2-latest.txt"
printf 'variant\tstate\ttimestamp\trun_dir\n' > "${QUEUE_ROOT}/status.tsv"

finish() {
  local rc=$?
  printf '%s\n' "${rc}" > "${QUEUE_ROOT}/exit.status"
}
trap finish EXIT

wait_for_idle_gpu() {
  local confirmations=0
  local used_mib util_pct
  while true; do
    IFS=, read -r used_mib util_pct < <(
      nvidia-smi \
        --id="${GPU_INDEX}" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
    used_mib="${used_mib//[[:space:]]/}"
    util_pct="${util_pct//[[:space:]]/}"
    if [[ "${used_mib}" -le "${GPU_MEMORY_START_THRESHOLD_MIB}" \
          && "${util_pct}" -le "${GPU_UTIL_START_THRESHOLD_PCT}" ]]; then
      confirmations=$((confirmations + 1))
    else
      confirmations=0
    fi
    printf '%s gpu=%s memory_used_mib=%s util_pct=%s idle_confirmations=%s/%s\n' \
      "$(date -u +%FT%TZ)" "${GPU_INDEX}" "${used_mib}" "${util_pct}" \
      "${confirmations}" "${GPU_IDLE_CONFIRMATIONS}"
    if [[ "${confirmations}" -ge "${GPU_IDLE_CONFIRMATIONS}" ]]; then
      return 0
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

overall_rc=0
for atoms in ${VARIANTS}; do
  run_name="imagenet-rqvae-no-perceptual-a${atoms}k2-${STAMP}"
  run_dir="${QUEUE_ROOT}/a${atoms}k2"
  mkdir -p "${run_dir}"
  printf 'a%sk2\twaiting_for_gpu\t%s\t%s\n' \
    "${atoms}" "$(date -u +%FT%TZ)" "${run_dir}" >> "${QUEUE_ROOT}/status.tsv"
  wait_for_idle_gpu
  printf 'a%sk2\trunning\t%s\t%s\n' \
    "${atoms}" "$(date -u +%FT%TZ)" "${run_dir}" >> "${QUEUE_ROOT}/status.tsv"

  set +e
  ATOM_VOCAB_SIZE="${atoms}" \
  SPARSITY_LEVEL=2 \
  STAMP="${STAMP}" \
  RUN_ROOT="${run_dir}" \
  RUN_NAME="${run_name}" \
  IMAGENET_DIR="${IMAGENET_DIR}" \
    bash scripts/run_imagenet_laser_a256k2_rqvae10.sh \
      2>&1 | tee "${run_dir}/driver.log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    state=completed
  else
    state="failed_${rc}"
    overall_rc=1
  fi
  printf 'a%sk2\t%s\t%s\t%s\n' \
    "${atoms}" "${state}" "$(date -u +%FT%TZ)" "${run_dir}" >> "${QUEUE_ROOT}/status.tsv"
done

exit "${overall_rc}"
