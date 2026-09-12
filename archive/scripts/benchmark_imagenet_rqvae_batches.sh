#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/Projects/laser}"
STEPS="${STEPS:-20}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
BENCH_ROOT="${BENCH_ROOT:-${REPO_ROOT}/outputs/bench-imagenet-rqvae-batches-${STAMP}}"
mkdir -p "${BENCH_ROOT}"

for batch_size in 32 64 128; do
  run_root="${BENCH_ROOT}/bs${batch_size}"
  mkdir -p "${run_root}"
  echo "BENCHMARK_START batch_size=${batch_size} steps=${STEPS}"
  start="$(date +%s)"
  env \
    WANDB_MODE=disabled \
    LASER_DISABLE_WANDB_MEDIA=1 \
    PYTHON="${REPO_ROOT}/.venv/bin/python" \
    BATCH_SIZE="${batch_size}" \
    ACCUMULATE_GRAD_BATCHES=1 \
    ALLOW_NON128_BATCH=1 \
    RUN_ROOT="${run_root}" \
    RUN_NAME="bench-rqvae-bs${batch_size}" \
    "${REPO_ROOT}/scripts/run_imagenet_laser_a256k2_rqvae10.sh" \
      train.max_steps="${STEPS}" \
      train.limit_val_batches=0 \
      train.val_check_interval=1.0 \
      train.run_test_after_fit=false \
      train.compute_rfid_after_fit=false \
      checkpoint.save_top_k=0 \
      checkpoint.save_last=false \
      checkpoint.upload_to_wandb=false \
      >"${run_root}/benchmark.log" 2>&1
  end="$(date +%s)"
  elapsed=$((end - start))
  images=$((batch_size * STEPS))
  ips="$(awk -v n="${images}" -v s="${elapsed}" 'BEGIN { if (s > 0) printf "%.3f", n/s; else print "nan" }')"
  echo "BENCHMARK_RESULT batch_size=${batch_size} steps=${STEPS} elapsed_seconds=${elapsed} images_per_second=${ips}"
done

echo "BENCHMARK_ROOT ${BENCH_ROOT}"
