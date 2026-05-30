#!/usr/bin/env bash
# Wait for the current p4s4 experiment to exit, then launch p2s2/k4/a4096.
set -euo pipefail

cd /home/xl598/Projects/laser

WAIT_PGID="${WAIT_PGID:-141815}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
DATA_DIR="${DATA_DIR:-/home/xl598/Projects/data/celeba_hq}"
WANDB_PROJECT="${WANDB_PROJECT:-laser}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-4096}"

echo "[$(date --iso-8601=seconds)] waiting for process group ${WAIT_PGID}"
while kill -0 "-${WAIT_PGID}" 2>/dev/null; do
  sleep "${WAIT_INTERVAL_SECONDS}"
done
echo "[$(date --iso-8601=seconds)] process group ${WAIT_PGID} exited"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/home/xl598/Projects/laser/runs/celebahq_p2s2_k4_a${NUM_EMBEDDINGS}_quant_${STAMP}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.nohup.log"

mkdir -p "${RUN_ROOT}"

echo "[$(date --iso-8601=seconds)] launching p2s2/k4/a${NUM_EMBEDDINGS} quant pipeline"
echo "RUN_ROOT=${RUN_ROOT}"
echo "PIPELINE_LOG=${PIPELINE_LOG}"

exec env \
  STAMP="${STAMP}" \
  RUN_ROOT="${RUN_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DATA_DIR="${DATA_DIR}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_GROUP="celebahq_p2s2_k4_a${NUM_EMBEDDINGS}_quant_${STAMP}" \
  NUM_EMBEDDINGS="${NUM_EMBEDDINGS}" \
  bash scripts/run_celebahq_p2s2_k4_quant_pipeline.sh >"${PIPELINE_LOG}" 2>&1
