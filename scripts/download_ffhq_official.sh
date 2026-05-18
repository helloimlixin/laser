#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${FFHQ_DIR:-/scratch/$USER/datasets/ffhq}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
THREADS="${THREADS:-16}"
SOURCE="${FFHQ_SOURCE:-hf_wds}"
DOWNLOAD_SCRIPT_URL="https://raw.githubusercontent.com/NVlabs/ffhq-dataset/master/download_ffhq.py"

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

download_hf_wds() {
  local shards_dir="$DATASET_DIR/hf_wds_shards"
  local images_dir="$DATASET_DIR/images1024x1024_webp"
  mkdir -p "$shards_dir" "$images_dir"

  for start in $(seq -f "%05g" 0 1000 69000); do
    local shard="$shards_dir/${start}.tar"
    local url="https://huggingface.co/datasets/gaunernst/ffhq-1024-wds/resolve/main/${start}.tar?download=true"
    echo "=== FFHQ HF shard ${start} ==="
    wget -c --progress=dot:giga -O "$shard" "$url"
    tar -xf "$shard" -C "$images_dir" --skip-old-files
  done

  find "$images_dir" -type f -name '*.webp' | wc -l > "$DATASET_DIR/image_count.txt"
  echo "FFHQ image count: $(cat "$DATASET_DIR/image_count.txt")"
}

if [[ "$SOURCE" == "hf_wds" || "$SOURCE" == "huggingface" ]]; then
  download_hf_wds
  exit 0
fi

if [[ ! -f download_ffhq.py ]]; then
  wget -O download_ffhq.py "$DOWNLOAD_SCRIPT_URL"
fi

CONTAINER_BIN=""
for candidate in singularity apptainer; do
  if command -v "$candidate" >/dev/null 2>&1; then
    CONTAINER_BIN="$candidate"
    break
  fi
done

if [[ -z "$CONTAINER_BIN" ]]; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
  module load apptainer 2>/dev/null || true
  for candidate in singularity apptainer; do
    if command -v "$candidate" >/dev/null 2>&1; then
      CONTAINER_BIN="$candidate"
      break
    fi
  done
fi

LOCAL_PYTHON="/home/$USER/.conda/envs/research/bin/python"
if [[ -x "$LOCAL_PYTHON" ]] && "$LOCAL_PYTHON" - <<'PY'
import importlib.util
raise SystemExit(0 if all(importlib.util.find_spec(name) for name in ("requests", "PIL", "numpy", "scipy")) else 1)
PY
then
  "$LOCAL_PYTHON" download_ffhq.py --json --images --num_threads "$THREADS"
elif [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec \
    --bind "/scratch/$USER" \
    --bind "/cache/home/$USER" \
    "$IMAGE" \
    bash -lc "python -m pip install --quiet --no-cache-dir requests pillow scipy && cd '$DATASET_DIR' && python download_ffhq.py --json --images --num_threads '$THREADS'"
else
  python3 -m pip install --user --quiet requests pillow scipy || true
  python3 download_ffhq.py --json --images --num_threads "$THREADS"
fi

find "$DATASET_DIR/images1024x1024" -type f -name '*.png' | wc -l > "$DATASET_DIR/image_count.txt"
echo "FFHQ image count: $(cat "$DATASET_DIR/image_count.txt")"
