#!/usr/bin/env bash
set -euo pipefail

ROOT="${MAESTRO_ROOT:-/scratch/$USER/datasets/maestro}"
URL="${MAESTRO_URL:-https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip}"
SHA256_EXPECTED="${MAESTRO_SHA256:-6680fea5be2339ea15091a249fbd70e49551246ddbd5ca50f1b2352c08c95291}"
ZIP="$ROOT/maestro-v3.0.0.zip"

mkdir -p "$ROOT"
wget -c --progress=dot:giga -O "$ZIP" "$URL"

echo "${SHA256_EXPECTED}  ${ZIP}" | sha256sum -c -
unzip -q -n "$ZIP" -d "$ROOT"

DATASET_DIR="$ROOT/maestro-v3.0.0"
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Expected extracted dataset directory missing: $DATASET_DIR" >&2
  exit 1
fi

find "$DATASET_DIR" -type f -name '*.wav' | wc -l > "$ROOT/wav_count.txt"
echo "MAESTRO WAV count: $(cat "$ROOT/wav_count.txt")"
