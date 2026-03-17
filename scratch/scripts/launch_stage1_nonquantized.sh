#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QUANTIZE_SPARSE_COEFFS=false
export WANDB_NAME="${WANDB_NAME:-proto_rqsd_rh_24g_100ep_dict_safe_ref_nonquantized}"
exec "$ROOT_DIR/scripts/launch_stage1_balanced.sh"
