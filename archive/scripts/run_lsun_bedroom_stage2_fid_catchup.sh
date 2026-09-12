#!/usr/bin/env bash
# One-shot 50k-sample FID for a preserved Stage-2 recovery checkpoint.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/tmp/laser-b300-venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/lsun-bedroom-a16384-k4-official-stage2-20260818-231500}"
STAGE2_OUT="$RUN_ROOT/stage2"
CHECKPOINT_DIR="$STAGE2_OUT/checkpoints"
FID_STEP="${FID_STEP:-15500}"
FID_STEP_PADDED="$(printf '%07d' "$FID_STEP")"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$CHECKPOINT_DIR/fid_catchup_step_${FID_STEP_PADDED}.pt}"
TOKEN_CACHE="$RUN_ROOT/token_cache/lsun_bedroom_train_a16384k4_compound_pairs.pt"
STAGE1_CHECKPOINT="$ROOT/outputs/lsun-bedroom-a16384-k4-imagenet5.08-ft1e-20260818-084230/bedroom256-rqvae-laser-8x8-a16384-k4-finetune/18082026_084335/best_rfid_slot1_model.pt"
FID_REFERENCE_STATS="$ROOT/third_party/rq-vae-transformer/assets/fid_stats/lsun_256_bedroom.npz"
WANDB_RUN_ID="${WANDB_RUN_ID:-lsunbedrooma16384k4s2-20260818231500}"
WANDB_NAME="${WANDB_NAME:-lsun-bedroom-a16384-k4-official-rqtransformer-600M-20260818-231500}"
STATUS_FILE="$RUN_ROOT/status.tsv"

for required in "$PYTHON_BIN" "$RESUME_CHECKPOINT" "$TOKEN_CACHE" \
  "$STAGE1_CHECKPOINT" "$FID_REFERENCE_STATS"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "fid_catchup" "$1" "$2" >> "$STATUS_FILE"
}

status starting "50000-sample official LSUN Bedroom FID at preserved step $FID_STEP"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$ROOT/scripts/train_official_rqtransformer_laser_stage2.py" \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --data "$RUN_ROOT/data/lsun" --dataset lsun_bedroom \
  --model-preset lsun-bedroom-600m --token-cache "$TOKEN_CACHE" \
  --output "$STAGE2_OUT" --checkpoint-dir "$CHECKPOINT_DIR" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --distributed-backend ddp --epochs 100 --batch-size 32 \
  --total-batch-size 2048 --num-atoms 16384 --sparsity-level 4 \
  --coeff-vocab-size 2048 --coeff-max 3 --coeff-scale 6.4 \
  --compound-tokens --coeff-target-mode soft --coeff-target-temperature 0.5 \
  --lr 0.0005 --lr-schedule cosine --lr-schedule-epochs 100 --min-lr 0 \
  --atom-temperature 1.0 --atom-top-k 250 --atom-top-p 1.0 \
  --coeff-temperature 1.0 --coeff-top-k 250 --coeff-top-p 1.0 \
  --fid-num-samples 50000 --fid-batch-size 250 --fid-every 10 \
  --metric-backend original-rqvae --fid-reference-stats "$FID_REFERENCE_STATS" \
  --save-ckpt-freq 10 --save-step-freq 500 \
  --sample-grid-every 5000 --sample-grid-size 64 \
  --sample-grid-batch-size 8 --sample-grid-samples-per-class 8 \
  --fid-only --wandb-mode online --wandb-entity helloimlixin-rutgers \
  --wandb-project laser --wandb-id "$WANDB_RUN_ID" --wandb-name "$WANDB_NAME" \
  2>&1 | tee -a "$RUN_ROOT/logs/fid-eval-step${FID_STEP}.log"
status complete "result=$STAGE2_OUT/evaluations/fid_step_${FID_STEP_PADDED}.json"

# Continue the main schedule, which now evaluates every ten epochs.
exec env RUN_ROOT="$RUN_ROOT" STAMP=20260818-231500 \
  WANDB_RUN_ID="$WANDB_RUN_ID" \
  bash "$ROOT/scripts/run_lsun_bedroom_a16384_k4_official_stage2.sh"
