#!/usr/bin/env bash
# Queue a longer p4s4/k8 adversarial follow-up after the current comparison.
set -euo pipefail

cd /home/xl598/Projects/laser

WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-laser_a2048_p4s4k8_ksvd_queued}"
PYTHON_BIN="${PYTHON_BIN:-/home/xl598/anaconda3/envs/laser/bin/python}"
BASE_GROUP="${BASE_GROUP:-celebahq_p4s4_k8_a2048_b035_c020_quant_20260530_135508_p4s4k8_b035c020}"
KSV_GROUP="${KSV_GROUP:-celebahq_p4s4_k8_a2048_b035_c020_ksvd_quant_20260530_142147_p4s4k8_ksvd}"
KSV_ROOT="${KSV_ROOT:-/home/xl598/Projects/laser/runs/celebahq_p4s4_k8_a2048_b035_c020_ksvd_quant_20260530_142147_p4s4k8_ksvd}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_p4s4k8_adv}"
RUN_ROOT="${RUN_ROOT:-/home/xl598/Projects/laser/runs/celebahq_p4s4_k8_a2048_adv_quant_${STAMP}}"
LOG="${LOG:-${RUN_ROOT}/pipeline.nohup.log}"
mkdir -p "${RUN_ROOT}"

echo "[$(date --iso-8601=seconds)] waiting for ${WAIT_FOR_SESSION}" | tee "${LOG}"
while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
  sleep 300
done

selection="$(
  BASE_GROUP="${BASE_GROUP}" KSV_GROUP="${KSV_GROUP}" KSV_ROOT="${KSV_ROOT}" "${PYTHON_BIN}" - <<'PY'
import math
import os
from pathlib import Path

use_ksvd = False
reason = "fallback: baseline"
try:
    import wandb

    api = wandb.Api()

    def stage1_summary(group: str):
        runs = list(api.runs("helloimlixin-rutgers/laser", filters={"group": group}))
        stage1 = [
            run for run in runs
            if run.config.get("training_stage") == "stage1"
            or run.config.get("stage_role") == "autoencoder_training"
        ]
        if not stage1:
            return None
        run = sorted(stage1, key=lambda item: item.created_at or "")[-1]
        psnr = run.summary.get("val/psnr")
        loss = run.summary.get("val/loss")
        try:
            psnr = float(psnr)
        except (TypeError, ValueError):
            psnr = float("nan")
        try:
            loss = float(loss)
        except (TypeError, ValueError):
            loss = float("nan")
        return run.state, psnr, loss, run.id

    base = stage1_summary(os.environ["BASE_GROUP"])
    ksvd = stage1_summary(os.environ["KSV_GROUP"])
    if base and ksvd and ksvd[0] == "finished":
        base_psnr = base[1]
        ksvd_psnr = ksvd[1]
        if math.isfinite(base_psnr) and math.isfinite(ksvd_psnr):
            use_ksvd = ksvd_psnr >= base_psnr
            reason = (
                f"wandb val/psnr ksvd={ksvd_psnr:.4f} "
                f"baseline={base_psnr:.4f}"
            )
        elif Path(os.environ["KSV_ROOT"]).joinpath("stage1").exists():
            use_ksvd = True
            reason = "ksvd finished; psnr unavailable"
    elif Path(os.environ["KSV_ROOT"]).joinpath("stage1/checkpoints").exists():
        final_ckpts = list(Path(os.environ["KSV_ROOT"]).glob("stage1/checkpoints/**/final.ckpt"))
        use_ksvd = bool(final_ckpts)
        reason = "local ksvd final checkpoint present" if use_ksvd else "ksvd not complete"
except Exception as exc:
    final_ckpts = list(Path(os.environ["KSV_ROOT"]).glob("stage1/checkpoints/**/final.ckpt"))
    use_ksvd = bool(final_ckpts)
    reason = f"wandb selection failed: {type(exc).__name__}; local final={use_ksvd}"

print("true" if use_ksvd else "false")
print(reason)
PY
)"
USE_KSVD="$(printf '%s\n' "${selection}" | sed -n '1p')"
SELECTION_REASON="$(printf '%s\n' "${selection}" | sed -n '2,$p')"

if [[ "${USE_KSVD}" == "true" ]]; then
  ONLINE_KSVD_ENABLED=true
  ONLINE_KSVD_START_STEP=500
  ONLINE_KSVD_INTERVAL_STEPS=500
  ONLINE_KSVD_STOP_STEP=3000
  ONLINE_KSVD_MAX_SAMPLES=512
  ONLINE_KSVD_MAX_ATOMS=256
  ONLINE_KSVD_BLEND=0.25
else
  ONLINE_KSVD_ENABLED=false
  ONLINE_KSVD_START_STEP=0
  ONLINE_KSVD_INTERVAL_STEPS=0
  ONLINE_KSVD_STOP_STEP=null
  ONLINE_KSVD_MAX_SAMPLES=512
  ONLINE_KSVD_MAX_ATOMS=256
  ONLINE_KSVD_BLEND=0.25
fi

{
  echo "[$(date --iso-8601=seconds)] selected ONLINE_KSVD_ENABLED=${ONLINE_KSVD_ENABLED}"
  echo "selection_reason=${SELECTION_REASON}"
  echo "[$(date --iso-8601=seconds)] launching longer adversarial p4s4/k8"
} | tee -a "${LOG}"

exec env \
  STAMP="${STAMP}" \
  RUN_ROOT="${RUN_ROOT}" \
  WANDB_GROUP="celebahq_p4s4_k8_a2048_adv_quant_${STAMP}" \
  NUM_EMBEDDINGS=2048 \
  PATCH_SIZE=4 \
  PATCH_STRIDE=4 \
  SPARSITY_LEVEL=8 \
  BOTTLENECK_LOSS_WEIGHT=0.35 \
  COMMITMENT_COST=0.20 \
  ONLINE_KSVD_ENABLED="${ONLINE_KSVD_ENABLED}" \
  ONLINE_KSVD_START_STEP="${ONLINE_KSVD_START_STEP}" \
  ONLINE_KSVD_INTERVAL_STEPS="${ONLINE_KSVD_INTERVAL_STEPS}" \
  ONLINE_KSVD_STOP_STEP="${ONLINE_KSVD_STOP_STEP}" \
  ONLINE_KSVD_MAX_SAMPLES="${ONLINE_KSVD_MAX_SAMPLES}" \
  ONLINE_KSVD_MAX_ATOMS="${ONLINE_KSVD_MAX_ATOMS}" \
  ONLINE_KSVD_BLEND="${ONLINE_KSVD_BLEND}" \
  STAGE1_EPOCHS=30 \
  STAGE2_EPOCHS=150 \
  STAGE2_MAX_STEPS=120000 \
  ADVERSARIAL_WEIGHT=0.05 \
  ADVERSARIAL_START_STEP=12000 \
  ADVERSARIAL_WARMUP_STEPS=12000 \
  DISCRIMINATOR_LR=5.0e-5 \
  DISCRIMINATOR_CHANNELS=64 \
  DISCRIMINATOR_LAYERS=3 \
  bash scripts/run_celebahq_p2s2_k4_quant_pipeline.sh
