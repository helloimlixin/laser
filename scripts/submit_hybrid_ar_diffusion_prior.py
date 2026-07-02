#!/usr/bin/env python3
"""Submit a hybrid stage-2 run: AR atom ids plus diffusion coefficients."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 8):
    raise SystemExit("ERROR: scripts/submit_hybrid_ar_diffusion_prior.py requires Python >= 3.8.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_multimodal_sweep as base  # noqa: E402


DEFAULT_STAGE1_CKPT = Path(
    "/scratch/xl598/Projects/laser/runs/stage1_patch_overcomplete_ffhq_20260607/"
    "laser-train-ffhq-f16-p2s2-k2-a8192-z64-oc32x-cm8-coeffl1w003-s1-3-20260607_010730/"
    "ffhq/stage1/checkpoints/run_20260607_010807/laser/final.ckpt"
)
DEFAULT_FFHQ_DIR = Path(f"/scratch/{os.environ.get('USER', 'xl598')}/datasets/ffhq")


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=base._scratch_path("submission_snapshots"))
    parser.add_argument("--run-root-base", default=base._scratch_path("Projects", "laser", "runs", "stage2_hybrid_ar_diffusion_ffhq_20260607"))
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--ffhq-dir", type=Path, default=DEFAULT_FFHQ_DIR)
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--exclude-nodes", default="gpuk[001-018]")
    parser.add_argument("--time-limit", default="2-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=64000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--python-bin", default="/home/xl598/.conda/envs/tinyvit/bin/python")
    parser.add_argument(
        "--pythonpath-extra",
        default="/scratch/xl598/Projects/cache_home_relocated/local/lib/python3.10/site-packages",
        help="Colon-separated site-packages paths prepended after the frozen snapshot.",
    )
    parser.add_argument("--label", default="ffhq-z64k8192-cm8-aratoms-diffcoeff")
    parser.add_argument("--cache-batch-size", type=int, default=32)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage2-max-steps", type=int, default=10000)
    parser.add_argument("--ar-batch-size", type=int, default=256)
    parser.add_argument("--diffusion-batch-size", type=int, default=512)
    parser.add_argument("--diffusion-hidden-channels", type=int, default=128)
    parser.add_argument("--diffusion-res-blocks", type=int, default=4)
    parser.add_argument("--diffusion-sample-steps", type=int, default=100)
    parser.add_argument("--sample-num-images", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_job_files(snapshot_path: Path, run_root: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    run_script = run_root / "run_hybrid_ar_diffusion.sh"
    sbatch_script = run_root / "sbatch_hybrid_ar_diffusion.sh"
    cache_path = run_root / "token_cache_real.pt"
    ar_dir = run_root / "ar_atoms"
    diffusion_dir = run_root / "diffusion_coeffs"
    hybrid_dir = run_root / "hybrid_samples"
    python_bin = Path(args.python_bin).expanduser()
    pythonpath_entries = [str(snapshot_path)]
    pythonpath_entries.extend(part for part in str(args.pythonpath_extra or "").split(":") if part)
    pythonpath = ":".join(pythonpath_entries)

    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONPATH="{pythonpath}${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONUSERBASE="${{PYTHONUSERBASE:-/scratch/{os.environ.get('USER', 'unknown')}/.pydeps/laser_src_py311}}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${{WANDB_MODE:-online}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-/scratch/{os.environ.get('USER', 'unknown')}/.cache/wandb}}"
export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{os.environ.get('USER', 'unknown')}/.config/wandb}}"
export TMPDIR="/tmp/laser_hybrid_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"

mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" {q(run_root)} {q(ar_dir)} {q(diffusion_dir)} {q(hybrid_dir)}
cd {q(snapshot_path)}

{q(python_bin)} - <<'PY'
import sys
import torch
print("python", sys.version.replace("\\n", " "))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this job.")
print("cuda_device", torch.cuda.get_device_name(0))
PY

echo "=== Extract real-valued sparse cache ==="
{q(python_bin)} cache.py \
  --stage1-checkpoint {q(args.stage1_checkpoint.expanduser().resolve())} \
  --output-path {q(cache_path)} \
  --dataset ffhq \
  --data-dir {q(args.ffhq_dir.expanduser().resolve())} \
  --image-size 256 \
  --batch-size {int(args.cache_batch_size)} \
  --num-workers {int(args.cpus_per_task)} \
  --seed 42 \
  --max-items 0 \
  --model-type laser \
  --coeff-bins 0 \
  --coeff-max 8.0

echo "=== Train AR atom-id prior only ==="
{q(python_bin)} train.py stage2 \
  token_cache_path={q(cache_path)} \
  output_dir={q(ar_dir)} \
  seed=42 \
  ar.type=sparse_spatial_depth \
  ar.autoregressive_coeffs=false \
  ar.coeff_loss_weight=0.0 \
  ar.coeff_loss_type=mse \
  ar.d_model=512 \
  ar.n_heads=8 \
  ar.n_layers=6 \
  ar.d_ff=2048 \
  ar.dropout=0.1 \
  ar.learning_rate=3.0e-4 \
  ar.warmup_steps=500 \
  ar.max_steps={int(args.stage2_max_steps)} \
  train_ar.batch_size={int(args.ar_batch_size)} \
  train_ar.max_epochs={int(args.stage2_epochs)} \
  train_ar.max_items=0 \
  train_ar.limit_val_batches=128 \
  train_ar.limit_test_batches=0 \
  train_ar.val_check_interval=0.25 \
  train_ar.run_test_after_fit=false \
  train_ar.save_final_samples_after_fit=false \
  train_ar.sample_num_images=0 \
  train_ar.devices={int(args.gpus)} \
  train_ar.strategy=auto \
  train_ar.precision=16-mixed \
  data.num_workers={int(args.cpus_per_task)} \
  wandb.project=laser-vision-stage2-hybrid-diagnostic \
  wandb.group={q(args.label)} \
  wandb.name=ar-atoms-z64-k8192-cm8 \
  wandb.append_timestamp=false \
  wandb.save_dir={q(ar_dir / "wandb")}

echo "=== Train diffusion prior over real coefficients ==="
{q(python_bin)} train_stage2_diffusion_prior.py \
  --token-cache-path {q(cache_path)} \
  --output-dir {q(diffusion_dir)} \
  --output-root {q(run_root)} \
  --seed 42 \
  --batch-size {int(args.diffusion_batch_size)} \
  --num-workers {int(args.cpus_per_task)} \
  --max-epochs {int(args.stage2_epochs)} \
  --max-steps {int(args.stage2_max_steps)} \
  --accelerator gpu \
  --devices {int(args.gpus)} \
  --precision 16-mixed \
  --gradient-clip-val 1.0 \
  --log-every-n-steps 25 \
  --val-check-interval 1.0 \
  --limit-val-batches 128 \
  --hidden-channels {int(args.diffusion_hidden_channels)} \
  --atom-embed-dim 16 \
  --time-embed-dim {int(args.diffusion_hidden_channels)} \
  --n-res-blocks {int(args.diffusion_res_blocks)} \
  --dropout 0.0 \
  --num-timesteps 1000 \
  --learning-rate 2.0e-4 \
  --weight-decay 1.0e-2 \
  --stats-items 8192 \
  --support-bank-size 512 \
  --sample-num-images 0 \
  --sample-steps {int(args.diffusion_sample_steps)}

AR_CKPT="$(find {q(ar_dir / "checkpoints")} -path '*/last.ckpt' -type f | sort | tail -1)"
DIFF_CKPT="$(find {q(diffusion_dir)} -path '*/checkpoints/last.ckpt' -type f | sort | tail -1)"
if [[ -z "$AR_CKPT" ]]; then
  echo "No AR last.ckpt found under {ar_dir}" >&2
  exit 1
fi
if [[ -z "$DIFF_CKPT" ]]; then
  echo "No diffusion last.ckpt found under {diffusion_dir}" >&2
  exit 1
fi
echo "AR checkpoint: $AR_CKPT"
echo "Diffusion checkpoint: $DIFF_CKPT"

echo "=== Hybrid sampling: AR atoms + diffusion coefficients ==="
{q(python_bin)} scripts/sample_hybrid_ar_diffusion.py \
  --token-cache-path {q(cache_path)} \
  --ar-checkpoint "$AR_CKPT" \
  --diffusion-checkpoint "$DIFF_CKPT" \
  --output-dir {q(hybrid_dir)} \
  --output-root {q(run_root)} \
  --num-images {int(args.sample_num_images)} \
  --temperature 1.0 \
  --top-k 0 \
  --diffusion-steps {int(args.diffusion_sample_steps)} \
  --coeff-temperature 1.0 \
  --seed 42
""",
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    sbatch_script.write_text(
        f"""#!/bin/bash
set -euo pipefail
bash {q(run_script)}
""",
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return run_script, sbatch_script


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
    ffhq_dir = args.ffhq_dir.expanduser().resolve()
    python_bin = Path(args.python_bin).expanduser().resolve()
    if not stage1_checkpoint.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_checkpoint}")
    if not ffhq_dir.exists():
        raise FileNotFoundError(f"FFHQ directory not found: {ffhq_dir}")
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python_bin}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = base._safe_label(args.label)
    snapshot_path = base.snapshot_repo(
        repo,
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_hybrid_ar_diffusion_{label}_{stamp}",
    )
    run_root = Path(args.run_root_base).expanduser().resolve() / f"{label}-{stamp}"
    _, sbatch_script = write_job_files(snapshot_path, run_root, args)

    log_base = run_root / "slurm"
    cmd = [
        "sbatch",
        f"--partition={args.partition}",
        "--job-name=lshyb-ffhq",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={int(args.cpus_per_task)}",
        f"--gres=gpu:{int(args.gpus)}",
        f"--mem={int(args.mem_mb)}",
        f"--time={args.time_limit}",
        f"--chdir={snapshot_path}",
        f"--output={log_base}_%j.out",
        f"--error={log_base}_%j.err",
    ]
    if str(args.exclude_nodes or "").strip():
        cmd.append(f"--exclude={str(args.exclude_nodes).strip()}")
    cmd.append(str(sbatch_script))

    if args.dry_run:
        print(" ".join(q(part) for part in cmd))
        job_id = "dry-run"
    else:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        job_id = (proc.stdout or proc.stderr).strip().split()[-1]

    print(f"Snapshot: {snapshot_path}")
    print(f"Run root:  {run_root}")
    print(f"Stage-1 checkpoint: {stage1_checkpoint}")
    print(f"FFHQ dir: {ffhq_dir}")
    print(f"job={job_id}")
    print(f"stdout={log_base}_{job_id}.out" if job_id != "dry-run" else f"stdout={log_base}_<jobid>.out")
    print(f"stderr={log_base}_{job_id}.err" if job_id != "dry-run" else f"stderr={log_base}_<jobid>.err")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
