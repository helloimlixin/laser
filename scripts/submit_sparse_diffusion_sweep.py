#!/usr/bin/env python3
"""Submit a compact sparse-coefficient diffusion prior sweep."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 8):
    raise SystemExit("ERROR: scripts/submit_sparse_diffusion_sweep.py requires Python >= 3.8.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_multimodal_sweep as base  # noqa: E402


DEFAULT_TOKEN_CACHE = Path(
    "/scratch/xl598/runs/celebahq256_patch_nooverlap_largepatch_quick_sweep/"
    "celebahq256_patch_noov_largepatch_5ep_p16r_p16s16_a6144_k12/"
    "20260327_080440/stage2/tokens_cache__rslurm-50580172-20260327_080440__h286fb91ff3.pt"
)


@dataclass(frozen=True)
class Variant:
    label: str
    hidden_channels: int
    n_res_blocks: int
    dropout: float = 0.0


VARIANTS = (
    Variant("h128-b4", hidden_channels=128, n_res_blocks=4),
    Variant("h192-b4", hidden_channels=192, n_res_blocks=4),
    Variant("h128-b8", hidden_channels=128, n_res_blocks=8),
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit sparse coefficient diffusion prior sweep jobs.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=base._scratch_path("submission_snapshots"))
    parser.add_argument("--run-root-base", default=base._scratch_path("runs", "sparse_diffusion_prior_sweep"))
    parser.add_argument("--output-root", default=base._scratch_path("runs"))
    parser.add_argument("--token-cache-path", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default="1-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=64000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-num-images", type=int, default=16)
    parser.add_argument("--support-bank-size", type=int, default=512)
    parser.add_argument("--stats-items", type=int, default=8192)
    parser.add_argument("--learning-rate", default="2.0e-4")
    parser.add_argument("--python-bin", default="/home/xl598/.conda/envs/tinyvit/bin/python")
    parser.add_argument(
        "--pythonpath-extra",
        default="/scratch/xl598/Projects/cache_home_relocated/local/lib/python3.10/site-packages",
        help="Colon-separated site-packages paths prepended after the frozen snapshot.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _variant_command(snapshot_path: Path, run_dir: Path, args: argparse.Namespace, variant: Variant) -> list[str]:
    return [
        str(Path(args.python_bin).expanduser()),
        "train_stage2_diffusion_prior.py",
        "--token-cache-path",
        str(args.token_cache_path.expanduser().resolve()),
        "--output-dir",
        str(run_dir),
        "--output-root",
        str(Path(args.output_root).expanduser().resolve()),
        "--seed",
        "42",
        "--batch-size",
        str(int(args.batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--max-epochs",
        str(int(args.max_epochs)),
        "--accelerator",
        "gpu",
        "--devices",
        str(int(args.gpus)),
        "--precision",
        "16-mixed",
        "--gradient-clip-val",
        "1.0",
        "--log-every-n-steps",
        "25",
        "--val-check-interval",
        "1.0",
        "--hidden-channels",
        str(int(variant.hidden_channels)),
        "--atom-embed-dim",
        "16",
        "--time-embed-dim",
        str(int(variant.hidden_channels)),
        "--n-res-blocks",
        str(int(variant.n_res_blocks)),
        "--dropout",
        str(float(variant.dropout)),
        "--num-timesteps",
        str(int(args.num_timesteps)),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        "1.0e-2",
        "--stats-items",
        str(int(args.stats_items)),
        "--support-bank-size",
        str(int(args.support_bank_size)),
        "--sample-num-images",
        str(int(args.sample_num_images)),
        "--sample-steps",
        str(int(args.sample_steps)),
    ]


def write_job_files(snapshot_path: Path, run_root: Path, args: argparse.Namespace, variant: Variant) -> tuple[Path, Path]:
    run_dir = run_root / variant.label
    run_dir.mkdir(parents=True, exist_ok=True)
    run_script = run_dir / "run_sparse_diffusion.sh"
    sbatch_script = run_dir / "sbatch_sparse_diffusion.sh"
    command = _variant_command(snapshot_path, run_dir, args, variant)
    quoted_command = " ".join(q(part) for part in command)
    pythonpath_entries = [str(snapshot_path)]
    pythonpath_entries.extend(part for part in str(args.pythonpath_extra or "").split(":") if part)
    pythonpath = ":".join(pythonpath_entries)
    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONPATH="{pythonpath}${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONUSERBASE="/scratch/{os.environ.get('USER', 'unknown')}/.pydeps/laser_src_py311"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${{WANDB_MODE:-offline}}"
export TMPDIR="/tmp/laser_sparse_diffusion_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"

mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" {q(run_dir)}

cd {q(snapshot_path)}

{q(args.python_bin)} - <<'PY'
import sys
import torch
print("python", sys.version.replace("\\n", " "))
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this job.")
print("cuda_device", torch.cuda.get_device_name(0))
PY

exec {quoted_command}
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
    return run_dir, sbatch_script


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    token_cache = args.token_cache_path.expanduser().resolve()
    python_bin = Path(args.python_bin).expanduser()
    if not token_cache.is_file():
        raise FileNotFoundError(f"Token cache not found: {token_cache}")
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python_bin}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = base.snapshot_repo(
        repo,
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_sparse_diffusion_prior_{stamp}",
    )
    run_root = Path(args.run_root_base).expanduser().resolve() / f"diffusion-p16r-{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    submissions = []
    for variant in VARIANTS:
        run_dir, sbatch_script = write_job_files(snapshot_path, run_root, args, variant)
        log_base = run_dir / "slurm"
        cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name=lsdiff-{variant.label}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={int(args.cpus_per_task)}",
            f"--gres=gpu:{int(args.gpus)}",
            f"--mem={int(args.mem_mb)}",
            f"--time={args.time_limit}",
            f"--chdir={snapshot_path}",
            f"--output={log_base}_%j.out",
            f"--error={log_base}_%j.err",
            str(sbatch_script),
        ]
        if args.dry_run:
            print(" ".join(q(part) for part in cmd))
            job_id = "dry-run"
        else:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
            job_id = (proc.stdout or proc.stderr).strip().split()[-1]
        submissions.append((variant.label, job_id, run_dir, log_base))

    print(f"Snapshot: {snapshot_path}")
    print(f"Run root:  {run_root}")
    print(f"Token cache: {token_cache}")
    for label, job_id, run_dir, log_base in submissions:
        out_path = f"{log_base}_{job_id}.out" if job_id != "dry-run" else f"{log_base}_<jobid>.out"
        err_path = f"{log_base}_{job_id}.err" if job_id != "dry-run" else f"{log_base}_<jobid>.err"
        print(f"[{label}] job={job_id} run_dir={run_dir} stdout={out_path} stderr={err_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
