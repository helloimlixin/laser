#!/usr/bin/env python3
"""Submit a sampling-parameter sweep for an existing stage-2 checkpoint."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 8):
    raise SystemExit("ERROR: scripts/submit_sampling_sweep.py requires Python >= 3.8.")

from submit_multimodal_sweep import snapshot_repo


def _user() -> str:
    return os.environ.get("USER", "unknown")


def _scratch_path(*parts: str) -> str:
    return str(Path("/scratch") / _user() / Path(*parts))


def _safe_slug(text: str, *, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text).strip()).strip("-")
    return (slug or "sampling-sweep")[:max_len].strip("-")


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=_scratch_path("submission_snapshots"))
    parser.add_argument("--run-root-base", default=_scratch_path("runs", "laser_sampling_sweeps"))
    parser.add_argument("--label", default="")
    parser.add_argument("--source-run", default="", help="Optional W&B source run path/id for naming metadata.")
    parser.add_argument("--ckpt", required=True, help="Stage-2 checkpoint to sample from.")
    parser.add_argument("--cache", required=True, help="Token cache used by the stage-2 run.")
    parser.add_argument("--out-root", required=True, help="Stage-1 output root used to locate the decoder.")
    parser.add_argument("--temps", default="0.55,0.65,0.75,0.85,0.95")
    parser.add_argument("--topks", default="0,128,256,512,1024")
    parser.add_argument("-n", "--num-samples", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=96000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--exclude-nodes", default="gpu018,gpuk[005-018]")
    parser.add_argument("--image", default="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--python-userbase", default=_scratch_path(".pydeps", "laser_src_py311"))
    parser.add_argument("--wandb-entity", default="helloimlixin-rutgers")
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_script(args: argparse.Namespace, snapshot: Path, run_root: Path, out_dir: Path) -> str:
    wandb_args = ""
    if not args.no_wandb:
        wandb_bits = [
            "--wandb",
            "--wandb-entity",
            args.wandb_entity,
            "--wandb-project",
            args.wandb_project,
            "--wandb-group",
            args.wandb_group or args.label,
            "--wandb-name",
            args.wandb_name or f"{args.label}-sampling-sweep",
        ]
        wandb_args = " ".join(_quote(bit) for bit in wandb_bits)

    sample_args = [
        args.python_bin,
        "scripts/sample_param_sweep.py",
        "--ckpt",
        str(Path(args.ckpt).expanduser().resolve()),
        "--cache",
        str(Path(args.cache).expanduser().resolve()),
        "--out-root",
        str(Path(args.out_root).expanduser().resolve()),
        "--out",
        str(out_dir),
        "-n",
        int(args.num_samples),
        "--nrow",
        int(args.nrow),
        "--seed",
        int(args.seed),
        "--temps",
        args.temps,
        "--topks",
        args.topks,
    ]
    sample_cmd = " ".join(_quote(bit) for bit in sample_args)
    if wandb_args:
        sample_cmd = f"{sample_cmd} {wandb_args}"

    python_userbase = str(Path(args.python_userbase).expanduser())
    snapshot_path = str(snapshot)
    wandb_dir = str(run_root / "wandb")
    user_cache = f"/scratch/{_user()}/.cache"
    user_config = f"/scratch/{_user()}/.config/wandb"

    return f"""#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="${{PYTHONUSERBASE:-{python_userbase}}}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot_path}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export WANDB_DIR="${{WANDB_DIR:-{wandb_dir}}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{user_cache}/wandb}}"
export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-{user_config}}}"
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-{user_cache}}}"
export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTHONUNBUFFERED=1
export TMPDIR="/tmp/laser_sampling_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$PYTHONUSERBASE" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$TMPDIR" {_quote(out_dir)}

PYTHON_BIN="{_quote(args.python_bin)}"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: $PYTHON_BIN must be Python >= 3.10." >&2
  exit 2
fi

run_user_pip_install() {{
  if command -v flock >/dev/null 2>&1; then
    (
      flock 9
      "$PYTHON_BIN" -m pip install --user --quiet "$@" 2>/dev/null || true
    ) 9>"$PYTHONUSERBASE/.install.lock"
  else
    "$PYTHON_BIN" -m pip install --user --quiet "$@" 2>/dev/null || true
  fi
}}

run_user_pip_install numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips pillow

"$PYTHON_BIN" - <<'PY'
import torch
print(f"CUDA available: {{torch.cuda.is_available()}}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this sampling job.")
print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")
PY

cd {_quote(snapshot)}
{sample_cmd}

echo "sampling-sweep-complete out={out_dir}"
"""


def _sbatch_script(args: argparse.Namespace, snapshot: Path, run_script: Path) -> str:
    return f"""#!/bin/bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
  elif [[ -f /etc/profile.d/modules.sh ]]; then
    set +u; source /etc/profile.d/modules.sh; set -u
  fi
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

export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-/scratch/{_user()}/Projects/laser/.cache/laser_container_shared}}"
export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}}"
mkdir -p "$APPTAINER_CACHEDIR" "$SINGULARITY_CACHEDIR"

IMAGE="${{IMAGE:-{args.image}}}"
nvidia-smi || true

if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec --nv \\
    --bind {_quote(snapshot)} \\
    --bind /scratch/{_user()} \\
    --bind /projects \\
    --bind /dev/shm \\
    "$IMAGE" bash {_quote(run_script)}
else
  bash {_quote(run_script)}
fi
"""


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = _safe_slug(args.label or args.source_run or Path(args.ckpt).stem)
    args.label = label
    run_root = Path(args.run_root_base).expanduser().resolve() / f"{label}_{stamp}"
    out_dir = run_root / "samples"
    run_root.mkdir(parents=True, exist_ok=True)

    snapshot = snapshot_repo(
        repo,
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_sampling_{label}_{stamp}",
    )
    run_script = run_root / "run_sampling_sweep.sh"
    sbatch_script = run_root / "sbatch_sampling_sweep.sh"
    stdout = run_root / "sampling_%j.out"
    stderr = run_root / "sampling_%j.err"

    _write(run_script, _run_script(args, snapshot, run_root, out_dir))
    _write(sbatch_script, _sbatch_script(args, snapshot, run_script))

    cmd = [
        "sbatch",
        f"--partition={args.partition}",
        "--job-name=sample-sweep",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        f"--cpus-per-task={int(args.cpus_per_task)}",
        f"--gres=gpu:{int(args.gpus)}",
        f"--mem={int(args.mem_mb)}",
        f"--time={args.time_limit}",
        f"--chdir={snapshot}",
        f"--output={stdout}",
        f"--error={stderr}",
    ]
    if args.exclude_nodes.strip():
        cmd.append(f"--exclude={args.exclude_nodes.strip()}")
    cmd.append(str(sbatch_script))

    print(" ".join(shlex.quote(part) for part in cmd))
    print(f"Snapshot: {snapshot}")
    print(f"Run root:  {run_root}")
    print(f"Samples:   {out_dir}")
    if args.dry_run:
        print("Job:       dry-run")
        print(f"Stdout:    {str(stdout).replace('%j', '<jobid>')}")
        print(f"Stderr:    {str(stderr).replace('%j', '<jobid>')}")
        return 0

    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(completed.stdout.strip())
    job_id = completed.stdout.strip().split()[-1]
    print(f"Job:       {job_id}")
    print(f"Stdout:    {str(stdout).replace('%j', job_id)}")
    print(f"Stderr:    {str(stderr).replace('%j', job_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
