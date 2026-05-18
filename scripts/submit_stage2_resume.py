#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


EXCLUDES = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tmp",
    ".tmp_*",
    "cluster_logs",
    "wandb",
    "runs",
    "source_snapshot_*",
    "pre_variation_snapshot_*",
    "*.out",
    "*.err",
    "*.pyc",
    "*.pyo",
    "*.swp",
)


@dataclass(frozen=True)
class Stage2ResumeRun:
    run_id: str
    job_label: str
    group: str
    token_cache: Path
    output_dir: Path
    batch_size: int
    learning_rate: str
    checkpoint: Path | None = None


def _user() -> str:
    return os.environ.get("USER", "unknown")


def _snapshot_ignore(repo: Path):
    repo = repo.resolve()

    def ignore(current_dir: str, names: Iterable[str]):
        rel_dir = Path(current_dir).resolve().relative_to(repo)
        ignored = set()
        for name in names:
            rel_path = (rel_dir / name) if rel_dir != Path(".") else Path(name)
            if len(rel_path.parts) >= 2 and rel_path.parts[0] == "configs" and rel_path.parts[1] == "wandb":
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDES):
                ignored.add(name)
        return ignored

    return ignore


def snapshot_repo(repo: Path, snapshot_root: Path, stem: str) -> Path:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / stem
    if snapshot_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing snapshot: {snapshot_path}")
    shutil.copytree(repo, snapshot_path, ignore=_snapshot_ignore(repo))
    return snapshot_path


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def bash_array(items: list[str]) -> str:
    return "\n".join(f"  {q(item)}" for item in items)


def default_resume_runs() -> list[Stage2ResumeRun]:
    highk = Path("/scratch/xl598/runs/laser_debugging_patchseq_highK_sweep")
    batched = Path("/scratch/xl598/runs/laser_debugging_patchseq_batched_sweep")
    p2_highk = highk / "laser-train-celebahq-10ep-laser-patch-p2-k2-a8192-highK-20260514_030507" / "celebahq"
    p4_highk = highk / "laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-highK-20260514_030520" / "celebahq"
    p2_batched = batched / "laser-train-celebahq-10ep-laser-patch-p2-k2-a1024-b8-s2b4-20260514_023350" / "celebahq"
    return [
        Stage2ResumeRun(
            run_id="ijbn2fip",
            job_label="ijbn2fip",
            group="laser-train-celebahq-10ep-laser-patch-p2-k2-a8192-highK-20260514_030507",
            token_cache=p2_highk / "token_cache.pt",
            output_dir=p2_highk / "stage2",
            checkpoint=p2_highk / "stage2" / "checkpoints" / "s2_20260514_032013" / "last.ckpt",
            batch_size=4,
            learning_rate="6.0e-4",
        ),
        Stage2ResumeRun(
            run_id="nlm5tz9b",
            job_label="nlm5tz9b",
            group="laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-highK-20260514_030520",
            token_cache=p4_highk / "token_cache.pt",
            output_dir=p4_highk / "stage2",
            checkpoint=None,
            batch_size=8,
            learning_rate="6.0e-4",
        ),
        Stage2ResumeRun(
            run_id="vs4tg95a",
            job_label="vs4tg95a",
            group="laser-train-celebahq-10ep-laser-patch-p2-k2-a1024-b8-s2b4-20260514_023350",
            token_cache=p2_batched / "token_cache.pt",
            output_dir=p2_batched / "stage2",
            checkpoint=p2_batched / "stage2" / "checkpoints" / "s2_20260514_024948" / "last.ckpt",
            batch_size=4,
            learning_rate="6.0e-4",
        ),
    ]


def validate_run(run: Stage2ResumeRun) -> None:
    if not run.token_cache.is_file():
        raise FileNotFoundError(f"{run.run_id}: token cache not found: {run.token_cache}")
    if run.checkpoint is not None and not run.checkpoint.is_file():
        raise FileNotFoundError(f"{run.run_id}: checkpoint not found: {run.checkpoint}")
    run.output_dir.mkdir(parents=True, exist_ok=True)


def stage2_overrides(run: Stage2ResumeRun) -> list[str]:
    overrides = [
        f"token_cache_path={run.token_cache}",
        f"output_dir={run.output_dir}",
        "seed=42",
        "ar.type=sparse_spatial_depth",
        "ar.max_steps=-1",
        "train_ar.max_epochs=10",
        f"train_ar.batch_size={run.batch_size}",
        "train_ar.max_items=0",
        "train_ar.limit_train_batches=1.0",
        "train_ar.limit_val_batches=1.0",
        "train_ar.limit_test_batches=1.0",
        "train_ar.log_every_n_steps=50",
        "train_ar.sample_every_n_epochs=1",
        "train_ar.sample_log_to_wandb=true",
        "train_ar.sample_num_images=64",
        "train_ar.generation_metric_num_samples=64",
        "train_ar.compute_generation_fid=true",
        "train_ar.compute_audio_generation_metrics=false",
        "train_ar.devices=3",
        "train_ar.strategy=ddp",
        "train_ar.precision=bf16-mixed",
        "train_ar.accelerator=gpu",
        "data.dataset=celebahq",
        "data.data_dir=/scratch/xl598/datasets/celebahq_packed_256",
        "data.image_size=256",
        "data.num_workers=0",
        f"ar.learning_rate={run.learning_rate}",
        "wandb.project=laser-debugging",
        f"wandb.group={run.group}",
        "wandb.name=celebahq-stage2-transformer",
        "wandb.tags=[train,laser,celebahq,stage2,transformer,generation,resume]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={run.output_dir / 'wandb'}",
        f"wandb.id={run.run_id}",
        "wandb.resume=allow",
    ]
    if run.checkpoint is not None:
        overrides.append(f"ckpt_path={run.checkpoint}")
    return overrides


def write_job_files(snapshot_path: Path, job_root: Path, run: Stage2ResumeRun) -> tuple[Path, Path]:
    run_root = job_root / run.job_label
    run_root.mkdir(parents=True, exist_ok=True)
    run_script = run_root / "run_stage2_resume.sh"
    sbatch_script = run_root / "sbatch_stage2_resume.sh"
    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="/scratch/{_user()}/.pydeps/laser_src_py311"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="{snapshot_path}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export TMPDIR="/tmp/laser_stage2_resume_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR" {q(run.output_dir / 'wandb')}

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

python_version() {{
  "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || printf 'unknown\\n'
}}

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: $PYTHON_BIN is Python $(python_version "$PYTHON_BIN"); LASER requires Python >= 3.10." >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install --user --quiet \\
  numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \\
  torch-fidelity matplotlib lpips soundfile 2>/dev/null || true

"$PYTHON_BIN" - <<'PY'
import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {{cuda_available}}")
if not cuda_available:
    raise SystemExit("CUDA is not available inside this job; failing before training.")
print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")
PY

cd {q(snapshot_path)}

STAGE2_ARGS=(
{bash_array(stage2_overrides(run))}
)

echo "=== Resume Stage 2: {run.run_id} ==="
"$PYTHON_BIN" train_stage2_prior.py "${{STAGE2_ARGS[@]}}"
""",
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)
    sbatch_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +u; source /usr/share/lmod/lmod/init/bash; set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u; source /usr/share/Modules/init/bash; set -u
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

IMAGE="${{IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}}"

if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec --nv \\
    --bind {q(snapshot_path)} \\
    --bind "/scratch/{_user()}" \\
    --bind {q(run.output_dir.parent)} \\
    --bind /dev/shm \\
    "$IMAGE" \\
    bash {q(run_script)}
else
  bash {q(run_script)}
fi
""",
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return run_script, sbatch_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume selected CelebA-HQ stage-2 W&B runs.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=f"/scratch/{_user()}/submission_snapshots")
    parser.add_argument("--job-root", default=f"/scratch/{_user()}/runs/laser_debugging_stage2_resume")
    parser.add_argument("--run-id", action="append", choices=[run.run_id for run in default_resume_runs()])
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default="1-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=64000)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.run_id or [])
    runs = [run for run in default_resume_runs() if not selected or run.run_id in selected]
    if not runs:
        raise SystemExit("No resume runs selected.")
    for run in runs:
        validate_run(run)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshot_repo(
        Path(args.repo).expanduser().resolve(),
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_stage2_resume_{stamp}",
    )
    job_root = Path(args.job_root).expanduser().resolve() / f"stage2-resume-{stamp}"
    job_root.mkdir(parents=True, exist_ok=True)

    submissions = []
    for run in runs:
        _, sbatch_script = write_job_files(snapshot_path, job_root, run)
        log_base = job_root / run.job_label / run.job_label
        cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name=s2res-{run.job_label}",
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
            job_id = "dry-run"
            print(" ".join(shlex.quote(part) for part in cmd))
        else:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
            text = (proc.stdout or proc.stderr).strip()
            job_id = text.split()[-1]
        submissions.append(
            {
                "run_id": run.run_id,
                "job_id": job_id,
                "checkpoint": str(run.checkpoint) if run.checkpoint else "none; restarting stage2",
                "stdout": f"{log_base}_{job_id}.out" if job_id != "dry-run" else f"{log_base}_<jobid>.out",
                "stderr": f"{log_base}_{job_id}.err" if job_id != "dry-run" else f"{log_base}_<jobid>.err",
            }
        )

    print(f"Snapshot: {snapshot_path}")
    print(f"Job root: {job_root}")
    for item in submissions:
        print(
            f"[{item['run_id']}] job={item['job_id']} checkpoint={item['checkpoint']} "
            f"stdout={item['stdout']} stderr={item['stderr']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
