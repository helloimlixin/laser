#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import os
import re
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
    ".cache",
    ".pydeps",
    ".tmp",
    ".tmp_*",
    "cluster_logs",
    "wandb",
    "runs",
    "submission_snapshots",
    "source_snapshot_*",
    "pre_variation_snapshot_*",
    "*.out",
    "*.err",
    "*.pyc",
    "*.pyo",
    "*.swp",
)

DATA_DIRS = {
    "celebahq": "Projects/data/celeba_hq",
    "ffhq": "Projects/data/ffhq",
    "imagenet": "Projects/data/imagenet",
}


@dataclass(frozen=True)
class VisualProbe:
    label: str
    dataset: str
    downsample: int
    sparsity: int
    checkpoint_stage: str
    checkpoint: Path
    run_dir: Path


def _user() -> str:
    return os.environ.get("USER", "unknown")


def _scratch_path(*parts: str) -> str:
    return str(Path("/scratch") / _user() / Path(*parts))


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


def _snapshot_repo(repo: Path, snapshot_root: Path, stem: str) -> Path:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / stem
    if snapshot_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing snapshot: {snapshot_path}")
    shutil.copytree(repo, snapshot_path, ignore=_snapshot_ignore(repo))
    return snapshot_path


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _bash_array(items: Iterable[str]) -> str:
    return "\n".join(f"  {_q(item)}" for item in items)


def _discover_probes(root: Path, *, checkpoint_stage: str) -> list[VisualProbe]:
    probes: list[VisualProbe] = []
    for dataset in ("celebahq", "ffhq", "imagenet"):
        for run_dir in sorted(root.glob(f"*/{dataset}")):
            match = re.search(r"nonpatch-d([56])k([34])", str(run_dir))
            if match is None:
                continue
            ckpts = sorted((run_dir / checkpoint_stage / "checkpoints").glob("run_*/laser/last.ckpt"))
            if not ckpts:
                continue
            downsample = int(match.group(1))
            sparsity = int(match.group(2))
            label = f"{dataset}-d{downsample}k{sparsity}"
            probes.append(
                VisualProbe(
                    label=label,
                    dataset=dataset,
                    downsample=downsample,
                    sparsity=sparsity,
                    checkpoint_stage=checkpoint_stage,
                    checkpoint=ckpts[-1],
                    run_dir=run_dir,
                )
            )
    deduped: dict[str, VisualProbe] = {}
    for probe in probes:
        # Prefer the newest retry when duplicate labels exist.
        previous = deduped.get(probe.label)
        if previous is None or probe.checkpoint.stat().st_mtime > previous.checkpoint.stat().st_mtime:
            deduped[probe.label] = probe
    return [deduped[key] for key in sorted(deduped)]


def _train_command(probe: VisualProbe, *, output_dir: Path, project: str, num_workers: int) -> list[str]:
    conditioning = "class" if probe.dataset == "imagenet" else "none"
    data_dir = Path("/scratch") / _user() / DATA_DIRS[probe.dataset]
    adversarial = "true" if probe.checkpoint_stage == "stage1_adv" else "false"
    return [
        "train.py",
        "--stage", "1",
        "--dataset", probe.dataset,
        "--modality", "image",
        "--conditioning", conditioning,
        "--adversarial", adversarial,
        "--num_gpus", "1",
        "--downsample_layers", str(probe.downsample),
        "--sparsity_level", str(probe.sparsity),
        "--data_dir", str(data_dir),
        "--image_size", "256",
        "--batch_size", "1",
        "--num_workers", str(num_workers),
        "--epochs", "1",
        "--max_steps", "2",
        "--output_dir", str(output_dir),
        "--run_name", f"visual-probe-{probe.label}",
        "--project", project,
        "init_ckpt_path=" + str(probe.checkpoint),
        "train.limit_train_batches=2",
        "train.limit_val_batches=4",
        "train.limit_test_batches=0",
        "train.val_check_interval=1.0",
        "train.log_every_n_steps=1",
        "train.run_test_after_fit=false",
        "model.compute_fid=false",
        "model.log_images_every_n_steps=1",
        "model.enable_val_latent_visuals=true",
        "checkpoint.save_top_k=0",
        "checkpoint.save_last=false",
        "checkpoint.save_final=false",
        "wandb.name=" + probe.checkpoint_stage + "-visual-probe-" + probe.label,
        "wandb.group=laser-" + probe.checkpoint_stage + "-visual-probes",
        "wandb.tags=[visual_probe,laser," + probe.checkpoint_stage + "," + probe.dataset + ",nonpatch]",
    ]


def _write_job_files(
    *,
    snapshot_path: Path,
    job_root: Path,
    probe: VisualProbe,
    project: str,
    num_workers: int,
) -> tuple[Path, Path]:
    run_root = job_root / probe.label
    run_root.mkdir(parents=True, exist_ok=True)
    output_dir = probe.run_dir / f"{probe.checkpoint_stage}_visual_probe_{job_root.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_script = run_root / "run_stage1_visual_probe.sh"
    sbatch_script = run_root / "sbatch_stage1_visual_probe.sh"
    command = _train_command(probe, output_dir=output_dir, project=project, num_workers=num_workers)
    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="${{PYTHONUSERBASE:-{_scratch_path('.pydeps', 'laser_src_py311')}}}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot_path}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-0}}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TMPDIR="/tmp/laser_visual_probe_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR" "{output_dir / 'wandb'}"
cd "{snapshot_path}"

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

CMD=(
  "$PYTHON_BIN"
{_bash_array(command)}
)

echo "=== {probe.checkpoint_stage} visual probe: {probe.label} ==="
"${{CMD[@]}}"
""",
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)
    sbatch_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

BOOTSTRAP_LOG="{run_root}/bootstrap_${{SLURM_JOB_ID:-manual}}.log"
exec > "$BOOTSTRAP_LOG" 2>&1
set -x
echo "bootstrap-start host=$(hostname) date=$(date) job=${{SLURM_JOB_ID:-manual}}"

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

CONTAINER_CACHE_DIR="${{CONTAINER_CACHE_DIR:-/scratch/{_user()}/Projects/laser/.cache/laser_container_shared}}"
export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-$CONTAINER_CACHE_DIR}}"
export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$CONTAINER_CACHE_DIR}}"
mkdir -p "$APPTAINER_CACHEDIR" "$SINGULARITY_CACHEDIR"

IMAGE="${{IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}}"
nvidia-smi || true

if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec --nv \\
    --bind "{snapshot_path}" \\
    --bind "/scratch/{_user()}" \\
    --bind "/projects" \\
    --bind /dev/shm \\
    "$IMAGE" \\
    bash "{run_script}"
else
  bash "{run_script}"
fi
""",
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return run_script, sbatch_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit short visual probes from existing non-patch image checkpoints.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-root", default=_scratch_path("runs", "laser_nonpatch_deep_cond_multidata"))
    parser.add_argument("--snapshot-root", default=_scratch_path("submission_snapshots"))
    parser.add_argument("--job-root", default=_scratch_path("runs", "laser_stage1_visual_probes"))
    parser.add_argument(
        "--checkpoint-stage",
        choices=("stage1", "stage1_adv"),
        default="stage1",
        help="Checkpoint tree to visualize.",
    )
    parser.add_argument("--cases", default="", help="Comma-separated labels to submit, e.g. celebahq-d5k3,ffhq-d5k3.")
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=96000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--project", default="laser")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probes = _discover_probes(Path(args.run_root).expanduser().resolve(), checkpoint_stage=str(args.checkpoint_stage))
    selected = {item.strip() for item in str(args.cases).split(",") if item.strip()}
    if selected:
        probes = [probe for probe in probes if probe.label in selected]
    if not probes:
        raise SystemExit("No visual probes found.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = _snapshot_repo(
        Path(args.repo).expanduser().resolve(),
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_stage1_visual_probes_{stamp}",
    )
    job_root = Path(args.job_root).expanduser().resolve() / f"{args.checkpoint_stage}-visual-probes-{stamp}"
    job_root.mkdir(parents=True, exist_ok=True)

    submissions = []
    for probe in probes:
        _, sbatch_script = _write_job_files(
            snapshot_path=snapshot_path,
            job_root=job_root,
            probe=probe,
            project=str(args.project),
            num_workers=max(0, int(args.num_workers)),
        )
        log_base = job_root / probe.label / probe.label
        cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name=vis-{probe.label}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={int(args.cpus_per_task)}",
            "--gres=gpu:1",
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
        submissions.append((probe, job_id, log_base))

    print(f"Snapshot: {snapshot_path}")
    print(f"Job root: {job_root}")
    for probe, job_id, log_base in submissions:
        stdout = f"{log_base}_{job_id}.out" if job_id != "dry-run" else f"{log_base}_<jobid>.out"
        stderr = f"{log_base}_{job_id}.err" if job_id != "dry-run" else f"{log_base}_<jobid>.err"
        print(f"[{probe.label}] job={job_id} ckpt={probe.checkpoint} stdout={stdout} stderr={stderr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
