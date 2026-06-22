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


@dataclass(frozen=True)
class AudioProbe:
    label: str
    downsample: int
    sparsity: int
    checkpoint: Path
    checkpoint_stage: str
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


def _preferred_checkpoint(vctk_dir: Path) -> tuple[Path, str] | None:
    for stage in ("stage1_adv", "stage1"):
        ckpt_root = vctk_dir / stage / "checkpoints"
        candidates = sorted(ckpt_root.glob("run_*/laser/final.ckpt"))
        if not candidates:
            candidates = sorted(ckpt_root.glob("run_*/laser/last.ckpt"))
        if candidates:
            return candidates[-1], stage
    return None


def _discover_probes(root: Path) -> list[AudioProbe]:
    probes: list[AudioProbe] = []
    for vctk_dir in sorted(root.glob("*/vctk")):
        match = re.search(r"vctk-wave-d([56])k([34])", str(vctk_dir))
        if match is None:
            continue
        selected = _preferred_checkpoint(vctk_dir)
        if selected is None:
            continue
        checkpoint, checkpoint_stage = selected
        downsample = int(match.group(1))
        sparsity = int(match.group(2))
        label = f"vctk-d{downsample}k{sparsity}"
        probes.append(
            AudioProbe(
                label=label,
                downsample=downsample,
                sparsity=sparsity,
                checkpoint=checkpoint,
                checkpoint_stage=checkpoint_stage,
                run_dir=vctk_dir,
            )
        )

    deduped: dict[str, AudioProbe] = {}
    for probe in probes:
        previous = deduped.get(probe.label)
        previous_rank = -1 if previous is None else (1 if previous.checkpoint_stage == "stage1_adv" else 0)
        rank = 1 if probe.checkpoint_stage == "stage1_adv" else 0
        if (
            previous is None
            or rank > previous_rank
            or (rank == previous_rank and probe.checkpoint.stat().st_mtime > previous.checkpoint.stat().st_mtime)
        ):
            deduped[probe.label] = probe
    return [deduped[key] for key in sorted(deduped)]


def _train_command(
    probe: AudioProbe,
    *,
    output_dir: Path,
    project: str,
    num_workers: int,
    accelerator: str,
) -> list[str]:
    data_dir = Path("/scratch") / _user() / "Projects" / "data" / "VCTK-Corpus" / "VCTK-Corpus"
    command = [
        "train.py",
        "--stage", "1",
        "--dataset", "vctk",
        "--modality", "audio",
        "--conditioning", "text",
        "--adversarial", "false",
        "--num_gpus", "1",
        "--downsample_layers", str(probe.downsample),
        "--sparsity_level", str(probe.sparsity),
        "--data_dir", str(data_dir),
        "--batch_size", "1",
        "--num_workers", str(num_workers),
        "--epochs", "1",
        "--max_steps", "1",
        "--learning_rate", "0.0",
        "--dict_learning_rate", "0.0",
        "--output_dir", str(output_dir),
        "--run_name", f"audio-recon-probe-{probe.label}",
        "--project", project,
        "init_ckpt_path=" + str(probe.checkpoint),
        "train.limit_train_batches=1",
        "train.limit_val_batches=4",
        "train.limit_test_batches=0",
        "train.val_check_interval=1.0",
        "train.log_every_n_steps=1",
        "train.run_test_after_fit=false",
        "train.deterministic=false",
        "model.compute_fid=false",
        "model.log_images_every_n_steps=1",
        "model.enable_val_latent_visuals=false",
        "checkpoint.save_top_k=0",
        "checkpoint.save_last=false",
        "checkpoint.save_final=false",
        "wandb.name=stage1-audio-recon-probe-" + probe.label,
        "wandb.group=laser-vctk-recon-audio-probes-20260617",
        "wandb.tags=[audio_recon_probe,laser,stage1,vctk,nonpatch]",
    ]
    if accelerator == "cpu":
        command.extend(
            [
                "train.accelerator=cpu",
                "train.devices=1",
                "train.strategy=auto",
                "train.precision=32",
            ]
        )
    return command


def _write_job_files(
    *,
    snapshot_path: Path,
    job_root: Path,
    probe: AudioProbe,
    project: str,
    num_workers: int,
    accelerator: str,
) -> tuple[Path, Path, Path]:
    run_root = job_root / probe.label
    run_root.mkdir(parents=True, exist_ok=True)
    output_dir = probe.run_dir / f"stage1_audio_recon_probe_{job_root.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_script = run_root / "run_vctk_audio_probe.sh"
    sbatch_script = run_root / "sbatch_vctk_audio_probe.sh"
    command = _train_command(
        probe,
        output_dir=output_dir,
        project=project,
        num_workers=num_workers,
        accelerator=accelerator,
    )
    nv_flag = "--nv" if accelerator == "gpu" else ""
    nvidia_probe = "nvidia-smi || true" if accelerator == "gpu" else "echo 'CPU probe: skipping nvidia-smi'"
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
export TMPDIR="/tmp/laser_audio_probe_${{SLURM_JOB_ID:-$$}}"
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

echo "=== VCTK audio reconstruction probe: {probe.label} ({probe.checkpoint_stage}, accelerator={accelerator}) ==="
"${{CMD[@]}}"
echo "Expected local WAV directory: {output_dir / 'wandb' / 'audio_media' / 'val'}"
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
{nvidia_probe}

if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec {nv_flag} \\
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
    return run_script, sbatch_script, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit short VCTK reconstruction audio probes from existing non-patch checkpoints.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-root", default=_scratch_path("runs", "laser_nonpatch_deep_cond_multidata"))
    parser.add_argument("--snapshot-root", default=_scratch_path("submission_snapshots"))
    parser.add_argument("--job-root", default=_scratch_path("runs", "laser_vctk_recon_audio_probes"))
    parser.add_argument("--cases", default="", help="Comma-separated labels to submit, e.g. vctk-d5k3,vctk-d6k4.")
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default="01:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=64000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--project", default="laser")
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probes = _discover_probes(Path(args.run_root).expanduser().resolve())
    selected = {item.strip() for item in str(args.cases).split(",") if item.strip()}
    if selected:
        probes = [probe for probe in probes if probe.label in selected]
    if not probes:
        raise SystemExit("No VCTK audio probes found.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = _snapshot_repo(
        Path(args.repo).expanduser().resolve(),
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_vctk_recon_audio_probes_{stamp}",
    )
    job_root = Path(args.job_root).expanduser().resolve() / f"vctk-recon-audio-probes-{stamp}"
    job_root.mkdir(parents=True, exist_ok=True)

    submissions = []
    for probe in probes:
        _, sbatch_script, output_dir = _write_job_files(
            snapshot_path=snapshot_path,
            job_root=job_root,
            probe=probe,
            project=str(args.project),
            num_workers=max(0, int(args.num_workers)),
            accelerator=str(args.accelerator),
        )
        log_base = job_root / probe.label / probe.label
        cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name=aud-{probe.label}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={int(args.cpus_per_task)}",
            f"--mem={int(args.mem_mb)}",
            f"--time={args.time_limit}",
            f"--chdir={snapshot_path}",
            f"--output={log_base}_%j.out",
            f"--error={log_base}_%j.err",
            str(sbatch_script),
        ]
        if str(args.accelerator) == "gpu":
            cmd.insert(7, "--gres=gpu:1")
        if args.dry_run:
            job_id = "dry-run"
            print(" ".join(shlex.quote(part) for part in cmd))
        else:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
            text = (proc.stdout or proc.stderr).strip()
            job_id = text.split()[-1]
        submissions.append((probe, job_id, log_base, output_dir))

    print(f"Snapshot: {snapshot_path}")
    print(f"Job root: {job_root}")
    for probe, job_id, log_base, output_dir in submissions:
        stdout = f"{log_base}_{job_id}.out" if job_id != "dry-run" else f"{log_base}_<jobid>.out"
        stderr = f"{log_base}_{job_id}.err" if job_id != "dry-run" else f"{log_base}_<jobid>.err"
        audio_dir = output_dir / "wandb" / "audio_media" / "val"
        print(
            f"[{probe.label}] job={job_id} source={probe.checkpoint_stage} "
            f"ckpt={probe.checkpoint} audio_dir={audio_dir} stdout={stdout} stderr={stderr}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
