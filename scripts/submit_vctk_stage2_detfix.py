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

SAMPLE_PROMPTS = (
    "The quick brown fox jumps over the lazy dog.",
    "A calm voice reads this sentence clearly.",
    "Please bring the warm tea to the table.",
    "The train arrived before sunrise.",
    "Several people waited outside the station.",
    "She opened the window and listened to the rain.",
    "This recording should follow the written text.",
    "A small boat moved slowly across the lake.",
)


@dataclass(frozen=True)
class VctkStage2Run:
    case: str
    token_cache: Path
    output_dir: Path
    group: str


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


def _default_runs() -> list[VctkStage2Run]:
    root = Path("/scratch") / _user() / "runs" / "laser_nonpatch_deep_cond_multidata"
    stems = {
        "d5k3": "laser-train-nonpatch-deepcond-20260616_144243-vctk-wave-d5k3-textcond-a8192-q256-seq384-s1e75-adve25-s2e300-20260616_144245",
        "d5k4": "laser-train-nonpatch-deepcond-20260616_144243-vctk-wave-d5k4-textcond-a8192-q256-seq512-s1e75-adve25-s2e300-20260616_144247",
        "d6k3": "laser-train-nonpatch-deepcond-20260616_144243-vctk-wave-d6k3-textcond-a8192-q256-seq192-s1e75-adve25-s2e300-20260616_144250",
    }
    runs: list[VctkStage2Run] = []
    for case, stem in stems.items():
        run_dir = root / stem / "vctk"
        runs.append(
            VctkStage2Run(
                case=case,
                token_cache=run_dir / "token_cache.pt",
                output_dir=run_dir / "stage2_detfix",
                group=f"laser-vctk-stage2-detfix-{case}-20260617",
            )
        )
    return runs


def _stage2_overrides(
    run: VctkStage2Run,
    *,
    project: str,
    vctk_dir: Path,
    devices: int,
    nodes: int,
) -> list[str]:
    strategy = "ddp" if devices * nodes > 1 else "auto"
    sample_prompts = "[" + ",".join(f'"{prompt}"' for prompt in SAMPLE_PROMPTS) + "]"
    return [
        f"token_cache_path={run.token_cache}",
        f"output_dir={run.output_dir}",
        "seed=42",
        "ar.type=sparse_spatial_depth",
        "ar.max_steps=300000",
        "ar.d_model=768",
        "ar.n_heads=12",
        "ar.n_layers=18",
        "ar.d_ff=3072",
        "ar.warmup_steps=1500",
        "ar.min_lr_ratio=0.03",
        "ar.n_global_spatial_tokens=16",
        "ar.coeff_loss_type=auto",
        "ar.coeff_huber_delta=0.25",
        "ar.sample_coeff_mode=gaussian",
        "ar.learning_rate=2.5e-4",
        "ar.text_conditional=true",
        "ar.text_prefix_length=16",
        "train_ar.max_epochs=300",
        "train_ar.batch_size=4",
        "train_ar.max_items=0",
        "train_ar.limit_train_batches=1.0",
        "train_ar.limit_val_batches=1.0",
        "train_ar.limit_test_batches=1.0",
        "train_ar.log_every_n_steps=50",
        "train_ar.devices=" + str(devices),
        "train_ar.num_nodes=" + str(nodes),
        f"train_ar.strategy={strategy}",
        "train_ar.precision=bf16-mixed",
        "train_ar.accelerator=gpu",
        "train_ar.deterministic=false",
        "train_ar.sample_top_k=0",
        "train_ar.sample_coeff_mode=gaussian",
        "train_ar.sample_every_n_epochs=2",
        "train_ar.sample_log_to_wandb=true",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=true",
        f"train_ar.sample_text_prompts={sample_prompts}",
        "train_ar.batch_size=4",
        "train_ar.sample_temperature=0.8",
        "train_ar.sample_num_images=8",
        "train_ar.compute_generation_fid=false",
        "train_ar.compute_audio_generation_metrics=true",
        "train_ar.generation_metric_num_samples=16",
        "data.dataset=vctk",
        f"data.data_dir={vctk_dir}",
        "data.image_size=128",
        "data.num_workers=2",
        f"wandb.project={project}",
        f"wandb.group={run.group}",
        f"wandb.name=vctk-stage2-detfix-{run.case}",
        "wandb.tags=[train,laser,vctk,stage2,transformer,text_conditional,detfix]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={run.output_dir / 'wandb'}",
    ]


def _write_job_files(
    *,
    snapshot_path: Path,
    job_root: Path,
    run: VctkStage2Run,
    project: str,
    vctk_dir: Path,
    devices: int,
    nodes: int,
) -> tuple[Path, Path]:
    run_root = job_root / run.case
    run_root.mkdir(parents=True, exist_ok=True)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run_script = run_root / "run_stage2_detfix.sh"
    sbatch_script = run_root / "sbatch_stage2_detfix.sh"
    stage2_args = _stage2_overrides(
        run,
        project=project,
        vctk_dir=vctk_dir,
        devices=devices,
        nodes=nodes,
    )
    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="${{PYTHONUSERBASE:-{_scratch_path('.pydeps', 'laser_src_py311')}}}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot_path}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-0}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TMPDIR="/tmp/laser_vctk_stage2_detfix_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR" "{run.output_dir / 'wandb'}"

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

cd "{snapshot_path}"

STAGE2_ARGS=(
{_bash_array(stage2_args)}
)

echo "=== VCTK Stage 2 deterministic fix retry ({run.case}) ==="
"$PYTHON_BIN" train.py stage2 "${{STAGE2_ARGS[@]}}"
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
LASER_NODES={nodes}
LAUNCH=()
if [[ "$LASER_NODES" -gt 1 ]]; then
  LAUNCH=(srun --nodes="$LASER_NODES" --ntasks="$LASER_NODES" --ntasks-per-node=1)
fi

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
  "${{LAUNCH[@]}}" "$CONTAINER_BIN" exec --nv \\
    --bind "{snapshot_path}" \\
    --bind "/scratch/{_user()}" \\
    --bind "/projects" \\
    --bind /dev/shm \\
    "$IMAGE" \\
    bash "{run_script}"
else
  "${{LAUNCH[@]}}" bash "{run_script}"
fi
""",
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return run_script, sbatch_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restart failed VCTK non-patch stage-2 jobs with deterministic algorithms disabled.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=_scratch_path("submission_snapshots"))
    parser.add_argument("--job-root", default=_scratch_path("runs", "laser_vctk_stage2_detfix"))
    parser.add_argument("--vctk-dir", default=_scratch_path("Projects", "data", "VCTK-Corpus", "VCTK-Corpus"))
    parser.add_argument("--cases", default="d5k3,d5k4,d6k3")
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=12)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--project", default="laser")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {item.strip() for item in str(args.cases).split(",") if item.strip()}
    runs = [run for run in _default_runs() if run.case in selected]
    if not runs:
        raise SystemExit("No VCTK stage-2 runs selected.")
    for run in runs:
        if not run.token_cache.is_file():
            raise FileNotFoundError(f"{run.case}: token cache not found: {run.token_cache}")
    if int(args.gpus) < 1 or int(args.nodes) < 1:
        raise SystemExit("--gpus and --nodes must be >= 1.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = _snapshot_repo(
        Path(args.repo).expanduser().resolve(),
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_vctk_stage2_detfix_{stamp}",
    )
    job_root = Path(args.job_root).expanduser().resolve() / f"vctk-stage2-detfix-{stamp}"
    job_root.mkdir(parents=True, exist_ok=True)

    submissions = []
    for run in runs:
        _, sbatch_script = _write_job_files(
            snapshot_path=snapshot_path,
            job_root=job_root,
            run=run,
            project=str(args.project),
            vctk_dir=Path(args.vctk_dir).expanduser().resolve(),
            devices=int(args.gpus),
            nodes=int(args.nodes),
        )
        log_base = job_root / run.case / run.case
        cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name=vctk-s2-{run.case}",
            f"--nodes={int(args.nodes)}",
            f"--ntasks={int(args.nodes)}",
            "--ntasks-per-node=1",
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
                "case": run.case,
                "job_id": job_id,
                "token_cache": str(run.token_cache),
                "output_dir": str(run.output_dir),
                "stdout": f"{log_base}_{job_id}.out" if job_id != "dry-run" else f"{log_base}_<jobid>.out",
                "stderr": f"{log_base}_{job_id}.err" if job_id != "dry-run" else f"{log_base}_<jobid>.err",
            }
        )

    print(f"Snapshot: {snapshot_path}")
    print(f"Job root: {job_root}")
    print(f"Nodes/job: {int(args.nodes)}")
    print(f"GPUs/node: {int(args.gpus)}")
    for item in submissions:
        print(
            f"[{item['case']}] job={item['job_id']} token_cache={item['token_cache']} "
            f"output_dir={item['output_dir']} stdout={item['stdout']} stderr={item['stderr']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
