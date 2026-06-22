#!/usr/bin/env python3

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
    raise SystemExit("ERROR: submit_celebahq_stage2_continue50.py requires Python >= 3.8.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_multimodal_sweep as base  # noqa: E402


@dataclass(frozen=True)
class ContinueRun:
    label: str
    group: str
    wandb_id: str
    token_cache: Path
    output_dir: Path
    checkpoint: Path
    batch_size: int
    max_epochs: int
    learning_rate: str = "6.0e-4"
    dependency: str = ""
    mem_mb: int = 160000


def _user() -> str:
    return os.environ.get("USER", "unknown")


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def bash_array(items: list[str]) -> str:
    return "\n".join(f"  {q(item)}" for item in items)


def continue_runs() -> list[ContinueRun]:
    ksvd = Path("/scratch/xl598/runs/laser_debugging_patchseq_ksvd_sweep")
    simple = Path("/scratch/xl598/runs/laser_debugging_simple_dict_patch_sweep")
    simple_root = simple / "laser-train-celebahq-10ep-simple-dict-patch-sweep-20260514_040939"
    highcap = Path("/scratch/xl598/runs/laser_debugging_celebahq_highcap_patch_sweep")
    highcap_root = highcap / "laser-train-celebahq-10s1-40s2-highcap-patch-20260514_054046"

    p4_clean = ksvd / "laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-clean-grad-20260514_034728" / "celebahq"
    p4_online = ksvd / "laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-onlineksvd-20260514_034804" / "celebahq"
    nopatch = simple_root / "nopatch-k1-a16384" / "celebahq"
    p2_simple = simple_root / "patch-p2-k2-a16384" / "celebahq"
    p4_simple = simple_root / "patch-p4-k4-a32768" / "celebahq"
    p8_highcap = highcap_root / "patch-p8-k32-a131072" / "celebahq"

    return [
        ContinueRun(
            label="ksvd-p4-clean-grad-plus50",
            group="laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-clean-grad-20260514_034728",
            wandb_id="zq5dayo7",
            token_cache=p4_clean / "token_cache.pt",
            output_dir=p4_clean / "stage2",
            checkpoint=p4_clean / "stage2/checkpoints/s2_20260514_134544/last.ckpt",
            batch_size=16,
            max_epochs=78,
        ),
        ContinueRun(
            label="ksvd-p4-onlineksvd-plus50",
            group="laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-onlineksvd-20260514_034804",
            wandb_id="h2te6xxo",
            token_cache=p4_online / "token_cache.pt",
            output_dir=p4_online / "stage2",
            checkpoint=p4_online / "stage2/checkpoints/s2_20260514_134435/last.ckpt",
            batch_size=16,
            max_epochs=77,
        ),
        ContinueRun(
            label="simple-nopatch-plus50",
            group="laser-train-celebahq-10ep-simple-dict-patch-sweep-20260514_040939-nopatch-k1-a16384",
            wandb_id="uag4a11r",
            token_cache=nopatch / "token_cache.pt",
            output_dir=nopatch / "stage2",
            checkpoint=nopatch / "stage2/checkpoints/s2_20260514_054032/last.ckpt",
            batch_size=8,
            max_epochs=97,
        ),
        ContinueRun(
            label="simple-p2-k2-a16384-plus50",
            group="laser-train-celebahq-10ep-simple-dict-patch-sweep-20260514_040939-patch-p2-k2-a16384",
            wandb_id="jhi3qxe2",
            token_cache=p2_simple / "token_cache.pt",
            output_dir=p2_simple / "stage2",
            checkpoint=p2_simple / "stage2/checkpoints/s2_20260514_134446/last.ckpt",
            batch_size=8,
            max_epochs=89,
        ),
        ContinueRun(
            label="simple-p4-k4-a32768-plus50",
            group="laser-train-celebahq-10ep-simple-dict-patch-sweep-20260514_040939-patch-p4-k4-a32768",
            wandb_id="08ibhyiv",
            token_cache=p4_simple / "token_cache.pt",
            output_dir=p4_simple / "stage2",
            checkpoint=p4_simple / "stage2/checkpoints/s2_20260514_134452/last.ckpt",
            batch_size=8,
            max_epochs=75,
        ),
        ContinueRun(
            label="highcap-p8-k32-a131072-plus50",
            group="laser-train-celebahq-10s1-40s2-highcap-patch-20260514_054046-patch-p8-k32-a131072",
            wandb_id="h4b0zwvs",
            token_cache=p8_highcap / "token_cache.pt",
            output_dir=p8_highcap / "stage2",
            checkpoint=p8_highcap / "stage2/checkpoints/s2_20260514_110038/last.ckpt",
            batch_size=6,
            max_epochs=90,
            dependency="afterok:54072645",
            mem_mb=192000,
        ),
    ]


def validate(run: ContinueRun, *, allow_pending_dependency: bool = True) -> None:
    if not run.token_cache.is_file():
        raise FileNotFoundError(f"{run.label}: missing token cache: {run.token_cache}")
    if not run.checkpoint.is_file():
        if allow_pending_dependency and run.dependency:
            return
        raise FileNotFoundError(f"{run.label}: missing checkpoint: {run.checkpoint}")
    run.output_dir.mkdir(parents=True, exist_ok=True)


def stage2_overrides(run: ContinueRun) -> list[str]:
    return [
        f"token_cache_path={run.token_cache}",
        f"output_dir={run.output_dir}",
        f"ckpt_path={run.checkpoint}",
        "seed=42",
        "ar.type=sparse_spatial_depth",
        "ar.max_steps=-1",
        f"train_ar.max_epochs={run.max_epochs}",
        f"train_ar.batch_size={run.batch_size}",
        "train_ar.max_items=0",
        "train_ar.limit_train_batches=1.0",
        "train_ar.limit_val_batches=1.0",
        "train_ar.limit_test_batches=1.0",
        "train_ar.log_every_n_steps=50",
        "train_ar.sample_every_n_epochs=1",
        "train_ar.sample_log_to_wandb=true",
        "train_ar.sample_num_images=64",
        "train_ar.generation_metric_num_samples=0",
        "train_ar.compute_generation_fid=false",
        "train_ar.compute_audio_generation_metrics=false",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=false",
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
        "wandb.tags=[train,laser,celebahq,stage2,transformer,generation,plus50]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={run.output_dir / 'wandb'}",
        f"wandb.id={run.wandb_id}",
        "wandb.resume=allow",
    ]


def write_job(snapshot_path: Path, job_root: Path, run: ContinueRun) -> tuple[Path, Path]:
    run_root = job_root / run.label
    run_root.mkdir(parents=True, exist_ok=True)
    run_script = run_root / "run_stage2_continue50.sh"
    sbatch_script = run_root / "sbatch_stage2_continue50.sh"
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
export TMPDIR="/tmp/laser_stage2_continue50_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR" {q(run.output_dir / 'wandb')}

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

"$PYTHON_BIN" -m pip install --user --quiet \\
  numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \\
  torch-fidelity matplotlib lpips soundfile 2>/dev/null || true

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this job; failing before training.")
print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")
PY

cd {q(snapshot_path)}

STAGE2_ARGS=(
{bash_array(stage2_overrides(run))}
)

echo "=== Continue CelebA-HQ Stage 2 for +50 epochs: {run.label} ==="
"$PYTHON_BIN" train.py stage2 "${{STAGE2_ARGS[@]}}"
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
    parser = argparse.ArgumentParser(description="Continue selected CelebA-HQ transformer runs for 50 more epochs.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=base._scratch_path("submission_snapshots"))
    parser.add_argument("--job-root", default=base._scratch_path("runs", "laser_debugging_celebahq_continue50"))
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--exclude-nodes", default="gpu018")
    parser.add_argument("--skip-dependent", action="store_true", help="Skip runs that are submitted with dependencies.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def submit(cmd: list[str], dry_run: bool) -> str:
    if dry_run:
        print(" ".join(shlex.quote(part) for part in cmd))
        return "dry-run"
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    text = (proc.stdout or proc.stderr).strip()
    return text.split()[-1]


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = base.snapshot_repo(
        Path(args.repo).expanduser().resolve(),
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_celebahq_stage2_continue50_{stamp}",
    )
    job_root = Path(args.job_root).expanduser().resolve() / f"continue50-{stamp}"
    job_root.mkdir(parents=True, exist_ok=True)

    submissions = []
    for run in continue_runs():
        if args.skip_dependent and run.dependency:
            continue
        validate(run)
        _, sbatch_script = write_job(snapshot_path, job_root, run)
        log_base = job_root / run.label / run.label
        cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name=s2p50-{run.label[:15]}",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=8",
            "--gres=gpu:3",
            f"--mem={run.mem_mb}",
            "--time=3-00:00:00",
            f"--chdir={snapshot_path}",
            f"--output={log_base}_%j.out",
            f"--error={log_base}_%j.err",
        ]
        if str(args.exclude_nodes or "").strip():
            cmd.append(f"--exclude={str(args.exclude_nodes).strip()}")
        if run.dependency:
            cmd.append(f"--dependency={run.dependency}")
        cmd.append(str(sbatch_script))
        job_id = submit(cmd, args.dry_run)
        submissions.append((run.label, job_id, f"{log_base}_{job_id}.out", f"{log_base}_{job_id}.err", run.dependency))

    print(f"Snapshot: {snapshot_path}")
    print(f"Job root: {job_root}")
    for label, job_id, stdout, stderr, dependency in submissions:
        suffix = f" dependency={dependency}" if dependency else ""
        print(f"[{label}] job={job_id}{suffix} stdout={stdout} stderr={stderr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
