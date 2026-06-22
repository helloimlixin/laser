#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 8):
    raise SystemExit("ERROR: submit_stl10_qtp_stage2_power.py requires Python >= 3.8.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_multimodal_sweep as base  # noqa: E402


SOURCE_RUN_ROOT = Path(
    "/scratch/xl598/runs/laser_stl10_classcond_sweep/"
    "laser-train-stl64-d2-p2s2-k6-a4096-e96-20260615_084848/stl10"
)
TOKEN_CACHE = SOURCE_RUN_ROOT / "token_cache.pt"
STL10_DIR = Path("/scratch/xl598/datasets/stl10")


def _user() -> str:
    return os.environ.get("USER", "unknown")


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def bash_array(items: list[str]) -> str:
    return "\n".join(f"  {q(item)}" for item in items)


def stage2_overrides(
    *,
    token_cache: Path,
    output_dir: Path,
    wandb_group: str,
    wandb_project: str,
) -> list[str]:
    return [
        f"token_cache_path={token_cache}",
        f"output_dir={output_dir}",
        "seed=42",
        "train_ar.max_epochs=300",
        "train_ar.max_items=0",
        "train_ar.limit_train_batches=1.0",
        "train_ar.limit_val_batches=1.0",
        "train_ar.limit_test_batches=1.0",
        "train_ar.sample_log_to_wandb=true",
        "train_ar.devices=2",
        "train_ar.strategy=ddp",
        "train_ar.precision=bf16-mixed",
        "train_ar.accelerator=gpu",
        "data.dataset=stl10",
        f"data.data_dir={STL10_DIR}",
        "data.image_size=64",
        "data.num_workers=4",
        f"wandb.project={wandb_project}",
        f"wandb.group={wandb_group}",
        "wandb.name=stl10-stage2-classcond-qtp-power",
        "wandb.tags=[train,laser,stl10,64x64,stage2,transformer,class_conditional,generation,qtp5wa8r_settings]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
        "ar.type=sparse_spatial_depth",
        "ar.autoregressive_coeffs=true",
        "ar.class_conditional=true",
        "ar.num_classes=10",
        "ar.max_steps=-1",
        "ar.d_model=768",
        "ar.n_heads=12",
        "ar.n_layers=18",
        "ar.d_ff=3072",
        "ar.n_global_spatial_tokens=16",
        "ar.dropout=0.1",
        "ar.learning_rate=2.5e-4",
        "ar.warmup_steps=1500",
        "ar.min_lr_ratio=0.03",
        "ar.coeff_loss_type=auto",
        "ar.coeff_loss_weight=1.0",
        "ar.coeff_huber_delta=0.25",
        "train_ar.batch_size=64",
        "train_ar.gradient_clip_val=1.0",
        "train_ar.val_check_interval=1.0",
        "train_ar.log_every_n_steps=20",
        "train_ar.checkpoint_save_top_k=1",
        "train_ar.checkpoint_save_last=false",
        "train_ar.checkpoint_keep_recent=1",
        "train_ar.checkpoint_every_n_epochs=25",
        "train_ar.sample_every_n_epochs=5",
        "train_ar.sample_num_images=20",
        "train_ar.sample_class_labels=[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]",
        "train_ar.sample_temperature=0.7",
        "train_ar.sample_top_k=0",
        "train_ar.compute_generation_fid=false",
        "train_ar.compute_audio_generation_metrics=false",
        "train_ar.generation_metric_num_samples=0",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=true",
    ]


def write_job_files(
    *,
    snapshot_path: Path,
    run_dir: Path,
    token_cache: Path,
    wandb_group: str,
    wandb_project: str,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "stage2"
    (output_dir / "wandb").mkdir(parents=True, exist_ok=True)
    run_script = run_dir / "run_stage2_qtp_power.sh"
    sbatch_script = run_dir / "sbatch_stage2_qtp_power.sh"
    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="${{PYTHONUSERBASE:-/scratch/{_user()}/.pydeps/laser_src_py311}}"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot_path}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{_user()}/.cache}}"
export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-$XDG_CACHE_HOME/wandb}}"
export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{_user()}/.config/wandb}}"
export MPLCONFIGDIR="${{MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}}"
export HYDRA_FULL_ERROR=1
export TMPDIR="/tmp/laser_stl10_qtp_stage2_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR" {q(output_dir / 'wandb')}

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

run_user_pip_install() {{
  mkdir -p "$PYTHONUSERBASE"
  if command -v flock >/dev/null 2>&1; then
    (
      flock 9
      "$PYTHON_BIN" -m pip install --user --quiet "$@" 2>/dev/null || true
    ) 9>"$PYTHONUSERBASE/.install.lock"
  else
    "$PYTHON_BIN" -m pip install --user --quiet "$@" 2>/dev/null || true
  fi
}}

run_user_pip_install \\
  numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \\
  torch-fidelity matplotlib lpips soundfile pystoi

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
{bash_array(stage2_overrides(token_cache=token_cache, output_dir=output_dir, wandb_group=wandb_group, wandb_project=wandb_project))}
)

echo "=== Stage 2: STL10 class-conditional qtp5wa8r-power prior ==="
"$PYTHON_BIN" train.py stage2 "${{STAGE2_ARGS[@]}}"
""",
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)
    sbatch_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

BOOTSTRAP_LOG="{run_dir}/bootstrap_${{SLURM_JOB_ID:-manual}}.log"
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
    --bind {q(snapshot_path)} \\
    --bind "/scratch/{_user()}" \\
    --bind "/projects" \\
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
    parser = argparse.ArgumentParser(
        description="Submit STL10 class-conditional stage-2 with qtp5wa8r image prior settings."
    )
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=base._scratch_path("submission_snapshots"))
    parser.add_argument("--run-root", default=base._scratch_path("runs", "laser_stl10_qtp_stage2_power"))
    parser.add_argument("--token-cache", default=str(TOKEN_CACHE))
    parser.add_argument("--project", default="laser")
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default="2-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=96000)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token_cache = Path(args.token_cache).expanduser().resolve()
    if not token_cache.is_file():
        raise FileNotFoundError(f"Missing token cache: {token_cache}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = base.snapshot_repo(
        Path(args.repo).expanduser().resolve(),
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_stl10_qtp_stage2_power_{stamp}",
    )
    group = f"laser-train-stl64-d2-p2s2-k6-a4096-e96-qtpprior-{stamp}"
    run_dir = Path(args.run_root).expanduser().resolve() / group / "stl10"
    _, sbatch_script = write_job_files(
        snapshot_path=snapshot_path,
        run_dir=run_dir,
        token_cache=token_cache,
        wandb_group=group,
        wandb_project=args.project,
    )
    log_base = run_dir / "stl10-qtp-stage2"
    cmd = [
        "sbatch",
        f"--partition={args.partition}",
        "--job-name=stl10-qtp-s2",
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

    print(f"Snapshot: {snapshot_path}")
    print(f"Run dir: {run_dir}")
    print(f"Token cache: {token_cache}")
    print(f"W&B group: {group}")
    print(f"Job: {job_id}")
    print(f"stdout: {log_base}_{job_id}.out")
    print(f"stderr: {log_base}_{job_id}.err")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
