#!/usr/bin/env python3
"""Submit VCTK speaker-ID conditional stage-2 training."""

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


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


DEFAULT_CACHE = (
    "/scratch/{user}/runs/laser_power_vctk_tts/"
    "laser-train-vctktts-21preiwb-speakercond-20260621_2045/"
    "vctk/token_cache_speaker.pt"
)
DEFAULT_CKPT = (
    "/scratch/{user}/runs/laser_power_vctk_tts/"
    "laser-train-vctktts-21preiwb-speakercond-20260621_2045/"
    "vctk/stage2/checkpoints/s2_20260621_203424/s2-epoch=018-val/loss=2.6113.ckpt"
)


def user():
    return os.environ.get("USER", "unknown")


def scratch_path(*parts):
    return str(Path("/scratch") / user() / Path(*parts))


def q(value):
    return shlex.quote(str(value))


def snapshot_ignore(repo):
    repo = repo.resolve()

    def ignore(current_dir, names):
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


def snapshot_repo(repo, snapshot_root, stem):
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / stem
    if snapshot_path.exists():
        shutil.rmtree(str(snapshot_path))
    shutil.copytree(str(repo), str(snapshot_path), ignore=snapshot_ignore(repo))
    return snapshot_path


def bash_array_lines(items):
    return "\n".join("  " + q(item) for item in items)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=scratch_path("submission_snapshots"))
    parser.add_argument("--run-root-base", default=scratch_path("runs", "vctk_speaker_stage2"))
    parser.add_argument("--token-cache", default=DEFAULT_CACHE.format(user=user()))
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT.format(user=user()))
    parser.add_argument("--vctk-dir", default=scratch_path("Projects", "data", "VCTK-Corpus"))
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=24)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--learning-rate", default="2.5e-4")
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--min-lr-ratio", default="0.05")
    parser.add_argument("--sample-every-n-epochs", type=int, default=1)
    parser.add_argument("--sample-num-images", type=int, default=12)
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--image", default="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
    parser.add_argument("--pydeps", default=scratch_path(".pydeps", "laser_src_py311"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1:
        raise SystemExit("--gpus must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.sample_every_n_epochs < 1:
        raise SystemExit("--sample-every-n-epochs must be >= 1")
    if args.sample_num_images < 1:
        raise SystemExit("--sample-num-images must be >= 1")
    return args


def main():
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    token_cache = Path(args.token_cache).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve() if str(args.ckpt_path).strip() else None
    vctk_dir = Path(args.vctk_dir).expanduser().resolve()
    if not token_cache.is_file():
        raise FileNotFoundError("Token cache not found: " + str(token_cache))
    if ckpt_path is not None and not ckpt_path.is_file():
        raise FileNotFoundError("Checkpoint not found: " + str(ckpt_path))
    if not vctk_dir.is_dir():
        raise FileNotFoundError("VCTK directory not found: " + str(vctk_dir))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = snapshot_repo(
        repo,
        Path(args.snapshot_root).expanduser().resolve(),
        "laser_vctk_speaker_stage2_" + stamp,
    )
    run_root = Path(args.run_root_base).expanduser().resolve() / ("vctk-speaker-stage2-" + stamp)
    stage2_dir = run_root / "stage2"
    slurm_dir = run_root / "slurm"
    run_root.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    strategy = "ddp" if int(args.gpus) > 1 else "auto"
    run_name = "vctk-stage2-speakercond-" + stamp
    overrides = [
        "token_cache_path=" + str(token_cache),
        "output_dir=" + str(stage2_dir),
        "seed=42",
        "ar.type=sparse_spatial_depth",
        "ar.autoregressive_coeffs=true",
        "ar.class_conditional=true",
        "ar.num_classes=109",
        "ar.text_conditional=false",
        "ar.max_steps=-1",
        "ar.d_model=768",
        "ar.n_heads=12",
        "ar.n_layers=18",
        "ar.d_ff=3072",
        "ar.n_global_spatial_tokens=16",
        "ar.dropout=0.1",
        "ar.learning_rate=" + str(args.learning_rate),
        "ar.warmup_steps=" + str(int(args.warmup_steps)),
        "ar.min_lr_ratio=" + str(args.min_lr_ratio),
        "ar.coeff_loss_type=auto",
        "ar.coeff_loss_weight=1.0",
        "ar.coeff_huber_delta=0.25",
        "train_ar.max_epochs=" + str(int(args.max_epochs)),
        "train_ar.batch_size=" + str(int(args.batch_size)),
        "train_ar.max_items=0",
        "train_ar.limit_train_batches=1.0",
        "train_ar.limit_val_batches=1.0",
        "train_ar.limit_test_batches=1.0",
        "train_ar.log_every_n_steps=50",
        "train_ar.devices=" + str(int(args.gpus)),
        "train_ar.num_nodes=1",
        "train_ar.strategy=" + strategy,
        "++train_ar.ddp_timeout_hours=6",
        "train_ar.precision=bf16-mixed",
        "train_ar.accelerator=gpu",
        "train_ar.deterministic=false",
        "train_ar.gradient_clip_val=1.0",
        "train_ar.sample_temperature=0.8",
        "train_ar.sample_top_k=0",
        "train_ar.sample_every_n_epochs=" + str(int(args.sample_every_n_epochs)),
        "train_ar.sample_num_images=" + str(int(args.sample_num_images)),
        "++train_ar.sample_class_labels=[0,10,25,50,75,100,0,10,25,50,75,100]",
        "train_ar.compute_generation_fid=false",
        "train_ar.compute_audio_generation_metrics=true",
        "train_ar.generation_metric_num_samples=18",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=true",
        "train_ar.sample_log_to_wandb=true",
        "++train_ar.checkpoint_save_top_k=1",
        "++train_ar.checkpoint_save_last=true",
        "train_ar.checkpoint_keep_recent=1",
        "train_ar.checkpoint_every_n_epochs=25",
        "data.dataset=vctk",
        "data.data_dir=" + str(vctk_dir),
        "data.image_size=128",
        "data.num_workers=2",
        "wandb.project=" + str(args.wandb_project),
        "wandb.group=vctk-speaker-stage2-" + stamp,
        "wandb.name=" + run_name,
        "wandb.tags=[train,laser,vctk,stage2,transformer,generation,speaker_conditioned]",
        "wandb.append_timestamp=false",
        "wandb.save_dir=" + str(stage2_dir / "wandb"),
    ]
    if ckpt_path is not None:
        overrides.insert(0, "ckpt_path='" + str(ckpt_path) + "'")

    run_script = run_root / "run_vctk_speaker_stage2.sh"
    sbatch_script = run_root / "sbatch_vctk_speaker_stage2.sh"
    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

echo "=== VCTK speaker-ID conditional stage 2 ==="
echo "run_root={run_root}"
echo "snapshot={snapshot}"
echo "token_cache={token_cache}"
echo "ckpt_path={ckpt_path if ckpt_path is not None else ''}"
echo "=== GPU inventory ==="
nvidia-smi
echo ""

export PYTHONUSERBASE={q(args.pydeps)}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-0}}"
export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-{stage2_dir / "wandb" / "data"}}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{stage2_dir / "wandb" / "cache"}}}"
export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-{stage2_dir / "wandb" / "artifacts"}}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/{user()}/tmp/laser_vctk_speaker_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$PYTHONUSERBASE" "$TMPDIR" {q(stage2_dir / "wandb")} "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

if command -v flock >/dev/null 2>&1; then
  (
    flock 9
    "$PYTHON_BIN" -m pip install --user --quiet numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips soundfile pystoi 2>/dev/null || true
  ) 9>"$PYTHONUSERBASE/.install.lock"
else
  "$PYTHON_BIN" -m pip install --user --quiet numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips soundfile pystoi 2>/dev/null || true
fi

cd {q(snapshot)}
STAGE2_ARGS=(
{bash_array_lines(overrides)}
)

printf 'Launching:'
printf ' %q' "$PYTHON_BIN" train.py stage2 "${{STAGE2_ARGS[@]}}"
printf '\\n'
exec "$PYTHON_BIN" train.py stage2 "${{STAGE2_ARGS[@]}}"
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
if ! command -v singularity >/dev/null 2>&1; then
  module load singularity 2>/dev/null || true
  module load singularityce 2>/dev/null || true
  module load singularity-ce 2>/dev/null || true
fi

if command -v singularity >/dev/null 2>&1; then
  singularity exec --nv \\
    --bind {q(snapshot)} \\
    --bind /scratch/{user()} \\
    --bind {q(token_cache.parent)} \\
    --bind {q(vctk_dir)} \\
    --bind {q(run_root)} \\
    --bind /dev/shm \\
    {q(args.image)} \\
    bash {q(run_script)}
else
  echo "Warning: singularity not found; running bare" >&2
  bash {q(run_script)}
fi
""",
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)

    log_base = slurm_dir / "vctk_speaker_stage2_%j"
    cmd = [
        "sbatch",
        "--partition=" + str(args.partition),
        "--job-name=vctk-spk-s2",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=" + str(int(args.cpus_per_task)),
        "--gres=gpu:" + str(int(args.gpus)),
        "--mem=" + str(int(args.mem_mb)),
        "--time=" + str(args.time_limit),
        "--chdir=" + str(snapshot),
        "--output=" + str(log_base) + ".out",
        "--error=" + str(log_base) + ".err",
        "--requeue",
    ]
    if str(args.constraint).strip():
        cmd.append("--constraint=" + str(args.constraint))
    cmd.append(str(sbatch_script))

    print("Snapshot:", snapshot)
    print("Run root:", run_root)
    print("Token cache:", token_cache)
    print("Checkpoint:", ckpt_path if ckpt_path is not None else "<fresh>")
    if args.dry_run:
        print("Dry run sbatch:", " ".join(q(part) for part in cmd))
        return 0

    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    job_id = (proc.stdout or proc.stderr).strip().split()[-1]
    print("Submitted job:", job_id)
    print("stdout:", slurm_dir / ("vctk_speaker_stage2_" + job_id + ".out"))
    print("stderr:", slurm_dir / ("vctk_speaker_stage2_" + job_id + ".err"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
