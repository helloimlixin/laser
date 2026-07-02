#!/usr/bin/env python3
"""Submit a 50-epoch FFHQ stage-1 adversarial continuation from yf7lf9zo."""

from __future__ import annotations

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path


USER = os.environ.get("USER", "xl598")
REPO = Path("/scratch/xl598/Projects/laser")
SNAPSHOT_ROOT = Path("/scratch/xl598/submission_snapshots")
RUN_ROOT_BASE = Path("/scratch/xl598/runs/ffhq_stage1_adv50_yf7lf9zo")
DATA_DIR = Path("/scratch/xl598/Projects/data/ffhq")
INIT_CKPT = Path(
    "/scratch/xl598/runs/ffhq_stage1_sweep_lpips005/"
    "ffhq-laser-paper-stage1-sweep-20260628_022906/k3-a8192/stage1/"
    "checkpoints/run_slurm57345280_4/laser/final.ckpt"
)
PYDEPS = Path("/scratch/xl598/.pydeps/laser_src_py311")
IMAGE = "docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
WANDB_ID = "yf7adv50"
WANDB_NAME = "ffhq-stage1-adv50-k3-a8192-from-yf7lf9zo"
WANDB_GROUP = "ffhq-laser-paper-stage1-sweep-20260628_022906"
VGG16_WEIGHTS = Path("/scratch/xl598/.cache/torch/hub/checkpoints/vgg16-397923af.pth")

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


def q(value: object) -> str:
    return shlex.quote(str(value))


def bash_array(items: list[str]) -> str:
    return "".join(f"  {q(item)}\n" for item in items)


def script_body(raw: str) -> str:
    lines = textwrap.dedent(raw).lstrip("\n").splitlines()
    cleaned = []
    for line in lines:
        if line.startswith("            "):
            line = line[12:]
        cleaned.append(line.rstrip())
    return "\n".join(cleaned) + "\n"


def snapshot_ignore(repo: Path):
    repo = repo.resolve()

    def ignore(current_dir: str, names: list[str]) -> set[str]:
        rel_dir = Path(current_dir).resolve().relative_to(repo)
        ignored: set[str] = set()
        for name in names:
            rel_path = (rel_dir / name) if rel_dir != Path(".") else Path(name)
            if len(rel_path.parts) >= 2 and rel_path.parts[0] == "configs" and rel_path.parts[1] == "wandb":
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDES):
                ignored.add(name)
        return ignored

    return ignore


def snapshot_repo(repo: Path, snapshot: Path) -> None:
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot}")
    shutil.copytree(repo, snapshot, ignore=snapshot_ignore(repo))


def stage1_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        "stage1",
        "seed=42",
        f"output_dir={output_dir}",
        f"init_ckpt_path={args.init_ckpt}",
        "model=laser_image_nonpatch_d5",
        "data=ffhq",
        f"data.data_dir={args.data_dir}",
        "data.batch_size=8",
        "data.eval_batch_size=8",
        "data.num_workers=8",
        "data.pin_memory=true",
        "data.prefetch_factor=6",
        "data.image_size=256",
        "data.train_crop_size=null",
        "data.augment=true",
        "train.accelerator=gpu",
        f"train.devices={int(args.gpus)}",
        "train.num_nodes=1",
        "train.strategy=ddp",
        "train.precision=bf16-mixed",
        f"train.max_epochs={int(args.epochs)}",
        "train.max_steps=-1",
        "train.limit_train_batches=1.0",
        "train.limit_val_batches=1.0",
        "train.limit_test_batches=0",
        "train.val_check_interval=1.0",
        "train.run_test_after_fit=false",
        "train.compute_rfid_after_fit=false",
        "train.rfid_split=val",
        "train.rfid_batch_size=32",
        "train.rfid_num_workers=8",
        "train.rfid_max_samples=0",
        "train.rfid_device=auto",
        "train.rfid_feature=2048",
        "train.learning_rate=4.0e-5",
        "train.beta=0.5",
        "train.beta2=0.9",
        "train.warmup_steps=0",
        "train.min_lr_ratio=1.0",
        "train.accumulate_grad_batches=1",
        "train.gradient_clip_val=1.0",
        "train.log_every_n_steps=20",
        "train.deterministic=false",
        "model.bottleneck_type=dictionary",
        "model.dropout=0.0",
        "model.embedding_dim=128",
        "model.commitment_cost=0.25",
        "model.bottleneck_loss_weight=0.75",
        "model.dictionary_loss_weight=null",
        "model.dict_learning_rate=2.5e-4",
        "model.coef_max=null",
        "model.patch_based=false",
        "model.patch_size=1",
        "model.patch_stride=1",
        "model.patch_reconstruction=tile",
        "model.data_init_from_first_batch=false",
        "model.recon_mse_weight=1.0",
        "model.recon_l1_weight=0.0",
        "model.recon_edge_weight=0.0",
        "model.perceptual_weight=0.05",
        "model.perceptual_start_step=5000",
        "model.perceptual_warmup_steps=10000",
        "model.adversarial_weight=0.75",
        "model.adversarial_start_step=0",
        "model.adversarial_warmup_steps=0",
        "model.disc_start_step=0",
        "model.disc_learning_rate=null",
        "model.discriminator_beta1=0.5",
        "model.discriminator_beta2=0.9",
        "model.disc_channels=64",
        "model.disc_num_layers=3",
        "model.disc_norm=group",
        "model.disc_spectral=false",
        "model.disc_loss=hinge",
        "model.use_adaptive_disc_weight=true",
        "model.disc_factor=1.0",
        "model.disc_weight_max=10000.0",
        "model.compute_fid=true",
        "model.log_images_every_n_steps=250",
        "model.diag_log_interval=250",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=4096",
        "model.num_embeddings=8192",
        "model.sparsity_level=3",
        "wandb.project=laser",
        f"wandb.name={args.wandb_name}",
        f"wandb.group={args.wandb_group}",
        "wandb.tags=[stage1_adv,ffhq,laser,dictionary,adversarial,continuation,yf7lf9zo,adv50]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
        "checkpoint.upload_to_wandb=false",
    ]


def write_job_files(args: argparse.Namespace, snapshot: Path, job_root: Path) -> tuple[Path, Path]:
    output_dir = job_root / "stage1_adv50"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_script = job_root / "run_yf7lf9zo_stage1_adv50.sh"
    sbatch_script = job_root / "sbatch_yf7lf9zo_stage1_adv50.sh"
    log_base = job_root / "yf7lf9zo_stage1_adv50"
    stage_args = bash_array(stage1_args(args, output_dir))

    run_script.write_text(
        script_body(
            f"""\
            #!/bin/bash
            set -euo pipefail

            export PYTHONUSERBASE={q(args.pydeps)}
            export PATH="$PYTHONUSERBASE/bin:$PATH"
            export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{q(snapshot)}${{PYTHONPATH:+:$PYTHONPATH}}"
            export WANDB_MODE="${{WANDB_MODE:-online}}"
            export WANDB_RUN_ID="{args.wandb_id}"
            export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
            export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-0}}"
            export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-{output_dir}/wandb/data}}"
            export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{output_dir}/wandb/cache}}"
            export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-{output_dir}/wandb/artifacts}}"
            export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{USER}/.config/wandb}}"
            export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{USER}/.cache}}"
            export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
            export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
            export MPLCONFIGDIR="${{MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}}"
            export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
            export HYDRA_FULL_ERROR=1
            export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
            export PYTHONUNBUFFERED=1
            export TMPDIR="/tmp/laser_yf7lf9zo_adv50_${{SLURM_JOB_ID:-$$}}"
            export TEMP="$TMPDIR"
            export TMP="$TMPDIR"
            if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" && -f {q(VGG16_WEIGHTS)} ]]; then
              export LASER_VGG16_WEIGHTS={q(VGG16_WEIGHTS)}
            fi

            mkdir -p "$PYTHONUSERBASE" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \\
              "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR" {q(output_dir / "wandb")}

            PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
            if [[ -z "$PYTHON_BIN" ]]; then
              echo "python3/python not found" >&2
              exit 127
            fi
            if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
              echo "ERROR: $PYTHON_BIN is too old; LASER requires Python >= 3.10." >&2
              exit 2
            fi

            if command -v nvidia-smi >/dev/null 2>&1; then
              nvidia-smi
            fi

            if command -v flock >/dev/null 2>&1; then
              (
                flock 9
                "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
              ) 9>"$PYTHONUSERBASE/.install.lock"
            else
              "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
            fi

            if [[ ! -f {q(args.init_ckpt)} ]]; then
              echo "Missing init checkpoint: {args.init_ckpt}" >&2
              exit 1
            fi

            cd {q(snapshot)}
            STAGE1_ARGS=(
            {stage_args}
            )

            echo "=== FFHQ stage1 adversarial continuation from yf7lf9zo ==="
            echo "init_checkpoint={args.init_ckpt}"
            echo "output_dir={output_dir}"
            echo "wandb_run_id={args.wandb_id}"
            echo "wandb_name={args.wandb_name}"
            echo "epochs={args.epochs}"
            printf 'Launching:'
            printf ' %q' "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
            printf '\\n'
            "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
            """
        ),
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    constraint_line = f"#SBATCH --constraint={args.constraint}\n" if str(args.constraint or "").strip() else ""
    sbatch_script.write_text(
        script_body(
            f"""\
            #!/bin/bash
            #SBATCH --job-name=ffhq-yf7-adv50
            #SBATCH --partition={args.partition}
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --ntasks-per-node=1
            #SBATCH --gres=gpu:{args.gpus}
            #SBATCH --cpus-per-task={args.cpus_per_task}
            #SBATCH --mem={args.mem_mb}
            #SBATCH --time={args.time_limit}
            {constraint_line}#SBATCH --output={log_base}_%j.out
            #SBATCH --error={log_base}_%j.err
            #SBATCH --requeue

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

            export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-/cache/home/{USER}/.apptainer/cache}}"
            export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}}"
            export APPTAINER_TMPDIR="${{APPTAINER_TMPDIR:-/scratch/{USER}/.apptainer/tmp_${{SLURM_JOB_ID:-manual}}}}"
            export SINGULARITY_TMPDIR="${{SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}}"
            mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

            IMAGE="${{IMAGE:-{args.image}}}"
            echo "container_bin=$CONTAINER_BIN"
            echo "image=$IMAGE"
            echo "snapshot={snapshot}"
            echo "job_root={job_root}"

            if [[ -n "$CONTAINER_BIN" ]]; then
              srun "$CONTAINER_BIN" exec --nv \\
                --bind /cache/home/{USER} \\
                --bind {q(snapshot)} \\
                --bind /scratch/{USER} \\
                --bind {q(args.data_dir)} \\
                --bind {q(job_root)} \\
                --bind {q(Path(args.init_ckpt).parent)} \\
                --bind /dev/shm \\
                "$IMAGE" \\
                bash {q(run_script)}
            else
              echo "Warning: singularity/apptainer not found; running bare" >&2
              srun bash {q(run_script)}
            fi
            """
        ),
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return run_script, sbatch_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--run-root-base", default=str(RUN_ROOT_BASE))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--init-ckpt", default=str(INIT_CKPT))
    parser.add_argument("--pydeps", default=str(PYDEPS))
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--wandb-id", default=WANDB_ID)
    parser.add_argument("--wandb-name", default=WANDB_NAME)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.gpus <= 0:
        raise SystemExit("--gpus must be positive")
    return args


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    init_ckpt = Path(args.init_ckpt).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"repo not found: {repo}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"FFHQ data dir not found: {data_dir}")
    if not init_ckpt.is_file():
        raise FileNotFoundError(f"init checkpoint not found: {init_ckpt}")

    args.data_dir = str(data_dir)
    args.init_ckpt = str(init_ckpt)
    args.pydeps = str(Path(args.pydeps).expanduser().resolve())
    args.image = str(args.image)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = Path(args.snapshot_root).expanduser().resolve() / f"laser_yf7lf9zo_stage1_adv50_{stamp}"
    job_root = Path(args.run_root_base).expanduser().resolve() / f"adv50_{stamp}"
    job_root.mkdir(parents=True, exist_ok=True)
    snapshot_repo(repo, snapshot)
    _, sbatch_script = write_job_files(args, snapshot, job_root)

    if args.dry_run:
        job_id = "dry-run"
    else:
        result = subprocess.run(["sbatch", str(sbatch_script)], check=True, text=True, capture_output=True)
        print(result.stdout.strip())
        job_id = (result.stdout or result.stderr).strip().split()[-1]

    print(f"Snapshot: {snapshot}")
    print(f"Job root: {job_root}")
    print(f"SBATCH: {sbatch_script}")
    print(f"Init checkpoint: {init_ckpt}")
    print(f"W&B run: helloimlixin-rutgers/laser/{args.wandb_id} ({args.wandb_name})")
    if job_id != "dry-run":
        print(f"stdout: {job_root / ('yf7lf9zo_stage1_adv50_' + job_id + '.out')}")
        print(f"stderr: {job_root / ('yf7lf9zo_stage1_adv50_' + job_id + '.err')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
