#!/usr/bin/env python3
"""Relaunch a mixed FFHQ k2-a16384 run from a clean output root."""

from __future__ import annotations

import argparse
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
RUN_ROOT_PARENT = Path("/scratch/xl598/runs")
DATA_DIR = Path("/scratch/xl598/Projects/data/ffhq")
CKPT = Path(
    "/scratch/xl598/runs/checkpoint_registry/"
    "ffhq_stage1_sweep_lpips005_k2-a16384/"
    "b5wjt19n/last-at-cancel-20260630_0205.ckpt"
)
PYDEPS = Path("/scratch/xl598/.pydeps/laser_src_py311")
VGG16_WEIGHTS = Path("/scratch/xl598/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
IMAGE = "docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
LABEL = "k2-a16384"


def q(value: object) -> str:
    return shlex.quote(str(value))


def hydra_string(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def bash_array(items: list[str]) -> str:
    return "".join(f"  {q(item)}\n" for item in items)


def script_body(raw: str) -> str:
    lines = raw.lstrip("\n").splitlines()
    cleaned = []
    for line in lines:
        if line.startswith("            "):
            line = line[12:]
        cleaned.append(line.rstrip())
    return "\n".join(cleaned) + "\n"


def snapshot_repo(repo: Path, snapshot: Path) -> None:
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot}")
    excludes = [
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        "__pycache__",
        ".pydeps",
        ".tmp",
        "submission_snapshots",
    ]
    cmd = ["rsync", "-a", "--delete"]
    for item in excludes:
        cmd.extend(["--exclude", item])
    cmd.extend([str(repo) + "/", str(snapshot) + "/"])
    subprocess.run(cmd, check=True)


def stage1_overrides(
    output_dir: Path,
    ckpt_path: Path,
    *,
    run_id: str,
    label: str,
    nodes: int,
    gpus_per_node: int,
) -> list[str]:
    return [
        "stage1",
        "seed=42",
        f"output_dir={output_dir}",
        "model=laser_image_nonpatch_d5",
        "data=ffhq",
        f"data.data_dir={DATA_DIR}",
        "data.batch_size=8",
        "data.eval_batch_size=8",
        "data.num_workers=8",
        "data.pin_memory=true",
        "data.prefetch_factor=6",
        "data.image_size=256",
        "data.train_crop_size=null",
        "data.augment=true",
        "train.accelerator=gpu",
        f"train.devices={int(gpus_per_node)}",
        f"train.num_nodes={int(nodes)}",
        "train.strategy=ddp",
        "train.precision=bf16-mixed",
        "train.max_epochs=150",
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
        "train.warmup_steps=9845",
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
        "model.data_init_from_first_batch=true",
        "model.recon_mse_weight=1.0",
        "model.recon_l1_weight=0.0",
        "model.recon_edge_weight=0.0",
        "model.perceptual_weight=0.05",
        "model.perceptual_start_step=5000",
        "model.perceptual_warmup_steps=10000",
        "model.adversarial_weight=0.0",
        "model.adversarial_start_step=1000000000",
        "model.adversarial_warmup_steps=0",
        "model.disc_start_step=1000000000",
        "model.compute_fid=true",
        "model.log_images_every_n_steps=250",
        "model.diag_log_interval=250",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=4096",
        "model.num_embeddings=16384",
        "model.sparsity_level=2",
        "wandb.project=laser",
        f"wandb.name=ffhq-stage1-{label}-ffhq-laser-paper-stage1-sweep-20260628_022906",
        "wandb.group=ffhq-laser-paper-stage1-sweep-20260628_022906",
        f"wandb.tags=[stage1,ffhq,laser,dictionary,no-adversarial,sweep,clean-relaunch,{run_id}]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
        "checkpoint.upload_to_wandb=false",
        f"ckpt_path={hydra_string(ckpt_path)}",
    ]


def write_job_files(args: argparse.Namespace, *, stamp: str, snapshot: Path, job_root: Path) -> tuple[Path, Path]:
    output_dir = job_root / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_script = job_root / f"run_{args.run_id}_clean_relaunch.sh"
    sbatch_script = job_root / f"sbatch_{args.run_id}_clean_relaunch.sh"
    log_base = job_root / f"{args.run_id}_clean_relaunch"
    stage1_args = bash_array(
        stage1_overrides(
            output_dir,
            Path(args.ckpt_path),
            run_id=args.run_id,
            label=args.label,
            nodes=args.nodes,
            gpus_per_node=args.gpus_per_node,
        )
    )
    srun_prefix = f"srun --ntasks={args.nodes} --ntasks-per-node=1 "

    run_script.write_text(
        script_body(
            f"""\
            #!/bin/bash
            set -euo pipefail

            export PYTHONUSERBASE={q(args.pydeps)}
            export PATH="$PYTHONUSERBASE/bin:$PATH"
            export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{q(snapshot)}${{PYTHONPATH:+:$PYTHONPATH}}"
            export WANDB_MODE="${{WANDB_MODE:-online}}"
            export WANDB_RUN_ID="{args.run_id}"
            export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
            export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-{output_dir}/wandb/data}}"
            export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{output_dir}/wandb/cache}}"
            export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-{output_dir}/wandb/artifacts}}"
            export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{USER}/.config/wandb}}"
            export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{USER}/.cache}}"
            export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
            export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
            export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
            export HYDRA_FULL_ERROR=1
            export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
            export PYTHONUNBUFFERED=1
            export TMPDIR="/tmp/laser_{args.run_id}_clean_${{SLURM_JOB_ID:-$$}}"
            export TEMP="$TMPDIR"
            export TMP="$TMPDIR"
            if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" && -f {q(VGG16_WEIGHTS)} ]]; then
              export LASER_VGG16_WEIGHTS={q(VGG16_WEIGHTS)}
            fi

            mkdir -p "$PYTHONUSERBASE" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \\
              "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" {q(output_dir / "wandb")}

            PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
            if [[ -z "$PYTHON_BIN" ]]; then
              echo "python3/python not found" >&2
              exit 127
            fi
            if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
              echo "ERROR: $PYTHON_BIN is too old; LASER requires Python >= 3.10." >&2
              exit 2
            fi

            echo "=== GPU inventory ==="
            nvidia-smi
            echo ""

            if command -v flock >/dev/null 2>&1; then
              (
                flock 9
                "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
              ) 9>"$PYTHONUSERBASE/.install.lock"
            else
              "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
            fi

            if [[ ! -f {q(args.ckpt_path)} ]]; then
              echo "Missing resume checkpoint: {args.ckpt_path}" >&2
              exit 1
            fi

            cd {q(snapshot)}
            STAGE1_ARGS=(
            {stage1_args}
            )

            echo "=== Clean relaunch W&B run {args.run_id} ({args.label}) ==="
            echo "resume_checkpoint={args.ckpt_path}"
            echo "output_dir={output_dir}"
            echo "hardware_plan=partition:{args.partition} constraint:{args.constraint or 'none'} nodes:{args.nodes} gpus_per_node:{args.gpus_per_node} total_gpus:{args.nodes * args.gpus_per_node}"
            printf 'Launching:'
            printf ' %q' "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
            printf '\\n'
            "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
            """
        ),
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    constraint_line = f"#SBATCH --constraint={args.constraint}\n" if args.constraint else ""
    sbatch_script.write_text(
        script_body(
            f"""\
            #!/bin/bash
            #SBATCH --job-name=ffhq-{args.run_id}-clean
            #SBATCH --partition={args.partition}
            #SBATCH --nodes={args.nodes}
            #SBATCH --ntasks={args.nodes}
            #SBATCH --ntasks-per-node=1
            #SBATCH --gres=gpu:{args.gpus_per_node}
            #SBATCH --cpus-per-task={args.cpus_per_task}
            #SBATCH --mem={args.mem_mb}
            #SBATCH --time={args.time_limit}
            {constraint_line}#SBATCH --output={log_base}_%j.out
            #SBATCH --error={log_base}_%j.err

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
              {srun_prefix}"$CONTAINER_BIN" exec --nv \\
                --bind /cache/home/{USER} \\
                --bind {q(snapshot)} \\
                --bind /scratch/{USER} \\
                --bind {q(DATA_DIR)} \\
                --bind {q(job_root)} \\
                --bind {q(Path(args.ckpt_path).parent)} \\
                --bind /dev/shm \\
                "$IMAGE" \\
                bash {q(run_script)}
            else
              echo "Warning: singularity/apptainer not found; running bare" >&2
              {srun_prefix}bash {q(run_script)}
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
    parser.add_argument("--run-root-base", default="")
    parser.add_argument("--run-id", default="b5wjt19n")
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--ckpt-path", default=str(CKPT))
    parser.add_argument("--pydeps", default=str(PYDEPS))
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="ampere")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=120000)
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = Path(args.snapshot_root).expanduser().resolve() / f"laser_{args.run_id}_clean_relaunch_{stamp}"
    if str(args.run_root_base).strip():
        run_root_base = Path(args.run_root_base).expanduser().resolve()
    else:
        run_root_base = RUN_ROOT_PARENT / f"ffhq_stage1_sweep_lpips005_clean_relaunch_{args.run_id}"
    job_root = run_root_base / f"relaunch_{stamp}"
    job_root.mkdir(parents=True, exist_ok=True)
    snapshot_repo(repo, snapshot)
    _, sbatch_script = write_job_files(args, stamp=stamp, snapshot=snapshot, job_root=job_root)
    if args.dry_run:
        print(f"Snapshot: {snapshot}")
        print(f"Job root: {job_root}")
        print(f"SBATCH: {sbatch_script}")
        return 0
    result = subprocess.run(["sbatch", str(sbatch_script)], check=True, text=True, capture_output=True)
    print(result.stdout.strip())
    print(f"Snapshot: {snapshot}")
    print(f"Job root: {job_root}")
    print(f"SBATCH: {sbatch_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
