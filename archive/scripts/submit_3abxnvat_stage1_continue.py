#!/usr/bin/env python3
"""Submit a fixed CelebA-HQ stage-1 continuation for W&B run 3abxnvat."""

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
RUN_ID = "3abxnvat"
LABEL = "k2-a4096"
WANDB_PROJECT = "laser"
WANDB_NAME = "celebahq-stage1-k2-a4096-celebahq-full-stage1-sweep-20260625_032139"
WANDB_GROUP = "celebahq-full-stage1-sweep-20260625_032139"

REPO = Path("/scratch/xl598/Projects/laser")
SNAPSHOT_ROOT = Path("/scratch/xl598/submission_snapshots")
RUN_ROOT_BASE = Path("/scratch/xl598/runs/celebahq_stage1_continue_3abxnvat")
DATA_DIR = Path("/scratch/xl598/Projects/data/celeba_hq")
CKPT = Path(
    "/scratch/xl598/runs/celebahq_stage1_sweep/"
    "celebahq-full-stage1-sweep-20260625_032139/k2-a4096/stage1/"
    "checkpoints/run_20260625_033706/laser/last.ckpt"
)
PYDEPS = Path("/scratch/xl598/.pydeps/laser_src_py311")
VGG16_WEIGHTS = Path("/scratch/xl598/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
DEFAULT_SIF = Path(
    "/cache/home/xl598/.apptainer/cache/oci-tmp/"
    "ac7c098a81512e719afa5d2d497f812d7db3498f340a4b819c69cb7b3b257126/"
    "pytorch_2.4.1-cuda12.1-cudnn9-runtime.sif"
)
DEFAULT_IMAGE = (
    str(DEFAULT_SIF)
    if DEFAULT_SIF.is_file()
    else "docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
)

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
    "outputs",
    "runs",
    "wandb",
    "submission_snapshots",
    "source_snapshot_*",
    "pre_variation_snapshot_*",
    "*.out",
    "*.err",
    "*.pyc",
    "*.pyo",
    "*.swp",
)


def q(value):
    return shlex.quote(str(value))


def bash_array(items):
    return "".join("  {}\n".format(q(item)) for item in items)


def script_body(raw):
    lines = textwrap.dedent(raw).lstrip("\n").splitlines()
    cleaned = []
    for line in lines:
        if line.startswith("            "):
            line = line[12:]
        cleaned.append(line.rstrip())
    return "\n".join(cleaned) + "\n"


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


def snapshot_repo(repo, snapshot):
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        raise FileExistsError("snapshot already exists: {}".format(snapshot))
    if shutil.which("rsync"):
        cmd = ["rsync", "-a"]
        cmd.extend(
            [
                "--include=/configs/",
                "--include=/configs/wandb/",
                "--include=/configs/wandb/***",
            ]
        )
        for pattern in EXCLUDES:
            cmd.append("--exclude={}".format(pattern))
        cmd.extend(["{}/".format(repo), "{}/".format(snapshot)])
        subprocess.run(cmd, check=True)
    else:
        shutil.copytree(str(repo), str(snapshot), ignore=snapshot_ignore(repo))


def stage1_overrides(args, output_dir):
    return [
        "stage1",
        "seed=42",
        "output_dir={}".format(output_dir),
        "ckpt_path=__RESUME_CKPT__",
        "model=laser_image_nonpatch_d5",
        "data=celebahq",
        "data.data_dir={}".format(args.data_dir),
        "data.batch_size=8",
        "data.eval_batch_size=8",
        "data.num_workers=4",
        "data.pin_memory=true",
        "data.prefetch_factor=4",
        "data.image_size=256",
        "data.train_crop_size=null",
        "data.augment=true",
        "train.accelerator=gpu",
        "train.devices={}".format(int(args.gpus)),
        "train.num_nodes=1",
        "train.strategy=ddp",
        "train.precision=bf16-mixed",
        "train.max_epochs={}".format(int(args.target_max_epochs)),
        "train.max_steps=-1",
        "train.limit_train_batches=1.0",
        "train.limit_val_batches=1.0",
        "train.limit_test_batches=0",
        "train.val_check_interval=1.0",
        "train.run_test_after_fit=false",
        "train.compute_rfid_after_fit=true",
        "train.rfid_split=val",
        "train.rfid_batch_size=32",
        "train.rfid_num_workers=8",
        "train.rfid_max_samples=0",
        "train.rfid_device=auto",
        "train.rfid_feature=2048",
        "train.learning_rate=4.0e-5",
        "train.beta=0.5",
        "train.beta2=0.9",
        "train.warmup_steps=4220",
        "train.min_lr_ratio=1.0",
        "train.accumulate_grad_batches=2",
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
        "model.perceptual_weight=1.0",
        "model.perceptual_start_step=0",
        "model.perceptual_warmup_steps=0",
        "model.adversarial_weight=0.0",
        "model.adversarial_start_step=1000000000",
        "model.adversarial_warmup_steps=0",
        "model.disc_start_step=1000000000",
        "model.compute_fid=true",
        "model.log_images_every_n_steps=0",
        "model.diag_log_interval=250",
        "model.enable_val_latent_visuals=false",
        "model.codebook_visual_max_vectors=4096",
        "model.num_embeddings=4096",
        "model.sparsity_level=2",
        "wandb.project={}".format(WANDB_PROJECT),
        "wandb.name={}".format(WANDB_NAME),
        "wandb.group={}".format(WANDB_GROUP),
        "wandb.tags=[stage1,celebahq,laser,dictionary,no-adversarial,sweep,continuation,fix-relaunch,from-3abxnvat,no-media]",
        "wandb.append_timestamp=false",
        "wandb.save_dir={}".format(output_dir / "wandb"),
        "checkpoint.save_top_k=1",
        "checkpoint.save_last=true",
        "checkpoint.upload_to_wandb=false",
    ]


def write_job_files(args, snapshot, job_root):
    output_dir = job_root / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_script = job_root / "run_3abxnvat_stage1_continue.sh"
    sbatch_script = job_root / "sbatch_3abxnvat_stage1_continue.sh"
    log_base = job_root / "3abxnvat_stage1_continue"
    stage_args = bash_array(stage1_overrides(args, output_dir))

    run_script.write_text(
        script_body(
            """\
            #!/bin/bash
            set -euo pipefail

            export PYTHONUSERBASE={pydeps}
            export PATH="$PYTHONUSERBASE/bin:$PATH"
            export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot}${{PYTHONPATH:+:$PYTHONPATH}}"
            export WANDB_MODE="${{WANDB_MODE:-online}}"
            export WANDB_RUN_ID="{run_id}"
            export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
            export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-1}}"
            export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-{output_dir}/wandb/data}}"
            export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{output_dir}/wandb/cache}}"
            export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-{output_dir}/wandb/artifacts}}"
            export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{user}/.config/wandb}}"
            export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{user}/.cache}}"
            export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
            export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
            export MPLCONFIGDIR="${{MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}}"
            export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
            export HYDRA_FULL_ERROR=1
            export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
            export PYTHONUNBUFFERED=1
            export TMPDIR="/tmp/laser_3abxnvat_continue_${{SLURM_JOB_ID:-$$}}"
            export TEMP="$TMPDIR"
            export TMP="$TMPDIR"
            if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" && -f {vgg16} ]]; then
              export LASER_VGG16_WEIGHTS={vgg16}
            fi

            mkdir -p "$PYTHONUSERBASE" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \\
              "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR" {wandb_dir}

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

            INITIAL_CKPT={ckpt}
            RESUME_CKPT="$INITIAL_CKPT"
            OWN_LATEST="$(find {checkpoint_root} -type f \\( -name '{run_id}-last.ckpt' -o -name 'last.ckpt' \\) -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
            if [[ -n "${{OWN_LATEST:-}}" && -f "$OWN_LATEST" ]]; then
              RESUME_CKPT="$OWN_LATEST"
            fi
            if [[ ! -f "$RESUME_CKPT" ]]; then
              echo "Missing resume checkpoint: $RESUME_CKPT" >&2
              exit 1
            fi
            if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" || ! -f "$LASER_VGG16_WEIGHTS" ]]; then
              echo "Missing LASER_VGG16_WEIGHTS: ${{LASER_VGG16_WEIGHTS:-<unset>}}" >&2
              exit 1
            fi

            cd {snapshot}

            STAGE1_ARGS=(
            {stage_args}
            )
            for i in "${{!STAGE1_ARGS[@]}}"; do
              if [[ "${{STAGE1_ARGS[$i]}}" == "ckpt_path=__RESUME_CKPT__" ]]; then
                STAGE1_ARGS[$i]="ckpt_path=$RESUME_CKPT"
              fi
            done

            echo "=== Continue W&B run {run_id} from local checkpoint ==="
            echo "initial_checkpoint={ckpt}"
            echo "auto_resume_checkpoint=$RESUME_CKPT"
            echo "source_checkpoint_note=latest valid local checkpoint is epoch 21; original W&B output reached later epochs but checkpoint writes stopped after quota errors"
            echo "target_max_epochs={target_max_epochs}"
            echo "output_dir={output_dir}"
            echo "wandb_run_id={run_id}"
            echo "media_disabled=$LASER_DISABLE_WANDB_MEDIA"
            echo "vgg16_weights=$LASER_VGG16_WEIGHTS"
            printf 'Launching:'
            printf ' %q' "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
            printf '\\n'
            "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
            """.format(
                pydeps=q(args.pydeps),
                snapshot=q(snapshot),
                run_id=RUN_ID,
                output_dir=q(output_dir),
                user=USER,
                vgg16=q(args.vgg16_weights),
                wandb_dir=q(output_dir / "wandb"),
                ckpt=q(args.ckpt_path),
                checkpoint_root=q(output_dir / "checkpoints"),
                stage_args=stage_args,
                target_max_epochs=int(args.target_max_epochs),
            )
        ),
        encoding="utf-8",
    )
    os.chmod(str(run_script), 0o755)

    constraint_line = "#SBATCH --constraint={}\n".format(args.constraint) if str(args.constraint or "").strip() else ""
    sbatch_script.write_text(
        script_body(
            """\
            #!/bin/bash
            #SBATCH --job-name=chq-3abx-cont
            #SBATCH --partition={partition}
            #SBATCH --nodes=1
            #SBATCH --ntasks=1
            #SBATCH --ntasks-per-node=1
            #SBATCH --gres=gpu:{gpus}
            #SBATCH --cpus-per-task={cpus}
            #SBATCH --mem={mem}
            #SBATCH --time={time_limit}
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

            export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-/cache/home/{user}/.apptainer/cache}}"
            export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}}"
            export APPTAINER_TMPDIR="${{APPTAINER_TMPDIR:-/scratch/{user}/.apptainer/tmp_${{SLURM_JOB_ID:-manual}}}}"
            export SINGULARITY_TMPDIR="${{SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}}"
            mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

            IMAGE="${{IMAGE:-{image}}}"
            echo "container_bin=$CONTAINER_BIN"
            echo "image=$IMAGE"
            echo "snapshot={snapshot}"
            echo "job_root={job_root}"

            if [[ -n "$CONTAINER_BIN" ]]; then
              srun "$CONTAINER_BIN" exec --nv \\
                --bind /cache/home/{user} \\
                --bind {snapshot} \\
                --bind /scratch/{user} \\
                --bind {data_dir} \\
                --bind {job_root} \\
                --bind {ckpt_dir} \\
                --bind {vgg16_dir} \\
                --bind /dev/shm \\
                "$IMAGE" \\
                bash {run_script}
            else
              echo "Warning: singularity/apptainer not found; running bare" >&2
              srun bash {run_script}
            fi
            """.format(
                partition=args.partition,
                gpus=int(args.gpus),
                cpus=int(args.cpus_per_task),
                mem=int(args.mem_mb),
                time_limit=args.time_limit,
                constraint_line=constraint_line,
                log_base=q(log_base),
                user=USER,
                image=q(args.image),
                snapshot=q(snapshot),
                job_root=q(job_root),
                data_dir=q(args.data_dir),
                ckpt_dir=q(Path(args.ckpt_path).parent),
                vgg16_dir=q(Path(args.vgg16_weights).parent),
                run_script=q(run_script),
            )
        ),
        encoding="utf-8",
    )
    os.chmod(str(sbatch_script), 0o755)
    return run_script, sbatch_script, log_base


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--run-root-base", default=str(RUN_ROOT_BASE))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--ckpt-path", default=str(CKPT))
    parser.add_argument("--pydeps", default=str(PYDEPS))
    parser.add_argument("--vgg16-weights", default=str(VGG16_WEIGHTS))
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem-mb", type=int, default=220000)
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--target-max-epochs", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate(args):
    checks = (
        (Path(args.repo), "repo", "is_dir"),
        (Path(args.data_dir), "CelebA-HQ data dir", "is_dir"),
        (Path(args.ckpt_path), "resume checkpoint", "is_file"),
        (Path(args.pydeps), "pydeps", "exists"),
        (Path(args.vgg16_weights), "VGG16 weights", "is_file"),
    )
    for path, label, predicate in checks:
        if predicate == "is_dir" and not path.is_dir():
            raise SystemExit("Missing {}: {}".format(label, path))
        if predicate == "is_file" and not path.is_file():
            raise SystemExit("Missing {}: {}".format(label, path))
        if predicate == "exists" and not path.exists():
            raise SystemExit("Missing {}: {}".format(label, path))
    if int(args.gpus) <= 0:
        raise SystemExit("--gpus must be positive")
    if int(args.target_max_epochs) <= 21:
        raise SystemExit("--target-max-epochs must exceed the source checkpoint epoch 21")


def main():
    args = parse_args()
    args.repo = str(Path(args.repo).expanduser().resolve())
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    args.ckpt_path = str(Path(args.ckpt_path).expanduser().resolve())
    args.pydeps = str(Path(args.pydeps).expanduser().resolve())
    args.vgg16_weights = str(Path(args.vgg16_weights).expanduser().resolve())
    validate(args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = Path(args.snapshot_root).expanduser().resolve() / "laser_3abxnvat_stage1_continue_{}".format(stamp)
    job_root = Path(args.run_root_base).expanduser().resolve() / "continue_{}".format(stamp)
    job_root.mkdir(parents=True, exist_ok=True)
    snapshot_repo(Path(args.repo), snapshot)
    _, sbatch_script, log_base = write_job_files(args, snapshot, job_root)

    if args.dry_run:
        print("Snapshot: {}".format(snapshot))
        print("Job root: {}".format(job_root))
        print("SBATCH: {}".format(sbatch_script))
        print("stdout: {}_%j.out".format(log_base))
        print("stderr: {}_%j.err".format(log_base))
        return 0

    result = subprocess.run(
        ["sbatch", str(sbatch_script)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    print("Snapshot: {}".format(snapshot))
    print("Job root: {}".format(job_root))
    print("SBATCH: {}".format(sbatch_script))
    print("stdout: {}_<jobid>.out".format(log_base))
    print("stderr: {}_<jobid>.err".format(log_base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
