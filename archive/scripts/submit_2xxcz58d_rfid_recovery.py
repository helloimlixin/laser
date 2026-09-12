#!/usr/bin/env python3
"""Submit post-fit rFID recovery for W&B run 2xxcz58d.

The original training reached max_epochs and saved final.ckpt, then failed during
the post-fit rFID subprocess because the frozen job snapshot could not find local
VGG16 feature weights. This submitter recomputes only rFID from final.ckpt and
logs the recovered scalars back to the same W&B run.
"""

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
RUN_ID = "2xxcz58d"
WANDB_ENTITY = "helloimlixin-rutgers"
WANDB_PROJECT = "laser"
WANDB_NAME = "celebahq-stage1-k2-a8192-celebahq-full-stage1-sweep-20260625_023220"
WANDB_GROUP = "celebahq-full-stage1-sweep-20260625_023220"

REPO = Path("/scratch/xl598/Projects/laser")
SNAPSHOT_ROOT = Path("/scratch/xl598/submission_snapshots")
RUN_ROOT_BASE = Path("/scratch/xl598/runs/celebahq_stage1_rfid_recovery_2xxcz58d")
SOURCE_ROOT = Path(
    "/scratch/xl598/runs/celebahq_stage1_sweep/"
    "celebahq-full-stage1-sweep-20260625_023220/k2-a8192/stage1"
)
CKPT = SOURCE_ROOT / "checkpoints/run_20260625_095259/laser/final.ckpt"
CONFIG = SOURCE_ROOT / "2026-06-25_09-52-59/.hydra/config.yaml"
DATA_DIR = Path("/scratch/xl598/Projects/data/celeba_hq")
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


def write_metric_logger(args, log_script, rfid_log):
    log_script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import re
            from pathlib import Path

            import wandb

            log_path = Path({rfid_log!r})
            if not log_path.is_file():
                raise SystemExit("Missing rFID log: {{}}".format(log_path))

            values = {{}}
            patterns = {{
                "processed_samples": re.compile(r"Processed samples:\\s*([0-9]+)"),
                "rfid": re.compile(r"\\brFID:\\s*([-+0-9.eE]+)"),
                "real_split_fid": re.compile(r"\\breal_split_FID:\\s*([-+0-9.eE]+)"),
                "adjusted_rfid": re.compile(r"\\badjusted_rFID:\\s*([-+0-9.eE]+)"),
            }}
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                for key, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        raw = match.group(1)
                        values[key] = int(raw) if key == "processed_samples" else float(raw)

            if "rfid" not in values:
                raise SystemExit("No rFID value found in {{}}".format(log_path))

            run = wandb.init(
                entity=os.environ.get("WANDB_ENTITY") or {entity!r},
                project=os.environ.get("WANDB_PROJECT") or {project!r},
                id=os.environ.get("WANDB_RUN_ID") or {run_id!r},
                resume=os.environ.get("WANDB_RESUME") or "allow",
                name={name!r},
                group={group!r},
                tags=[
                    "stage1",
                    "celebahq",
                    "laser",
                    "dictionary",
                    "no-adversarial",
                    "sweep",
                    "rfid-recovery",
                    "from-2xxcz58d",
                ],
            )

            payload = {{
                "val/rfid_full": values["rfid"],
                "recovery/rfid_recomputed": 1,
                "recovery/rfid_log_path": str(log_path),
            }}
            if "processed_samples" in values:
                payload["val/rfid_processed_samples"] = values["processed_samples"]
            if "real_split_fid" in values:
                payload["val/rfid_real_split"] = values["real_split_fid"]
            if "adjusted_rfid" in values:
                payload["val/rfid_adjusted"] = values["adjusted_rfid"]

            wandb.log(payload)
            for key, value in payload.items():
                run.summary[key] = value
            run.finish()
            print("Logged recovered rFID metrics to W&B run {{}}: {{}}".format(run.id, payload))
            """.format(
                rfid_log=str(rfid_log),
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                run_id=RUN_ID,
                name=WANDB_NAME,
                group=WANDB_GROUP,
            )
        ),
        encoding="utf-8",
    )
    os.chmod(str(log_script), 0o755)


def write_job_files(args, snapshot, job_root):
    output_dir = job_root / "rfid_recovery"
    wandb_dir = output_dir / "wandb"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_script = job_root / "run_2xxcz58d_rfid_recovery.sh"
    log_script = job_root / "log_2xxcz58d_rfid_to_wandb.py"
    sbatch_script = job_root / "sbatch_2xxcz58d_rfid_recovery.sh"
    log_base = job_root / "2xxcz58d_rfid_recovery"
    rfid_log = Path(args.ckpt_path).parent / "rfid.log"

    write_metric_logger(args, log_script, rfid_log)

    run_script.write_text(
        script_body(
            """\
            #!/bin/bash
            set -euo pipefail

            export PYTHONUSERBASE={pydeps}
            export PATH="$PYTHONUSERBASE/bin:$PATH"
            export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot}${{PYTHONPATH:+:$PYTHONPATH}}"
            export WANDB_ENTITY="${{WANDB_ENTITY:-{entity}}}"
            export WANDB_PROJECT="${{WANDB_PROJECT:-{project}}}"
            export WANDB_MODE="${{WANDB_MODE:-online}}"
            export WANDB_RUN_ID="{run_id}"
            export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
            export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-1}}"
            export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-{wandb_dir}/data}}"
            export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{wandb_dir}/cache}}"
            export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-{wandb_dir}/artifacts}}"
            export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{user}/.config/wandb}}"
            export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{user}/.cache}}"
            export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
            export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
            export MPLCONFIGDIR="${{MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}}"
            export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
            export HYDRA_FULL_ERROR=1
            export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
            export PYTHONUNBUFFERED=1
            export TMPDIR="/tmp/laser_2xxcz58d_rfid_${{SLURM_JOB_ID:-$$}}"
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

            if command -v flock >/dev/null 2>&1; then
              (
                flock 9
                "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
              ) 9>"$PYTHONUSERBASE/.install.lock"
            else
              "$PYTHON_BIN" -m pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
            fi

            CKPT={ckpt}
            CONFIG={config}
            DATA_DIR={data_dir}
            RFID_LOG={rfid_log}
            if [[ ! -f "$CKPT" ]]; then
              echo "Missing checkpoint: $CKPT" >&2
              exit 1
            fi
            if [[ ! -f "$CONFIG" ]]; then
              echo "Missing config: $CONFIG" >&2
              exit 1
            fi
            if [[ ! -d "$DATA_DIR" ]]; then
              echo "Missing data dir: $DATA_DIR" >&2
              exit 1
            fi
            if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" || ! -f "$LASER_VGG16_WEIGHTS" ]]; then
              echo "Missing LASER_VGG16_WEIGHTS: ${{LASER_VGG16_WEIGHTS:-<unset>}}" >&2
              exit 1
            fi

            echo "=== GPU inventory ==="
            nvidia-smi
            echo ""
            echo "=== Recover post-fit rFID for W&B run {run_id} ==="
            echo "checkpoint=$CKPT"
            echo "config=$CONFIG"
            echo "data_dir=$DATA_DIR"
            echo "rfid_log=$RFID_LOG"
            echo "vgg16_weights=$LASER_VGG16_WEIGHTS"
            echo "snapshot={snapshot}"
            echo "job_root={job_root}"

            cd {snapshot}
            "$PYTHON_BIN" scripts/tools/compute_rfid.py \\
              --ckpt "$CKPT" \\
              --config "$CONFIG" \\
              --dataset celebahq \\
              --data-dir "$DATA_DIR" \\
              --image-size 256 \\
              --split val \\
              --batch-size {rfid_batch_size} \\
              --num-workers {rfid_num_workers} \\
              --max-samples 0 \\
              --device auto \\
              --feature 2048 \\
              --mean 0.5 0.5 0.5 \\
              --std 0.5 0.5 0.5

            "$PYTHON_BIN" {log_script}
            """.format(
                pydeps=q(args.pydeps),
                snapshot=q(snapshot),
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                run_id=RUN_ID,
                wandb_dir=q(wandb_dir),
                user=USER,
                vgg16=q(args.vgg16_weights),
                ckpt=q(args.ckpt_path),
                config=q(args.config_path),
                data_dir=q(args.data_dir),
                rfid_log=q(rfid_log),
                job_root=q(job_root),
                rfid_batch_size=int(args.rfid_batch_size),
                rfid_num_workers=int(args.rfid_num_workers),
                log_script=q(log_script),
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
            #SBATCH --job-name=chq-2xx-rfid
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
    return run_script, log_script, sbatch_script, log_base


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--run-root-base", default=str(RUN_ROOT_BASE))
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--ckpt-path", default=str(CKPT))
    parser.add_argument("--config-path", default=str(CONFIG))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--pydeps", default=str(PYDEPS))
    parser.add_argument("--vgg16-weights", default=str(VGG16_WEIGHTS))
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=12)
    parser.add_argument("--mem-mb", type=int, default=120000)
    parser.add_argument("--time-limit", default="12:00:00")
    parser.add_argument("--rfid-batch-size", type=int, default=32)
    parser.add_argument("--rfid-num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate(args):
    checks = (
        (Path(args.repo), "repo", "is_dir"),
        (Path(args.source_root), "source run root", "is_dir"),
        (Path(args.ckpt_path), "final checkpoint", "is_file"),
        (Path(args.config_path), "Hydra config", "is_file"),
        (Path(args.data_dir), "CelebA-HQ data dir", "is_dir"),
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
    if int(args.rfid_batch_size) <= 0:
        raise SystemExit("--rfid-batch-size must be positive")
    if int(args.rfid_num_workers) < 0:
        raise SystemExit("--rfid-num-workers must be non-negative")


def main():
    args = parse_args()
    args.repo = str(Path(args.repo).expanduser().resolve())
    args.snapshot_root = str(Path(args.snapshot_root).expanduser().resolve())
    args.run_root_base = str(Path(args.run_root_base).expanduser().resolve())
    args.source_root = str(Path(args.source_root).expanduser().resolve())
    args.ckpt_path = str(Path(args.ckpt_path).expanduser().resolve())
    args.config_path = str(Path(args.config_path).expanduser().resolve())
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    args.pydeps = str(Path(args.pydeps).expanduser().resolve())
    args.vgg16_weights = str(Path(args.vgg16_weights).expanduser().resolve())
    validate(args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = Path(args.snapshot_root) / "laser_2xxcz58d_rfid_recovery_{}".format(stamp)
    job_root = Path(args.run_root_base) / "rfid_recovery_{}".format(stamp)
    job_root.mkdir(parents=True, exist_ok=True)
    snapshot_repo(Path(args.repo), snapshot)
    _, _, sbatch_script, log_base = write_job_files(args, snapshot, job_root)

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
