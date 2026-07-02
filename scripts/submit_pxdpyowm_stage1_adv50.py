#!/usr/bin/env python3
"""Submit a 50-epoch FFHQ stage-1 adversarial run from pxdpyowm."""

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple


USER = os.environ.get("USER", "xl598")
SOURCE_RUN_ID = "pxdpyowm"
ADV_RUN_ID = "pxdadv50"
WANDB_PROJECT = "laser"
WANDB_GROUP = "ffhq-laser-paper"
WANDB_NAME = "ffhq-stage1-adv50-from-pxdpyowm"

REPO = Path("/scratch/xl598/Projects/laser")
SNAPSHOT_ROOT = Path("/scratch/xl598/submission_snapshots")
RUN_ROOT_BASE = Path("/scratch/xl598/runs/ffhq_pxdpyowm_stage1_adv50")
DATA_DIR = Path("/scratch/xl598/Projects/data/ffhq")
INIT_CKPT = Path(
    "/scratch/xl598/runs/ffhq_laser_paper_stage1_continue_pxdpyowm/"
    "continue15_20260629_144546/stage1/checkpoints/run_slurm57714706/"
    "laser/pxdpyowm-final.ckpt"
)
PYDEPS = Path("/scratch/xl598/.pydeps/laser_src_py311")
VGG16_WEIGHTS = Path("/scratch/xl598/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
DEFAULT_SIF = Path(
    "/cache/home/xl598/.apptainer/cache/oci-tmp/"
    "ac7c098a81512e719afa5d2d497f812d7db3498f340a4b819c69cb7b3b257126/"
    "pytorch_2.4.1-cuda12.1-cudnn9-runtime.sif"
)
DEFAULT_IMAGE = str(DEFAULT_SIF) if DEFAULT_SIF.is_file() else "docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"

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
    "outputs",
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


def bash_array(items: List[str]) -> str:
    return "\n".join("  " + q(item) for item in items)


def snapshot_ignore(repo: Path):
    repo = repo.resolve()

    def ignore(current_dir: str, names: List[str]) -> Set[str]:
        rel_dir = Path(current_dir).resolve().relative_to(repo)
        ignored: Set[str] = set()
        for name in names:
            rel_path = (rel_dir / name) if rel_dir != Path(".") else Path(name)
            if len(rel_path.parts) >= 2 and rel_path.parts[0] == "configs" and rel_path.parts[1] == "wandb":
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDES):
                ignored.add(name)
        return ignored

    return ignore


def snapshot_repo(repo: Path, snapshot_root: Path, stem: str) -> Path:
    if not repo.is_dir():
        raise FileNotFoundError(f"repo not found: {repo}")
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_root / stem
    if snapshot.exists():
        raise FileExistsError(f"snapshot already exists: {snapshot}")
    if shutil.which("rsync"):
        cmd = ["rsync", "-a"]
        cmd.extend(
            [
                "--include=/configs/",
                "--include=/configs/wandb/",
                "--include=/configs/wandb/***",
            ]
        )
        cmd.extend(f"--exclude={pattern}" for pattern in EXCLUDES)
        cmd.extend([f"{repo}/", str(snapshot)])
        subprocess.run(cmd, check=True)
    else:
        shutil.copytree(str(repo), str(snapshot), ignore=snapshot_ignore(repo))
    return snapshot


def stage1_adv_overrides(args: argparse.Namespace, output_dir: Path) -> List[str]:
    effective_batch = int(args.gpus) * int(args.batch_size) * int(args.accumulate_grad_batches)
    return [
        "stage1",
        "seed=42",
        f"output_dir={output_dir}",
        f"init_ckpt_path={args.init_ckpt}",
        "model=laser_image_nonpatch_d5",
        "data=ffhq",
        "data.dataset=ffhq",
        f"data.data_dir={args.data_dir}",
        "data.image_size=256",
        "data.train_crop_size=null",
        f"data.batch_size={int(args.batch_size)}",
        f"data.eval_batch_size={int(args.batch_size)}",
        f"data.num_workers={int(args.num_workers)}",
        "data.pin_memory=true",
        "data.prefetch_factor=6",
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
        "train.log_every_n_steps=20",
        "train.run_test_after_fit=false",
        "train.compute_rfid_after_fit=false",
        "train.rfid_split=val",
        "train.rfid_batch_size=32",
        "train.rfid_num_workers=8",
        "train.rfid_max_samples=0",
        "train.rfid_device=auto",
        "train.rfid_feature=2048",
        f"train.learning_rate={args.learning_rate}",
        "train.beta=0.5",
        "train.beta2=0.9",
        "train.warmup_steps=0",
        "train.min_lr_ratio=1.0",
        "train.gradient_clip_val=1.0",
        f"train.accumulate_grad_batches={int(args.accumulate_grad_batches)}",
        "train.deterministic=false",
        "model.type=laser",
        "model.backbone=ddpm",
        "model.num_downsamples=5",
        "model.channel_multipliers=[1,1,2,2,4,4]",
        "model.backbone_latent_channels=512",
        "model.attn_resolutions=[8,16]",
        "model.decoder_extra_residual_layers=2",
        "model.use_mid_attention=true",
        "model.in_channels=3",
        "model.dropout=0.0",
        "model.num_hiddens=128",
        "model.num_residual_blocks=3",
        "model.num_residual_hiddens=96",
        "model.bottleneck_type=dictionary",
        "model.num_embeddings=4096",
        "model.embedding_dim=128",
        "model.sparsity_level=4",
        "model.commitment_cost=0.25",
        "model.bottleneck_loss_weight=0.75",
        "model.dictionary_loss_weight=null",
        f"model.dict_learning_rate={args.dict_learning_rate}",
        "model.coef_max=null",
        "model.data_init_from_first_batch=false",
        "model.patch_based=false",
        "model.patch_size=1",
        "model.patch_stride=1",
        "model.patch_reconstruction=tile",
        "model.sparsity_reg_weight=0.0",
        "model.recon_mse_weight=1.0",
        "model.recon_l1_weight=0.0",
        "model.recon_edge_weight=0.0",
        "model.perceptual_weight=1.0",
        "model.perceptual_start_step=0",
        "model.perceptual_warmup_steps=0",
        f"model.adversarial_weight={args.adversarial_weight}",
        "model.adversarial_start_step=0",
        "model.adversarial_warmup_steps=0",
        "model.disc_start_step=0",
        "model.disc_learning_rate=5e-05",
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
        "model.fid_feature=2048",
        "model.log_images_every_n_steps=250",
        "model.diag_log_interval=250",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=4096",
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.name={args.wandb_name}",
        f"+wandb.id={args.wandb_id}",
        "+wandb.resume=allow",
        f"wandb.group={args.wandb_group}",
        (
            "wandb.tags=[stage1_adv,ffhq,laser,dictionary,adversarial,"
            f"pxdpyowm,adv50,gpu-redhat,{int(args.gpus)}gpu,effbatch{effective_batch}]"
        ),
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
        "checkpoint.upload_to_wandb=true",
    ]


def write_job_files(job_root: Path, snapshot: Path, args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    output_dir = job_root / "stage1_adv50"
    log_base = job_root / "pxdpyowm_stage1_adv50"
    run_script = job_root / "run_pxdpyowm_stage1_adv50.sh"
    sbatch_script = job_root / "sbatch_pxdpyowm_stage1_adv50.sh"
    job_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_args = bash_array(stage1_adv_overrides(args, output_dir))
    effective_batch = int(args.gpus) * int(args.batch_size) * int(args.accumulate_grad_batches)

    run_script.write_text(
        f"""#!/bin/bash
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
export TMPDIR="/tmp/laser_pxdpyowm_adv50_${{SLURM_JOB_ID:-$$}}"
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

echo "=== GPU inventory ==="
nvidia-smi || true
echo ""

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

echo "=== FFHQ stage1 adversarial continuation from {SOURCE_RUN_ID} ==="
echo "source_run=helloimlixin-rutgers/laser/{SOURCE_RUN_ID}"
echo "init_checkpoint={args.init_ckpt}"
echo "output_dir={output_dir}"
echo "wandb_run_id={args.wandb_id}"
echo "wandb_name={args.wandb_name}"
echo "epochs={int(args.epochs)}"
echo "hardware_plan=partition:{args.partition} gpus:{int(args.gpus)} per_gpu_batch:{int(args.batch_size)} accumulate:{int(args.accumulate_grad_batches)} effective_batch:{effective_batch}"
echo "lr_plan=effective batch preserved at 40; train.learning_rate={args.learning_rate}; model.dict_learning_rate={args.dict_learning_rate}"
printf 'Launching:'
printf ' %q' "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
printf '\\n'
"$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
""",
        encoding="utf-8",
    )
    os.chmod(str(run_script), 0o755)

    sbatch_script.write_text(
        f"""#!/bin/bash
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
echo "output_dir={output_dir}"

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
""",
        encoding="utf-8",
    )
    os.chmod(str(sbatch_script), 0o755)
    return run_script, sbatch_script, log_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--run-root-base", default=str(RUN_ROOT_BASE))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--init-ckpt", default=str(INIT_CKPT))
    parser.add_argument("--pydeps", default=str(PYDEPS))
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--exclude", default="gpu029")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--accumulate-grad-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", default="4e-05")
    parser.add_argument("--dict-learning-rate", default="2.5e-4")
    parser.add_argument("--adversarial-weight", default="0.75")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--job-name", default="ffhq-pxd-adv50")
    parser.add_argument("--wandb-id", default=ADV_RUN_ID)
    parser.add_argument("--wandb-name", default=WANDB_NAME)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not str(args.dict_learning_rate).strip():
        args.dict_learning_rate = args.learning_rate
    return args


def validate(args: argparse.Namespace) -> None:
    checks = (
        (Path(args.repo), "repo"),
        (Path(args.data_dir), "FFHQ data dir"),
        (Path(args.init_ckpt), "init checkpoint"),
        (Path(args.pydeps), "pydeps"),
    )
    for path, label in checks:
        if not path.exists():
            raise SystemExit(f"Missing {label}: {path}")
    if int(args.gpus) <= 0:
        raise SystemExit("--gpus must be positive")
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive")
    if int(args.accumulate_grad_batches) <= 0:
        raise SystemExit("--accumulate-grad-batches must be positive")
    if int(args.epochs) <= 0:
        raise SystemExit("--epochs must be positive")


def main() -> int:
    args = parse_args()
    validate(args)
    args.repo = Path(args.repo).expanduser().resolve()
    args.snapshot_root = Path(args.snapshot_root).expanduser().resolve()
    args.run_root_base = Path(args.run_root_base).expanduser().resolve()
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.init_ckpt = Path(args.init_ckpt).expanduser().resolve()
    args.pydeps = Path(args.pydeps).expanduser().resolve()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = snapshot_repo(
        args.repo,
        args.snapshot_root,
        f"laser_pxdpyowm_stage1_adv50_{stamp}",
    )
    job_root = args.run_root_base / f"adv50_{stamp}"
    run_script, sbatch_script, log_base = write_job_files(job_root, snapshot, args)

    cmd = [
        "sbatch",
        f"--partition={args.partition}",
        f"--job-name={args.job_name}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={int(args.cpus_per_task)}",
        f"--gres=gpu:{int(args.gpus)}",
        f"--mem={int(args.mem_mb)}",
        f"--time={args.time_limit}",
        f"--chdir={snapshot}",
        f"--output={log_base}_%j.out",
        f"--error={log_base}_%j.err",
        "--requeue",
    ]
    if str(args.constraint).strip():
        cmd.append(f"--constraint={str(args.constraint).strip()}")
    if str(args.exclude).strip():
        cmd.append(f"--exclude={str(args.exclude).strip()}")
    cmd.append(str(sbatch_script))

    effective_batch = int(args.gpus) * int(args.batch_size) * int(args.accumulate_grad_batches)
    print(f"Snapshot: {snapshot}")
    print(f"Job root: {job_root}")
    print(f"Run script: {run_script}")
    print(f"Init checkpoint: {args.init_ckpt}")
    print(
        "Batch plan: gpus={} per_gpu_batch={} accumulate={} effective_batch={} lr={}".format(
            int(args.gpus),
            int(args.batch_size),
            int(args.accumulate_grad_batches),
            effective_batch,
            args.learning_rate,
        )
    )
    print(f"Adversarial plan: epochs={int(args.epochs)} weight={args.adversarial_weight}")
    print(f"W&B run: helloimlixin-rutgers/laser/{args.wandb_id} ({args.wandb_name})")
    print("Submit command: {}".format(" ".join(q(part) for part in cmd)))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    if proc.returncode != 0:
        return proc.returncode
    text = (proc.stdout or proc.stderr).strip()
    job_id = text.split()[-1] if text else "unknown"
    print(f"stdout: {log_base}_{job_id}.out")
    print(f"stderr: {log_base}_{job_id}.err")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
