#!/usr/bin/env python3
"""Submit a +75 epoch stage-1 continuation for W&B run t1n66dfp."""

import argparse
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


USER = os.environ.get("USER", "xl598")
RUN_DIR = Path(
    "/scratch/xl598/runs/celebahq_simple_adv_stage2/"
    "celebahq-simple-adv-stage2-20260625_220220/k2-a8192"
)
STAGE1_ADV_DIR = RUN_DIR / "stage1_adv"
DEFAULT_SNAPSHOT = Path("/scratch/xl598/submission_snapshots/laser_celebahq_simple_adv_stage2_20260625_220220")
CKPT = STAGE1_ADV_DIR / "checkpoints/run_20260626_061848/laser/last.ckpt"
DATA_DIR = Path("/scratch/xl598/Projects/data/celeba_hq")
PYDEPS = Path("/scratch/xl598/.pydeps/laser_src_py311")
WANDB_ID = "t1n66dfp"
WANDB_NAME = "celebahq-stage1-adv-k2-a8192-from-ab5i0nam"
WANDB_GROUP = "celebahq-simple-adv-stage2-20260625_220220"


def q(value):
    return shlex.quote(str(value))


def bash_array(items):
    return "\n".join("  " + q(item) for item in items)


def stage1_args(max_epochs, gpus, output_dir, ckpt_path):
    return [
        "stage1",
        "seed=42",
        "output_dir={}".format(output_dir),
        "ckpt_path={}".format(ckpt_path),
        "model=laser_image_nonpatch_d5",
        "data=celebahq",
        "data.data_dir={}".format(DATA_DIR),
        "data.batch_size=11",
        "data.eval_batch_size=11",
        "data.num_workers=8",
        "data.pin_memory=true",
        "data.prefetch_factor=4",
        "data.image_size=256",
        "data.train_crop_size=null",
        "data.augment=true",
        "train.accelerator=gpu",
        "train.devices={}".format(int(gpus)),
        "train.num_nodes=1",
        "train.strategy=ddp",
        "train.precision=bf16-mixed",
        "train.max_epochs={}".format(int(max_epochs)),
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
        "model.perceptual_weight=1.0",
        "model.perceptual_start_step=0",
        "model.perceptual_warmup_steps=0",
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
        "model.compute_fid=true",
        "model.log_images_every_n_steps=250",
        "model.diag_log_interval=250",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=4096",
        "model.backbone=simple",
        "model.num_embeddings=8192",
        "model.sparsity_level=2",
        "wandb.project=laser",
        "wandb.name={}".format(WANDB_NAME),
        "wandb.group={}".format(WANDB_GROUP),
        "wandb.tags=[stage1_adv,celebahq,laser,dictionary,adversarial,simple,continuation,continue75]",
        "wandb.append_timestamp=false",
        "wandb.save_dir={}".format(output_dir / "wandb"),
        "checkpoint.upload_to_wandb=true",
    ]


def write_job_files(job_root, max_epochs, gpus, image, snapshot, output_dir, ckpt_path, pydeps):
    job_root.mkdir(parents=True, exist_ok=True)
    run_script = job_root / "run_t1n66dfp_continue75.sh"
    sbatch_script = job_root / "sbatch_t1n66dfp_continue75.sh"
    run_script.write_text(
        """#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE={pydeps}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export WANDB_RUN_ID="{wandb_id}"
export WANDB_RESUME="allow"
export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-0}}"
export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-{output_dir}/wandb/data}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{output_dir}/wandb/cache}}"
export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-{output_dir}/wandb/artifacts}}"
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{user}/.cache}}"
export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{user}/.config/wandb}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
export PYTHONUNBUFFERED=1
export TMPDIR="/tmp/laser_t1n66dfp_continue75_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$PYTHONUSERBASE" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \\
  "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" {wandb_dir}

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

if [[ ! -f {ckpt} ]]; then
  echo "Missing resume checkpoint: {ckpt}" >&2
  exit 1
fi

cd {snapshot}

STAGE1_ARGS=(
{stage1_args}
)

echo "=== Continue W&B run {wandb_id} for +75 epochs (25 -> {max_epochs}) ==="
printf 'Launching:'
printf ' %q' "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
printf '\\n'
"$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
""".format(
            pydeps=q(pydeps),
            snapshot=q(snapshot),
            wandb_id=WANDB_ID,
            output_dir=q(output_dir),
            user=USER,
            wandb_dir=q(output_dir / "wandb"),
            ckpt=q(ckpt_path),
            stage1_args=bash_array(stage1_args(max_epochs, gpus, output_dir, ckpt_path)),
            max_epochs=int(max_epochs),
        ),
        encoding="utf-8",
    )
    os.chmod(str(run_script), 0o755)

    sbatch_script.write_text(
        """#!/bin/bash
JOB_INTERNAL_LOG="/tmp/laser_t1n66dfp_batch_${{SLURM_JOB_ID:-manual}}.log"
mkdir -p "$(dirname "$JOB_INTERNAL_LOG")"
exec > >(tee -a "$JOB_INTERNAL_LOG") 2>&1

set -euo pipefail

echo "batch_job=${{SLURM_JOB_ID:-unknown}} host=$(hostname) start=$(date -Is)"
export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-/scratch/{user}/.apptainer/cache}}"
export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}}"
export APPTAINER_TMPDIR="${{APPTAINER_TMPDIR:-/scratch/{user}/.apptainer/tmp_${{SLURM_JOB_ID:-manual}}}}"
export SINGULARITY_TMPDIR="${{SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if ! command -v module >/dev/null 2>&1; then
  module_init_status=0
  if [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    set +e
    set +u
    source /usr/share/lmod/lmod/init/bash
    module_init_status=$?
    set -u
    set -e
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +e
    set +u
    source /usr/share/Modules/init/bash
    module_init_status=$?
    set -u
    set -e
  fi
  if [[ "$module_init_status" -ne 0 ]]; then
    echo "WARNING: module initialization returned $module_init_status; continuing without module setup"
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
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
    module load apptainer 2>/dev/null || true
  fi
  for candidate in singularity apptainer; do
    if command -v "$candidate" >/dev/null 2>&1; then
      CONTAINER_BIN="$candidate"
      break
    fi
  done
fi

IMAGE="${{IMAGE:-{image}}}"

echo "container_bin=$CONTAINER_BIN"
echo "image=$IMAGE"
echo "snapshot={snapshot}"

if [[ -n "$CONTAINER_BIN" ]]; then
    "$CONTAINER_BIN" exec --nv \\
    --bind /cache/home/{user} \\
    --bind {snapshot} \\
    --bind /scratch/{user} \\
    --bind {data_dir} \\
    --bind {run_dir} \\
    --bind /dev/shm \\
    "$IMAGE" \\
    bash {run_script}
else
  bash {run_script}
fi
""".format(
            job_root=q(job_root),
            image=image,
            snapshot=q(snapshot),
            user=USER,
            data_dir=q(DATA_DIR),
            run_dir=q(RUN_DIR),
            run_script=q(run_script),
        ),
        encoding="utf-8",
    )
    os.chmod(str(sbatch_script), 0o755)
    return run_script, sbatch_script


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="cgpu-redhat")
    parser.add_argument("--constraint", default="ampere")
    parser.add_argument("--nodelist", default="")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=24)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--job-root", default=str(STAGE1_ADV_DIR / "continue75_jobs"))
    parser.add_argument("--snapshot-path", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--slurm-chdir", default="")
    parser.add_argument("--ckpt-path", default=str(CKPT))
    parser.add_argument("--output-dir", default=str(STAGE1_ADV_DIR))
    parser.add_argument("--pydeps", default=str(PYDEPS))
    parser.add_argument("--image", default="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate(snapshot, ckpt_path, pydeps):
    for path, label in ((snapshot, "snapshot"), (ckpt_path, "resume checkpoint"), (DATA_DIR, "data dir"), (pydeps, "pydeps")):
        if not path.exists():
            raise SystemExit("Missing {}: {}".format(label, path))


def main():
    args = parse_args()
    snapshot = Path(args.snapshot_path).expanduser().resolve()
    slurm_chdir = Path(args.slurm_chdir).expanduser().resolve() if str(args.slurm_chdir).strip() else snapshot
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    pydeps = Path(args.pydeps).expanduser().resolve()
    validate(snapshot, ckpt_path, pydeps)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_root = Path(args.job_root).expanduser().resolve() / "t1n66dfp-continue75-{}".format(stamp)
    _, sbatch_script = write_job_files(job_root, args.max_epochs, args.gpus, args.image, snapshot, output_dir, ckpt_path, pydeps)
    log_base = job_root / "t1n66dfp_continue75"
    cmd = [
        "sbatch",
        "--partition={}".format(args.partition),
        "--job-name=t1n66dfp-p75",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task={}".format(int(args.cpus_per_task)),
        "--gres=gpu:{}".format(int(args.gpus)),
        "--mem={}".format(int(args.mem_mb)),
        "--time={}".format(args.time_limit),
        "--chdir={}".format(slurm_chdir),
        "--output={}_%j.out".format(log_base),
        "--error={}_%j.err".format(log_base),
    ]
    if str(args.constraint).strip():
        cmd.append("--constraint={}".format(str(args.constraint).strip()))
    if str(args.nodelist).strip():
        cmd.append("--nodelist={}".format(str(args.nodelist).strip()))
    cmd.append(str(sbatch_script))

    if args.dry_run:
        print(" ".join(q(part) for part in cmd))
        print("Job root: {}".format(job_root))
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
    print("Job root: {}".format(job_root))
    print("stdout: {}_{}.out".format(log_base, job_id))
    print("stderr: {}_{}.err".format(log_base, job_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
