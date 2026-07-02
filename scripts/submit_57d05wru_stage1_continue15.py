#!/usr/bin/env python3
"""Submit a checkpointed continuation for W&B run 57d05wru.

The first continuation attempt from the downloaded W&B artifact reached local
epoch-2 checkpoints before Slurm preempted it.  This launcher resumes from that
latest local checkpoint and keeps the interrupted continuation target
(``train.max_epochs=17``).  The default resource shape targets the shared
scratch gpu-redhat partition: 2 Ampere GPUs, per-GPU batch 21, and accumulation
3, giving effective batch 126, essentially the original batch-128 recipe.
"""

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


USER = os.environ.get("USER", "xl598")
RUN_ID = "57d05wru"
WANDB_PROJECT = "laser"
WANDB_GROUP = "imagenet-nonpatch-d5-k4-scratch-h200"
WANDB_NAME = "imagenet-nonpatch-d5-k4-scratch-stage1"
SOURCE_RUN_MAX_EPOCHS = 7
DEFAULT_CONTINUE_EPOCHS = 15
DEFAULT_TARGET_MAX_EPOCHS = 17
CHECKPOINT_EPOCH = 2
CHECKPOINT_GLOBAL_STEP = 72996

REPO = Path("/scratch/xl598/Projects/laser")
SNAPSHOT_ROOT = Path("/scratch/xl598/submission_snapshots")
RUN_ROOT_BASE = Path("/scratch/xl598/runs/imagenet_nonpatch_d5_k4_scratch_continue_57d05wru")
DATA_DIR = Path("/scratch/xl598/Projects/data/imagenet")
CKPT = Path(
    "/scratch/xl598/runs/imagenet_nonpatch_d5_k4_scratch_continue_57d05wru/"
    "continue15_20260629_212338/stage1/checkpoints/run_slurm57603510/laser/"
    "57d05wru-last.ckpt"
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


def q(value):
    return shlex.quote(str(value))


def bash_array(items):
    return "\n".join("  " + q(item) for item in items)


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
    if not repo.is_dir():
        raise FileNotFoundError("repo not found: {}".format(repo))
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_root / stem
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
        cmd.extend("--exclude={}".format(pattern) for pattern in EXCLUDES)
        cmd.extend(["{}/".format(repo), str(snapshot)])
        subprocess.run(cmd, check=True)
    else:
        shutil.copytree(str(repo), str(snapshot), ignore=snapshot_ignore(repo))
    return snapshot


def stage1_overrides(args, output_dir):
    effective_batch = int(args.gpus) * int(args.batch_size) * int(args.accumulate_grad_batches)
    # Original run: 1 H200, per-device batch 64, accumulate=2 => effective batch 128.
    # Default continuation: 2 RTX 3090-class GPUs, per-device batch 21,
    # accumulate=3 => effective batch 126, so the LR remains effectively 4e-5.
    return [
        "stage1",
        "seed=42",
        "output_dir={}".format(output_dir),
        "ckpt_path=__RESUME_CKPT__",
        "model=laser_image_nonpatch_d5",
        "data=imagenet",
        "data.dataset=imagenet",
        "data.data_dir={}".format(args.data_dir),
        "data.image_size=256",
        "data.train_crop_size=null",
        "data.batch_size={}".format(int(args.batch_size)),
        "data.eval_batch_size={}".format(int(args.batch_size)),
        "data.num_workers={}".format(int(args.num_workers)),
        "data.pin_memory=true",
        "data.prefetch_factor=4",
        "data.augment=true",
        "data.mean=[0.5,0.5,0.5]",
        "data.std=[0.5,0.5,0.5]",
        "train.accelerator=gpu",
        "train.devices={}".format(int(args.gpus)),
        "train.num_nodes=1",
        "train.strategy=ddp",
        "train.precision=bf16-mixed",
        "train.max_epochs={}".format(int(args.target_max_epochs)),
        "train.max_steps=-1",
        "train.limit_train_batches=1.0",
        "train.limit_val_batches=256",
        "train.limit_test_batches=0",
        "train.val_check_interval=5000",
        "train.log_every_n_steps=20",
        "train.run_test_after_fit=false",
        "train.compute_rfid_after_fit=true",
        "train.rfid_split=val",
        "train.rfid_batch_size=64",
        "train.rfid_num_workers=8",
        "train.rfid_max_samples=0",
        "train.rfid_device=auto",
        "train.rfid_feature=2048",
        "train.learning_rate={}".format(args.learning_rate),
        "train.beta=0.5",
        "train.beta2=0.9",
        "train.warmup_steps=5005",
        "train.min_lr_ratio=1.0",
        "train.gradient_clip_val=1.0",
        "train.accumulate_grad_batches={}".format(int(args.accumulate_grad_batches)),
        "train.deterministic=false",
        "model.type=laser",
        "model.backbone=ddpm",
        "model.num_downsamples=5",
        "model.channel_multipliers=[1,1,2,2,4,4]",
        "model.backbone_latent_channels=256",
        "model.attn_resolutions=[8]",
        "model.decoder_extra_residual_layers=0",
        "model.use_mid_attention=true",
        "model.in_channels=3",
        "model.dropout=0.0",
        "model.num_hiddens=128",
        "model.num_residual_blocks=2",
        "model.num_residual_hiddens=96",
        "model.bottleneck_type=dictionary",
        "model.num_embeddings=8192",
        "model.embedding_dim=128",
        "model.sparsity_level=4",
        "model.commitment_cost=0.25",
        "model.bottleneck_loss_weight=0.75",
        "model.dictionary_loss_weight=0.75",
        "model.dict_learning_rate={}".format(args.dict_learning_rate),
        "model.coef_max=16.0",
        "model.data_init_from_first_batch=true",
        "model.patch_based=false",
        "model.patch_size=1",
        "model.patch_stride=1",
        "model.patch_reconstruction=tile",
        "model.sparsity_reg_weight=0.01",
        "model.recon_mse_weight=0.25",
        "model.recon_l1_weight=1.0",
        "model.recon_edge_weight=0.5",
        "model.perceptual_weight=1.0",
        "model.perceptual_start_step=0",
        "model.perceptual_warmup_steps=0",
        "model.adversarial_weight=0.0",
        "model.adversarial_start_step=1000000000",
        "model.adversarial_warmup_steps=0",
        "model.disc_start_step=1000000000",
        "model.disc_learning_rate=5e-05",
        "model.disc_channels=64",
        "model.disc_num_layers=3",
        "model.disc_norm=group",
        "model.disc_spectral=false",
        "model.disc_loss=hinge",
        "model.use_adaptive_disc_weight=true",
        "model.disc_factor=1.0",
        "model.compute_fid=true",
        "model.fid_feature=2048",
        "model.log_images_every_n_steps=200",
        "model.diag_log_interval=100",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=1024",
        "wandb.project={}".format(WANDB_PROJECT),
        "wandb.name={}".format(WANDB_NAME),
        "wandb.group={}".format(WANDB_GROUP),
        "wandb.tags=[train,laser,imagenet,stage1,autoencoder,nonpatch,d5,k4,scratch,h200,continuation,continue15,gpu-redhat,{}gpu,effbatch{}]".format(
            int(args.gpus),
            effective_batch,
        ),
        "wandb.append_timestamp=false",
        "wandb.save_dir={}".format(output_dir / "wandb"),
        "checkpoint.upload_to_wandb=true",
    ]


def write_job_files(job_root, snapshot, args):
    output_dir = job_root / "stage1"
    log_base = job_root / "57d05wru_continue15"
    run_script = job_root / "run_57d05wru_continue15.sh"
    sbatch_script = job_root / "sbatch_57d05wru_continue15.sh"
    job_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_script.write_text(
        """#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE={pydeps}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{snapshot}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export WANDB_RUN_ID="{run_id}"
export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-{output_dir}/wandb/data}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-{output_dir}/wandb/cache}}"
export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-{output_dir}/wandb/artifacts}}"
export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-/scratch/{user}/.config/wandb}}"
export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-/scratch/{user}/.cache}}"
export TORCH_HOME="${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}"
export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
export PYTHONUNBUFFERED=1
if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" && -f {vgg16} ]]; then
  export LASER_VGG16_WEIGHTS={vgg16}
fi
export TMPDIR="/tmp/laser_57d05wru_continue15_${{SLURM_JOB_ID:-$$}}"
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
OWN_CKPT={checkpoint_root}/run_slurm${{SLURM_JOB_ID:-manual}}/laser/{run_id}-last.ckpt
RESUME_CKPT="$INITIAL_CKPT"
if [[ -f "$OWN_CKPT" ]]; then
  RESUME_CKPT="$OWN_CKPT"
fi
if [[ ! -f "$RESUME_CKPT" ]]; then
  echo "Missing resume checkpoint: $RESUME_CKPT" >&2
  exit 1
fi

cd {snapshot}

STAGE1_ARGS=(
{stage1_args}
)
for i in "${{!STAGE1_ARGS[@]}}"; do
  if [[ "${{STAGE1_ARGS[$i]}}" == "ckpt_path=__RESUME_CKPT__" ]]; then
    STAGE1_ARGS[$i]="ckpt_path=$RESUME_CKPT"
  fi
done

echo "=== Continue W&B run {run_id} ==="
echo "source_run_max_epochs={source_epochs}"
echo "initial_checkpoint={ckpt}"
echo "auto_resume_checkpoint=$RESUME_CKPT"
echo "checkpoint_epoch={ckpt_epoch}"
echo "checkpoint_global_step={ckpt_step}"
echo "target_max_epochs={target_epochs}"
echo "hardware_plan=partition:{partition} gpus:{gpus} per_gpu_batch:{batch} accumulate:{accum} effective_batch:{effective_batch}"
echo "lr_plan=original effective batch 128; selected effective batch {effective_batch}; train.learning_rate={lr}, model.dict_learning_rate={dict_lr}"
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
            vgg16=q(VGG16_WEIGHTS),
            wandb_dir=q(output_dir / "wandb"),
            ckpt=q(args.ckpt_path),
            checkpoint_root=q(output_dir / "checkpoints"),
            stage1_args=bash_array(stage1_overrides(args, output_dir)),
            source_epochs=SOURCE_RUN_MAX_EPOCHS,
            ckpt_epoch=CHECKPOINT_EPOCH,
            ckpt_step=CHECKPOINT_GLOBAL_STEP,
            target_epochs=int(args.target_max_epochs),
            partition=args.partition,
            gpus=int(args.gpus),
            batch=int(args.batch_size),
            accum=int(args.accumulate_grad_batches),
            effective_batch=int(args.gpus) * int(args.batch_size) * int(args.accumulate_grad_batches),
            lr=args.learning_rate,
            dict_lr=args.dict_learning_rate,
        ),
        encoding="utf-8",
    )
    os.chmod(str(run_script), 0o755)

    sbatch_script.write_text(
        """#!/bin/bash
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

export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-/cache/home/{user}/.apptainer/cache}}"
export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}}"
export APPTAINER_TMPDIR="${{APPTAINER_TMPDIR:-/scratch/{user}/.apptainer/tmp_${{SLURM_JOB_ID:-manual}}}}"
export SINGULARITY_TMPDIR="${{SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

IMAGE="${{IMAGE:-{image}}}"
echo "container_bin=$CONTAINER_BIN"
echo "image=$IMAGE"
echo "snapshot={snapshot}"
echo "output_dir={output_dir}"

if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec --nv \\
    --bind /cache/home/{user} \\
    --bind {snapshot} \\
    --bind /scratch/{user} \\
    --bind {data_dir} \\
    --bind {job_root} \\
    --bind {ckpt_dir} \\
    --bind /dev/shm \\
    "$IMAGE" \\
    bash {run_script}
else
  echo "Warning: singularity/apptainer not found; running bare" >&2
  bash {run_script}
fi
""".format(
            user=USER,
            image=args.image,
            snapshot=q(snapshot),
            output_dir=q(output_dir),
            data_dir=q(args.data_dir),
            job_root=q(job_root),
            ckpt_dir=q(Path(args.ckpt_path).parent),
            run_script=q(run_script),
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
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="ampere")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=24)
    parser.add_argument("--mem-mb", type=int, default=80000)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=21)
    parser.add_argument("--accumulate-grad-batches", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--learning-rate", default="4e-05")
    parser.add_argument("--dict-learning-rate", default="")
    parser.add_argument("--continue-epochs", type=int, default=DEFAULT_CONTINUE_EPOCHS)
    parser.add_argument("--target-max-epochs", type=int, default=DEFAULT_TARGET_MAX_EPOCHS)
    parser.add_argument("--job-name", default="imnet-57d05wru-p15")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not str(args.dict_learning_rate).strip():
        args.dict_learning_rate = args.learning_rate
    return args


def validate(args):
    for path, label in (
        (Path(args.repo), "repo"),
        (Path(args.data_dir), "ImageNet data dir"),
        (Path(args.ckpt_path), "resume checkpoint"),
        (Path(args.pydeps), "pydeps"),
    ):
        if not path.exists():
            raise SystemExit("Missing {}: {}".format(label, path))
    if int(args.gpus) <= 0:
        raise SystemExit("--gpus must be positive")
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive")
    if int(args.accumulate_grad_batches) <= 0:
        raise SystemExit("--accumulate-grad-batches must be positive")
    if int(args.target_max_epochs) <= CHECKPOINT_EPOCH:
        raise SystemExit("--target-max-epochs must be greater than {}".format(CHECKPOINT_EPOCH))


def main():
    args = parse_args()
    validate(args)
    args.repo = Path(args.repo).expanduser().resolve()
    args.snapshot_root = Path(args.snapshot_root).expanduser().resolve()
    args.run_root_base = Path(args.run_root_base).expanduser().resolve()
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    args.pydeps = Path(args.pydeps).expanduser().resolve()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = snapshot_repo(
        args.repo,
        args.snapshot_root,
        "laser_57d05wru_stage1_continue15_{}".format(stamp),
    )
    job_root = args.run_root_base / "continue15_{}".format(stamp)
    run_script, sbatch_script, log_base = write_job_files(job_root, snapshot, args)

    cmd = [
        "sbatch",
        "--partition={}".format(args.partition),
        "--job-name={}".format(args.job_name),
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task={}".format(int(args.cpus_per_task)),
        "--gres=gpu:{}".format(int(args.gpus)),
        "--mem={}".format(int(args.mem_mb)),
        "--time={}".format(args.time_limit),
        "--chdir={}".format(snapshot),
        "--output={}_%j.out".format(log_base),
        "--error={}_%j.err".format(log_base),
        "--requeue",
    ]
    if str(args.constraint).strip():
        cmd.append("--constraint={}".format(str(args.constraint).strip()))
    cmd.append(str(sbatch_script))

    print("Snapshot: {}".format(snapshot))
    print("Job root: {}".format(job_root))
    print("Run script: {}".format(run_script))
    effective_batch = int(args.gpus) * int(args.batch_size) * int(args.accumulate_grad_batches)
    print("Batch plan: gpus={} per_gpu_batch={} accumulate={} effective_batch={} lr={}".format(
        int(args.gpus),
        int(args.batch_size),
        int(args.accumulate_grad_batches),
        effective_batch,
        args.learning_rate,
    ))
    print("Target max epochs: {} (interrupted continuation target; requested +{})".format(
        int(args.target_max_epochs),
        int(args.continue_epochs),
    ))
    print("Resume checkpoint: {} (epoch {}, global_step {})".format(
        args.ckpt_path,
        CHECKPOINT_EPOCH,
        CHECKPOINT_GLOBAL_STEP,
    ))
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
    print("stdout: {}_{}.out".format(log_base, job_id))
    print("stderr: {}_{}.err".format(log_base, job_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
