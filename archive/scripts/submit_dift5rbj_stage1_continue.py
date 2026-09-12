#!/usr/bin/env python3
"""Submit a +15 epoch checkpointed continuation for W&B run dift5rbj.

The crashed job was preempted after the original 7-epoch ImageNet stage-1
target.  The latest local checkpoint is ``laser-epoch=006.ckpt``/``last.ckpt``.
This launcher extends the run to 22 total epochs by default and keeps the
memory-safe 3-GPU L40S/Ada shape.  It can request either one node with three
GPUs or three nodes with one GPU each; per-GPU batch 21 with accumulation 2
gives effective batch 126, essentially the original batch-128 recipe.
"""

import argparse
import math
import os
import shlex
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

from submit_imagenet_stage1_sweep import snapshot_repo


USER = os.environ.get("USER", "xl598")
RUN_ID = "dift5rbj"
WANDB_PROJECT = "laser"
WANDB_NAME = "imagenet-stage1-k3-a4096-imagenet-full-stage1-sweep-20260628_013825"
WANDB_GROUP = "imagenet-full-stage1-sweep-20260628_013825"
SOURCE_RUN_MAX_EPOCHS = 7
DEFAULT_CONTINUE_EPOCHS = 15
DEFAULT_TARGET_MAX_EPOCHS = SOURCE_RUN_MAX_EPOCHS + DEFAULT_CONTINUE_EPOCHS
CHECKPOINT_EPOCH = 6
IMAGENET_TRAIN_IMAGES = 1_281_167

REPO = Path("/scratch/xl598/Projects/laser")
SNAPSHOT_ROOT = Path("/scratch/xl598/submission_snapshots")
RUN_ROOT_BASE = Path("/scratch/xl598/runs/imagenet_stage1_sweep_lpips005_continue15_dift5rbj")
DATA_DIR = Path("/scratch/xl598/Projects/data/imagenet")
CKPT = Path(
    "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/"
    "imagenet-full-stage1-sweep-20260628_013825/k3-a4096/stage1/"
    "checkpoints/run_slurm57342779_2/laser/last.ckpt"
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


def q(value):
    return shlex.quote(str(value))


def bash_array(items):
    return "\n".join("  " + q(item) for item in items)


def effective_batch(args):
    return total_gpus(args) * int(args.batch_size) * int(args.accumulate_grad_batches)


def total_gpus(args):
    return int(args.nodes) * int(args.gpus)


def train_batches_per_epoch(args):
    return int(math.ceil(IMAGENET_TRAIN_IMAGES / float(total_gpus(args) * int(args.batch_size))))


def optimizer_updates_per_epoch(args):
    return int(math.ceil(train_batches_per_epoch(args) / float(int(args.accumulate_grad_batches))))


def resolved_continue_epochs(args):
    return max(0, int(args.target_max_epochs) - SOURCE_RUN_MAX_EPOCHS)


def stage1_overrides(args, output_dir):
    batch_size = int(args.batch_size)
    eff_batch = effective_batch(args)
    return [
        "stage1",
        "seed=42",
        f"output_dir={output_dir}",
        f"ckpt_path={args.ckpt_path}",
        "model=laser_image_nonpatch_d5",
        "data=imagenet",
        "data.dataset=imagenet",
        f"data.data_dir={args.data_dir}",
        "data.image_size=256",
        "data.train_crop_size=null",
        f"data.batch_size={batch_size}",
        f"data.eval_batch_size={batch_size}",
        f"data.num_workers={int(args.num_workers)}",
        "data.pin_memory=true",
        "data.prefetch_factor=4",
        "data.augment=true",
        "data.mean=[0.5,0.5,0.5]",
        "data.std=[0.5,0.5,0.5]",
        "train.accelerator=gpu",
        f"train.devices={int(args.gpus)}",
        f"train.num_nodes={int(args.nodes)}",
        "train.strategy=ddp",
        "train.precision=bf16-mixed",
        f"train.max_epochs={int(args.target_max_epochs)}",
        "train.max_steps=-1",
        "train.limit_train_batches=1.0",
        "train.limit_val_batches=256",
        "train.limit_test_batches=0",
        "train.val_check_interval=5000",
        "train.run_test_after_fit=true",
        "train.compute_rfid_after_fit=false",
        "train.rfid_split=val",
        "train.rfid_batch_size=64",
        "train.rfid_num_workers=8",
        "train.rfid_max_samples=0",
        "train.rfid_device=auto",
        "train.rfid_feature=2048",
        f"train.learning_rate={args.learning_rate}",
        "train.beta=0.5",
        "train.beta2=0.9",
        f"train.warmup_steps={int(args.warmup_steps)}",
        f"train.min_lr_ratio={args.min_lr_ratio}",
        "train.gradient_clip_val=1.0",
        f"train.accumulate_grad_batches={int(args.accumulate_grad_batches)}",
        "train.deterministic=true",
        "train.log_every_n_steps=100",
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
        "model.num_embeddings=4096",
        "model.embedding_dim=128",
        "model.sparsity_level=3",
        "model.commitment_cost=0.25",
        "model.bottleneck_loss_weight=0.75",
        "model.dictionary_loss_weight=null",
        f"model.dict_learning_rate={args.dict_learning_rate}",
        "model.coef_max=16.0",
        "model.data_init_from_first_batch=true",
        "model.patch_based=false",
        "model.patch_size=1",
        "model.patch_stride=1",
        "model.patch_reconstruction=tile",
        "model.sparsity_reg_weight=0.0",
        "model.recon_mse_weight=0.25",
        "model.recon_l1_weight=1.0",
        "model.recon_edge_weight=0.5",
        "model.perceptual_weight=0.05",
        "model.perceptual_start_step=5000",
        "model.perceptual_warmup_steps=10000",
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
        "model.disc_weight_max=10000.0",
        "model.compute_fid=true",
        "model.fid_feature=2048",
        "model.log_images_every_n_steps=500",
        "model.diag_log_interval=100",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=4096",
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.name={WANDB_NAME}",
        f"wandb.group={WANDB_GROUP}",
        f"wandb.tags=[stage1,imagenet,laser,full,sweep,continuation,continue15,gpu-redhat,{int(args.nodes)}node,{total_gpus(args)}gpu,effbatch{eff_batch}]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={output_dir / 'wandb'}",
        "checkpoint.upload_to_wandb=true",
    ]


def write_job_files(job_root, snapshot, args):
    output_dir = job_root / "stage1"
    log_base = job_root / "dift5rbj_continue15"
    run_script = job_root / "run_dift5rbj_continue15.sh"
    sbatch_script = job_root / "sbatch_dift5rbj_continue15.sh"
    job_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_args = bash_array(stage1_overrides(args, output_dir))
    eff_batch = effective_batch(args)
    batches_per_epoch = train_batches_per_epoch(args)
    updates_per_epoch = optimizer_updates_per_epoch(args)
    continue_epochs = resolved_continue_epochs(args)
    remaining_batches = continue_epochs * batches_per_epoch
    remaining_updates = continue_epochs * updates_per_epoch
    schedule_total_steps = int(args.target_max_epochs) * batches_per_epoch
    srun_prefix = (
        f"srun --ntasks={int(args.nodes)} --ntasks-per-node=1 "
        if int(args.nodes) > 1
        else ""
    )

    run_script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail

            export PYTHONUSERBASE={q(args.pydeps)}
            export PATH="$PYTHONUSERBASE/bin:$PATH"
            export PYTHONPATH="$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONUSERBASE/lib/python3.12/site-packages:{q(snapshot)}${{PYTHONPATH:+:$PYTHONPATH}}"
            export WANDB_MODE="${{WANDB_MODE:-online}}"
            export WANDB_RUN_ID="{RUN_ID}"
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
            export TMPDIR="/tmp/laser_dift5rbj_continue_${{SLURM_JOB_ID:-$$}}"
            export TEMP="$TMPDIR"
            export TMP="$TMPDIR"
            if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" && -f {q(VGG16_WEIGHTS)} ]]; then
              export LASER_VGG16_WEIGHTS={q(VGG16_WEIGHTS)}
            fi

            mkdir -p "$PYTHONUSERBASE" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \\
              "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" {q(output_dir / 'wandb')}

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

            echo "=== Continue W&B run {RUN_ID} for +{continue_epochs} epochs ({SOURCE_RUN_MAX_EPOCHS} -> {int(args.target_max_epochs)}) ==="
            echo "resume_checkpoint={args.ckpt_path}"
            echo "target_max_epochs={int(args.target_max_epochs)}"
            echo "hardware_plan=partition:{args.partition} constraint:{args.constraint or 'none'} nodes:{int(args.nodes)} gpus_per_node:{int(args.gpus)} total_gpus:{total_gpus(args)} cpus_per_task:{int(args.cpus_per_task)} mem_mb_per_node:{int(args.mem_mb)} time:{args.time_limit}"
            echo "batch_plan=per_gpu_batch:{int(args.batch_size)} accumulate:{int(args.accumulate_grad_batches)} effective_batch:{eff_batch} train_batches_per_epoch:{batches_per_epoch} optimizer_updates_per_epoch:{updates_per_epoch}"
            echo "continuation_work=epochs:{continue_epochs} train_batches:{remaining_batches} optimizer_updates:{remaining_updates}"
            echo "lr_plan=train.learning_rate={args.learning_rate} model.dict_learning_rate={args.dict_learning_rate} warmup_steps:{int(args.warmup_steps)} min_lr_ratio:{args.min_lr_ratio} schedule_total_steps:{schedule_total_steps} post_warmup:constant"
            printf 'Launching:'
            printf ' %q' "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
            printf '\\n'
            "$PYTHON_BIN" train.py "${{STAGE1_ARGS[@]}}"
            """
        ),
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    sbatch_script.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
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
                --bind {q(args.data_dir)} \\
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
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=21)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--accumulate-grad-batches", type=int, default=2)
    parser.add_argument("--continue-epochs", type=int, default=DEFAULT_CONTINUE_EPOCHS)
    parser.add_argument("--target-max-epochs", type=int)
    parser.add_argument("--learning-rate", default="4.0e-5")
    parser.add_argument("--dict-learning-rate", default="4.0e-5")
    parser.add_argument("--warmup-steps", type=int, default=5005)
    parser.add_argument("--min-lr-ratio", default="1.0")
    parser.add_argument("--job-name", default="imnet-dift5rbj-cont")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_max_epochs is None:
        args.target_max_epochs = SOURCE_RUN_MAX_EPOCHS + int(args.continue_epochs)
    return args


def validate(args):
    for path, label in (
        (Path(args.repo), "repo"),
        (Path(args.data_dir), "ImageNet data dir"),
        (Path(args.ckpt_path), "resume checkpoint"),
        (Path(args.pydeps), "pydeps"),
    ):
        if not path.exists():
            raise SystemExit(f"Missing {label}: {path}")
    if int(args.nodes) <= 0:
        raise SystemExit("--nodes must be positive")
    if int(args.gpus) <= 0:
        raise SystemExit("--gpus must be positive")
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive")
    if int(args.num_workers) < 0:
        raise SystemExit("--num-workers must be non-negative")
    if int(args.accumulate_grad_batches) <= 0:
        raise SystemExit("--accumulate-grad-batches must be positive")
    if int(args.continue_epochs) <= 0:
        raise SystemExit("--continue-epochs must be positive")
    if int(args.target_max_epochs) <= CHECKPOINT_EPOCH:
        raise SystemExit(f"--target-max-epochs must be greater than checkpoint epoch {CHECKPOINT_EPOCH}")


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
        f"laser_dift5rbj_stage1_continue15_{stamp}",
    )
    job_root = args.run_root_base / f"continue15_{stamp}"
    run_script, sbatch_script, log_base = write_job_files(job_root, snapshot, args)

    cmd = [
        "sbatch",
        f"--partition={args.partition}",
        f"--job-name={args.job_name}",
        f"--nodes={int(args.nodes)}",
        f"--ntasks={int(args.nodes)}",
        "--ntasks-per-node=1",
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
    cmd.append(str(sbatch_script))

    eff_batch = effective_batch(args)
    batches_per_epoch = train_batches_per_epoch(args)
    updates_per_epoch = optimizer_updates_per_epoch(args)
    continue_epochs = resolved_continue_epochs(args)
    print(f"Snapshot: {snapshot}")
    print(f"Job root: {job_root}")
    print(f"Run script: {run_script}")
    print(f"Resume checkpoint: {args.ckpt_path}")
    print(
        "Batch plan: "
        f"nodes={int(args.nodes)} gpus_per_node={int(args.gpus)} total_gpus={total_gpus(args)} "
        f"per_gpu_batch={int(args.batch_size)} "
        f"accumulate={int(args.accumulate_grad_batches)} effective_batch={eff_batch} "
        f"train_batches_per_epoch={batches_per_epoch} optimizer_updates_per_epoch={updates_per_epoch}"
    )
    print(
        "LR schedule: "
        f"lr={args.learning_rate} dict_lr={args.dict_learning_rate} warmup_steps={int(args.warmup_steps)} "
        f"min_lr_ratio={args.min_lr_ratio} schedule_total_steps={int(args.target_max_epochs) * batches_per_epoch} "
        "post_warmup=constant"
    )
    print(
        f"Target max epochs: {int(args.target_max_epochs)} "
        f"(source run max {SOURCE_RUN_MAX_EPOCHS}, continue +{continue_epochs})"
    )
    print(f"W&B resume: id={RUN_ID} resume=allow")
    print("Submit command: " + " ".join(q(part) for part in cmd))

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
