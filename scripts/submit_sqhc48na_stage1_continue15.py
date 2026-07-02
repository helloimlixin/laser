#!/usr/bin/env python3
"""Submit a +15 epoch checkpointed continuation for W&B run sqhc48na.

This is the k4/a4096 ImageNet stage-1 member of the LPIPS sweep.  It reuses the
same continuation launcher shape as dift5rbj, but resumes sqhc48na's completed
7-epoch checkpoint and extends the run to 22 total epochs by default.
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

import submit_dift5rbj_stage1_continue as base


RUN_ID = "sqhc48na"
WANDB_NAME = "imagenet-stage1-k4-a4096-imagenet-full-stage1-sweep-20260628_013825"
RUN_ROOT_BASE = Path("/scratch/xl598/runs/imagenet_stage1_sweep_lpips005_continue15_sqhc48na")
CKPT = Path(
    "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/"
    "imagenet-full-stage1-sweep-20260628_013825/k4-a4096/stage1/"
    "checkpoints/run_slurm57342779_5/laser/last.ckpt"
)
EXCLUDE_NODES = os.environ.get("LASER_EXCLUDE_NODES", "gpu029").strip()


_stage1_overrides = base.stage1_overrides
_write_job_files = base.write_job_files
_snapshot_repo = base.snapshot_repo


def stage1_overrides(args, output_dir):
    overrides = _stage1_overrides(args, output_dir)
    return [
        "model.sparsity_level=4" if item == "model.sparsity_level=3" else item
        for item in overrides
    ]


def write_job_files(job_root, snapshot, args):
    run_script, sbatch_script, _log_base = _write_job_files(job_root, snapshot, args)
    new_run_script = job_root / "run_sqhc48na_continue15.sh"
    new_sbatch_script = job_root / "sbatch_sqhc48na_continue15.sh"
    new_log_base = job_root / "sqhc48na_continue15"

    run_text = run_script.read_text(encoding="utf-8").replace("dift5rbj", RUN_ID)
    new_run_script.write_text(run_text, encoding="utf-8")
    os.chmod(new_run_script, 0o755)
    run_script.unlink()

    sbatch_text = (
        sbatch_script.read_text(encoding="utf-8")
        .replace(str(run_script), str(new_run_script))
        .replace("dift5rbj", RUN_ID)
    )
    new_sbatch_script.write_text(sbatch_text, encoding="utf-8")
    os.chmod(new_sbatch_script, 0o755)
    sbatch_script.unlink()

    return new_run_script, new_sbatch_script, new_log_base


def snapshot_repo(repo, snapshot_root, name):
    return _snapshot_repo(repo, snapshot_root, name.replace("dift5rbj", RUN_ID))


def add_default_arg(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def main():
    base.RUN_ID = RUN_ID
    base.WANDB_NAME = WANDB_NAME
    base.RUN_ROOT_BASE = RUN_ROOT_BASE
    base.CKPT = CKPT
    base.stage1_overrides = stage1_overrides
    base.write_job_files = write_job_files
    base.snapshot_repo = snapshot_repo

    add_default_arg("--job-name", "imnet-sqhc48na-cont")
    add_default_arg("--run-root-base", str(RUN_ROOT_BASE))
    add_default_arg("--ckpt-path", str(CKPT))

    args = base.parse_args()
    base.validate(args)
    args.repo = Path(args.repo).expanduser().resolve()
    args.snapshot_root = Path(args.snapshot_root).expanduser().resolve()
    args.run_root_base = Path(args.run_root_base).expanduser().resolve()
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    args.pydeps = Path(args.pydeps).expanduser().resolve()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = base.snapshot_repo(
        args.repo,
        args.snapshot_root,
        "laser_{}_stage1_continue15_{}".format(RUN_ID, stamp),
    )
    job_root = args.run_root_base / "continue15_{}".format(stamp)
    run_script, sbatch_script, log_base = base.write_job_files(job_root, snapshot, args)

    cmd = [
        "sbatch",
        "--partition={}".format(args.partition),
        "--job-name={}".format(args.job_name),
        "--nodes={}".format(int(args.nodes)),
        "--ntasks={}".format(int(args.nodes)),
        "--ntasks-per-node=1",
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
    if EXCLUDE_NODES:
        cmd.append("--exclude={}".format(EXCLUDE_NODES))
    cmd.append(str(sbatch_script))

    eff_batch = base.effective_batch(args)
    batches_per_epoch = base.train_batches_per_epoch(args)
    updates_per_epoch = base.optimizer_updates_per_epoch(args)
    continue_epochs = base.resolved_continue_epochs(args)
    print("Snapshot: {}".format(snapshot))
    print("Job root: {}".format(job_root))
    print("Run script: {}".format(run_script))
    print("Resume checkpoint: {}".format(args.ckpt_path))
    print(
        "Batch plan: "
        "nodes={} gpus_per_node={} total_gpus={} per_gpu_batch={} "
        "accumulate={} effective_batch={} train_batches_per_epoch={} "
        "optimizer_updates_per_epoch={}".format(
            int(args.nodes),
            int(args.gpus),
            base.total_gpus(args),
            int(args.batch_size),
            int(args.accumulate_grad_batches),
            eff_batch,
            batches_per_epoch,
            updates_per_epoch,
        )
    )
    print(
        "LR schedule: "
        "lr={} dict_lr={} warmup_steps={} min_lr_ratio={} "
        "schedule_total_steps={} post_warmup=constant".format(
            args.learning_rate,
            args.dict_learning_rate,
            int(args.warmup_steps),
            args.min_lr_ratio,
            int(args.target_max_epochs) * batches_per_epoch,
        )
    )
    print(
        "Target max epochs: {} (source run max {}, continue +{})".format(
            int(args.target_max_epochs),
            base.SOURCE_RUN_MAX_EPOCHS,
            continue_epochs,
        )
    )
    print("W&B resume: id={} resume=allow".format(RUN_ID))
    if EXCLUDE_NODES:
        print("Excluded nodes: {}".format(EXCLUDE_NODES))
    print("Submit command: " + " ".join(base.q(part) for part in cmd))

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
