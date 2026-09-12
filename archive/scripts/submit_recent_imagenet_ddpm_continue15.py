#!/usr/bin/env python3
"""Submit +15 epoch continuations for completed 7-epoch ImageNet DDPM runs."""

import argparse
import copy
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

import submit_dift5rbj_stage1_continue as base


USER = os.environ.get("USER", "xl598")
REPO = Path("/scratch/xl598/Projects/laser")
SNAPSHOT_ROOT = Path("/scratch/xl598/submission_snapshots")
RUN_ROOT_BASE = Path("/scratch/xl598/runs/imagenet_recent_ddpm_continue15")
DATA_DIR = Path("/scratch/xl598/Projects/data/imagenet")
PYDEPS = Path("/scratch/xl598/.pydeps/laser_src_py311")
EXCLUDE_NODES = os.environ.get("LASER_EXCLUDE_NODES", "gpu029").strip()

SOURCE_RUN_MAX_EPOCHS = 7
DEFAULT_CONTINUE_EPOCHS = 15
DEFAULT_TARGET_MAX_EPOCHS = SOURCE_RUN_MAX_EPOCHS + DEFAULT_CONTINUE_EPOCHS

ORIG_STAGE1_OVERRIDES = base.stage1_overrides
ORIG_WRITE_JOB_FILES = base.write_job_files
ORIG_SNAPSHOT_REPO = base.snapshot_repo


RUNS = [
    {
        "label": "hgjcypgh",
        "run_id": "hgjcypgh",
        "variant": "k2-a16384",
        "config": "/scratch/xl598/runs/imagenet_stage1_sweep/imagenet-full-stage1-sweep-20260626_132200/k2-a16384/stage1/2026-06-27_06-46-08/.hydra/config.yaml",
        "ckpt": "/scratch/xl598/runs/imagenet_stage1_sweep/imagenet-full-stage1-sweep-20260626_132200/k2-a16384/stage1/checkpoints/run_slurm57327128_0/laser/last.ckpt",
        "wandb_name": "imagenet-stage1-k2-a16384-imagenet-full-stage1-sweep-20260626_132200",
        "wandb_group": "imagenet-full-stage1-sweep-20260626_132200",
    },
    {
        "label": "quzzl7ik",
        "run_id": "quzzl7ik",
        "variant": "k2-a8192",
        "config": "/scratch/xl598/runs/imagenet_stage1_sweep/imagenet-full-stage1-sweep-20260626_132835/k2-a8192/stage1/2026-06-28_08-18-10/.hydra/config.yaml",
        "ckpt": "/scratch/xl598/runs/imagenet_stage1_sweep/imagenet-full-stage1-sweep-20260626_132835/k2-a8192/stage1/checkpoints/run_slurm57327129_1/laser/last.ckpt",
        "wandb_name": "imagenet-stage1-k2-a8192-imagenet-full-stage1-sweep-20260626_132835",
        "wandb_group": "imagenet-full-stage1-sweep-20260626_132835",
    },
    {
        "label": "x7howl2l",
        "run_id": "x7howl2l",
        "variant": "k2-a4096",
        "config": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k2-a4096/stage1/2026-06-28_08-18-09/.hydra/config.yaml",
        "ckpt": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k2-a4096/stage1/checkpoints/run_slurm57342779_0/laser/last.ckpt",
        "wandb_name": "imagenet-stage1-k2-a4096-imagenet-full-stage1-sweep-20260628_013825",
        "wandb_group": "imagenet-full-stage1-sweep-20260628_013825",
    },
    {
        "label": "xyslsylu",
        "run_id": "xyslsylu",
        "variant": "k2-a8192",
        "config": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k2-a8192/stage1/2026-06-28_08-34-28/.hydra/config.yaml",
        "ckpt": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k2-a8192/stage1/checkpoints/run_slurm57342779_1/laser/last.ckpt",
        "wandb_name": "imagenet-stage1-k2-a8192-imagenet-full-stage1-sweep-20260628_013825",
        "wandb_group": "imagenet-full-stage1-sweep-20260628_013825",
    },
    {
        "label": "ppekc6zq",
        "run_id": "ppekc6zq",
        "variant": "k3-a8192",
        "config": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k3-a8192/stage1/2026-06-29_12-02-56/.hydra/config.yaml",
        "ckpt": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k3-a8192/stage1/checkpoints/run_slurm57342779_3/laser/last.ckpt",
        "wandb_name": "imagenet-stage1-k3-a8192-imagenet-full-stage1-sweep-20260628_013825",
        "wandb_group": "imagenet-full-stage1-sweep-20260628_013825",
    },
    {
        "label": "wezni79m",
        "run_id": "wezni79m",
        "variant": "k3-a16384",
        "config": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k3-a16384/stage1/2026-06-29_13-05-49/.hydra/config.yaml",
        "ckpt": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k3-a16384/stage1/checkpoints/run_slurm57342779_4/laser/last.ckpt",
        "wandb_name": "imagenet-stage1-k3-a16384-imagenet-full-stage1-sweep-20260628_013825",
        "wandb_group": "imagenet-full-stage1-sweep-20260628_013825",
    },
    {
        "label": "wslurm57342779_6",
        "run_id": "wandb_slurm57342779_6",
        "variant": "k4-a8192",
        "config": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k4-a8192/stage1/2026-06-30_15-56-45/.hydra/config.yaml",
        "ckpt": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k4-a8192/stage1/checkpoints/run_slurm57342779_6/laser/wandb_slurm57342779_6-last.ckpt",
        "wandb_name": "imagenet-stage1-k4-a8192-imagenet-full-stage1-sweep-20260628_013825",
        "wandb_group": "imagenet-full-stage1-sweep-20260628_013825",
    },
    {
        "label": "wslurm57342779_7",
        "run_id": "wandb_slurm57342779_7",
        "variant": "k4-a16384",
        "config": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k4-a16384/stage1/2026-06-30_17-55-56/.hydra/config.yaml",
        "ckpt": "/scratch/xl598/runs/imagenet_stage1_sweep_lpips005/imagenet-full-stage1-sweep-20260628_013825/k4-a16384/stage1/checkpoints/run_slurm57342779_7/laser/wandb_slurm57342779_7-last.ckpt",
        "wandb_name": "imagenet-stage1-k4-a16384-imagenet-full-stage1-sweep-20260628_013825",
        "wandb_group": "imagenet-full-stage1-sweep-20260628_013825",
    },
]


COPY_KEYS = set(
    [
        "data.batch_size",
        "data.eval_batch_size",
        "data.num_workers",
        "data.pin_memory",
        "data.prefetch_factor",
        "data.augment",
        "data.mean",
        "data.std",
        "train.learning_rate",
        "train.beta",
        "train.beta2",
        "train.warmup_steps",
        "train.min_lr_ratio",
        "train.gradient_clip_val",
        "train.accumulate_grad_batches",
        "train.deterministic",
        "train.log_every_n_steps",
        "train.limit_val_batches",
        "train.val_check_interval",
        "train.run_test_after_fit",
        "model.num_embeddings",
        "model.embedding_dim",
        "model.sparsity_level",
        "model.commitment_cost",
        "model.bottleneck_loss_weight",
        "model.dictionary_loss_weight",
        "model.dict_learning_rate",
        "model.coef_max",
        "model.data_init_from_first_batch",
        "model.patch_based",
        "model.patch_size",
        "model.patch_stride",
        "model.patch_reconstruction",
        "model.sparsity_reg_weight",
        "model.recon_mse_weight",
        "model.recon_l1_weight",
        "model.recon_edge_weight",
        "model.perceptual_weight",
        "model.perceptual_start_step",
        "model.perceptual_warmup_steps",
        "model.adversarial_weight",
        "model.adversarial_start_step",
        "model.adversarial_warmup_steps",
        "model.disc_start_step",
    ]
)


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = str(key) if not prefix else prefix + "." + str(key)
            flatten(child_prefix, child, out)
    else:
        out[prefix] = value


def load_source(candidate):
    path = Path(candidate["config"])
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    flat = {}
    flatten("", cfg, flat)
    return cfg, flat


def hydra_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    return str(value)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def short_job_label(value):
    label = safe_name(value)
    if len(label) <= 12:
        return label
    return label[:8] + label[-4:]


def apply_source_config(overrides, flat):
    patched = []
    for item in overrides:
        key = item.split("=", 1)[0]
        if key in COPY_KEYS and key in flat:
            patched.append(key + "=" + hydra_value(flat[key]))
        else:
            patched.append(item)
    return patched


def stage1_overrides_for(candidate, args, output_dir):
    _cfg, flat = load_source(candidate)
    overrides = ORIG_STAGE1_OVERRIDES(args, output_dir)
    return apply_source_config(overrides, flat)


def write_job_files_for(candidate, job_root, snapshot, args):
    base.RUN_ID = candidate["run_id"]
    base.WANDB_NAME = candidate["wandb_name"]
    base.WANDB_GROUP = candidate["wandb_group"]
    base.stage1_overrides = lambda current_args, output_dir: stage1_overrides_for(
        candidate, current_args, output_dir
    )

    run_script, sbatch_script, _log_base = ORIG_WRITE_JOB_FILES(job_root, snapshot, args)
    label = safe_name(candidate["label"])
    new_run_script = job_root / ("run_" + label + "_continue15.sh")
    new_sbatch_script = job_root / ("sbatch_" + label + "_continue15.sh")
    new_log_base = job_root / (label + "_continue15")

    run_text = run_script.read_text(encoding="utf-8").replace("dift5rbj", label)
    new_run_script.write_text(run_text, encoding="utf-8")
    os.chmod(str(new_run_script), 0o755)
    run_script.unlink()

    sbatch_text = (
        sbatch_script.read_text(encoding="utf-8")
        .replace(str(run_script), str(new_run_script))
        .replace("dift5rbj", label)
    )
    new_sbatch_script.write_text(sbatch_text, encoding="utf-8")
    os.chmod(str(new_sbatch_script), 0o755)
    sbatch_script.unlink()

    return new_run_script, new_sbatch_script, new_log_base


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--run-root-base", default=str(RUN_ROOT_BASE))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--pydeps", default=str(PYDEPS))
    parser.add_argument("--partition", default="gpu-redhat,legacy-gpu")
    parser.add_argument("--constraint", default="ampere")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--continue-epochs", type=int, default=DEFAULT_CONTINUE_EPOCHS)
    parser.add_argument("--target-max-epochs", type=int)
    parser.add_argument("--image", default=base.DEFAULT_IMAGE)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_max_epochs is None:
        args.target_max_epochs = SOURCE_RUN_MAX_EPOCHS + int(args.continue_epochs)
    return args


def args_for_candidate(common, candidate):
    _cfg, flat = load_source(candidate)
    args = copy.copy(common)
    args.repo = Path(common.repo).expanduser().resolve()
    args.snapshot_root = Path(common.snapshot_root).expanduser().resolve()
    args.run_root_base = Path(common.run_root_base).expanduser().resolve()
    args.data_dir = Path(common.data_dir).expanduser().resolve()
    args.pydeps = Path(common.pydeps).expanduser().resolve()
    args.ckpt_path = Path(candidate["ckpt"]).expanduser().resolve()
    args.batch_size = int(flat.get("data.batch_size", 21))
    args.num_workers = int(flat.get("data.num_workers", 6))
    args.accumulate_grad_batches = int(flat.get("train.accumulate_grad_batches", 2))
    args.learning_rate = hydra_value(flat.get("train.learning_rate", "4.0e-5"))
    args.dict_learning_rate = hydra_value(flat.get("model.dict_learning_rate", "4.0e-5"))
    args.warmup_steps = int(flat.get("train.warmup_steps", 5005))
    args.min_lr_ratio = hydra_value(flat.get("train.min_lr_ratio", "1.0"))
    args.job_name = "imnet-" + short_job_label(candidate["label"]) + "-cont"
    return args


def validate_candidate(args, candidate):
    for path, label in (
        (args.repo, "repo"),
        (args.data_dir, "ImageNet data dir"),
        (args.pydeps, "pydeps"),
        (args.ckpt_path, "resume checkpoint"),
        (Path(candidate["config"]), "source Hydra config"),
    ):
        if not Path(path).exists():
            raise SystemExit("Missing {}: {}".format(label, path))
    _cfg, flat = load_source(candidate)
    if flat.get("data.dataset") != "imagenet":
        raise SystemExit("{} is not ImageNet".format(candidate["label"]))
    if flat.get("model.backbone") != "ddpm":
        raise SystemExit("{} is not DDPM".format(candidate["label"]))
    if int(flat.get("train.max_epochs", -1)) != SOURCE_RUN_MAX_EPOCHS:
        raise SystemExit("{} was not a 7-epoch source run".format(candidate["label"]))


def make_sbatch_cmd(args, snapshot, log_base, sbatch_script):
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
    return cmd


def selected_runs(args):
    if not args.only:
        return RUNS
    wanted = set(args.only)
    return [
        item
        for item in RUNS
        if item["label"] in wanted or item["run_id"] in wanted or item["variant"] in wanted
    ]


def main():
    common = parse_args()
    runs = selected_runs(common)
    if not runs:
        raise SystemExit("No runs selected")

    prepared = []
    for candidate in runs:
        args = args_for_candidate(common, candidate)
        validate_candidate(args, candidate)
        prepared.append((candidate, args))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = ORIG_SNAPSHOT_REPO(
        Path(common.repo).expanduser().resolve(),
        Path(common.snapshot_root).expanduser().resolve(),
        "laser_recent_imagenet_ddpm_continue15_{}".format(stamp),
    )
    print("Snapshot: {}".format(snapshot))
    print("Selected runs: {}".format(", ".join(item["label"] for item, _args in prepared)))

    submitted = []
    for candidate, args in prepared:
        label = safe_name(candidate["label"])
        job_root = Path(args.run_root_base) / ("{}_continue15_{}".format(label, stamp))
        run_script, sbatch_script, log_base = write_job_files_for(
            candidate, job_root, snapshot, args
        )
        cmd = make_sbatch_cmd(args, snapshot, log_base, sbatch_script)
        eff_batch = base.effective_batch(args)
        print("")
        print("Run: {} ({}, {})".format(candidate["label"], candidate["variant"], candidate["run_id"]))
        print("Job root: {}".format(job_root))
        print("Run script: {}".format(run_script))
        print("Resume checkpoint: {}".format(args.ckpt_path))
        print(
            "Batch plan: nodes={} gpus_per_node={} total_gpus={} per_gpu_batch={} "
            "accumulate={} effective_batch={}".format(
                int(args.nodes),
                int(args.gpus),
                base.total_gpus(args),
                int(args.batch_size),
                int(args.accumulate_grad_batches),
                eff_batch,
            )
        )
        print(
            "LR plan: lr={} dict_lr={} warmup_steps={} min_lr_ratio={}".format(
                args.learning_rate,
                args.dict_learning_rate,
                int(args.warmup_steps),
                args.min_lr_ratio,
            )
        )
        print(
            "Target max epochs: {} (source {}, continue +{})".format(
                int(args.target_max_epochs),
                SOURCE_RUN_MAX_EPOCHS,
                int(args.target_max_epochs) - SOURCE_RUN_MAX_EPOCHS,
            )
        )
        print("Submit command: " + " ".join(base.q(part) for part in cmd))
        if common.dry_run:
            submitted.append((candidate, "dry-run", job_root, log_base))
            continue
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        text = (proc.stdout or proc.stderr).strip()
        job_id = text.split()[-1] if text else "unknown"
        print("stdout: {}_{}.out".format(log_base, job_id))
        print("stderr: {}_{}.err".format(log_base, job_id))
        submitted.append((candidate, job_id, job_root, log_base))

    print("")
    print("Summary:")
    for candidate, job_id, job_root, log_base in submitted:
        print(
            "{} {} job={} root={} logs={}_<job>.out/.err".format(
                candidate["label"], candidate["variant"], job_id, job_root, log_base
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
