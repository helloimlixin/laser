#!/usr/bin/env python3
"""Submit image stage-1 dictionary/sparsity sweeps on Amarel."""

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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


def user():
    return os.environ.get("USER", "unknown")


def scratch_path(*parts):
    return str(Path("/scratch") / user() / Path(*parts))


def _face_model_overrides():
    return [
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
    ]


DATASET_SPECS = {
    "imagenet": {
        "data_config": "imagenet",
        "data_dir": scratch_path("Projects", "data", "imagenet"),
        "run_root_base": scratch_path("runs", "imagenet_stage1_competitive_sweep"),
        "run_prefix": "imagenet-competitive-stage1-sweep",
        "snapshot_prefix": "laser_imagenet_competitive_stage1_sweep",
        "job_name": "imnet-comp",
        "batch_size": 8,
        "num_workers": 8,
        "prefetch_factor": 4,
        "accumulate_grad_batches": 3,
        "max_epochs": 10,
        "max_steps": -1,
        "limit_val_batches": "256",
        "val_check_interval": "1.0",
        "run_test_after_fit": "false",
        "rfid_batch_size": 32,
        "rfid_num_workers": 8,
        "learning_rate": "4.0e-5",
        "warmup_steps": 4500,
        "min_lr_ratio": "0.10",
        "gradient_clip_val": "1.0",
        "log_every_n_steps": "50",
        "tags": "[stage1,imagenet,laser,full,sweep,competitive,lpips1,gan,rqvae-budget]",
        "model_overrides": [
            "model.backbone=ddpm",
            "model.num_downsamples=5",
            "model.channel_multipliers=[1,1,2,2,4,4]",
            "model.backbone_latent_channels=256",
            "model.attn_resolutions=[8]",
            "model.decoder_extra_residual_layers=0",
            "model.use_mid_attention=true",
            "model.dropout=0.0",
            "model.num_hiddens=128",
            "model.num_residual_blocks=2",
            "model.num_residual_hiddens=96",
            "model.embedding_dim=256",
            "model.commitment_cost=0.25",
            "model.bottleneck_loss_weight=0.75",
            "model.dictionary_loss_weight=null",
            "model.patch_based=false",
            "model.patch_size=1",
            "model.patch_stride=1",
            "model.dict_learning_rate=4.0e-5",
            "model.coef_max=16.0",
            "model.sparsity_reg_weight=0.0",
            "model.recon_mse_weight=0.5",
            "model.recon_l1_weight=1.0",
            "model.recon_edge_weight=0.25",
            "model.compute_fid=true",
            "model.log_images_every_n_steps=1000",
            "model.enable_val_latent_visuals=false",
            "model.codebook_visual_max_vectors=2048",
            "model.perceptual_weight=1.0",
            "model.perceptual_start_step=0",
            "model.perceptual_warmup_steps=0",
            "model.adversarial_weight=0.75",
            "model.adversarial_start_step=0",
            "model.adversarial_warmup_steps=0",
            "model.disc_start_step=0",
            "model.disc_learning_rate=4.0e-5",
            "model.discriminator_beta1=0.5",
            "model.discriminator_beta2=0.9",
            "model.disc_channels=64",
            "model.disc_num_layers=2",
            "model.disc_norm=group",
            "model.disc_spectral=false",
            "model.disc_loss=hinge",
            "model.use_adaptive_disc_weight=true",
            "model.disc_factor=1.0",
            "model.disc_weight_max=10000.0",
        ],
    },
    "ffhq": {
        "data_config": "ffhq",
        "data_dir": scratch_path("Projects", "data", "ffhq"),
        "run_root_base": scratch_path("runs", "ffhq_stage1_sweep"),
        "run_prefix": "ffhq-laser-paper-stage1-sweep",
        "snapshot_prefix": "laser_ffhq_stage1_sweep",
        "job_name": "ffhq-sweep",
        "batch_size": 11,
        "num_workers": 8,
        "prefetch_factor": 6,
        "accumulate_grad_batches": 1,
        "max_epochs": 150,
        "max_steps": -1,
        "limit_val_batches": "1.0",
        "val_check_interval": "1.0",
        "run_test_after_fit": "false",
        "rfid_batch_size": 32,
        "rfid_num_workers": 8,
        "learning_rate": "4.0e-5",
        "warmup_steps": 9845,
        "min_lr_ratio": "1.0",
        "deterministic": "false",
        "gradient_clip_val": "1.0",
        "log_every_n_steps": "20",
        "tags": "[stage1,ffhq,laser,dictionary,no-adversarial,sweep]",
        "model_overrides": _face_model_overrides(),
    },
    "celebahq": {
        "data_config": "celebahq",
        "data_dir": scratch_path("Projects", "data", "celeba_hq"),
        "run_root_base": scratch_path("runs", "celebahq_stage1_sweep"),
        "run_prefix": "celebahq-full-stage1-sweep",
        "snapshot_prefix": "laser_celebahq_stage1_sweep",
        "job_name": "chq-sweep",
        "batch_size": 11,
        "num_workers": 8,
        "prefetch_factor": 4,
        "accumulate_grad_batches": 1,
        "max_epochs": 150,
        "max_steps": -1,
        "limit_val_batches": "1.0",
        "val_check_interval": "1.0",
        "run_test_after_fit": "false",
        "rfid_batch_size": 32,
        "rfid_num_workers": 8,
        "learning_rate": "4.0e-5",
        "warmup_steps": 4220,
        "min_lr_ratio": "1.0",
        "deterministic": "false",
        "gradient_clip_val": "1.0",
        "log_every_n_steps": "20",
        "tags": "[stage1,celebahq,laser,dictionary,no-adversarial,sweep]",
        "model_overrides": _face_model_overrides(),
    },
}


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
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / stem
    if snapshot_path.exists():
        shutil.rmtree(str(snapshot_path))
    shutil.copytree(str(repo), str(snapshot_path), ignore=snapshot_ignore(repo))
    return snapshot_path


def q(value):
    return shlex.quote(str(value))


def bash_array_lines(items):
    return "\n".join("  " + q(item) for item in items)


def parse_csv_ints(raw, name):
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers: {raw!r}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{name} values must be positive: {raw!r}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


def _parse_resume_task_ids(raw):
    text = str(raw or "").strip()
    if not text:
        return set()
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"--resume-task-ids must be comma-separated integers: {raw!r}") from exc
        if value < 0:
            raise argparse.ArgumentTypeError(f"--resume-task-ids values must be >= 0: {raw!r}")
        values.append(value)
    return set(values)


def _variant_rows_from_run_root(run_root):
    variant_file = Path(run_root).expanduser().resolve() / "variants.tsv"
    if not variant_file.is_file():
        raise FileNotFoundError(f"variants.tsv not found under resume run root: {run_root}")
    rows = []
    for line in variant_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 5:
            task_id, label, sparsity, atoms, run_dir = parts
            patch_size = 0
        elif len(parts) == 6:
            task_id, label, patch_size, sparsity, atoms, run_dir = parts
        else:
            raise ValueError(f"Unexpected variants.tsv row in {variant_file}: {line!r}")
        rows.append(
            {
                "task_id": int(task_id),
                "label": label,
                "patch_size": int(patch_size),
                "sparsity": int(sparsity),
                "atoms": int(atoms),
                "run_dir": Path(run_dir).expanduser().resolve(),
            }
        )
    return rows


def _latest_checkpoint(run_dir, *, min_bytes):
    checkpoint_root = Path(run_dir) / "stage1" / "checkpoints"
    patterns = (
        "run_*/laser/last.ckpt",
        "run_*/laser/final.ckpt",
        "run_*/laser/*.ckpt",
    )
    candidates = []
    for pattern in patterns:
        for path in checkpoint_root.glob(pattern):
            if not path.is_file():
                continue
            stat = path.stat()
            if stat.st_size < int(min_bytes):
                continue
            candidates.append((stat.st_mtime, path.name == "last.ckpt", path.resolve()))
        if candidates:
            break
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1], str(item[2])), reverse=True)[0][2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="imagenet", choices=sorted(DATASET_SPECS))
    parser.add_argument("--backbone", default="ddpm", choices=("ddpm", "simple"))
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=scratch_path("submission_snapshots"))
    parser.add_argument("--run-root-base", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--imagenet-dir", default=None, help="Compatibility alias for --data-dir when --dataset=imagenet.")
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--constraint", default="adalovelace")
    parser.add_argument("--time-limit", default="3-00:00:00")
    parser.add_argument("--nodes", type=int, default=1, help="Number of SLURM nodes per array task.")
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem-mb", type=int, default=240000)
    parser.add_argument("--gpus-per-node", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-GPU batch size.")
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-val-batches", default=None)
    parser.add_argument("--val-check-interval", default=None)
    parser.add_argument(
        "--log-images-every-n-steps",
        type=int,
        default=None,
        help="Override model image-grid logging interval; 0 disables image grids.",
    )
    parser.add_argument(
        "--enable-val-latent-visuals",
        action="store_true",
        help="Enable validation latent/codebook visualizations.",
    )
    parser.add_argument(
        "--codebook-visual-max-vectors",
        type=int,
        default=None,
        help="Override maximum vectors used in codebook visualizations.",
    )
    parser.add_argument("--sparsity-levels", default="2,3,4")
    parser.add_argument("--dictionary-sizes", default="4096,8192,16384")
    parser.add_argument("--patch-sizes", default="", help="Comma-separated latent patch sizes; empty keeps per-site coding.")
    parser.add_argument("--only-labels", default="", help="Comma-separated labels such as k2-a4096,p4-k3-a8192.")
    parser.add_argument("--max-concurrent", type=int, default=0, help="0 means no Slurm array throttle.")
    parser.add_argument("--image", default="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
    parser.add_argument("--pydeps", default=scratch_path(".pydeps", "laser_src_py311"))
    parser.add_argument("--wandb-project", default="laser")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--resume-from-run-root", default="", help="Existing run root with variants.tsv/checkpoints to continue.")
    parser.add_argument("--resume-task-ids", default="", help="Comma-separated original array task ids to continue.")
    parser.add_argument("--resume-min-ckpt-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--disable-wandb-media", action="store_true")
    parser.add_argument("--disable-wandb-checkpoint-upload", action="store_true")
    parser.add_argument("--skip-postfit-rfid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = DATASET_SPECS[str(args.dataset)]
    args.spec = spec
    args.run_prefix = spec["run_prefix"]
    args.snapshot_prefix = spec["snapshot_prefix"]
    args.job_name = spec["job_name"]
    args.wandb_tags = spec["tags"]
    args.patch_sizes = parse_csv_ints(args.patch_sizes, "--patch-sizes") if str(args.patch_sizes).strip() else []
    if args.backbone == "simple":
        args.run_prefix = args.run_prefix + "-simple"
        args.snapshot_prefix = args.snapshot_prefix + "_simple"
        args.job_name = args.job_name + "-simple"
        if args.wandb_tags.endswith("]"):
            args.wandb_tags = args.wandb_tags[:-1] + ",simple]"
    if args.patch_sizes:
        args.run_prefix = args.run_prefix + "-patch-d5"
        args.snapshot_prefix = args.snapshot_prefix + "_patch_d5"
        args.job_name = args.job_name + "-patch"
        if args.wandb_tags.endswith("]"):
            args.wandb_tags = args.wandb_tags[:-1] + ",patch,d5]"
    if args.run_root_base is None:
        run_root_base = spec["run_root_base"]
        if args.backbone == "simple" or args.patch_sizes:
            base = Path(run_root_base)
            base_name = base.name
            if args.backbone == "simple":
                if base_name.endswith("_stage1_sweep"):
                    base_name = base_name[: -len("_stage1_sweep")] + "_stage1_simple_sweep"
                else:
                    base_name = base_name + "_simple"
            if args.patch_sizes:
                if base_name.endswith("_stage1_simple_sweep"):
                    base_name = base_name[: -len("_stage1_simple_sweep")] + "_stage1_simple_patch_sweep"
                elif base_name.endswith("_stage1_sweep"):
                    base_name = base_name[: -len("_stage1_sweep")] + "_stage1_patch_sweep"
                elif base_name.endswith("_sweep"):
                    base_name = base_name[: -len("_sweep")] + "_patch_sweep"
                else:
                    base_name = base_name + "_patch"
            run_root_base = str(base.with_name(base_name))
        args.run_root_base = run_root_base
    if args.data_dir is None:
        args.data_dir = args.imagenet_dir if args.imagenet_dir and args.dataset == "imagenet" else spec["data_dir"]
    if args.batch_size is None:
        args.batch_size = int(spec["batch_size"])
    if args.accumulate_grad_batches is None:
        args.accumulate_grad_batches = int(spec["accumulate_grad_batches"])
    if args.num_workers is None:
        args.num_workers = int(spec["num_workers"])
    if args.max_epochs is None:
        args.max_epochs = int(spec["max_epochs"])
    if args.max_steps is None:
        args.max_steps = int(spec["max_steps"])
    if args.limit_val_batches is None:
        args.limit_val_batches = str(spec["limit_val_batches"])
    if args.val_check_interval is None:
        args.val_check_interval = str(spec["val_check_interval"])

    args.sparsity_levels = parse_csv_ints(args.sparsity_levels, "--sparsity-levels")
    args.dictionary_sizes = parse_csv_ints(args.dictionary_sizes, "--dictionary-sizes")
    args.only_labels = set(part.strip() for part in str(args.only_labels).split(",") if part.strip())
    if args.nodes <= 0:
        raise SystemExit("--nodes must be positive")
    if args.gpus_per_node <= 0:
        raise SystemExit("--gpus-per-node must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.cpus_per_task <= 0:
        raise SystemExit("--cpus-per-task must be positive")
    if args.mem_mb <= 0:
        raise SystemExit("--mem-mb must be positive")
    if args.max_concurrent < 0:
        raise SystemExit("--max-concurrent must be >= 0")
    if args.log_images_every_n_steps is not None and args.log_images_every_n_steps < 0:
        raise SystemExit("--log-images-every-n-steps must be >= 0")
    if args.codebook_visual_max_vectors is not None and args.codebook_visual_max_vectors <= 0:
        raise SystemExit("--codebook-visual-max-vectors must be positive")
    return args


def write_job_files(snapshot, run_root, variants, args):
    slurm_dir = run_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    variant_file = run_root / "variants.tsv"
    runner = run_root / "run_variant.sh"
    sbatch_script = run_root / "sbatch_array.sh"

    rows = []
    for task_id, variant in enumerate(variants):
        if len(variant) == 4:
            label, patch_size, sparsity, atoms = variant
            resume_ckpt = ""
        else:
            label, patch_size, sparsity, atoms, resume_ckpt = variant
        run_dir = run_root / label
        rows.append(
            "\t".join(
                [
                    str(task_id),
                    label,
                    str(patch_size),
                    str(sparsity),
                    str(atoms),
                    str(run_dir),
                    str(resume_ckpt or ""),
                ]
            )
        )
        run_dir.mkdir(parents=True, exist_ok=True)
    variant_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    spec = args.spec
    model_overrides = list(spec["model_overrides"])
    if args.backbone == "simple":
        model_overrides = [
            override
            for override in model_overrides
            if not override.startswith("model.backbone=")
        ]
        model_overrides.append("model.backbone=simple")
    if args.patch_sizes:
        model_overrides = [
            override
            for override in model_overrides
            if not (
                override.startswith("model.patch_based=")
                or override.startswith("model.patch_size=")
                or override.startswith("model.patch_stride=")
                or override.startswith("model.patch_reconstruction=")
            )
        ]
        model_overrides.extend(
            [
                "model.patch_based=true",
                "model.patch_reconstruction=tile",
            ]
        )
    if args.log_images_every_n_steps is not None:
        model_overrides = [
            override
            for override in model_overrides
            if not override.startswith("model.log_images_every_n_steps=")
        ]
        model_overrides.append(f"model.log_images_every_n_steps={int(args.log_images_every_n_steps)}")
    if args.enable_val_latent_visuals:
        model_overrides = [
            override
            for override in model_overrides
            if not override.startswith("model.enable_val_latent_visuals=")
        ]
        model_overrides.append("model.enable_val_latent_visuals=true")
        if args.codebook_visual_max_vectors is None and not any(
            override.startswith("model.codebook_visual_max_vectors=") for override in model_overrides
        ):
            model_overrides.append("model.codebook_visual_max_vectors=4096")
    if args.codebook_visual_max_vectors is not None:
        model_overrides = [
            override
            for override in model_overrides
            if not override.startswith("model.codebook_visual_max_vectors=")
        ]
        model_overrides.append(f"model.codebook_visual_max_vectors={int(args.codebook_visual_max_vectors)}")
    constant_overrides = [
        f"model=laser_image_nonpatch_d5",
        f"data={spec['data_config']}",
        f"data.data_dir={args.data_dir}",
        f"data.batch_size={int(args.batch_size)}",
        f"data.eval_batch_size={int(args.batch_size)}",
        f"data.num_workers={int(args.num_workers)}",
        "data.pin_memory=true",
        f"data.prefetch_factor={int(spec['prefetch_factor'])}",
        "data.image_size=256",
        "data.train_crop_size=null",
        "data.augment=true",
        "train.accelerator=gpu",
        f"train.devices={int(args.gpus_per_node)}",
        f"train.num_nodes={int(args.nodes)}",
        "train.strategy=ddp",
        "train.precision=bf16-mixed",
        f"train.max_epochs={int(args.max_epochs)}",
        f"train.max_steps={int(args.max_steps)}",
        "train.limit_train_batches=1.0",
        f"train.limit_val_batches={args.limit_val_batches}",
        "train.limit_test_batches=0",
        f"train.val_check_interval={args.val_check_interval}",
        f"train.run_test_after_fit={spec['run_test_after_fit']}",
        "train.compute_rfid_after_fit=true",
        "train.rfid_split=val",
        f"train.rfid_batch_size={int(spec['rfid_batch_size'])}",
        f"train.rfid_num_workers={int(spec['rfid_num_workers'])}",
        "train.rfid_max_samples=0",
        "train.rfid_device=auto",
        "train.rfid_feature=2048",
        f"train.learning_rate={spec['learning_rate']}",
        "train.beta=0.5",
        "train.beta2=0.9",
        f"train.warmup_steps={int(spec['warmup_steps'])}",
        f"train.min_lr_ratio={spec['min_lr_ratio']}",
        f"train.accumulate_grad_batches={int(args.accumulate_grad_batches)}",
        f"train.gradient_clip_val={spec['gradient_clip_val']}",
        f"train.log_every_n_steps={spec['log_every_n_steps']}",
    ]
    if spec.get("deterministic") is not None:
        constant_overrides.append(f"train.deterministic={spec['deterministic']}")
    constant_overrides.extend(model_overrides)
    constant_lines = bash_array_lines(constant_overrides)
    patch_arg_lines = ""
    if args.patch_sizes:
        patch_arg_lines = '  model.patch_size="$PATCH_SIZE"\n  model.patch_stride="$PATCH_SIZE"'

    runner.write_text(
        f"""#!/bin/bash
set -euo pipefail

TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
VARIANT_LINE="$(awk -v id="$TASK_ID" 'BEGIN {{ FS="\\t" }} $1 == id {{ print; found=1 }} END {{ if (!found) exit 2 }}' {q(variant_file)})"
IFS=$'\\t' read -r TASK_ID LABEL PATCH_SIZE SPARSITY NUM_EMBEDDINGS RUN_DIR RESUME_CKPT <<< "$VARIANT_LINE"

mkdir -p "$RUN_DIR/stage1"
RUN_ID="{args.dataset}-stage1-${{LABEL}}-{run_root.name}"
echo "=== {args.dataset} stage-1 sweep variant ==="
echo "task_id=$TASK_ID label=$LABEL patch_size=$PATCH_SIZE sparsity=$SPARSITY num_embeddings=$NUM_EMBEDDINGS"
echo "wandb_run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "resume_ckpt=${{RESUME_CKPT:-}}"
echo "slurm_job_id=${{SLURM_JOB_ID:-unknown}} array_task=${{SLURM_ARRAY_TASK_ID:-none}}"
echo "=== GPU inventory ==="
nvidia-smi
echo ""

export PYTHONUSERBASE={q(args.pydeps)}
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH={q(snapshot)}${{PYTHONPATH:+:$PYTHONPATH}}
if [[ -z "${{LASER_VGG16_WEIGHTS:-}}" && -f {q(scratch_path(".cache", "torch", "hub", "checkpoints", "vgg16-397923af.pth"))} ]]; then
  export LASER_VGG16_WEIGHTS={q(scratch_path(".cache", "torch", "hub", "checkpoints", "vgg16-397923af.pth"))}
fi
export WANDB_MODE={q(args.wandb_mode)}
export WANDB_RUN_ID="$RUN_ID"
export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
export LASER_DISABLE_WANDB_MEDIA="${{LASER_DISABLE_WANDB_MEDIA:-{1 if args.disable_wandb_media else 0}}}"
export WANDB_DATA_DIR="${{WANDB_DATA_DIR:-$RUN_DIR/stage1/wandb/data}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-$RUN_DIR/stage1/wandb/cache}}"
export WANDB_ARTIFACT_DIR="${{WANDB_ARTIFACT_DIR:-$RUN_DIR/stage1/wandb/artifacts}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-4}}"
mkdir -p "$PYTHONUSERBASE" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

hydra_quote_path() {{
  printf '"%s"' "$1"
}}

if command -v flock >/dev/null 2>&1; then
  (
    flock 9
    pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
  ) 9>"$PYTHONUSERBASE/.install.lock"
else
  pip install --user --quiet scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' torch-fidelity matplotlib lpips 2>/dev/null || true
fi

cd {q(snapshot)}

CMD=(
  python train.py stage1
  seed=42
  output_dir="$RUN_DIR/stage1"
{constant_lines}
  model.num_embeddings="$NUM_EMBEDDINGS"
  model.sparsity_level="$SPARSITY"
{patch_arg_lines}
  wandb.project={q(args.wandb_project)}
  wandb.name="{args.dataset}-stage1-${{LABEL}}-{run_root.name}"
  +wandb.id="$WANDB_RUN_ID"
  +wandb.resume=allow
  wandb.group="{run_root.name}"
  {q("wandb.tags=" + args.wandb_tags)}
  wandb.append_timestamp=false
  wandb.save_dir="$RUN_DIR/stage1/wandb"
)

if [[ -n "${{RESUME_CKPT// /}}" ]]; then
  if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "Missing resume checkpoint: $RESUME_CKPT" >&2
    exit 1
  fi
  CMD+=("ckpt_path=$(hydra_quote_path "$RESUME_CKPT")")
fi

INIT_CKPT="${{{args.dataset.upper()}_STAGE1_INIT_CKPT:-${{STAGE1_INIT_CKPT:-}}}}"
if [[ -z "${{RESUME_CKPT// /}}" && -n "${{INIT_CKPT// /}}" ]]; then
  CMD+=("init_ckpt_path=$(hydra_quote_path "$INIT_CKPT")")
fi
if [[ {1 if (args.disable_wandb_checkpoint_upload or args.resume_from_run_root) else 0} -eq 1 ]]; then
  CMD+=("checkpoint.upload_to_wandb=false")
fi
if [[ {1 if (args.skip_postfit_rfid or args.resume_from_run_root) else 0} -eq 1 ]]; then
  CMD+=("train.compute_rfid_after_fit=false")
fi
if [[ {1 if args.disable_wandb_media else 0} -eq 1 ]]; then
  CMD+=("model.log_images_every_n_steps=0" "model.enable_val_latent_visuals=false")
fi

printf 'Launching:'
printf ' %q' "${{CMD[@]}}"
printf '\\n'
exec "${{CMD[@]}}"
""",
        encoding="utf-8",
    )
    os.chmod(runner, 0o755)

    srun_prefix = f"srun --ntasks={int(args.nodes)} --ntasks-per-node=1 " if int(args.nodes) > 1 else ""
    sbatch_script.write_text(
        f"""#!/bin/bash
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

export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-/cache/home/{user()}/.apptainer/cache}}"
export SINGULARITY_CACHEDIR="${{SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}}"
export APPTAINER_TMPDIR="${{APPTAINER_TMPDIR:-/scratch/{user()}/.apptainer/tmp_${{SLURM_JOB_ID:-manual}}}}"
export SINGULARITY_TMPDIR="${{SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [[ -n "$CONTAINER_BIN" ]]; then
  {srun_prefix}"$CONTAINER_BIN" exec --nv \\
    --bind /cache/home/{user()} \\
    --bind {q(snapshot)} \\
    --bind /scratch/{user()} \\
    --bind {q(args.data_dir)} \\
    --bind {q(run_root)} \\
    --bind /dev/shm \\
    {q(args.image)} \\
    bash {q(runner)}
else
  echo "Warning: singularity/apptainer not found; running bare" >&2
  {srun_prefix}bash {q(runner)}
fi
""",
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return variant_file, sbatch_script


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{args.dataset} directory not found: {data_dir}")
    args.data_dir = str(data_dir)
    spec = args.spec

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = snapshot_repo(
        repo,
        Path(args.snapshot_root).expanduser().resolve(),
        f"{args.snapshot_prefix}_{stamp}",
    )
    run_root = Path(args.run_root_base).expanduser().resolve() / f"{args.run_prefix}-{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    variants = []
    resume_rows = []
    skipped_resume_rows = []
    if str(args.resume_from_run_root).strip():
        resume_task_ids = _parse_resume_task_ids(args.resume_task_ids)
        resume_rows = _variant_rows_from_run_root(args.resume_from_run_root)
        for row in resume_rows:
            if resume_task_ids and row["task_id"] not in resume_task_ids:
                continue
            if args.only_labels and row["label"] not in args.only_labels:
                continue
            ckpt = _latest_checkpoint(row["run_dir"], min_bytes=args.resume_min_ckpt_bytes)
            if ckpt is None:
                skipped_resume_rows.append(row)
                continue
            variants.append((row["label"], row["patch_size"], row["sparsity"], row["atoms"], str(ckpt)))
    else:
        patch_sizes = args.patch_sizes if args.patch_sizes else [0]
        for patch_size in patch_sizes:
            for sparsity in args.sparsity_levels:
                for atoms in args.dictionary_sizes:
                    label = f"k{sparsity}-a{atoms}"
                    if args.patch_sizes:
                        label = f"p{patch_size}-{label}"
                    variants.append((label, patch_size, sparsity, atoms))
    if args.only_labels:
        requested = set(args.only_labels)
        variants = [variant for variant in variants if variant[0] in requested]
        found = {variant[0] for variant in variants}
        missing = sorted(requested - found)
        if missing and not str(args.resume_from_run_root).strip():
            raise SystemExit("Unknown --only-labels value(s): " + ", ".join(missing))
    if not variants:
        if skipped_resume_rows:
            skipped = ", ".join(row["label"] for row in skipped_resume_rows)
            raise SystemExit("No variants selected with valid resume checkpoints. Skipped: " + skipped)
        raise SystemExit("No variants selected.")
    variant_file, sbatch_script = write_job_files(snapshot, run_root, variants, args)

    array_spec = f"0-{len(variants) - 1}"
    if args.max_concurrent:
        array_spec = f"{array_spec}%{args.max_concurrent}"
    log_base = run_root / "slurm" / f"{args.dataset}_stage1_%A_%a"
    cmd = [
        "sbatch",
        f"--partition={args.partition}",
        f"--job-name={args.job_name}",
        f"--nodes={int(args.nodes)}",
        f"--ntasks={int(args.nodes)}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={int(args.cpus_per_task)}",
        f"--gres=gpu:{int(args.gpus_per_node)}",
        f"--mem={int(args.mem_mb)}",
        f"--time={args.time_limit}",
        f"--array={array_spec}",
        f"--chdir={snapshot}",
        f"--output={log_base}.out",
        f"--error={log_base}.err",
        "--requeue",
    ]
    if args.constraint:
        cmd.append(f"--constraint={args.constraint}")
    cmd.append(str(sbatch_script))

    print(f"Snapshot: {snapshot}")
    print(f"Run root: {run_root}")
    print(f"Variants: {variant_file}")
    total_gpus = int(args.nodes) * int(args.gpus_per_node)
    effective_batch = total_gpus * int(args.batch_size) * int(args.accumulate_grad_batches)
    print(
        f"Resources: partition={args.partition} constraint={args.constraint or '<none>'} "
        f"nodes={args.nodes} gpus_per_node={args.gpus_per_node} total_gpus={total_gpus} "
        f"batch_per_gpu={args.batch_size} accumulate={args.accumulate_grad_batches} "
        f"effective_batch={effective_batch}"
    )
    if str(args.resume_from_run_root).strip():
        print(f"Resume source: {Path(args.resume_from_run_root).expanduser().resolve()}")
        if skipped_resume_rows:
            skipped = ", ".join(row["label"] for row in skipped_resume_rows)
            print(f"Skipped without valid checkpoint: {skipped}")
    if args.dry_run:
        print("Dry run sbatch:", " ".join(q(part) for part in cmd))
        for task_id, variant in enumerate(variants):
            label, patch_size, sparsity, atoms = variant[:4]
            resume_ckpt = variant[4] if len(variant) > 4 else ""
            patch_text = f" patch_size={patch_size}" if int(patch_size) > 0 else ""
            resume_text = f" resume={resume_ckpt}" if resume_ckpt else ""
            print(f"[task {task_id}] {label}:{patch_text} sparsity={sparsity} num_embeddings={atoms}{resume_text}")
        return 0

    proc = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    job_id = (proc.stdout or proc.stderr).strip().split()[-1]
    print(f"Submitted array job: {job_id}")
    for task_id, variant in enumerate(variants):
        label, patch_size, sparsity, atoms = variant[:4]
        resume_ckpt = variant[4] if len(variant) > 4 else ""
        out_path = run_root / "slurm" / f"{args.dataset}_stage1_{job_id}_{task_id}.out"
        err_path = run_root / "slurm" / f"{args.dataset}_stage1_{job_id}_{task_id}.err"
        patch_text = f" patch_size={patch_size}" if int(patch_size) > 0 else ""
        resume_text = f" resume={resume_ckpt}" if resume_ckpt else ""
        print(
            f"[task {task_id}] {label}:{patch_text} sparsity={sparsity} num_embeddings={atoms} "
            f"stdout={out_path} stderr={err_path}{resume_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
