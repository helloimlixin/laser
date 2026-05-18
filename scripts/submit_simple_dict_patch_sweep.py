#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_multimodal_sweep as base  # noqa: E402


@dataclass(frozen=True)
class Variant:
    label: str
    patch_based: bool
    patch_size: int
    atoms: int
    sparsity: int
    stage1_batch_size: int = 8
    stage2_batch_size: int = 8


VARIANTS = (
    # No-patch has a minimum combined atom+coeff token depth of 2, so it is
    # intentionally the one over-budget control in this sweep.
    Variant("nopatch-k1-a16384", False, 1, 16384, 1),
    Variant("patch-p2-k2-a16384", True, 2, 16384, 2),
    Variant("patch-p4-k4-a32768", True, 4, 32768, 4),
    Variant("patch-p8-k8-a65536", True, 8, 65536, 8),
    # p16 already has a very wide patch dictionary; 32k atoms keeps memory sane
    # while still giving substantially more dictionary parameters than p8.
    Variant("patch-p16-k16-a32768", True, 16, 32768, 16, stage1_batch_size=6, stage2_batch_size=6),
)


def stage1_overrides(variant: Variant) -> tuple[str, ...]:
    overrides = [
        "data.num_workers=0",
        f"data.batch_size={variant.stage1_batch_size}",
        "train.learning_rate=1.6e-3",
        "model.dict_learning_rate=4.0e-4",
        "model.dictionary_update_mode=gradient",
        "model.dictionary_usage_ema_decay=0.0",
        "model.dictionary_usage_grad_scale=0.0",
        f"model.patch_based={'true' if variant.patch_based else 'false'}",
        f"model.num_embeddings={variant.atoms}",
        f"model.sparsity_level={variant.sparsity}",
        "model.coef_max=3.0",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=1024",
        "model.log_images_every_n_steps=100",
    ]
    if variant.patch_based:
        overrides.extend(
            [
                f"model.patch_size={variant.patch_size}",
                f"model.patch_stride={variant.patch_size}",
                "model.patch_reconstruction=tile",
            ]
        )
    return tuple(overrides)


def stage2_overrides(variant: Variant) -> tuple[str, ...]:
    return (
        "data.num_workers=0",
        f"train_ar.batch_size={variant.stage2_batch_size}",
        "ar.learning_rate=6.0e-4",
        "train_ar.sample_log_to_wandb=true",
        "train_ar.sample_every_n_epochs=1",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit the simple gradient dictionary patch sweep.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=base._scratch_path("submission_snapshots"))
    parser.add_argument("--run-root-base", default=base._scratch_path("runs", "laser_debugging_simple_dict_patch_sweep"))
    parser.add_argument("--project", default="laser-debugging")
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--time-limit", default="2-00:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem-mb", type=int, default=128000)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = base.snapshot_repo(
        repo,
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_simple_dict_patch_sweep_{stamp}",
    )

    vctk_dir = Path(base._scratch_path("datasets", "VCTK-Corpus-smoke"))
    coco_dir = Path(base._scratch_path("data", "coco"))
    cases = base._dataset_cases(vctk_dir, coco_dir=coco_dir, model_family="laser")
    celebahq = next(case for case in cases if case.name == "celebahq")

    group_name = f"laser-train-celebahq-10ep-simple-dict-patch-sweep-{stamp}"
    run_root = Path(args.run_root_base).expanduser().resolve() / group_name
    run_root.mkdir(parents=True, exist_ok=True)

    submissions = []
    for variant in VARIANTS:
        variant_root = run_root / variant.label
        run_dir, _, sbatch_script = base.write_job_files(
            snapshot_path=snapshot_path,
            run_root=variant_root,
            group_name=f"{group_name}-{variant.label}",
            project=args.project,
            case=celebahq,
            synthetic_vctk=False,
            full_training=True,
            model_family="laser",
            stage1_epochs=int(args.stage1_epochs),
            stage2_epochs=int(args.stage2_epochs),
            num_gpus=int(args.gpus),
            stage1_only=False,
            extra_stage1_overrides=stage1_overrides(variant),
            extra_stage2_overrides=stage2_overrides(variant),
        )
        log_base = run_dir / celebahq.name
        cmd = [
            "sbatch",
            f"--partition={args.partition}",
            f"--job-name=lsdict-{variant.label[:16]}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={int(args.cpus_per_task)}",
            f"--gres=gpu:{int(args.gpus)}",
            f"--mem={int(args.mem_mb)}",
            f"--time={args.time_limit}",
            f"--chdir={snapshot_path}",
            f"--output={log_base}_%j.out",
            f"--error={log_base}_%j.err",
            str(sbatch_script),
        ]
        if args.dry_run:
            job_id = "dry-run"
            print(" ".join(shlex.quote(part) for part in cmd))
        else:
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
            text = (proc.stdout or proc.stderr).strip()
            job_id = text.split()[-1]
        submissions.append(
            {
                "variant": variant.label,
                "job_id": job_id,
                "run_dir": str(run_dir),
                "stdout": f"{log_base}_{job_id}.out" if job_id != "dry-run" else f"{log_base}_<jobid>.out",
                "stderr": f"{log_base}_{job_id}.err" if job_id != "dry-run" else f"{log_base}_<jobid>.err",
            }
        )

    print(f"Snapshot: {snapshot_path}")
    print(f"Run root:  {run_root}")
    for item in submissions:
        print(
            f"[{item['variant']}] job={item['job_id']} "
            f"stdout={item['stdout']} stderr={item['stderr']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
