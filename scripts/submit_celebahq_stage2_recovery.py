#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 8):
    raise SystemExit("ERROR: submit_celebahq_stage2_recovery.py requires Python >= 3.8.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_multimodal_sweep as base  # noqa: E402


@dataclass(frozen=True)
class ResumeRun:
    label: str
    group: str
    wandb_id: str
    token_cache: Path
    output_dir: Path
    checkpoint: Path
    batch_size: int
    max_epochs: int
    learning_rate: str = "6.0e-4"


@dataclass(frozen=True)
class HighCapVariant:
    label: str
    patch_size: int
    atoms: int
    sparsity: int
    stage1_batch_size: int
    stage2_batch_size: int


def _user() -> str:
    return os.environ.get("USER", "unknown")


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def bash_array(items: list[str]) -> str:
    return "\n".join(f"  {q(item)}" for item in items)


def default_resume_runs() -> list[ResumeRun]:
    ksvd = Path("/scratch/xl598/runs/laser_debugging_patchseq_ksvd_sweep")
    simple = Path("/scratch/xl598/runs/laser_debugging_simple_dict_patch_sweep")
    p4_clean = ksvd / "laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-clean-grad-20260514_034728" / "celebahq"
    p4_online = ksvd / "laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-onlineksvd-20260514_034804" / "celebahq"
    simple_root = simple / "laser-train-celebahq-10ep-simple-dict-patch-sweep-20260514_040939"
    nopatch = simple_root / "nopatch-k1-a16384" / "celebahq"
    p2_simple = simple_root / "patch-p2-k2-a16384" / "celebahq"
    p4_simple = simple_root / "patch-p4-k4-a32768" / "celebahq"
    return [
        ResumeRun(
            label="ksvd-p4-clean-grad",
            group="laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-clean-grad-20260514_034728",
            wandb_id="zq5dayo7",
            token_cache=p4_clean / "token_cache.pt",
            output_dir=p4_clean / "stage2",
            checkpoint=p4_clean / "stage2/checkpoints/s2_20260514_041237/last.ckpt",
            batch_size=16,
            max_epochs=50,
        ),
        ResumeRun(
            label="ksvd-p4-onlineksvd",
            group="laser-train-celebahq-10ep-laser-patch-p4-k4-a32768-onlineksvd-20260514_034804",
            wandb_id="h2te6xxo",
            token_cache=p4_online / "token_cache.pt",
            output_dir=p4_online / "stage2",
            checkpoint=p4_online / "stage2/checkpoints/s2_20260514_050212/last.ckpt",
            batch_size=16,
            max_epochs=50,
        ),
        ResumeRun(
            label="simple-nopatch",
            group="laser-train-celebahq-10ep-simple-dict-patch-sweep-20260514_040939-nopatch-k1-a16384",
            wandb_id="uag4a11r",
            token_cache=nopatch / "token_cache.pt",
            output_dir=nopatch / "stage2",
            checkpoint=nopatch / "stage2/checkpoints/s2_20260514_042534/last.ckpt",
            batch_size=8,
            max_epochs=48,
        ),
        ResumeRun(
            label="simple-p2-k2-a16384",
            group="laser-train-celebahq-10ep-simple-dict-patch-sweep-20260514_040939-patch-p2-k2-a16384",
            wandb_id="jhi3qxe2",
            token_cache=p2_simple / "token_cache.pt",
            output_dir=p2_simple / "stage2",
            checkpoint=p2_simple / "stage2/checkpoints/s2_20260514_042331/last.ckpt",
            batch_size=8,
            max_epochs=50,
        ),
        ResumeRun(
            label="simple-p4-k4-a32768",
            group="laser-train-celebahq-10ep-simple-dict-patch-sweep-20260514_040939-patch-p4-k4-a32768",
            wandb_id="08ibhyiv",
            token_cache=p4_simple / "token_cache.pt",
            output_dir=p4_simple / "stage2",
            checkpoint=p4_simple / "stage2/checkpoints/s2_20260514_043418/last.ckpt",
            batch_size=8,
            max_epochs=50,
        ),
    ]


def highcap_variants() -> list[HighCapVariant]:
    return [
        HighCapVariant(
            label="patch-p8-k32-a131072",
            patch_size=8,
            atoms=131072,
            sparsity=32,
            stage1_batch_size=6,
            stage2_batch_size=6,
        ),
        HighCapVariant(
            label="patch-p16-k64-a65536",
            patch_size=16,
            atoms=65536,
            sparsity=64,
            stage1_batch_size=3,
            stage2_batch_size=6,
        ),
    ]


def validate_resume(run: ResumeRun) -> None:
    for path in (run.token_cache, run.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"{run.label}: missing {path}")
    run.output_dir.mkdir(parents=True, exist_ok=True)


def stage2_resume_overrides(run: ResumeRun) -> list[str]:
    return [
        f"token_cache_path={run.token_cache}",
        f"output_dir={run.output_dir}",
        f"ckpt_path={run.checkpoint}",
        "seed=42",
        "ar.type=sparse_spatial_depth",
        "ar.max_steps=-1",
        f"train_ar.max_epochs={run.max_epochs}",
        f"train_ar.batch_size={run.batch_size}",
        "train_ar.max_items=0",
        "train_ar.limit_train_batches=1.0",
        "train_ar.limit_val_batches=1.0",
        "train_ar.limit_test_batches=1.0",
        "train_ar.log_every_n_steps=50",
        "train_ar.sample_every_n_epochs=1",
        "train_ar.sample_log_to_wandb=true",
        "train_ar.sample_num_images=64",
        "train_ar.generation_metric_num_samples=0",
        "train_ar.compute_generation_fid=false",
        "train_ar.compute_audio_generation_metrics=false",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=false",
        "train_ar.devices=3",
        "train_ar.strategy=ddp",
        "train_ar.precision=bf16-mixed",
        "train_ar.accelerator=gpu",
        "data.dataset=celebahq",
        "data.data_dir=/scratch/xl598/datasets/celebahq_packed_256",
        "data.image_size=256",
        "data.num_workers=0",
        f"ar.learning_rate={run.learning_rate}",
        "wandb.project=laser-debugging",
        f"wandb.group={run.group}",
        "wandb.name=celebahq-stage2-transformer",
        "wandb.tags=[train,laser,celebahq,stage2,transformer,generation,resume40,recovery]",
        "wandb.append_timestamp=false",
        f"wandb.save_dir={run.output_dir / 'wandb'}",
        f"wandb.id={run.wandb_id}",
        "wandb.resume=allow",
    ]


def write_resume_job(snapshot_path: Path, job_root: Path, run: ResumeRun) -> tuple[Path, Path]:
    run_root = job_root / run.label
    run_root.mkdir(parents=True, exist_ok=True)
    run_script = run_root / "run_stage2_resume40.sh"
    sbatch_script = run_root / "sbatch_stage2_resume40.sh"
    run_script.write_text(
        f"""#!/bin/bash
set -euo pipefail

export PYTHONUSERBASE="/scratch/{_user()}/.pydeps/laser_src_py311"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="{snapshot_path}${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export WANDB_RESUME="${{WANDB_RESUME:-allow}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export HYDRA_FULL_ERROR=1
export TMPDIR="/tmp/laser_stage2_resume40_${{SLURM_JOB_ID:-$$}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

mkdir -p "$TMPDIR" {q(run.output_dir / 'wandb')}

PYTHON_BIN="${{PYTHON_BIN:-$(command -v python3 || command -v python || true)}}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3/python not found" >&2
  exit 127
fi

"$PYTHON_BIN" -m pip install --user --quiet \\
  numpy scipy wandb lightning omegaconf hydra-core rich 'torchmetrics[image]' \\
  torch-fidelity matplotlib lpips soundfile 2>/dev/null || true

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this job; failing before training.")
print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")
PY

cd {q(snapshot_path)}

STAGE2_ARGS=(
{bash_array(stage2_resume_overrides(run))}
)

echo "=== Resume Stage 2 for another 40 epochs: {run.label} ==="
"$PYTHON_BIN" train_stage2_prior.py "${{STAGE2_ARGS[@]}}"
""",
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)
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

IMAGE="${{IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}}"

if [[ -n "$CONTAINER_BIN" ]]; then
  "$CONTAINER_BIN" exec --nv \\
    --bind {q(snapshot_path)} \\
    --bind "/scratch/{_user()}" \\
    --bind {q(run.output_dir.parent)} \\
    --bind /dev/shm \\
    "$IMAGE" \\
    bash {q(run_script)}
else
  bash {q(run_script)}
fi
""",
        encoding="utf-8",
    )
    os.chmod(sbatch_script, 0o755)
    return run_script, sbatch_script


def highcap_stage1_overrides(variant: HighCapVariant) -> tuple[str, ...]:
    return (
        "data.num_workers=0",
        f"data.batch_size={variant.stage1_batch_size}",
        "train.learning_rate=1.2e-3",
        "model.dict_learning_rate=3.0e-4",
        "model.embedding_dim=64",
        "model.patch_based=true",
        f"model.patch_size={variant.patch_size}",
        f"model.patch_stride={variant.patch_size}",
        "model.patch_reconstruction=tile",
        f"model.num_embeddings={variant.atoms}",
        f"model.sparsity_level={variant.sparsity}",
        "model.coef_max=3.0",
        "model.enable_val_latent_visuals=true",
        "model.codebook_visual_max_vectors=1024",
        "model.log_images_every_n_steps=100",
    )


def highcap_stage2_overrides(variant: HighCapVariant) -> tuple[str, ...]:
    return (
        "data.num_workers=0",
        f"train_ar.batch_size={variant.stage2_batch_size}",
        "ar.learning_rate=6.0e-4",
        "train_ar.sample_log_to_wandb=true",
        "train_ar.sample_every_n_epochs=1",
        "train_ar.sample_num_images=64",
        "train_ar.generation_metric_num_samples=0",
        "train_ar.compute_generation_fid=false",
        "train_ar.run_test_after_fit=false",
        "train_ar.save_final_samples_after_fit=false",
    )


def submit(cmd: list[str], dry_run: bool) -> str:
    if dry_run:
        print(" ".join(shlex.quote(part) for part in cmd))
        return "dry-run"
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    text = (proc.stdout or proc.stderr).strip()
    return text.split()[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit CelebA-HQ stage-2 recovery jobs.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--snapshot-root", default=base._scratch_path("submission_snapshots"))
    parser.add_argument("--resume-root", default=base._scratch_path("runs", "laser_debugging_celebahq_recovery"))
    parser.add_argument("--highcap-root", default=base._scratch_path("runs", "laser_debugging_celebahq_highcap_patch_sweep"))
    parser.add_argument("--project", default="laser-debugging")
    parser.add_argument("--partition", default="gpu-redhat")
    parser.add_argument("--mode", choices=("all", "resume", "highcap"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = base.snapshot_repo(
        Path(args.repo).expanduser().resolve(),
        Path(args.snapshot_root).expanduser().resolve(),
        stem=f"laser_celebahq_stage2_recovery_{stamp}",
    )

    submissions = []
    if args.mode in {"all", "resume"}:
        resume_root = Path(args.resume_root).expanduser().resolve() / f"resume40-{stamp}"
        resume_root.mkdir(parents=True, exist_ok=True)
        for run in default_resume_runs():
            validate_resume(run)
            _, sbatch_script = write_resume_job(snapshot_path, resume_root, run)
            log_base = resume_root / run.label / run.label
            cmd = [
                "sbatch",
                f"--partition={args.partition}",
                f"--job-name=s2r40-{run.label[:16]}",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=8",
                "--gres=gpu:3",
                "--mem=96000",
                "--time=1-12:00:00",
                f"--chdir={snapshot_path}",
                f"--output={log_base}_%j.out",
                f"--error={log_base}_%j.err",
                str(sbatch_script),
            ]
            job_id = submit(cmd, args.dry_run)
            submissions.append((run.label, job_id, f"{log_base}_{job_id}.out", f"{log_base}_{job_id}.err"))

    if args.mode in {"all", "highcap"}:
        vctk_dir = Path(base._scratch_path("datasets", "VCTK-Corpus-smoke"))
        coco_dir = Path(base._scratch_path("data", "coco"))
        cases = base._dataset_cases(vctk_dir, coco_dir=coco_dir, model_family="laser")
        celebahq = next(case for case in cases if case.name == "celebahq")
        group_name = f"laser-train-celebahq-10s1-40s2-highcap-patch-{stamp}"
        highcap_root = Path(args.highcap_root).expanduser().resolve() / group_name
        highcap_root.mkdir(parents=True, exist_ok=True)
        for variant in highcap_variants():
            variant_root = highcap_root / variant.label
            run_dir, _, sbatch_script = base.write_job_files(
                snapshot_path=snapshot_path,
                run_root=variant_root,
                group_name=f"{group_name}-{variant.label}",
                project=args.project,
                case=celebahq,
                synthetic_vctk=False,
                full_training=True,
                model_family="laser",
                stage1_epochs=10,
                stage2_epochs=40,
                num_gpus=3,
                stage1_only=False,
                extra_stage1_overrides=highcap_stage1_overrides(variant),
                extra_stage2_overrides=highcap_stage2_overrides(variant),
            )
            log_base = run_dir / celebahq.name
            cmd = [
                "sbatch",
                f"--partition={args.partition}",
                f"--job-name=lshicap-{variant.label[:14]}",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=8",
                "--gres=gpu:3",
                "--mem=192000",
                "--time=3-00:00:00",
                f"--chdir={snapshot_path}",
                f"--output={log_base}_%j.out",
                f"--error={log_base}_%j.err",
                str(sbatch_script),
            ]
            job_id = submit(cmd, args.dry_run)
            submissions.append((variant.label, job_id, f"{log_base}_{job_id}.out", f"{log_base}_{job_id}.err"))

    print(f"Snapshot: {snapshot_path}")
    for label, job_id, stdout, stderr in submissions:
        print(f"[{label}] job={job_id} stdout={stdout} stderr={stderr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
