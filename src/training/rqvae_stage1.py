"""Run the preserved rFID4.21 autoencoder trainer through the public CLI."""

from pathlib import Path
import os
import subprocess
import sys

from omegaconf import OmegaConf

from src.training.paths import ROOT


def run(cfg):
    runtime = Path(str(cfg.runtime_root)).expanduser().resolve()
    if not (runtime / "third_party/rq-vae-transformer/rqvae").is_dir():
        raise FileNotFoundError(f"Preserved RQ-VAE runtime missing: {runtime}")
    output = Path(str(cfg.output_dir)).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipe = OmegaConf.create(OmegaConf.to_container(cfg.recipe, resolve=True))
    devices = int(cfg.devices)
    if devices * recipe.experiment.batch_size != recipe.experiment.total_batch_size:
        raise ValueError("Per-GPU batch size times devices must equal total_batch_size")
    recipe_path = output / "training-config.yaml"
    OmegaConf.save(recipe, recipe_path)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={devices}",
               str(ROOT / "scripts/tools/train_imagenet_stage1_rfid421.py"),
               "--runtime-root", str(runtime), "--config", str(recipe_path),
               "--output", str(output)]
    if cfg.get("smoke", False):
        command.append("--smoke")
    if cfg.get("resume"):
        command.extend(["--resume", str(Path(str(cfg.resume)).expanduser().resolve())])
    elif (output / "last_model.pt").exists():
        raise FileExistsError("Output contains a checkpoint; use resume or a fresh output_dir")
    # Keep torchrun in the detached supervisor's process group.
    subprocess.run(command, cwd=ROOT, env=dict(os.environ), check=True)
