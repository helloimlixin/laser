#!/usr/bin/env python3
"""Fail closed if an ImageNet 8x8xK Stage-1 run drifts from RQ-VAE."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


def require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def require_source_text(path: Path, needles: tuple[str, ...], label: str):
    source = path.read_text()
    for needle in needles:
        if needle not in source:
            raise RuntimeError(f"{label} is missing required behavior: {needle}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(8, 16), required=True)
    parser.add_argument("--local-batch-size", type=int, required=True)
    parser.add_argument("--sparsity-level", type=int, choices=(4, 8), required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    rq_root = source_root / "third_party" / "rq-vae-transformer"
    reference_path = (
        rq_root
        / "configs"
        / "imagenet256"
        / "stage1"
        / "in256-rqvae-8x8x4.yaml"
    )
    config_path = args.config.resolve()
    for required in (rq_root, reference_path, config_path):
        if not required.exists():
            raise RuntimeError(f"missing frozen input: {required}")

    sys.path[:0] = [str(rq_root), str(source_root)]
    import lmdb  # noqa: F401 -- preflight the complete ImageNet dependency set.
    import rqvae.img_datasets  # noqa: F401
    import rqvae.models  # noqa: F401
    import rqvae.optimizer  # noqa: F401
    import rqvae.trainers  # noqa: F401
    import rqvae.utils.setup  # noqa: F401
    from rqvae.losses.vqgan.discriminator import NLayerDiscriminator
    from rqvae.models.rqvae.modules import Decoder, Encoder
    from src.models.dictionary_learner import DictionaryLearning

    expected_modules = (rq_root / "rqvae/models/rqvae/modules.py").resolve()
    expected_discriminator = (
        rq_root / "rqvae/losses/vqgan/discriminator.py"
    ).resolve()
    require_equal(Path(inspect.getfile(Encoder)).resolve(), expected_modules, "encoder")
    require_equal(Path(inspect.getfile(Decoder)).resolve(), expected_modules, "decoder")
    require_equal(
        Path(inspect.getfile(NLayerDiscriminator)).resolve(),
        expected_discriminator,
        "discriminator",
    )

    config = OmegaConf.load(config_path)
    reference = OmegaConf.load(reference_path)
    require_equal(
        OmegaConf.to_container(config.arch.ddconfig, resolve=True),
        OmegaConf.to_container(reference.arch.ddconfig, resolve=True),
        "encoder/decoder config",
    )
    require_equal(
        OmegaConf.to_container(config.gan.disc.arch, resolve=True),
        OmegaConf.to_container(reference.gan.disc.arch, resolve=True),
        "PatchGAN architecture",
    )
    require_equal(
        OmegaConf.to_container(config.gan.loss, resolve=True),
        OmegaConf.to_container(reference.gan.loss, resolve=True),
        "GAN loss and start dynamics",
    )

    checks = {
        "arch.type": (config.arch.type, "rq-vae"),
        "bottleneck": (config.arch.hparams.bottleneck_type, "laser"),
        "embed_dim": (int(config.arch.hparams.embed_dim), 256),
        "n_embed": (int(config.arch.hparams.n_embed), 16384),
        "latent_shape": (list(config.arch.hparams.latent_shape), [8, 8, 256]),
        "code_shape": (
            list(config.arch.hparams.code_shape),
            [8, 8, args.sparsity_level],
        ),
        "sparsity": (
            int(config.arch.hparams.sparsity_level),
            args.sparsity_level,
        ),
        "progressive_loss": (bool(config.arch.hparams.progressive_loss), True),
        "commitment_cost": (float(config.arch.hparams.commitment_cost), 1.0),
        "latent_loss_weight": (float(config.arch.hparams.latent_loss_weight), 0.25),
        "global_batch": (int(config.experiment.total_batch_size), 128),
        "epochs": (int(config.experiment.epochs), 10),
        "checkpoint_frequency": (int(config.experiment.save_ckpt_freq), 1),
        "validation_frequency": (int(config.experiment.test_freq), 1),
        "rfid": (bool(config.experiment.compute_rfid), True),
        "rfid_backend": (config.experiment.rfid_backend, "original-rqvae"),
        "recovery_frequency": (int(config.experiment.recovery_ckpt_freq_steps), 250),
        "amp": (bool(config.experiment.amp), False),
        "generator_lr": (float(config.optimizer.init_lr), 4.0e-5),
        "discriminator_lr": (float(config.gan.disc.optimizer.init_lr), 4.0e-5),
        "dictionary_lr": (float(config.arch.hparams.dict_learning_rate), 4.0e-5),
        "generator_betas": (list(config.optimizer.betas), [0.5, 0.9]),
        "discriminator_betas": (
            list(config.gan.disc.optimizer.betas),
            [0.5, 0.9],
        ),
        "generator_warmup": (float(config.optimizer.warmup.epoch), 0.5),
        "discriminator_warmup": (
            float(config.gan.disc.optimizer.warmup.epoch),
            0.5,
        ),
        "generator_warmup_from_zero": (
            bool(config.optimizer.warmup.start_from_zero),
            True,
        ),
        "discriminator_warmup_from_zero": (
            bool(config.gan.disc.optimizer.warmup.start_from_zero),
            True,
        ),
    }
    for label, (actual, expected) in checks.items():
        require_equal(actual, expected, label)
    require_equal(
        args.local_batch_size * args.world_size,
        128,
        "effective batch",
    )

    learner = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=args.sparsity_level,
        commitment_cost=1.0,
        progressive_loss=True,
    )
    require_equal(
        learner.sparsity_level,
        args.sparsity_level,
        "runtime sparsity",
    )
    require_equal(learner.commitment_cost, 1.0, "runtime commitment weight")
    require_equal(learner.progressive_loss, True, "runtime progressive objective")

    require_source_text(
        rq_root / "rqvae/trainers/trainer_rqvae.py",
        (
            "self.gan_start_epoch = gan_config.loss.disc_start",
            "use_discriminator = True if epoch >= self.gan_start_epoch else False",
            "self.disc_optimizer.step()",
            "self.disc_scheduler.step()",
            "g_weight * self.disc_weight * loss_gen",
        ),
        "discriminator trainer",
    )
    require_source_text(
        rq_root / "rqvae/trainers/trainer.py",
        (
            "best_rfid_slot{rank_idx}_model.pt",
            "'save_top_k': 3",
            "self.writer.upload_checkpoint_files(ckpt_path, best)",
        ),
        "checkpoint retention",
    )
    require_source_text(
        rq_root / "rqvae/utils/writer.py",
        (
            "WANDB_MODE",
            "WANDB_CHECKPOINT_UPLOAD",
            "def upload_checkpoint_files",
            "upload_paths.extend(result_path / item['path'] for item in best)",
        ),
        "W&B checkpoint upload",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "model": "ImageNet LASER Stage 1 matched to RQ-VAE",
                "latent_shape": [8, 8, 256],
                "code_shape": [8, 8, args.sparsity_level],
                "atoms": 16384,
                "sparsity": args.sparsity_level,
                "world_size": args.world_size,
                "local_batch_size": args.local_batch_size,
                "global_batch_size": 128,
                "generator_lr": 4.0e-5,
                "discriminator_lr": 4.0e-5,
                "discriminator": {
                    "layers": 2,
                    "loss": "hinge",
                    "generator_loss": "vanilla",
                    "start_epoch": 0,
                    "updates_per_generator_step": 1,
                    "adaptive_generator_weight": True,
                },
                "checkpoint_policy": "last plus three lowest original-RQ-VAE rFID",
                "wandb_checkpoint_upload": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
