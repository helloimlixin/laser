#!/usr/bin/env python3
"""Fail closed if the frozen Stage-1 duplicate drifts from its specification."""

import argparse
import inspect
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


def require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(8, 16), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    rq_root = source_root / "third_party" / "rq-vae-transformer"
    config_path = args.config.resolve()
    if not rq_root.is_dir() or not config_path.is_file():
        raise RuntimeError(f"missing frozen source or config: {rq_root}, {config_path}")

    sys.path[:0] = [str(rq_root), str(source_root)]
    from rqvae.losses.vqgan.discriminator import NLayerDiscriminator
    from rqvae.models.rqvae.modules import Decoder, Encoder
    from rqvae.models.rqvae.rqvae import RQVAE
    from src.models.dictionary_learner import DictionaryLearning

    expected_modules = (rq_root / "rqvae" / "models" / "rqvae" / "modules.py").resolve()
    expected_rqvae = (rq_root / "rqvae" / "models" / "rqvae" / "rqvae.py").resolve()
    expected_discriminator = (
        rq_root / "rqvae" / "losses" / "vqgan" / "discriminator.py"
    ).resolve()
    require_equal(Path(inspect.getfile(Encoder)).resolve(), expected_modules, "encoder source")
    require_equal(Path(inspect.getfile(Decoder)).resolve(), expected_modules, "decoder source")
    require_equal(Path(inspect.getfile(RQVAE)).resolve(), expected_rqvae, "RQVAE source")
    require_equal(
        Path(inspect.getfile(NLayerDiscriminator)).resolve(),
        expected_discriminator,
        "discriminator source",
    )

    config = OmegaConf.load(config_path)
    checks = {
        "arch.type": (config.arch.type, "rq-vae"),
        "bottleneck": (config.arch.hparams.bottleneck_type, "laser"),
        "embed_dim": (config.arch.hparams.embed_dim, 256),
        "n_embed": (config.arch.hparams.n_embed, 16384),
        "latent_shape": (list(config.arch.hparams.latent_shape), [8, 8, 256]),
        "code_shape": (list(config.arch.hparams.code_shape), [8, 8, 4]),
        "sparsity": (config.arch.hparams.sparsity_level, 4),
        "progressive_loss": (config.arch.hparams.progressive_loss, True),
        "commitment_cost": (float(config.arch.hparams.commitment_cost), 1.0),
        "latent_loss_weight": (float(config.arch.hparams.latent_loss_weight), 0.25),
        "ch_mult": (list(config.arch.ddconfig.ch_mult), [1, 1, 2, 2, 4, 4]),
        "num_res_blocks": (config.arch.ddconfig.num_res_blocks, 2),
        "attention": (list(config.arch.ddconfig.attn_resolutions), [8]),
        "disc_layers": (config.gan.disc.arch.num_layers, 2),
        "disc_ndf": (config.gan.disc.arch.ndf, 64),
        "disc_weight": (float(config.gan.loss.disc_weight), 0.75),
        "perceptual_weight": (float(config.gan.loss.perceptual_weight), 1.0),
        "batch_per_gpu": (config.experiment.batch_size, 32),
        "rfid_backend": (config.experiment.rfid_backend, "original-rqvae"),
    }
    for label, (actual, expected) in checks.items():
        require_equal(actual, expected, label)

    expected_global_batch = 32 * args.world_size
    expected_lr = 4.0e-5 * (expected_global_batch / 128)
    if abs(args.learning_rate - expected_lr) > 1e-12:
        raise RuntimeError(
            f"scaled LR: expected {expected_lr} for global batch "
            f"{expected_global_batch}, got {args.learning_rate}"
        )

    learner = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=4,
        commitment_cost=1.0,
        progressive_loss=True,
    )
    require_equal(learner.commitment_cost, 1.0, "runtime commitment weight")
    require_equal(learner.progressive_loss, True, "runtime progressive objective")

    rqvae_source = expected_rqvae.read_text()
    optimizer_source = (rq_root / "rqvae" / "optimizer" / "optimizer.py").read_text()
    for needle in (
        "_last_bottleneck_objective_for_backward",
        "loss_total = loss_recon + self.latent_loss_weight * loss_latent",
    ):
        if needle not in rqvae_source:
            raise RuntimeError(f"missing LASER/RQ-VAE loss integration: {needle}")
    for needle in ("name': 'dictionary'", "'lr': float(dict_lr)"):
        if needle not in optimizer_source:
            raise RuntimeError(f"missing dictionary optimizer group: {needle}")

    print(
        json.dumps(
            {
                "status": "ok",
                "source_root": str(source_root),
                "encoder_decoder": str(expected_modules),
                "discriminator": str(expected_discriminator),
                "world_size": args.world_size,
                "batch_per_gpu": 32,
                "global_batch": expected_global_batch,
                "generator_lr": args.learning_rate,
                "discriminator_lr": args.learning_rate,
                "dictionary_lr": args.learning_rate,
                "bottleneck_terms": {
                    "dictionary": 1.0,
                    "commitment": 1.0,
                    "outer_latent_weight": 0.25,
                    "progressive_depth_average": True,
                },
                "rfid_backend": "original-rqvae",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
