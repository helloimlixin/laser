"""Train an experimental diffusion prior over real-valued sparse coefficients."""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

if sys.version_info < (3, 10):
    raise SystemExit(
        "ERROR: train_stage2_diffusion_prior.py requires Python >= 3.10. "
        "Set PYTHON_BIN to a supported environment or run through scripts/run.sh."
    )

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision.utils import save_image

from src.data.token_cache import TokenCacheDataModule
from src.models.sparse_diffusion_prior import (
    SparseCoeffDiffusionModule,
    build_sparse_coeff_diffusion_prior_from_cache,
)
from src.stage2_compat import (
    decode_stage2_outputs,
    ensure_stage2_cache_metadata,
    load_stage1_decoder_bundle,
)
from src.stage2_metrics import build_stage2_metrics_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a support-conditional DDPM prior over real-valued LASER sparse coefficients. "
            "The token cache must contain coeffs_flat, i.e. be extracted with --coeff-bins 0."
        )
    )
    parser.add_argument("--token-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffusion"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--validation-split", type=float, default=0.05)
    parser.add_argument("--test-split", type=float, default=0.05)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--crop-h-sites", type=int, default=0)
    parser.add_argument("--crop-w-sites", type=int, default=0)

    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--atom-embed-dim", type=int, default=16)
    parser.add_argument("--time-embed-dim", type=int, default=128)
    parser.add_argument("--n-res-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1.0e-4)
    parser.add_argument("--beta-end", type=float, default=2.0e-2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)

    parser.add_argument("--stats-items", type=int, default=8192)
    parser.add_argument("--support-bank-size", type=int, default=512)
    parser.add_argument("--sample-num-images", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=1.0)

    # W&B logging (training loss + post-fit generation FID). Without --wandb-project
    # the trainer keeps its previous no-logger behaviour.
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-save-dir", type=str, default="")

    # Post-fit generation FID (image datasets only; reuses the AR prior's
    # src/stage2_metrics.build_stage2_metrics_payload, currently celeba/celebahq).
    parser.add_argument("--compute-generation-fid", action="store_true")
    parser.add_argument("--generation-metric-num-samples", type=int, default=0)
    parser.add_argument("--fid-dataset", type=str, default="")
    parser.add_argument("--fid-data-dir", type=str, default="")
    parser.add_argument("--fid-image-size", type=int, default=256)
    parser.add_argument("--fid-chunk-size", type=int, default=32)
    return parser.parse_args()


def _dataset_item_tensors(item) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(item, (tuple, list)) or len(item) < 2:
        raise ValueError("Expected dataset items to contain tokens and real-valued coefficients")
    tokens = item[0]
    coeffs = item[1]
    if not torch.is_tensor(tokens) or not torch.is_tensor(coeffs):
        raise ValueError("Expected token and coefficient tensors in the token-cache dataset")
    return tokens, coeffs


def _select_indices(total: int, limit: int, *, seed: int) -> list[int]:
    total = int(total)
    limit = total if int(limit) <= 0 else min(total, int(limit))
    if limit >= total:
        return list(range(total))
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(total, generator=generator)[:limit].tolist()


def _estimate_coeff_stats(dataset, *, max_items: int, seed: int) -> tuple[float, float]:
    coeff_chunks = []
    for idx in _select_indices(len(dataset), max_items, seed=seed):
        _, coeffs = _dataset_item_tensors(dataset[idx])
        coeff_chunks.append(coeffs.reshape(-1).to(torch.float32))
    if not coeff_chunks:
        raise ValueError("Cannot estimate coefficient stats from an empty training dataset")
    coeffs_flat = torch.cat(coeff_chunks, dim=0)
    mean = float(coeffs_flat.mean().item())
    std = float(coeffs_flat.std(unbiased=False).clamp_min(1.0e-6).item())
    return mean, std


def _build_support_bank(dataset, *, max_items: int, seed: int) -> torch.Tensor:
    tokens = []
    for idx in _select_indices(len(dataset), max_items, seed=seed):
        item_tokens, _ = _dataset_item_tensors(dataset[idx])
        tokens.append(item_tokens.reshape(-1).to(torch.long))
    if not tokens:
        return torch.empty(0, 0, dtype=torch.long)
    return torch.stack(tokens, dim=0).contiguous()


def _devices_arg(raw: str):
    text = str(raw).strip().lower()
    if text in {"", "auto"}:
        return "auto"
    try:
        return int(text)
    except ValueError:
        return raw


def _sample_preview(
    model: SparseCoeffDiffusionModule,
    cache: dict,
    *,
    token_cache_path: Path,
    output_root: Path,
    out_dir: Path,
    sample_num_images: int,
    sample_steps: Optional[int],
    sample_temperature: float,
    device: torch.device,
) -> Optional[Path]:
    if int(sample_num_images) <= 0:
        return None
    try:
        s1 = load_stage1_decoder_bundle(
            cache,
            token_cache_path=token_cache_path,
            device=device,
            output_root=output_root,
        )
    except Exception as exc:
        print(f"Skipping decoded diffusion preview because stage-1 decoder loading failed: {exc}")
        return None

    model = model.eval().to(device)
    atoms, coeffs = model.generate_sparse_codes(
        int(sample_num_images),
        temperature=float(sample_temperature),
        steps=sample_steps,
    )
    cfg = model.prior.cfg
    atom_grid = atoms.view(int(sample_num_images), int(cfg.H), int(cfg.W), int(cfg.D))
    coeff_grid = coeffs.view(int(sample_num_images), int(cfg.H), int(cfg.W), int(cfg.D))
    with torch.no_grad():
        images = decode_stage2_outputs(s1, atom_grid, coeff_grid, device=device).detach().cpu()

    out_dir.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.ceil(math.sqrt(int(sample_num_images)))))
    png_path = out_dir / "diffusion_samples_final.png"
    save_image(images, png_path, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))
    torch.save(
        {
            "atom_ids": atom_grid.detach().cpu(),
            "coeffs": coeff_grid.detach().cpu(),
            "images": images,
            "shape": (int(cfg.H), int(cfg.W), int(cfg.D)),
            "token_cache": str(token_cache_path),
            "stage1_checkpoint": str(s1.checkpoint_path),
        },
        out_dir / "diffusion_samples_final.pt",
    )
    return png_path


def _generate_decoded_images(
    model: SparseCoeffDiffusionModule,
    cache: dict,
    *,
    token_cache_path: Path,
    output_root: Path,
    n: int,
    sample_steps: Optional[int],
    sample_temperature: float,
    device: torch.device,
    chunk: int = 32,
) -> Optional[torch.Tensor]:
    """Generate + decode ``n`` images for FID, chunked to bound decoder memory."""
    if int(n) <= 0:
        return None
    s1 = load_stage1_decoder_bundle(
        cache,
        token_cache_path=token_cache_path,
        device=device,
        output_root=output_root,
    )
    model = model.eval().to(device)
    cfg = model.prior.cfg
    out: list[torch.Tensor] = []
    remaining = int(n)
    while remaining > 0:
        b = min(max(1, int(chunk)), remaining)
        atoms, coeffs = model.generate_sparse_codes(
            b, temperature=float(sample_temperature), steps=sample_steps
        )
        atom_grid = atoms.view(b, int(cfg.H), int(cfg.W), int(cfg.D))
        coeff_grid = coeffs.view(b, int(cfg.H), int(cfg.W), int(cfg.D))
        with torch.no_grad():
            imgs = decode_stage2_outputs(s1, atom_grid, coeff_grid, device=device).detach().cpu()
        out.append(imgs)
        remaining -= b
    return torch.cat(out, dim=0) if out else None


def main() -> None:
    args = _parse_args()
    pl.seed_everything(int(args.seed), workers=True)

    token_cache_path = args.token_cache_path.expanduser().resolve()
    run_name = "stage2-diffusion-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.expanduser().resolve() / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("STAGE 2: SPARSE COEFFICIENT DIFFUSION PRIOR")
    print("=" * 64)
    print(f"Token cache: {token_cache_path}")
    print(f"Run dir:     {run_dir}")

    dm = TokenCacheDataModule(
        cache_path=str(token_cache_path),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        validation_fraction=float(args.validation_split),
        test_fraction=float(args.test_split),
        max_items=int(args.max_items),
        crop_h_sites=int(args.crop_h_sites),
        crop_w_sites=int(args.crop_w_sites),
    )
    dm.setup("fit")
    cache = ensure_stage2_cache_metadata(
        dm.cache,
        token_cache_path=token_cache_path,
        output_root=args.output_root.expanduser().resolve(),
    )
    dm.cache = cache
    if cache.get("coeffs_flat") is None:
        raise ValueError(
            "Sparse coefficient diffusion requires a real-valued cache with coeffs_flat. "
            "Rebuild the cache with cache.py --coeff-bins 0."
        )

    coeff_mean, coeff_std = _estimate_coeff_stats(
        dm.train_dataset,
        max_items=int(args.stats_items),
        seed=int(args.seed),
    )
    support_bank = _build_support_bank(
        dm.train_dataset,
        max_items=int(args.support_bank_size),
        seed=int(args.seed) + 17,
    )
    print(f"Token shape: {dm.token_shape}")
    print(f"Coeff stats: mean={coeff_mean:.6f}, std={coeff_std:.6f}")
    print(f"Support bank: {tuple(support_bank.shape)}")

    cache_for_build = dict(cache)
    cache_for_build["shape"] = dm.token_shape
    cache_for_build["tokens_flat"] = support_bank
    cache_for_build["coeffs_flat"] = torch.zeros_like(support_bank, dtype=torch.float32)
    prior = build_sparse_coeff_diffusion_prior_from_cache(
        cache_for_build,
        hidden_channels=int(args.hidden_channels),
        atom_embed_dim=int(args.atom_embed_dim),
        time_embed_dim=int(args.time_embed_dim),
        n_res_blocks=int(args.n_res_blocks),
        dropout=float(args.dropout),
        num_timesteps=int(args.num_timesteps),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        support_bank=support_bank,
    )
    model = SparseCoeffDiffusionModule(
        prior,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    if dm.val_dataset is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:03d}",
            save_top_k=-1,
            save_last=True,
        )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]

    # Optional W&B logger so the diffusion run's loss + post-fit FID land in the
    # same project/group as the AR-prior runs. Guarded so a logger failure never
    # kills training (falls back to the previous no-logger behaviour).
    logger = False
    if str(args.wandb_project).strip():
        try:
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(
                project=str(args.wandb_project),
                name=str(args.wandb_name) or None,
                group=str(args.wandb_group) or None,
                save_dir=str(args.wandb_save_dir) or str(run_dir),
            )
        except Exception as exc:  # pragma: no cover - logging is best-effort
            print(f"Warning: could not create W&B logger, continuing without it ({exc})")
            logger = False

    trainer = pl.Trainer(
        max_epochs=int(args.max_epochs),
        max_steps=int(args.max_steps),
        accelerator=str(args.accelerator),
        devices=_devices_arg(args.devices),
        precision=str(args.precision),
        gradient_clip_val=float(args.gradient_clip_val),
        log_every_n_steps=int(args.log_every_n_steps),
        val_check_interval=float(args.val_check_interval),
        limit_train_batches=float(args.limit_train_batches),
        limit_val_batches=float(args.limit_val_batches),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(run_dir),
    )
    trainer.fit(model, datamodule=dm)

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sample_steps = None if int(args.sample_steps) <= 0 else int(args.sample_steps)
    sample_path = _sample_preview(
        model,
        cache,
        token_cache_path=token_cache_path,
        output_root=args.output_root.expanduser().resolve(),
        out_dir=run_dir,
        sample_num_images=int(args.sample_num_images),
        sample_steps=sample_steps,
        sample_temperature=float(args.sample_temperature),
        device=device,
    )
    if sample_path is not None:
        print(f"Saved diffusion sample preview: {sample_path}")

    # Post-fit generation FID (image datasets only). Generates a larger batch than
    # the preview, decodes through the stage-1 decoder, and reuses the AR prior's
    # FID helper so the scalar is comparable to the AR runs' generation/fid.
    fid_n = int(args.generation_metric_num_samples)
    if bool(args.compute_generation_fid) and fid_n > 0 and str(args.fid_data_dir).strip():
        try:
            from omegaconf import OmegaConf

            fid_images = _generate_decoded_images(
                model,
                cache,
                token_cache_path=token_cache_path,
                output_root=args.output_root.expanduser().resolve(),
                n=fid_n,
                sample_steps=sample_steps,
                sample_temperature=float(args.sample_temperature),
                device=device,
                chunk=int(args.fid_chunk_size),
            )
            if fid_images is not None:
                fid_cfg = OmegaConf.create(
                    {
                        "seed": int(args.seed),
                        "data": {
                            "dataset": str(args.fid_dataset),
                            "data_dir": str(args.fid_data_dir),
                            "image_size": int(args.fid_image_size),
                            "num_workers": int(args.num_workers),
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5],
                        },
                        "train_ar": {"batch_size": int(args.fid_chunk_size)},
                    }
                )
                payload = build_stage2_metrics_payload(
                    fid_images,
                    cfg=fid_cfg,
                    cache=cache,
                    max_items=fid_n,
                    compute_fid=True,
                    compute_audio=False,
                )
                if payload:
                    print(f"Diffusion generation metrics (n={fid_n}): {payload}")
                    if logger is not False and logger is not None:
                        try:
                            logger.log_metrics(payload, step=int(trainer.global_step))
                        except Exception as exc:  # pragma: no cover - best effort
                            print(f"Warning: could not log FID to W&B ({exc})")
                else:
                    print(
                        f"Generation FID skipped (dataset='{args.fid_dataset}' is not "
                        "celeba/celebahq, or no real images found)."
                    )
        except Exception as exc:  # pragma: no cover - metrics are best-effort
            print(f"Skipping diffusion generation FID: {exc}")

    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
