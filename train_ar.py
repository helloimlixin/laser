"""
Training entrypoint for autoregressive models on LASER-derived tokens.

Supports two maintained paths:
1. `pattern`: legacy pattern-quantizer AR transformer.
2. `sparse_spatial_depth` / `sparse_mingpt`: quantized sparse-token priors
   trained from a cached token grid.
"""

import math
import os
import warnings
from datetime import datetime
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings("ignore", message=".*TF32.*")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

from src.hydra_argparse_compat import patch_argparse_for_hydra_on_py314

patch_argparse_for_hydra_on_py314()
import hydra

from src.lightning_warning_filters import register as register_lightning_warning_filters

register_lightning_warning_filters()
import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src.data.celeba import CelebADataModule
from src.data.cifar10 import CIFAR10DataModule
from src.data.config import DataConfig
from src.data.imagenette2 import Imagenette2DataModule
from src.data.pattern_dataset import PatternDataModule
from src.data.token_cache import TokenCacheDataModule
from src.pl_trainer_util import resolve_val_check_interval
from src.models.ar_transformer import ARTransformer
from src.models.laser import LASER
from src.models.sparse_token_prior import (
    SparseTokenPriorModule,
    build_sparse_prior_from_cache,
    infer_sparse_vocab_sizes,
)

torch.set_float32_matmul_precision("medium")

progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="green1",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    ),
    leave=True,
)


def _build_base_datamodule(cfg: DictConfig):
    if cfg.data.dataset == "cifar10":
        return CIFAR10DataModule(DataConfig.from_dict(cfg.data))
    if cfg.data.dataset == "imagenette2":
        return Imagenette2DataModule(DataConfig.from_dict(cfg.data))
    if cfg.data.dataset == "celeba":
        return CelebADataModule(DataConfig.from_dict(cfg.data))
    raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")


def _to_tuple(val):
    if isinstance(val, (list, tuple)):
        return int(val[0]), int(val[1])
    return int(val), int(val)


def _pattern_expected_seq_len(cfg: DictConfig, laser_model: LASER) -> int:
    patch_h, patch_w = _to_tuple(laser_model.hparams.patch_size)
    patch_stride_cfg = getattr(laser_model.hparams, "patch_stride", None)
    stride_h, stride_w = _to_tuple(patch_stride_cfg) if patch_stride_cfg is not None else (patch_h, patch_w)
    latent_h = math.ceil(cfg.data.image_size / 4)
    latent_w = math.ceil(cfg.data.image_size / 4)
    pad_h = (patch_h - (latent_h % patch_h)) % patch_h
    pad_w = (patch_w - (latent_w % patch_w)) % patch_w
    padded_h = latent_h + pad_h
    padded_w = latent_w + pad_w
    n_h = (padded_h - patch_h) // stride_h + 1
    n_w = (padded_w - patch_w) // stride_w + 1
    return n_h * n_w


def _build_pattern_model(cfg: DictConfig):
    if not cfg.laser_ckpt:
        raise ValueError("pattern mode requires laser_ckpt")

    print(f"\nLoading LASER model from: {cfg.laser_ckpt}")
    laser_model = LASER.load_from_checkpoint(cfg.laser_ckpt, map_location="cpu")
    laser_model.eval()

    if not laser_model.use_pattern_quantizer:
        raise ValueError(
            "LASER bottleneck quantization is disabled. Use the external sparse-code "
            "+ k-means quantization pipeline to create patterns before AR training."
        )

    resolved_vocab_size = int(laser_model.bottleneck.num_patterns)
    if cfg.ar.vocab_size is None or int(cfg.ar.vocab_size) != resolved_vocab_size:
        print(
            f"Adjusting AR vocab_size from {cfg.ar.vocab_size} to {resolved_vocab_size} "
            "to match the LASER pattern vocabulary"
        )
        cfg.ar.vocab_size = resolved_vocab_size

    datamodule = PatternDataModule(
        laser_checkpoint=cfg.laser_ckpt,
        base_datamodule=_build_base_datamodule(cfg),
        batch_size=cfg.train_ar.batch_size,
        num_workers=cfg.data.num_workers,
        cache=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    expected_seq_len = _pattern_expected_seq_len(cfg, laser_model)
    if cfg.ar.seq_len != expected_seq_len:
        print(f"Adjusting AR seq_len from {cfg.ar.seq_len} to {expected_seq_len} to match the patch grid")
        cfg.ar.seq_len = expected_seq_len

    model = ARTransformer(
        vocab_size=cfg.ar.vocab_size,
        seq_len=cfg.ar.seq_len,
        d_model=cfg.ar.d_model,
        n_heads=cfg.ar.n_heads,
        n_layers=cfg.ar.n_layers,
        d_ff=cfg.ar.d_ff,
        dropout=cfg.ar.dropout,
        learning_rate=cfg.ar.learning_rate,
        weight_decay=cfg.ar.weight_decay,
        warmup_steps=cfg.ar.warmup_steps,
        max_steps=cfg.ar.max_steps,
        use_bos=cfg.ar.use_bos,
        use_eos=cfg.ar.use_eos,
    )
    model.set_laser_model(laser_model, log_images_every_n_epochs=1, num_samples=8)
    return model, datamodule


def _build_sparse_model(cfg: DictConfig, mode: str):
    if not cfg.token_cache_path:
        raise ValueError("Sparse prior modes require token_cache_path")

    datamodule = TokenCacheDataModule(
        cache_path=cfg.token_cache_path,
        batch_size=cfg.train_ar.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
        validation_fraction=getattr(cfg.train_ar, "validation_split", 0.05),
        test_fraction=getattr(cfg.train_ar, "test_split", 0.05),
        max_items=getattr(cfg.train_ar, "max_items", 0),
    )
    datamodule.setup("fit")
    architecture = "spatial_depth" if mode == "sparse_spatial_depth" else "mingpt"
    total_vocab_size, atom_vocab_size, coeff_vocab_size = infer_sparse_vocab_sizes(
        datamodule.cache,
        total_vocab_size=cfg.ar.vocab_size,
        atom_vocab_size=cfg.ar.atom_vocab_size,
        coeff_vocab_size=cfg.ar.coeff_vocab_size,
    )
    if cfg.ar.vocab_size != total_vocab_size:
        print(f"Adjusting sparse vocab_size from {cfg.ar.vocab_size} to {total_vocab_size} from cache metadata")
        cfg.ar.vocab_size = total_vocab_size
    if cfg.ar.atom_vocab_size != atom_vocab_size:
        print(f"Resolved atom_vocab_size = {atom_vocab_size}")
        cfg.ar.atom_vocab_size = atom_vocab_size
    if cfg.ar.coeff_vocab_size != coeff_vocab_size:
        print(f"Resolved coeff_vocab_size = {coeff_vocab_size}")
        cfg.ar.coeff_vocab_size = coeff_vocab_size

    prior = build_sparse_prior_from_cache(
        datamodule.cache,
        architecture=architecture,
        total_vocab_size=total_vocab_size,
        atom_vocab_size=atom_vocab_size,
        coeff_vocab_size=coeff_vocab_size,
        d_model=cfg.ar.d_model,
        n_heads=cfg.ar.n_heads,
        n_layers=cfg.ar.n_layers,
        d_ff=cfg.ar.d_ff,
        dropout=cfg.ar.dropout,
        n_global_spatial_tokens=cfg.ar.n_global_spatial_tokens,
    )

    model = SparseTokenPriorModule(
        prior=prior,
        learning_rate=cfg.ar.learning_rate,
        weight_decay=cfg.ar.weight_decay,
        warmup_steps=cfg.ar.warmup_steps,
        min_lr_ratio=cfg.ar.min_lr_ratio,
        atom_loss_weight=cfg.ar.atom_loss_weight,
        coeff_loss_weight=cfg.ar.coeff_loss_weight,
        coeff_depth_weighting=cfg.ar.coeff_depth_weighting,
        coeff_focal_gamma=cfg.ar.coeff_focal_gamma,
    )
    return model, datamodule


@hydra.main(config_path="configs", config_name="config_ar", version_base="1.2")
def train_ar(cfg: DictConfig):
    mode = str(getattr(cfg.ar, "type", "pattern")).strip().lower()

    print("\n" + "=" * 60)
    print("TOKEN MODEL TRAINING")
    print("=" * 60)
    print(f"\nMode: {mode}")
    if mode == "pattern":
        print(f"LASER Checkpoint: {cfg.laser_ckpt}")
    else:
        print(f"Token Cache: {cfg.token_cache_path}")

    print("\nModel Config:")
    print(f"  Vocab Size: {cfg.ar.vocab_size}")
    print(f"  Model Dim: {cfg.ar.d_model}")
    print(f"  Heads: {cfg.ar.n_heads}")
    print(f"  Layers: {cfg.ar.n_layers}")
    print(f"  FF Dim: {cfg.ar.d_ff}")
    print(f"  Dropout: {cfg.ar.dropout}")
    if mode == "pattern":
        print(f"  Sequence Length: {cfg.ar.seq_len}")
    else:
        print(f"  Atom Vocab Size: {cfg.ar.atom_vocab_size}")
        print(f"  Coeff Vocab Size: {cfg.ar.coeff_vocab_size}")
        print(f"  Global Spatial Tokens: {cfg.ar.n_global_spatial_tokens}")

    print("\nTraining Config:")
    print(f"  Learning Rate: {cfg.ar.learning_rate}")
    print(f"  Warmup Steps: {cfg.ar.warmup_steps}")
    print(f"  Max Steps: {cfg.ar.max_steps}")
    print(f"  Batch Size: {cfg.train_ar.batch_size}")
    print(f"  Max Epochs: {cfg.train_ar.max_epochs}")
    print("=" * 60 + "\n")

    pl.seed_everything(cfg.seed)

    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    if mode == "pattern":
        model, datamodule = _build_pattern_model(cfg)
    elif mode in {"sparse_spatial_depth", "sparse_mingpt"}:
        model, datamodule = _build_sparse_model(cfg, mode)
        print(f"Sparse token grid shape: {datamodule.token_shape}")
        if cfg.laser_ckpt:
            print("Sparse prior mode ignores laser_ckpt for visualization because the maintained src LASER path does not decode sparse token caches yet.")
    else:
        raise ValueError(f"Unsupported ar.type: {cfg.ar.type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = "ar" if mode == "pattern" else mode
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"{run_prefix}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint directory: {ckpt_dir}")

    if mode == "pattern":
        run_name = f"ar_transformer_{cfg.data.dataset}_{timestamp}"
    else:
        run_name = f"{mode}_{Path(str(cfg.token_cache_path)).stem}_{timestamp}"

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        save_dir=cfg.wandb.save_dir,
    )
    wandb_logger.log_hyperparams(
        {
            "training_mode": mode,
            "laser_ckpt": cfg.laser_ckpt,
            "token_cache_path": cfg.token_cache_path,
            "ar_vocab_size": cfg.ar.vocab_size,
            "ar_seq_len": getattr(cfg.ar, "seq_len", None),
            "ar_d_model": cfg.ar.d_model,
            "ar_n_heads": cfg.ar.n_heads,
            "ar_n_layers": cfg.ar.n_layers,
            "ar_d_ff": cfg.ar.d_ff,
            "ar_dropout": cfg.ar.dropout,
            "dataset": cfg.data.dataset,
        }
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="ar-{epoch:03d}-{val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        progress_bar,
    ]

    val_check_interval = resolve_val_check_interval(
        datamodule, getattr(cfg.train_ar, "val_check_interval", 1.0)
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train_ar.max_epochs,
        accelerator=cfg.train_ar.accelerator,
        devices=cfg.train_ar.devices,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=cfg.train_ar.precision,
        gradient_clip_val=cfg.train_ar.gradient_clip_val,
        log_every_n_steps=cfg.train_ar.log_every_n_steps,
        val_check_interval=val_check_interval,
        deterministic=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
    )

    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)

    print("\nRunning test evaluation...")
    trainer.test(model, datamodule=datamodule)

    print("\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    return callbacks[0].best_model_path


if __name__ == "__main__":
    train_ar()
