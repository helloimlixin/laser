"""
Training entrypoint for autoregressive models on LASER-derived tokens.

Supports the maintained sparse-token prior paths:
1. `sparse_spatial_depth`
2. `sparse_mingpt`
"""

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
from torchvision.utils import save_image

from src.data.token_cache import TokenCacheDataModule, load_token_cache
from src.pl_trainer_util import resolve_val_check_interval
from src.models.laser import LASER
from src.models.sparse_token_prior import (
    SparseTokenPriorModule,
    build_sparse_prior_from_cache,
    infer_sparse_vocab_sizes,
    token_cache_grid_shape,
)
from src.stage2_paths import infer_latest_token_cache

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


class PeriodicStage2SamplingCallback(pl.Callback):
    def __init__(
        self,
        *,
        token_cache_path: str,
        sample_dir: str,
        every_n_steps: int = 0,
        num_samples: int = 4,
        temperature: float = 1.0,
        top_k: int = 0,
    ):
        super().__init__()
        self.token_cache_path = str(token_cache_path)
        self.sample_dir = Path(sample_dir).expanduser().resolve()
        self.every_n_steps = max(0, int(every_n_steps))
        self.num_samples = max(1, int(num_samples))
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self._last_sampled_step = -1
        self._cache = None
        self._stage1 = None

    def _ensure_ready(self, device: torch.device):
        if self._cache is None:
            self._cache = load_token_cache(self.token_cache_path)
        if self._stage1 is None:
            meta = self._cache.get("meta", {}) if isinstance(self._cache, dict) else {}
            stage1_checkpoint = meta.get("stage1_checkpoint")
            if not stage1_checkpoint:
                raise RuntimeError(
                    f"Token cache {self.token_cache_path} is missing meta.stage1_checkpoint for periodic sampling."
                )
            self._stage1 = LASER.load_from_checkpoint(stage1_checkpoint, map_location="cpu").eval().to(device)
        else:
            self._stage1 = self._stage1.to(device)

    @torch.no_grad()
    def _sample_and_save(self, trainer: pl.Trainer, pl_module: pl.LightningModule, step: int):
        device = pl_module.device
        self._ensure_ready(device)
        cache = self._cache
        meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
        token_h, token_w, token_depth = token_cache_grid_shape(cache)
        coeff_vocab_size = int(meta.get("coeff_vocab_size") or meta.get("n_bins") or 0)
        coeff_bin_values = meta.get("coeff_bin_values")
        if coeff_vocab_size <= 0 or coeff_bin_values is None:
            raise RuntimeError(
                f"Token cache {self.token_cache_path} is missing coefficient quantization metadata for sampling."
            )
        latent_hw = meta.get("latent_hw")
        if isinstance(latent_hw, (tuple, list)) and len(latent_hw) == 2:
            latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
        else:
            latent_hw = None

        top_k = None if self.top_k <= 0 else self.top_k
        token_grid = pl_module.generate_tokens(
            self.num_samples,
            temperature=self.temperature,
            top_k=top_k,
        ).view(self.num_samples, token_h, token_w, token_depth)
        images = self._stage1.decode_from_tokens(
            token_grid.to(device=device, dtype=torch.long),
            latent_hw=latent_hw,
            coeff_vocab_size=coeff_vocab_size,
            coeff_bin_values=torch.as_tensor(coeff_bin_values, dtype=torch.float32, device=device),
        ).detach().cpu()

        self.sample_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.sample_dir / f"step_{step:07d}.png"
        auto_path = self.sample_dir / f"step_{step:07d}_autocontrast.png"
        nrow = max(1, int(self.num_samples ** 0.5))
        save_image(images, raw_path, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))
        save_image(images, auto_path, nrow=nrow, normalize=True, scale_each=True)
        print(f"Saved stage-2 samples at step {step}: {raw_path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0 or not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_sampled_step or (step % self.every_n_steps) != 0:
            return
        self._sample_and_save(trainer, pl_module, step)
        self._last_sampled_step = step

def _build_sparse_model(cfg: DictConfig, mode: str):
    if not cfg.token_cache_path:
        inferred = infer_latest_token_cache(ar_output_dir=cfg.output_dir)
        if inferred is None:
            raise ValueError(
                f"Sparse prior modes require token_cache_path, and no cache could be inferred under {Path(str(cfg.output_dir)).expanduser().resolve() / 'token_cache'}"
            )
        cfg.token_cache_path = str(inferred)
        print(f"Inferred token_cache_path: {cfg.token_cache_path}")

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
    model.save_hyperparameters(
        {
            "token_cache_path": str(Path(str(cfg.token_cache_path)).expanduser().resolve()),
            "resolved_atom_vocab_size": int(atom_vocab_size),
            "resolved_coeff_vocab_size": int(coeff_vocab_size),
            "resolved_total_vocab_size": int(total_vocab_size),
        }
    )
    return model, datamodule


@hydra.main(config_path="configs", config_name="config_ar", version_base="1.2")
def train_ar(cfg: DictConfig):
    mode = str(getattr(cfg.ar, "type", "sparse_spatial_depth")).strip().lower()

    print("\n" + "=" * 60)
    print("TOKEN MODEL TRAINING")
    print("=" * 60)
    print(f"\nMode: {mode}")
    print(f"Token Cache: {cfg.token_cache_path}")

    print("\nModel Config:")
    print(f"  Vocab Size: {cfg.ar.vocab_size}")
    print(f"  Model Dim: {cfg.ar.d_model}")
    print(f"  Heads: {cfg.ar.n_heads}")
    print(f"  Layers: {cfg.ar.n_layers}")
    print(f"  FF Dim: {cfg.ar.d_ff}")
    print(f"  Dropout: {cfg.ar.dropout}")
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

    if mode in {"sparse_spatial_depth", "sparse_mingpt"}:
        model, datamodule = _build_sparse_model(cfg, mode)
        print(f"Sparse token grid shape: {datamodule.token_shape}")
    else:
        raise ValueError(f"Unsupported ar.type: {cfg.ar.type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"{mode}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint directory: {ckpt_dir}")

    run_name = f"{mode}_{Path(str(cfg.token_cache_path)).stem}_{timestamp}"

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        save_dir=cfg.wandb.save_dir,
    )
    wandb_logger.log_hyperparams(
        {
            "training_mode": mode,
            "token_cache_path": cfg.token_cache_path,
            "ar_vocab_size": cfg.ar.vocab_size,
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
    sample_every_n_steps = int(getattr(cfg.train_ar, "sample_every_n_steps", 0) or 0)
    if sample_every_n_steps > 0:
        callbacks.append(
            PeriodicStage2SamplingCallback(
                token_cache_path=str(cfg.token_cache_path),
                sample_dir=os.path.join(cfg.output_dir, "samples", f"{mode}_{timestamp}"),
                every_n_steps=sample_every_n_steps,
                num_samples=int(getattr(cfg.train_ar, "sample_num_images", 4) or 4),
                temperature=float(getattr(cfg.train_ar, "sample_temperature", 1.0) or 1.0),
                top_k=int(getattr(cfg.train_ar, "sample_top_k", 0) or 0),
            )
        )

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
