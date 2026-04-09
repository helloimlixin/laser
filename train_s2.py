"""
Train the maintained stage-2 sparse prior.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings("ignore", message=".*TF32.*")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

from src.hydra_argparse_compat import patch_argparse_for_hydra_on_py314

patch_argparse_for_hydra_on_py314()
import hydra

from src.lightning_warning_filters import register as reg_lit_warn

reg_lit_warn()
import lightning as pl
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src.data.token_cache import TokenCacheDataModule, load_token_cache
from src.models.sparse_token_prior import (
    SparseTokenPriorModule,
    build_sparse_prior_from_cache,
    infer_sparse_vocab_sizes,
)
from src.pl_trainer_util import resolve_val_check_interval
from src.s2 import sample as sample_s2, save_grid
from src.stage2_compat import (
    ensure_stage2_cache_metadata as add_cache_meta,
    load_stage1_decoder_bundle as load_s1,
)
from src.stage2_paths import infer_latest_token_cache as pick_cache

torch.set_float32_matmul_precision("medium")

bar = RichProgressBar(
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


class SampleCb(pl.Callback):
    def __init__(
        self,
        *,
        cache_pt: str,
        out_dir: str,
        step_every: int = 0,
        epoch_every: int = 0,
        n: int = 4,
        temp: float = 1.0,
        top_k: int = 0,
        ctemp=None,
        cmode: Optional[str] = None,
        s1_root: str = "outputs",
        use_wandb: bool = False,
    ):
        super().__init__()
        self.cache_pt = str(cache_pt)
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.step_every = max(0, int(step_every))
        self.epoch_every = max(0, int(epoch_every))
        self.n = max(1, int(n))
        self.temp = float(temp)
        self.top_k = int(top_k)
        self.ctemp = None if ctemp is None else float(ctemp)
        self.cmode = None if cmode is None else str(cmode).strip().lower()
        self.s1_root = str(s1_root)
        self.use_wandb = bool(use_wandb)
        self._last_step = -1
        self._last_epoch = -1
        self._cache = None
        self._s1 = None
        self._shape = None

    def _ready(self, dev: torch.device):
        if self._cache is None:
            raw = load_token_cache(self.cache_pt)
            self._cache = add_cache_meta(
                raw,
                token_cache_path=self.cache_pt,
                output_root=self.s1_root,
            )
            self._shape = tuple(int(v) for v in self._cache["shape"])
        if self._s1 is None:
            self._s1 = load_s1(
                self._cache,
                token_cache_path=self.cache_pt,
                device=dev,
                output_root=self.s1_root,
            )
        else:
            self._s1.model = self._s1.model.to(dev)

    def _log(self, trainer: pl.Trainer, *, step: int, epoch: Optional[int], raw: Path, auto: Path):
        if not self.use_wandb:
            return
        exp = getattr(getattr(trainer, "logger", None), "experiment", None)
        if exp is None or not hasattr(exp, "log"):
            return
        if epoch is not None:
            step = int(step) + 1
        try:
            step = max(int(step), int(getattr(exp, "step", step)))
        except (TypeError, ValueError):
            step = int(step)
        try:
            step = max(int(step), int(getattr(exp, "_step", step)))
        except (TypeError, ValueError):
            step = int(step)

        bits = [f"step={step}"]
        if epoch is not None:
            bits.append(f"epoch={epoch}")
        cap = " ".join(bits)
        exp.log(
            {
                "s2/raw": wandb.Image(str(raw), caption=cap),
                "s2/auto": wandb.Image(str(auto), caption=cap),
                "s2/epoch": epoch,
            },
            step=step,
        )

    @torch.no_grad()
    def _save(self, trainer: pl.Trainer, mod: pl.LightningModule, *, step: int, epoch: Optional[int] = None):
        dev = mod.device
        self._ready(dev)
        batch = sample_s2(
            mod,
            self._s1,
            self._shape,
            n=self.n,
            temp=self.temp,
            top_k=self.top_k,
            ctemp=self.ctemp,
            cmode=self.cmode,
            dev=dev,
        )

        stem = f"s{step:07d}" if epoch is None else f"e{epoch:03d}_s{step:07d}"
        raw, auto = save_grid(batch.imgs, self.out_dir, stem=stem)
        self._log(trainer, step=step, epoch=epoch, raw=raw, auto=auto)
        if epoch is None:
            print(f"Saved s2 samples at step {step}: {raw}")
        else:
            print(f"Saved s2 samples at epoch {epoch}, step {step}: {raw}")

    def on_train_batch_end(self, trainer, mod, outputs, batch, batch_idx):
        if self.step_every <= 0 or not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_step or (step % self.step_every) != 0:
            return
        self._save(trainer, mod, step=step)
        self._last_step = step

    def on_train_epoch_end(self, trainer, mod):
        if self.epoch_every <= 0 or not trainer.is_global_zero:
            return
        epoch = int(trainer.current_epoch) + 1
        step = int(trainer.global_step)
        if epoch <= 0 or epoch == self._last_epoch or (epoch % self.epoch_every) != 0:
            return
        if step <= 0 or step == self._last_step:
            return
        self._save(trainer, mod, step=step, epoch=epoch)
        self._last_epoch = epoch
        self._last_step = step


def arch_name(raw) -> str:
    text = str(raw or "sparse_spatial_depth").strip().lower()
    if text in {"sparse_spatial_depth", "spatial_depth"}:
        return "spatial_depth"
    if text in {"mingpt", "gpt"}:
        return "gpt"
    raise ValueError(f"Unsupported stage-2 architecture: {raw!r}")


def build(cfg: DictConfig):
    if not cfg.token_cache_path:
        cache_pt = pick_cache(ar_output_dir=cfg.output_dir)
        if cache_pt is None:
            need = Path(str(cfg.output_dir)).expanduser().resolve() / "token_cache"
            raise ValueError(
                f"Sparse prior runs require token_cache_path, and no cache could be inferred under {need}"
            )
        cfg.token_cache_path = str(cache_pt)
        print(f"Inferred token_cache_path: {cfg.token_cache_path}")

    dm = TokenCacheDataModule(
        cache_path=cfg.token_cache_path,
        batch_size=cfg.train_ar.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
        validation_fraction=getattr(cfg.train_ar, "validation_split", 0.05),
        test_fraction=getattr(cfg.train_ar, "test_split", 0.05),
        max_items=getattr(cfg.train_ar, "max_items", 0),
        crop_h_sites=getattr(cfg.train_ar, "crop_h_sites", 0),
        crop_w_sites=getattr(cfg.train_ar, "crop_w_sites", 0),
    )
    dm.setup("fit")
    dm.cache = add_cache_meta(
        dm.cache,
        token_cache_path=cfg.token_cache_path,
        output_root=Path(str(cfg.output_dir)).expanduser().resolve().parent,
    )

    arch = arch_name(getattr(cfg.ar, "type", "sparse_spatial_depth"))
    real = dm.cache.get("coeffs_flat") is not None
    if real and arch != "spatial_depth":
        raise ValueError("Real-valued sparse-token caches are only supported with ar.type=sparse_spatial_depth.")
    if arch == "gpt":
        H, W, D = dm.token_shape
        seq_len = int(H * W * D)
        window_sites = int(getattr(cfg.ar, "window_sites", 0) or 0)
        if seq_len > 8192 and window_sites <= 0:
            raise ValueError(
                "ar.type=gpt without ar.window_sites is a full-sequence GPT over the flattened sparse token stream and "
                f"is not practical for sequence length {seq_len} from token shape {(H, W, D)}. "
                "Use a much shorter token grid, set ar.window_sites for local sliding-window attention, "
                "or keep ar.type=sparse_spatial_depth for this cache."
            )

    prior = build_sparse_prior_from_cache(
        dm.cache,
        architecture=arch,
        total_vocab_size=cfg.ar.vocab_size,
        atom_vocab_size=cfg.ar.atom_vocab_size,
        coeff_vocab_size=cfg.ar.coeff_vocab_size,
        grid_shape=dm.token_shape,
        window_sites=getattr(cfg.ar, "window_sites", 0),
        d_model=cfg.ar.d_model,
        n_heads=cfg.ar.n_heads,
        n_layers=cfg.ar.n_layers,
        d_ff=cfg.ar.d_ff,
        dropout=cfg.ar.dropout,
        n_global_spatial_tokens=cfg.ar.n_global_spatial_tokens,
        autoregressive_coeffs=cfg.ar.autoregressive_coeffs,
    )
    if real:
        atom_vocab = int(prior.atom_vocab_size)
        vocab = int(prior.cfg.vocab_size)
        coeff_vocab = 0
    else:
        vocab, atom_vocab, coeff_vocab = infer_sparse_vocab_sizes(
            dm.cache,
            total_vocab_size=cfg.ar.vocab_size,
            atom_vocab_size=cfg.ar.atom_vocab_size,
            coeff_vocab_size=cfg.ar.coeff_vocab_size,
        )
    if cfg.ar.vocab_size != vocab:
        print(f"Adjusting sparse vocab_size from {cfg.ar.vocab_size} to {vocab} from cache metadata")
        cfg.ar.vocab_size = vocab
    if cfg.ar.atom_vocab_size != atom_vocab:
        print(f"Resolved atom_vocab_size = {atom_vocab}")
        cfg.ar.atom_vocab_size = atom_vocab
    coeff_cfg = None if real else int(coeff_vocab)
    if cfg.ar.coeff_vocab_size != coeff_cfg:
        print(f"Resolved coeff_vocab_size = {coeff_cfg}")
        cfg.ar.coeff_vocab_size = coeff_cfg

    coeff_type = getattr(cfg.ar, "coeff_loss_type", "auto")
    coeff_key = "" if coeff_type is None else str(coeff_type).strip().lower()
    if real and coeff_key == "gt_atom_recon_mse" and bool(getattr(cfg.ar, "autoregressive_coeffs", True)):
        warnings.warn(
            "ar.coeff_loss_type='gt_atom_recon_mse' on a real-valued autoregressive stage-2 prior "
            "optimizes coefficients against ground-truth atom supports but samples with predicted "
            "supports at generation time. This mismatch can produce stripe/zebra artifacts in "
            "unconditional samples; prefer ar.coeff_loss_type='recon_mse' for generation-focused runs."
        )

    try:
        s1 = load_s1(
            dm.cache,
            token_cache_path=cfg.token_cache_path,
            device="cpu",
            output_root=Path(str(cfg.output_dir)).expanduser().resolve().parent,
        )
    except Exception as err:
        s1 = None
        if real and coeff_key in {"recon_mse", "gt_atom_recon_mse"}:
            raise
        print(f"Warning: could not load stage-1 decoder bundle ({err}); recon viz will be disabled.")

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
        coeff_loss_type=cfg.ar.coeff_loss_type,
        coeff_huber_delta=cfg.ar.coeff_huber_delta,
        sample_coeff_temperature=cfg.ar.sample_coeff_temperature,
        sample_coeff_mode=cfg.ar.sample_coeff_mode,
        stage1_decoder_bundle=s1,
        log_recon_every_n_steps=int(getattr(cfg.train_ar, "log_recon_every_n_steps", 500) or 0),
    )
    model.save_hyperparameters(
        {
            "token_cache_path": str(Path(str(cfg.token_cache_path)).expanduser().resolve()),
            "resolved_atom_vocab_size": int(atom_vocab),
            "resolved_coeff_vocab_size": int(coeff_vocab),
            "resolved_total_vocab_size": int(vocab),
            "token_cache_real_valued": bool(real),
        }
    )
    return model, dm


@hydra.main(config_path="configs", config_name="config_ar", version_base="1.2")
def main(cfg: DictConfig):
    mode = arch_name(getattr(cfg.ar, "type", "sparse_spatial_depth"))
    cfg.ar.type = mode

    print("\n" + "=" * 60)
    print("S2 TRAINING")
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
    print(f"  Window Sites: {getattr(cfg.ar, 'window_sites', 0)}")
    print(f"  Global Spatial Tokens: {cfg.ar.n_global_spatial_tokens}")
    print(f"  Autoregressive Coeffs: {cfg.ar.autoregressive_coeffs}")
    print(f"  Coeff Loss Type: {cfg.ar.coeff_loss_type}")

    print("\nTrain Config:")
    print(f"  Learning Rate: {cfg.ar.learning_rate}")
    print(f"  Warmup Steps: {cfg.ar.warmup_steps}")
    print(f"  Max Steps: {cfg.ar.max_steps}")
    print(f"  Batch Size: {cfg.train_ar.batch_size}")
    print(f"  Max Epochs: {cfg.train_ar.max_epochs}")
    print(f"  Crop H Sites: {getattr(cfg.train_ar, 'crop_h_sites', 0)}")
    print(f"  Crop W Sites: {getattr(cfg.train_ar, 'crop_w_sites', 0)}")
    print("=" * 60 + "\n")

    pl.seed_everything(cfg.seed)

    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    model, dm = build(cfg)
    print(f"S2 grid shape: {dm.token_shape}")
    cache_meta = dm.metadata
    cache_dataset = str(cache_meta.get("dataset") or "").strip()
    cfg_dataset = str(getattr(cfg.data, "dataset", "") or "").strip()
    if cache_dataset:
        if cfg_dataset and cfg_dataset.lower() != cache_dataset.lower():
            print(f"Stage-2 cache dataset: {cache_dataset} (ignoring data.dataset={cfg_dataset})")
        else:
            print(f"Stage-2 cache dataset: {cache_dataset}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"s2_{stamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint dir: {ckpt_dir}")

    run = str(getattr(cfg.wandb, "name", "") or "").strip() or f"s2_{stamp}"
    wb = WandbLogger(
        project=cfg.wandb.project,
        name=run,
        save_dir=cfg.wandb.save_dir,
    )
    wb.log_hyperparams(
        {
            "training_mode": "s2",
            "token_cache_path": cfg.token_cache_path,
            "ar_vocab_size": cfg.ar.vocab_size,
            "ar_d_model": cfg.ar.d_model,
            "ar_n_heads": cfg.ar.n_heads,
            "ar_n_layers": cfg.ar.n_layers,
            "ar_d_ff": cfg.ar.d_ff,
            "ar_dropout": cfg.ar.dropout,
            "dataset": cache_dataset or cfg_dataset or None,
            "config_dataset": cfg_dataset or None,
        }
    )

    cbs = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="s2-{epoch:03d}-{val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        bar,
    ]
    step_every = int(getattr(cfg.train_ar, "sample_every_n_steps", 0) or 0)
    epoch_every = int(getattr(cfg.train_ar, "sample_every_n_epochs", 0) or 0)
    if step_every > 0 or epoch_every > 0:
        cbs.append(
            SampleCb(
                cache_pt=str(cfg.token_cache_path),
                out_dir=os.path.join(cfg.output_dir, "samples", f"s2_{stamp}"),
                step_every=step_every,
                epoch_every=epoch_every,
                n=int(getattr(cfg.train_ar, "sample_num_images", 4) or 4),
                temp=float(getattr(cfg.train_ar, "sample_temperature", 1.0) or 1.0),
                top_k=int(getattr(cfg.train_ar, "sample_top_k", 0) or 0),
                ctemp=getattr(cfg.train_ar, "sample_coeff_temperature", cfg.ar.sample_coeff_temperature),
                cmode=getattr(cfg.train_ar, "sample_coeff_mode", cfg.ar.sample_coeff_mode),
                s1_root=str(Path(str(cfg.output_dir)).expanduser().resolve().parent),
                use_wandb=bool(getattr(cfg.train_ar, "sample_log_to_wandb", False)),
            )
        )

    val_every = resolve_val_check_interval(dm, getattr(cfg.train_ar, "val_check_interval", 1.0))
    trainer = pl.Trainer(
        max_epochs=cfg.train_ar.max_epochs,
        accelerator=cfg.train_ar.accelerator,
        devices=cfg.train_ar.devices,
        strategy=getattr(cfg.train_ar, "strategy", "auto"),
        logger=wb,
        callbacks=cbs,
        precision=cfg.train_ar.precision,
        gradient_clip_val=cfg.train_ar.gradient_clip_val,
        log_every_n_steps=cfg.train_ar.log_every_n_steps,
        val_check_interval=val_every,
        deterministic=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
    )

    print("\nStarting training...")
    trainer.fit(model, datamodule=dm)

    print("\nRunning test evaluation...")
    trainer.test(model, datamodule=dm)

    print("\nTraining complete!")
    print(f"Best checkpoint: {cbs[0].best_model_path}")
    return cbs[0].best_model_path


train_ar = main


if __name__ == "__main__":
    main()
