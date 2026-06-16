"""Train the maintained stage-2 transformer prior and save generation previews."""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit(
        "ERROR: train_stage2_prior.py requires Python >= 3.10. "
        "Set PYTHON_BIN to a supported environment or run through scripts/run.sh."
    )

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
from lightning.pytorch.plugins.environments import LightningEnvironment
from omegaconf import DictConfig, OmegaConf

from src.data.token_cache import TokenCacheDataModule
from src.models.sparse_token_prior import (
    SparseTokenPriorModule,
    build_sparse_prior_from_cache,
    infer_sparse_vocab_sizes,
)
from src.pl_trainer_util import resolve_val_check_interval
from src.stage2_compat import ensure_stage2_cache_metadata as add_cache_meta
from src.stage2_metrics import build_stage2_metrics_payload
from src.stage2_paths import infer_latest_token_cache as pick_cache
from src.stage2_preview import (
    Stage2SamplePreviewCallback,
    save_final_generation_preview,
)
from src.wandb_media import log_wandb_payload

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

STAGE2_TITLE = "STAGE 2: TRANSFORMER PRIOR TRAINING + GENERATION"
STAGE2_MODE = "stage2_transformer_generation"


def _default_stage2_run_name() -> str:
    return "stage2-transformer"


def _optional_container(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def arch_name(raw) -> str:
    text = str(raw or "sparse_spatial_depth").strip().lower()
    if text in {"sparse_spatial_depth", "spatial_depth"}:
        return "spatial_depth"
    if text in {"mingpt", "gpt"}:
        return "gpt"
    raise ValueError(f"Unsupported stage-2 architecture: {raw!r}")


def _preferred_module_device(module: torch.nn.Module) -> torch.device:
    """Return the module device, falling back to the current CUDA device."""
    try:
        param = next(module.parameters())
        if param.device.type != "cpu":
            return param.device
    except StopIteration:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


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
        if int(D) % 2 != 0:
            raise ValueError(
                "ar.type=gpt requires an even token depth because it models an interleaved atom/coeff stream. "
                f"Got token shape {(H, W, D)}. VQ-VAE caches use depth 1, so keep ar.type=sparse_spatial_depth."
            )
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
    print(STAGE2_TITLE)
    print("=" * 60)
    print(f"\nTransformer Mode: {mode}")
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
    print(f"  Num Nodes: {getattr(cfg.train_ar, 'num_nodes', 1)}")
    print(f"  Batch Size: {cfg.train_ar.batch_size}")
    print(f"  Max Epochs: {cfg.train_ar.max_epochs}")
    print(f"  Limit Train Batches: {getattr(cfg.train_ar, 'limit_train_batches', 1.0)}")
    print(f"  Limit Val Batches: {getattr(cfg.train_ar, 'limit_val_batches', 1.0)}")
    print(f"  Limit Test Batches: {getattr(cfg.train_ar, 'limit_test_batches', 1.0)}")
    print(f"  Crop H Sites: {getattr(cfg.train_ar, 'crop_h_sites', 0)}")
    print(f"  Crop W Sites: {getattr(cfg.train_ar, 'crop_w_sites', 0)}")
    print("=" * 60 + "\n")

    pl.seed_everything(cfg.seed, workers=True)

    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    model, dm = build(cfg)
    print(f"Transformer Token Grid Shape: {dm.token_shape}")
    cache_meta = dm.metadata
    cache_dataset = str(cache_meta.get("dataset") or "").strip()
    cfg_dataset = str(getattr(cfg.data, "dataset", "") or "").strip()
    if cache_dataset:
        if cfg_dataset and cfg_dataset.lower() != cache_dataset.lower():
            print(f"Stage-2 transformer cache dataset: {cache_dataset} (ignoring data.dataset={cfg_dataset})")
        else:
            print(f"Stage-2 transformer cache dataset: {cache_dataset}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"s2_{stamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nCheckpoint dir: {ckpt_dir}")

    base_run = str(getattr(cfg.wandb, "name", "") or "").strip() or _default_stage2_run_name()
    if bool(getattr(cfg.wandb, "append_timestamp", False)):
        run = f"{base_run}_{stamp}"
    else:
        run = base_run
    run_group = str(getattr(cfg.wandb, "group", "") or "").strip() or None
    run_tags = list(getattr(cfg.wandb, "tags", []) or [])
    devices_cfg = cfg.train_ar.devices
    try:
        num_devices = int(devices_cfg) if isinstance(devices_cfg, (int, str)) else len(devices_cfg)
    except Exception:
        num_devices = 1
    if num_devices > 1:
        wandb.setup()
    wandb_id = str(getattr(cfg.wandb, "id", "") or "").strip() or None
    wandb_resume = str(getattr(cfg.wandb, "resume", "") or "").strip() or None
    wandb_kwargs = {}
    if wandb_resume:
        wandb_kwargs["resume"] = wandb_resume
    wb = WandbLogger(
        project=cfg.wandb.project,
        name=run,
        save_dir=cfg.wandb.save_dir,
        group=run_group,
        tags=run_tags if run_tags else None,
        id=wandb_id,
        **wandb_kwargs,
    )
    wb.log_hyperparams(
        {
            "training_stage": "stage2",
            "stage_role": "transformer_generation",
            "training_mode": STAGE2_MODE,
            "legacy_training_mode": "s2",
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
    sample_out_dir = os.path.join(cfg.output_dir, "samples", f"s2_{stamp}")
    sample_variants = _optional_container(getattr(cfg.train_ar, "sample_variants", None))
    if step_every > 0 or epoch_every > 0:
        cbs.append(
            Stage2SamplePreviewCallback(
                cache_pt=str(cfg.token_cache_path),
                out_dir=sample_out_dir,
                step_every=step_every,
                epoch_every=epoch_every,
                n=int(getattr(cfg.train_ar, "sample_num_images", 4) or 4),
                temp=float(getattr(cfg.train_ar, "sample_temperature", 1.0) or 1.0),
                top_k=int(getattr(cfg.train_ar, "sample_top_k", 0) or 0),
                ctemp=getattr(cfg.train_ar, "sample_coeff_temperature", cfg.ar.sample_coeff_temperature),
                cmode=getattr(cfg.train_ar, "sample_coeff_mode", cfg.ar.sample_coeff_mode),
                sample_variants=sample_variants,
                s1_root=str(Path(str(cfg.output_dir)).expanduser().resolve().parent),
                use_wandb=bool(getattr(cfg.train_ar, "sample_log_to_wandb", False)),
            )
        )

    val_every = resolve_val_check_interval(dm, getattr(cfg.train_ar, "val_check_interval", 1.0))
    strategy_cfg = getattr(cfg.train_ar, "strategy", "auto")
    strategy_lower = str(strategy_cfg).lower()
    if num_devices <= 1 and strategy_lower in ("ddp", "ddp_spawn", "ddp_notebook"):
        strategy_cfg = "auto"
        strategy_lower = "auto"
    trainer_plugins = [LightningEnvironment()] if num_devices > 1 and strategy_lower.startswith("ddp") else None
    trainer = pl.Trainer(
        max_epochs=cfg.train_ar.max_epochs,
        max_steps=int(getattr(cfg.ar, "max_steps", -1) or -1),
        accelerator=cfg.train_ar.accelerator,
        num_nodes=int(getattr(cfg.train_ar, "num_nodes", 1) or 1),
        devices=cfg.train_ar.devices,
        strategy=strategy_cfg,
        plugins=trainer_plugins,
        logger=wb,
        callbacks=cbs,
        precision=cfg.train_ar.precision,
        gradient_clip_val=cfg.train_ar.gradient_clip_val,
        log_every_n_steps=cfg.train_ar.log_every_n_steps,
        val_check_interval=val_every,
        limit_train_batches=getattr(cfg.train_ar, "limit_train_batches", 1.0),
        limit_val_batches=getattr(cfg.train_ar, "limit_val_batches", 1.0),
        limit_test_batches=getattr(cfg.train_ar, "limit_test_batches", 1.0),
        deterministic=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
    )

    ckpt_path = str(getattr(cfg, "ckpt_path", "") or "").strip() or None
    if ckpt_path:
        print(f"\nResuming transformer prior training from: {ckpt_path}")
    else:
        print("\nStarting transformer prior training...")
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    run_test_cfg = getattr(cfg.train_ar, "run_test_after_fit", None)
    run_test_after_fit = (num_devices <= 1) if run_test_cfg is None else bool(run_test_cfg)
    if run_test_after_fit:
        print("\nRunning transformer prior test evaluation...")
        trainer.test(model, datamodule=dm)
    elif trainer.is_global_zero:
        print("\nSkipping post-fit transformer test evaluation for multi-device training.")

    compute_generation_fid = bool(getattr(cfg.train_ar, "compute_generation_fid", False))
    compute_audio_generation = bool(getattr(cfg.train_ar, "compute_audio_generation_metrics", False))
    generation_metric_samples = int(getattr(cfg.train_ar, "generation_metric_num_samples", 0) or 0)
    sample_count = int(getattr(cfg.train_ar, "sample_num_images", 4) or 0)
    if compute_generation_fid or compute_audio_generation:
        sample_count = max(sample_count, generation_metric_samples)
    final_samples_cfg = getattr(cfg.train_ar, "save_final_samples_after_fit", None)
    save_final_samples = (num_devices <= 1) if final_samples_cfg is None else bool(final_samples_cfg)
    if not save_final_samples and trainer.is_global_zero and sample_count > 0:
        print("Skipping final post-fit generation preview for multi-device training.")
    if save_final_samples and trainer.is_global_zero and sample_count > 0:
        try:
            sample_device = _preferred_module_device(model)
            if sample_device.type != "cpu":
                model.to(sample_device)
            final_result = save_final_generation_preview(
                trainer=trainer,
                mod=model,
                cache_pt=str(cfg.token_cache_path),
                out_dir=sample_out_dir,
                n=sample_count,
                temp=float(getattr(cfg.train_ar, "sample_temperature", 1.0) or 1.0),
                top_k=int(getattr(cfg.train_ar, "sample_top_k", 0) or 0),
                ctemp=getattr(cfg.train_ar, "sample_coeff_temperature", cfg.ar.sample_coeff_temperature),
                cmode=getattr(cfg.train_ar, "sample_coeff_mode", cfg.ar.sample_coeff_mode),
                s1_root=str(Path(str(cfg.output_dir)).expanduser().resolve().parent),
                use_wandb=bool(getattr(cfg.train_ar, "sample_log_to_wandb", False)),
                return_batch=(compute_generation_fid or compute_audio_generation),
            )
            if isinstance(final_result, tuple):
                final_raw, final_batch, final_cache = final_result
            else:
                final_raw, final_batch, final_cache = final_result, None, None
            print(f"\nSaved final transformer generation preview: {final_raw}")
            if final_batch is not None and final_cache is not None and generation_metric_samples > 0:
                metric_payload = build_stage2_metrics_payload(
                    final_batch.imgs.to(sample_device),
                    cfg=cfg,
                    cache=final_cache,
                    max_items=generation_metric_samples,
                    compute_fid=compute_generation_fid,
                    compute_audio=compute_audio_generation,
                )
                if metric_payload:
                    step = max(1, int(getattr(trainer, "global_step", 0) or 0))
                    log_wandb_payload(wb, metric_payload, step=step)
                    print(f"Logged final generation metrics: {sorted(metric_payload)}")
        except Exception as err:
            print(f"\nWarning: could not save final transformer generation preview ({err})")
    if save_final_samples:
        Stage2SamplePreviewCallback._barrier_if_needed(trainer)

    print("\nTransformer training complete.")
    print(f"Best checkpoint: {cbs[0].best_model_path}")
    return cbs[0].best_model_path


train_ar = main


if __name__ == "__main__":
    main()
