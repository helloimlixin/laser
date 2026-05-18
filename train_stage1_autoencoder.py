"""Train the maintained stage-1 autoencoder."""

import os
import sys
import warnings

if sys.version_info < (3, 10):
    raise SystemExit(
        "ERROR: train_stage1_autoencoder.py requires Python >= 3.10. "
        "Set PYTHON_BIN to a supported environment or run through scripts/run.sh."
    )

# Windows: PyTorch (LLVM OpenMP) and MKL/NumPy (Intel OpenMP) can both load and trigger OMP #15.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Suppress TF32 deprecation warnings (PyTorch 2.9 with Lightning compatibility)
warnings.filterwarnings('ignore', message='.*TF32.*')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import torch

from src.hydra_argparse_compat import patch_argparse_for_hydra_on_py314

patch_argparse_for_hydra_on_py314()
import hydra
from omegaconf import DictConfig

from src.lightning_warning_filters import register as register_lightning_warning_filters

register_lightning_warning_filters()
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
import wandb
from datetime import datetime

# Reduce DeepSpeed info logs
os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "warning")
# Required by cuBLAS for deterministic kernels on supported CUDA paths.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

torch.set_float32_matmul_precision('medium')

from src.pl_trainer_util import resolve_val_check_interval
from src.stage1_setup import (
    build_stage1_datamodule,
    build_stage1_model,
    data_config_from_section,
    infer_data_channels,
)

# Configure progress bar theme
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="green1",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82"
    ),
    leave=True
)

STAGE1_TITLE = "STAGE 1: AUTOENCODER TRAINING"
STAGE1_MODE = "stage1_autoencoder"


def _default_stage1_run_name(model_type: str) -> str:
    return f"{str(model_type).strip().lower()}-autoencoder"


def _resolve_ckpt_file(path: str) -> str:
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        preferred = [
            "final.ckpt",
            "last.ckpt",
            "mp_rank_00_model_states.pt",
            "model.pth",
            "model.pt",
            "state_dict.pth",
            "state_dict.pt",
            "weights.pt",
            "weights.pth",
        ]
        for name in preferred:
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                return cand
        for root, _, files in os.walk(path):
            for filename in sorted(files):
                if filename.endswith((".pt", ".pth", ".ckpt", ".bin")):
                    return os.path.join(root, filename)
        raise FileNotFoundError(f"No checkpoint file found under directory: {path}")
    return path


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    """
    Main training function using Hydra for configuration.
    
    Args:
        cfg: Hydra configuration object containing model and training parameters
    """
    ckpt_path = getattr(cfg, "ckpt_path", None)
    if ckpt_path:
        ckpt_path = _resolve_ckpt_file(ckpt_path)
        print(f"\nResume checkpoint: {ckpt_path}")
    deterministic = bool(getattr(cfg.train, "deterministic", False))
    resolved_in_channels = infer_data_channels(cfg.data)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    # Print detailed experiment configuration
    print("\n" + "=" * 60)
    print(STAGE1_TITLE)
    print("=" * 60)
    print("\nExperiment Configuration:")
    
    print("\nGeneral Settings:")
    print("Stage Role: autoencoder training")
    print(f"Random Seed: {cfg.seed}")
    print(f"Output Directory: {cfg.output_dir}")
    
    print("\nDataset Configuration:")
    print(f"Dataset: {cfg.data.dataset}")
    print(f"Data Directory: {cfg.data.data_dir}")
    print(f"Batch Size: {cfg.data.batch_size}")
    print(f"Number of Workers: {cfg.data.num_workers}")
    print(f"Image Size: {cfg.data.image_size}")
    print(f"Mean: {cfg.data.mean}")
    print(f"Std: {cfg.data.std}")
    if str(cfg.data.dataset).strip().lower() in {"vctk", "maestro"}:
        print(f"Audio Representation: {getattr(cfg.data, 'audio_representation', 'spectrogram')}")
        print(f"Sample Rate: {cfg.data.sample_rate}")
        print(f"Audio Samples Per Clip: {cfg.data.audio_num_samples}")
        print(f"STFT FFT Size: {cfg.data.stft_n_fft}")
        print(f"STFT Hop Length: {cfg.data.stft_hop_length}")

    print("\nModel Configuration:")
    print(f"Model Type: {cfg.model.type}")
    print(f"Input Channels: {resolved_in_channels}")
    print(f"Hidden Dimensions: {cfg.model.num_hiddens}")
    print(f"Embedding Dimensions: {cfg.model.embedding_dim}")
    print(f"Number of Residual Blocks: {cfg.model.num_residual_blocks}")
    print(f"Residual Hidden Dimensions: {cfg.model.num_residual_hiddens}")
    if cfg.model.type == "laser":
        print(f"Backbone: {getattr(cfg.model, 'backbone', 'simple')}")
        print(f"Dictionary Size: {cfg.model.num_embeddings}")
        print(f"Sparsity: {cfg.model.sparsity_level}")
        print(f"Bypass Bottleneck: {bool(getattr(cfg.model, 'bypass_bottleneck', False))}")
        print(f"Coefficient Bound: {getattr(cfg.model, 'coef_max', None)}")
        print(
            "Bounded OMP Refine Steps: "
            f"{getattr(cfg.model, 'bounded_omp_refine_steps', 8)}"
        )
        print(f"Variational Coefficients: {bool(getattr(cfg.model, 'variational_coeffs', False))}")
        if bool(getattr(cfg.model, 'variational_coeffs', False)):
            print(
                "Variational Coeff Refine Weight: "
                f"{float(getattr(cfg.model, 'variational_coeff_refine_weight', 0.0))}"
            )
            print(
                "Variational Coeff Target Std: "
                f"{float(getattr(cfg.model, 'variational_coeff_target_std', 0.25))}"
            )
            print(
                "Variational Coeff Min Std: "
                f"{float(getattr(cfg.model, 'variational_coeff_min_std', 0.01))}"
            )
        print(
            "Dictionary Through Decoder: "
            f"{bool(getattr(cfg.model, 'dictionary_through_decoder', False))}"
        )
        if str(getattr(cfg.model, 'backbone', 'simple')).strip().lower() != "simple":
            print(f"Downsamples: {getattr(cfg.model, 'num_downsamples', 2)}")
            print(f"Attention Resolutions: {tuple(getattr(cfg.model, 'attn_resolutions', ())) or ()}")
            print(f"Use Mid Attention: {bool(getattr(cfg.model, 'use_mid_attention', True))}")
            channel_multipliers = getattr(cfg.model, 'channel_multipliers', None)
            if channel_multipliers not in (None, "", ()):
                print(f"Channel Multipliers: {tuple(channel_multipliers)}")
            print(
                "Backbone Latent Channels: "
                f"{getattr(cfg.model, 'backbone_latent_channels', cfg.model.embedding_dim)}"
            )
            print(f"Max Channel Multiplier: {getattr(cfg.model, 'max_ch_mult', 2)}")
    elif cfg.model.type == "vqvae":
        print(f"Number of Embeddings: {cfg.model.num_embeddings}")
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
    
    print("\nTraining Configuration:")
    print(f"Learning Rate: {cfg.train.learning_rate}")
    print(f"Reconstruction MSE Weight: {float(getattr(cfg.model, 'recon_mse_weight', 1.0))}")
    print(f"Reconstruction L1 Weight: {float(getattr(cfg.model, 'recon_l1_weight', 0.0))}")
    print(f"Reconstruction Edge Weight: {float(getattr(cfg.model, 'recon_edge_weight', 0.0))}")
    print(f"Audio Multi-Resolution Loss Weight: {float(getattr(cfg.model, 'audio_multires_loss_weight', 0.0))}")
    print(f"Audio Multi-Resolution Scales: {tuple(getattr(cfg.model, 'audio_multires_scales', (1, 2, 4, 8)))}")
    print(f"Perceptual Weight: {float(getattr(cfg.model, 'perceptual_weight', 0.0))}")
    print(f"Perceptual Start Step: {int(getattr(cfg.model, 'perceptual_start_step', 0))}")
    print(f"Perceptual Warmup Steps: {int(getattr(cfg.model, 'perceptual_warmup_steps', 0))}")
    print(f"Beta: {cfg.train.beta}")
    print(f"Max Epochs: {cfg.train.max_epochs}")
    print(f"Max Steps: {getattr(cfg.train, 'max_steps', -1)}")
    print(f"Accelerator: {cfg.train.accelerator}")
    print(f"Num Nodes: {getattr(cfg.train, 'num_nodes', 1)}")
    print(f"Devices: {cfg.train.devices}")
    print(f"Precision: {cfg.train.precision}")
    print(f"Gradient Clip Value: {cfg.train.gradient_clip_val}")
    print(f"Deterministic: {deterministic}")
    print(f"Limit Train Batches: {getattr(cfg.train, 'limit_train_batches', 1.0)}")
    print(f"Limit Val Batches: {getattr(cfg.train, 'limit_val_batches', 1.0)}")
    print(f"Limit Test Batches: {getattr(cfg.train, 'limit_test_batches', 1.0)}")
    print(f"Run Test After Fit: {bool(getattr(cfg.train, 'run_test_after_fit', False))}")
    
    print("\nWandB Configuration:")
    print(f"Project: {cfg.wandb.project}")
    print(f"Run Name: {cfg.wandb.name}")
    print(f"Save Directory: {cfg.wandb.save_dir}")
    
    # Resolve checkpoint directory (base from config + run timestamp + model type)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_ckpt_dir = getattr(cfg.checkpoint, "dirpath", os.path.join(cfg.output_dir, "checkpoints"))
    run_ckpt_dir = os.path.join(base_ckpt_dir, f'run_{timestamp}', cfg.model.type)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    # Determine monitor key and mode (configurable, with safe defaults per model)
    configured_monitor = getattr(cfg.checkpoint, "monitor", None)
    if configured_monitor:
        monitor_key = configured_monitor
    else:
        monitor_key = "val/loss"
    monitor_mode = getattr(cfg.checkpoint, "mode", "min")
    filename_template = getattr(cfg.checkpoint, "filename", f"{cfg.model.type}-{{epoch:03d}}")

    print("\nCheckpoint Configuration:")
    print(f"Base Save Directory: {base_ckpt_dir}")
    print(f"Run Save Directory:  {run_ckpt_dir}")
    print(f"Filename Template:   {filename_template}")
    print(f"Monitor:             {monitor_key} (mode={monitor_mode})")
    print(f"Save Top K:          {cfg.checkpoint.save_top_k}")
    print(f"Save Last:           {cfg.checkpoint.save_last}")
    print("=" * 60 + "\n")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # Print GPU information
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Initialize data module
    print(f"Initializing data module for dataset: {cfg.data.dataset}")
    data_config = data_config_from_section(cfg.data)
    datamodule = build_stage1_datamodule(data_config)

    # Print dataset info for debugging
    print(f"Using dataset: {cfg.data.dataset}")
    print(f"Data module type: {type(datamodule).__name__}")

    if int(cfg.model.in_channels) != resolved_in_channels:
        print(
            f"Adjusting model input channels from {int(cfg.model.in_channels)} "
            f"to {resolved_in_channels} to match dataset {cfg.data.dataset}."
        )
    
    model = build_stage1_model(cfg.model, cfg.train, cfg.data)

    if ckpt_path:
        # Older checkpoints may contain metric module state such as
        # val_rfid/test_fid. Current models instantiate those lazily, so strict
        # resume would reject otherwise valid training checkpoints.
        model.strict_loading = False

    # Initialize wandb logger
    base_run_name = str(getattr(cfg.wandb, "name", "") or "").strip() or _default_stage1_run_name(cfg.model.type)
    if bool(getattr(cfg.wandb, "append_timestamp", False)):
        run_name = f"{base_run_name}_{timestamp}"
    else:
        run_name = base_run_name
    run_group = str(getattr(cfg.wandb, "group", "") or "").strip() or None
    run_tags = list(getattr(cfg.wandb, "tags", []) or [])
    devices_cfg = cfg.train.devices
    try:
        num_devices = int(devices_cfg) if isinstance(devices_cfg, (int, str)) else len(devices_cfg)
    except Exception:
        num_devices = 1
    if num_devices > 1:
        wandb.setup()
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        save_dir=cfg.wandb.save_dir,
        group=run_group,
        tags=run_tags if run_tags else None,
    )
    wandb_logger.log_hyperparams(
        {
            "training_stage": "stage1",
            "stage_role": "autoencoder_training",
            "training_mode": STAGE1_MODE,
            "model_type": cfg.model.type,
            "dataset": cfg.data.dataset,
            "input_channels": resolved_in_channels,
        }
    )

    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=run_ckpt_dir,
            filename=filename_template,
            save_top_k=cfg.checkpoint.save_top_k,
            monitor=monitor_key,
            mode=monitor_mode,
            save_last=cfg.checkpoint.save_last
        ),
        LearningRateMonitor(logging_interval='step'),
        progress_bar
    ]
    # Add EarlyStopping only if configured
    if getattr(cfg.train, "early_stopping_patience", None):
        callbacks.insert(1, EarlyStopping(
            monitor=monitor_key,
            patience=cfg.train.early_stopping_patience,
            mode=monitor_mode
        ))

    # Initialize trainer
    # Choose DDP only when using >1 device. Single-GPU DDP still inits torch.distributed (NCCL on CUDA),
    # which is unavailable on many Windows PyTorch builds — use auto instead.
    strategy_cfg = getattr(cfg.train, "strategy", None)
    if strategy_cfg is None:
        if cfg.model.type == "vqvae" and num_devices and num_devices > 1:
            strategy_cfg = "ddp"
    # Lightning rejects strategy=None; null / unset in config means default (auto).
    if strategy_cfg is None:
        strategy_cfg = "auto"
    strat_lower = str(strategy_cfg).lower()
    if num_devices <= 1 and strat_lower in ("ddp", "ddp_spawn", "ddp_notebook"):
        strategy_cfg = "auto"
        strat_lower = "auto"
    val_check_interval = resolve_val_check_interval(
        datamodule, getattr(cfg.train, "val_check_interval", 1.0)
    )
    max_steps = int(getattr(cfg.train, "max_steps", -1) or -1)
    trainer_plugins = [LightningEnvironment()] if num_devices > 1 and strat_lower.startswith("ddp") else None
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        max_steps=max_steps,
        accelerator=cfg.train.accelerator,
        num_nodes=int(getattr(cfg.train, "num_nodes", 1) or 1),
        devices=cfg.train.devices,
        strategy=strategy_cfg,
        plugins=trainer_plugins,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=val_check_interval,
        limit_train_batches=getattr(cfg.train, "limit_train_batches", 1.0),
        limit_val_batches=getattr(cfg.train, "limit_val_batches", 1.0),
        limit_test_batches=getattr(cfg.train, "limit_test_batches", 1.0),
        deterministic=deterministic,
        enable_progress_bar=True,
        enable_model_summary=(str(cfg.train.precision) == "32"),
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )

    # Train and test model (use PyTorch defaults for matmul precision to avoid API mixing)
    print("\nStarting autoencoder training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print("\nAutoencoder training complete.")

    final_ckpt_path = os.path.join(run_ckpt_dir, "final.ckpt")
    # In DDP, Lightning's checkpoint path may involve strategy collectives. All
    # ranks need to enter the call; Lightning handles rank-zero-only file writes.
    trainer.save_checkpoint(final_ckpt_path)
    if trainer.is_global_zero:
        print(f"Saved final stage-1 checkpoint: {final_ckpt_path}")

    if not bool(getattr(cfg.train, "run_test_after_fit", False)):
        return

    # Keep DDP test evaluation distributed. Launching a new single-rank trainer
    # while the original process group/environment is still alive can hang when
    # model logs use sync_dist=True.
    if num_devices > 1 and str(strategy_cfg).lower().startswith("ddp"):
        if trainer.is_global_zero:
            print("\nRunning autoencoder test evaluation with the DDP trainer...")
        trainer.test(model, datamodule=datamodule)
        if trainer.is_global_zero:
            print("\nAutoencoder evaluation complete.")
        return

    if not trainer.is_global_zero:
        return

    print("\nRunning autoencoder test evaluation...")
    test_trainer = pl.Trainer(
        accelerator=('gpu' if (cfg.train.accelerator == 'gpu' and torch.cuda.is_available()) else 'cpu'),
        devices=1,
        logger=wandb_logger,
        precision=cfg.train.precision,
        deterministic=deterministic,
        limit_test_batches=getattr(cfg.train, "limit_test_batches", 1.0),
        enable_progress_bar=True,
        enable_model_summary=(str(cfg.train.precision) == "32")
    )
    test_trainer.test(model, datamodule=datamodule)
    print("\nAutoencoder evaluation complete.")

if __name__ == "__main__":
    train()
