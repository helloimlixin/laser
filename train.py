import os
import torch
import hydra
from omegaconf import DictConfig
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime

# Reduce DeepSpeed info logs
os.environ.setdefault("DEEPSPEED_LOG_LEVEL", "warning")

from src.models.dlvae import DLVAE
from src.models.vqvae import VQVAE
from src.data.cifar10 import CIFAR10DataModule
from src.data.config import DataConfig
from src.data.imagenette2 import Imagenette2DataModule
from src.data.celeba import CelebADataModule

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

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    """
    Main training function using Hydra for configuration.
    
    Args:
        cfg: Hydra configuration object containing model and training parameters
    """
    # Print detailed experiment configuration
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)
    
    print("\nGeneral Settings:")
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
    
    print("\nModel Configuration:")
    print(f"Model Type: {cfg.model.type}")
    print(f"Input Channels: {cfg.model.in_channels}")
    print(f"Hidden Dimensions: {cfg.model.num_hiddens}")
    print(f"Embedding Dimensions: {cfg.model.embedding_dim}")
    print(f"Number of Residual Blocks: {cfg.model.num_residual_blocks}")
    print(f"Residual Hidden Dimensions: {cfg.model.num_residual_hiddens}")
    if cfg.model.type == "dlvae":
        print(f"Dictionary Size: {cfg.model.num_embeddings}")
        print(f"Sparsity: {cfg.model.sparsity_level}")
        if hasattr(cfg.model, "omp_tolerance"):
            print(f"OMP Tolerance: {cfg.model.omp_tolerance}")
        if hasattr(cfg.model, "omp_debug"):
            print(f"OMP Debug: {cfg.model.omp_debug}")
    elif cfg.model.type == "vqvae":
        print(f"Number of Embeddings: {cfg.model.num_embeddings}")
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
    
    print("\nTraining Configuration:")
    print(f"Learning Rate: {cfg.train.learning_rate}")
    print(f"Beta: {cfg.train.beta}")
    print(f"Max Epochs: {cfg.train.max_epochs}")
    print(f"Accelerator: {cfg.train.accelerator}")
    print(f"Devices: {cfg.train.devices}")
    print(f"Precision: {cfg.train.precision}")
    print(f"Gradient Clip Value: {cfg.train.gradient_clip_val}")
    
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
        monitor_key = "val/loss_epoch"
    monitor_mode = getattr(cfg.checkpoint, "mode", "min")
    filename_template = getattr(cfg.checkpoint, "filename", f"{cfg.model.type}-{{epoch:03d}}")

    print("\nCheckpoint Configuration:")
    print(f"Base Save Directory: {base_ckpt_dir}")
    print(f"Run Save Directory:  {run_ckpt_dir}")
    print(f"Filename Template:   {filename_template}")
    print(f"Monitor:             {monitor_key} (mode={monitor_mode})")
    print(f"Save Top K:          {cfg.checkpoint.save_top_k}")
    print(f"Save Last:           {cfg.checkpoint.save_last}")
    print("="*50 + "\n")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    # Print GPU information
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Initialize data module
    print(f"Initializing data module for dataset: {cfg.data.dataset}")
    if cfg.data.dataset == 'cifar10':
        datamodule = CIFAR10DataModule(DataConfig.from_dict(cfg.data))
    elif cfg.data.dataset == 'imagenette2':
        datamodule = Imagenette2DataModule(DataConfig.from_dict(cfg.data))
    elif cfg.data.dataset == 'celeba':
        datamodule = CelebADataModule(DataConfig.from_dict(cfg.data))
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    # Print dataset info for debugging
    print(f"Using dataset: {cfg.data.dataset}")
    print(f"Data module type: {type(datamodule).__name__}")

    def _resolve_ckpt_file(path: str) -> str:
        if os.path.isdir(path):
            preferred = [
                "mp_rank_00_model_states.pt",
                "model.pth",
                "model.pt",
                "state_dict.pth",
                "state_dict.pt",
                "weights.pt",
                "weights.pth",
                "last.ckpt",
            ]
            for name in preferred:
                cand = os.path.join(path, name)
                if os.path.isfile(cand):
                    return cand
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith((".pt", ".pth", ".ckpt", ".bin")):
                        return os.path.join(root, f)
            raise FileNotFoundError(f"No checkpoint file found under directory: {path}")
        return path

    def _extract_state_dict(container):
        if isinstance(container, dict):
            if isinstance(container.get('state_dict'), dict):
                return container['state_dict']
            module_blob = container.get('module')
            if isinstance(module_blob, dict):
                return module_blob.get('state_dict', module_blob)
            for key in ('model', 'ema', 'model_state_dict', 'net', 'generator'):
                payload = container.get(key)
                if isinstance(payload, dict):
                    return payload
        return container

    def _load_module_state(module, ckpt_path: str, label: str):
        load_path = _resolve_ckpt_file(ckpt_path)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        try:
            state = torch.load(load_path, map_location='cpu', weights_only=False)
        except TypeError:
            state = torch.load(load_path, map_location='cpu')
        sd = _extract_state_dict(state)
        if not isinstance(sd, dict):
            raise RuntimeError(f"Checkpoint at {load_path} does not contain a valid state_dict")
        missing, unexpected = module.load_state_dict(sd, strict=False)
        print(f"Loaded {label} checkpoint from {load_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        return missing, unexpected

    def _fetch_sample_batch():
        datamodule.setup("fit")
        loader = datamodule.train_dataloader()
        first = next(iter(loader))
        sample = first[0] if isinstance(first, (list, tuple)) else first
        return sample[:2].to('cpu')
    
    # Initialize model based on type
    if cfg.model.type == "dlvae":
        model_params = {
            'in_channels': cfg.model.in_channels,
            'num_hiddens': cfg.model.num_hiddens,
            'num_embeddings': cfg.model.num_embeddings,
            'embedding_dim': cfg.model.embedding_dim,
            'num_residual_blocks': cfg.model.num_residual_blocks,
            'num_residual_hiddens': cfg.model.num_residual_hiddens,
            'commitment_cost': cfg.model.commitment_cost,
            'decay': cfg.model.decay,
            'perceptual_weight': cfg.model.perceptual_weight,
            'learning_rate': cfg.train.learning_rate,
            'beta': cfg.train.beta,
            'compute_fid': cfg.model.compute_fid,
            'sparsity_level': cfg.model.sparsity_level,
            'omp_tolerance': getattr(cfg.model, 'omp_tolerance', 1e-7),
            'omp_debug': getattr(cfg.model, 'omp_debug', False),
        }
        model = DLVAE(**model_params)
    elif cfg.model.type == "vqvae":
        model_params = {
            'in_channels': cfg.model.in_channels,
            'num_hiddens': cfg.model.num_hiddens,
            'num_embeddings': cfg.model.num_embeddings,
            'embedding_dim': cfg.model.embedding_dim,
            'num_residual_blocks': cfg.model.num_residual_blocks,
            'num_residual_hiddens': cfg.model.num_residual_hiddens,
            'commitment_cost': cfg.model.commitment_cost,
            'decay': cfg.model.decay,
            'perceptual_weight': cfg.model.perceptual_weight,
            'learning_rate': cfg.train.learning_rate,
            'beta': cfg.train.beta,
            'compute_fid': cfg.model.compute_fid,
        }
        model = VQVAE(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")

    if hasattr(cfg.model, 'log_images_every_n_steps'):
        setattr(model, 'log_images_every_n_steps', cfg.model.log_images_every_n_steps)

    # Initialize wandb logger
    run_name = f"{cfg.wandb.name}_{cfg.model.type}_{timestamp}"
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        save_dir=cfg.wandb.save_dir
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
            monitor="val/loss_epoch",
            patience=cfg.train.early_stopping_patience,
            mode="min"
        ))

    # Initialize trainer
    # Choose DDP strategy only when using >1 device (GPU or CPU). Respect explicit config if provided.
    strategy_cfg = getattr(cfg.train, "strategy", None)
    if strategy_cfg is None:
        # Determine effective number of devices requested
        devices_cfg = cfg.train.devices
        try:
            num_devices = int(devices_cfg) if isinstance(devices_cfg, (int, str)) else len(devices_cfg)
        except Exception:
            num_devices = 1
        # Use DDP for these model types only when running multi-device
        if cfg.model.type in ("dlvae", "vqvae") and num_devices and num_devices > 1:
            strategy_cfg = "ddp"
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=strategy_cfg,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=getattr(cfg.train, "val_check_interval", 1.0),
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=(str(cfg.train.precision) == "32"),
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )

    # Train and test model (use PyTorch defaults for matmul precision to avoid API mixing)
    trainer.fit(model, datamodule=datamodule)
    # Run test on a single device to avoid DistributedSampler duplications
    test_trainer = pl.Trainer(
        accelerator=('gpu' if (cfg.train.accelerator == 'gpu' and torch.cuda.is_available()) else 'cpu'),
        devices=1,
        logger=wandb_logger,
        precision=cfg.train.precision,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=(str(cfg.train.precision) == "32")
    )
    test_trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    train()
