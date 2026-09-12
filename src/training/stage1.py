"""Autoencoder training."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
from src.training.paths import ROOT
from src.training.common import (
    _configure_training_tempdir,
    _dataset_key,
    _dedupe_tags,
    _make_selected_checkpoint_artifact_callback,
    _make_selected_checkpoint_file_callback,
    _stage1_fit_checkpoint_kwargs,
    _stage1_wandb_tags,
)

POST_FIT_RFID_DATASETS = {"celeba", "celebahq", "cifar10", "ffhq", "imagenet", "imagenette2", "lsun_bedroom", "lsun_church", "lsun_cat"}

import os
import subprocess
import sys
import warnings

if sys.version_info < (3, 10):
    raise SystemExit(
        "ERROR: train.py stage1 requires Python >= 3.10. "
        "Use a Python >= 3.10 environment."
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
from omegaconf import DictConfig, open_dict

from src.lightning_warning_filters import register as register_lightning_warning_filters

register_lightning_warning_filters()
import lightning as pl
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
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
CHECKPOINT_SAVE_TOP_K = 3
CHECKPOINT_SAVE_LAST = True
CHECKPOINT_EVERY_N_EPOCHS = 1
CHECKPOINT_UPLOAD_TO_WANDB = True
CHECKPOINT_UPLOAD_EVERY_N_EPOCHS = 1


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

def _scalar_image_size(value: object) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("data.image_size cannot be empty for post-fit rFID")
        return int(value[0])
    return int(value)


def _stat_override_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.strip("[]() ").split(",") if part.strip()]
    else:
        try:
            raw_values = list(value)  # type: ignore[arg-type]
        except TypeError:
            raw_values = [value]
    if len(raw_values) != 3:
        return []
    return [str(float(v)) for v in raw_values]


def _run_post_fit_rfid(final_ckpt_path: str, cfg: DictConfig, resolved_in_channels: int) -> None:
    if not bool(getattr(cfg.train, "compute_rfid_after_fit", False)):
        return

    dataset = _dataset_key(getattr(cfg.data, "dataset", ""))
    if resolved_in_channels != 3:
        print("\nSkipping post-fit rFID: rFID is only defined for RGB image datasets.")
        return
    if dataset not in POST_FIT_RFID_DATASETS:
        print(f"\nSkipping post-fit rFID: dataset {dataset!r} is not supported by compute_rfid.py.")
        return

    data_dir = str(getattr(cfg.data, "data_dir", "") or "").strip()
    if not data_dir:
        raise ValueError("Post-fit rFID requires data.data_dir to be set.")

    rfid_batch_size = int(getattr(cfg.train, "rfid_batch_size", 100) or 100)
    rfid_num_workers_raw = getattr(cfg.train, "rfid_num_workers", None)
    if rfid_num_workers_raw is None:
        rfid_num_workers_raw = getattr(cfg.data, "num_workers", 4)
    rfid_num_workers = int(rfid_num_workers_raw or 0)
    rfid_max_samples = int(getattr(cfg.train, "rfid_max_samples", 0) or 0)
    rfid_split = str(getattr(cfg.train, "rfid_split", "val") or "val")
    rfid_device = str(getattr(cfg.train, "rfid_device", "auto") or "auto")
    rfid_feature = int(getattr(cfg.train, "rfid_feature", 2048) or 2048)

    script_path = ROOT / "compute_rfid.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--ckpt",
        str(final_ckpt_path),
        "--dataset",
        dataset,
        "--data-dir",
        data_dir,
        "--image-size",
        str(_scalar_image_size(getattr(cfg.data, "image_size", 256))),
        "--split",
        rfid_split,
        "--batch-size",
        str(rfid_batch_size),
        "--num-workers",
        str(rfid_num_workers),
        "--max-samples",
        str(rfid_max_samples),
        "--device",
        rfid_device,
        "--feature",
        str(rfid_feature),
    ]

    mean_values = _stat_override_values(getattr(cfg.data, "mean", None))
    std_values = _stat_override_values(getattr(cfg.data, "std", None))
    if mean_values:
        cmd.extend(["--mean", *mean_values])
    if std_values:
        cmd.extend(["--std", *std_values])

    if rfid_max_samples <= 0:
        print("\nRunning post-fit paper-style rFID on the full validation split...")
    else:
        print("\nRunning post-fit debug rFID...")
    print(shlex.join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print("Post-fit rFID complete. See rfid.log next to the evaluated checkpoint.")


def run(cfg: DictConfig):
    """
    Main training function using Hydra for configuration.

    Args:
        cfg: Hydra configuration object containing model and training parameters
    """
    ckpt_path = getattr(cfg, "ckpt_path", None)
    if ckpt_path:
        ckpt_path = _resolve_ckpt_file(ckpt_path)
        print(f"\nResume checkpoint: {ckpt_path}")
    init_ckpt_path = getattr(cfg, "init_ckpt_path", None)
    if init_ckpt_path:
        init_ckpt_path = _resolve_ckpt_file(init_ckpt_path)
        print(f"\nInitialize weights from checkpoint: {init_ckpt_path}")
    if ckpt_path and init_ckpt_path:
        raise ValueError("Use ckpt_path to resume training or init_ckpt_path to initialize weights, not both.")
    deterministic = bool(getattr(cfg.train, "deterministic", False))
    accumulate_grad_batches = max(1, int(getattr(cfg.train, "accumulate_grad_batches", 1) or 1))
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
        if str(getattr(cfg.data, "audio_representation", "")).strip().lower() == "waveform":
            print(f"Audio Backbone: {getattr(cfg.model, 'audio_backbone', 'waveform')}")
            if str(getattr(cfg.model, "audio_backbone", "")).strip().lower() == "meta_encodec":
                print(
                    "Meta EnCodec Initialization: "
                    f"pretrained={bool(getattr(cfg.model, 'meta_encodec_pretrained', True))}, "
                    f"trainable={bool(getattr(cfg.model, 'meta_encodec_trainable', True))}"
                )
        print(f"Dictionary Size: {cfg.model.num_embeddings}")
        print(f"Sparsity: {cfg.model.sparsity_level}")
        print(f"Bypass Bottleneck: {bool(getattr(cfg.model, 'bypass_bottleneck', False))}")
        print(f"Coefficient Quantization Bound: {getattr(cfg.model, 'coef_max', None)}")
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
    print(f"Adversarial Weight: {float(getattr(cfg.model, 'adversarial_weight', 0.0))}")
    print(f"Adversarial Start Step: {int(getattr(cfg.model, 'adversarial_start_step', 0))}")
    print(f"Adversarial Warmup Steps: {int(getattr(cfg.model, 'adversarial_warmup_steps', 0))}")
    print(f"Adversarial Start Recon MSE: {getattr(cfg.model, 'adversarial_start_recon_mse', None)}")
    print(f"Adversarial Quality EMA Decay: {float(getattr(cfg.model, 'adversarial_quality_ema_decay', 0.99))}")
    print(f"Beta: {cfg.train.beta}")
    print(f"Beta2: {float(getattr(cfg.train, 'beta2', 0.999))}")
    print(f"Accumulate Grad Batches: {int(getattr(cfg.train, 'accumulate_grad_batches', 1) or 1)}")
    print(f"Max Epochs: {cfg.train.max_epochs}")
    print(f"Max Steps: {getattr(cfg.train, 'max_steps', -1)}")
    print(f"Accelerator: {cfg.train.accelerator}")
    print(f"Num Nodes: {getattr(cfg.train, 'num_nodes', 1)}")
    print(f"Devices: {cfg.train.devices}")
    print(f"Precision: {cfg.train.precision}")
    print(f"Accumulate Grad Batches: {accumulate_grad_batches}")
    print(f"Gradient Clip Value: {cfg.train.gradient_clip_val}")
    print(f"Deterministic: {deterministic}")
    print(f"Limit Train Batches: {getattr(cfg.train, 'limit_train_batches', 1.0)}")
    print(f"Limit Val Batches: {getattr(cfg.train, 'limit_val_batches', 1.0)}")
    print(f"Limit Test Batches: {getattr(cfg.train, 'limit_test_batches', 1.0)}")
    print(f"Run Test After Fit: {bool(getattr(cfg.train, 'run_test_after_fit', False))}")
    print(f"Compute Post-Fit rFID: {bool(getattr(cfg.train, 'compute_rfid_after_fit', False))}")
    print(f"Post-Fit rFID Split: {getattr(cfg.train, 'rfid_split', 'val')}")
    print(f"Post-Fit rFID Max Samples: {int(getattr(cfg.train, 'rfid_max_samples', 0) or 0)}")

    print("\nWandB Configuration:")
    print(f"Project: {cfg.wandb.project}")
    print(f"Run Name: {cfg.wandb.name}")
    print(f"Save Directory: {cfg.wandb.save_dir}")

    # Resolve checkpoint directory. Maintenance resumes can opt into the
    # checkpoint's existing directory so Lightning restores its full top-k
    # monitor state instead of silently starting a new ranking after every
    # process restart.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_ckpt_dir = getattr(cfg.checkpoint, "dirpath", os.path.join(cfg.output_dir, "checkpoints"))
    checkpoint_resume_in_place = bool(
        getattr(cfg.checkpoint, "resume_in_place", False)
    )
    if ckpt_path and checkpoint_resume_in_place:
        resume_checkpoint = Path(str(ckpt_path)).expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot resume checkpoints in place; checkpoint not found: {resume_checkpoint}"
            )
        run_ckpt_dir = str(resume_checkpoint.parent)
    else:
        run_ckpt_dir = os.path.join(base_ckpt_dir, f'run_{timestamp}', cfg.model.type)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    temp_dir = _configure_training_tempdir(cfg.output_dir)

    # Determine monitor key and mode (configurable, with safe defaults per model)
    configured_monitor = getattr(cfg.checkpoint, "monitor", None)
    if configured_monitor:
        monitor_key = configured_monitor
    else:
        monitor_key = "val/loss"
    monitor_mode = getattr(cfg.checkpoint, "mode", "min")
    if monitor_key in {"val/audio_visqol", "val/audio_visqol_audio48k"}:
        from src.audio_logging import is_visqol_available

        if not is_visqol_available():
            raise RuntimeError(
                f"checkpoint.monitor={monitor_key} requires the official "
                "ViSQOL Python package or a `visqol` CLI on PATH (alternatively "
                "set VISQOL_BINARY). Refusing to run without the monitored metric."
            )
    filename_template = getattr(cfg.checkpoint, "filename", f"{cfg.model.type}-{{epoch:03d}}")
    checkpoint_save_top_k = int(getattr(cfg.checkpoint, "save_top_k", CHECKPOINT_SAVE_TOP_K))
    checkpoint_save_last = bool(getattr(cfg.checkpoint, "save_last", CHECKPOINT_SAVE_LAST))
    checkpoint_every_n_epochs = max(
        1,
        int(getattr(cfg.checkpoint, "every_n_epochs", CHECKPOINT_EVERY_N_EPOCHS) or 1),
    )
    checkpoint_upload_to_wandb = bool(
        getattr(cfg.checkpoint, "upload_to_wandb", CHECKPOINT_UPLOAD_TO_WANDB)
    )
    checkpoint_upload_every_n_epochs = max(
        1,
        int(getattr(cfg.checkpoint, "upload_every_n_epochs", CHECKPOINT_UPLOAD_EVERY_N_EPOCHS) or 1),
    )
    checkpoint_upload_mode = str(
        getattr(cfg.checkpoint, "upload_mode", "artifact") or "artifact"
    ).strip().lower()
    if checkpoint_upload_mode not in {"artifact", "files"}:
        raise ValueError(
            "checkpoint.upload_mode must be 'artifact' or 'files', "
            f"got {checkpoint_upload_mode!r}"
        )
    with open_dict(cfg):
        cfg.checkpoint.save_top_k = checkpoint_save_top_k
        cfg.checkpoint.save_last = checkpoint_save_last
        cfg.checkpoint.every_n_epochs = checkpoint_every_n_epochs
        cfg.checkpoint.upload_to_wandb = checkpoint_upload_to_wandb
        cfg.checkpoint.upload_every_n_epochs = checkpoint_upload_every_n_epochs
        cfg.checkpoint.upload_mode = checkpoint_upload_mode

    print("\nCheckpoint Configuration:")
    print(f"Base Save Directory: {base_ckpt_dir}")
    print(f"Run Save Directory:  {run_ckpt_dir}")
    print(f"Temp Directory:      {temp_dir}")
    print(f"Filename Template:   {filename_template}")
    print(f"Monitor:             {monitor_key} (mode={monitor_mode})")
    print(f"Save Top K:          {cfg.checkpoint.save_top_k}")
    print(f"Save Last:           {cfg.checkpoint.save_last}")
    print(f"Resume In Place:     {checkpoint_resume_in_place}")
    print(f"Every N Epochs:      {checkpoint_every_n_epochs}")
    print(f"W&B Selected Checkpoint Upload: {checkpoint_upload_to_wandb}")
    print(f"W&B Checkpoint Upload Mode:     {checkpoint_upload_mode}")
    print(f"W&B Upload Every N Epochs:      {checkpoint_upload_every_n_epochs}")
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

    if (
        str(getattr(cfg.model, "type", "")).strip().lower() == "vqvae"
        and bool(getattr(cfg.model, "speaker_conditioning", False))
        and int(getattr(cfg.model, "speaker_conditioning_num_speakers", 0) or 0) <= 0
        and str(getattr(cfg.data, "dataset", "")).strip().lower() in {"vctk", "maestro"}
    ):
        datamodule.prepare_data()
        datamodule.setup("fit")
        num_speakers = int(getattr(datamodule, "num_speakers", 0) or 0)
        if num_speakers <= 0:
            raise RuntimeError("Speaker conditioning was requested, but no speakers were found in the data module.")
        with open_dict(cfg):
            cfg.model.speaker_conditioning_num_speakers = num_speakers
        print(f"Inferred VQ-VAE speaker conditioning classes: {num_speakers}")

    if int(cfg.model.in_channels) != resolved_in_channels:
        print(
            f"Adjusting model input channels from {int(cfg.model.in_channels)} "
            f"to {resolved_in_channels} to match dataset {cfg.data.dataset}."
        )

    model = build_stage1_model(cfg.model, cfg.train, cfg.data)
    if init_ckpt_path:
        from src.checkpoint_io import extract_state_dict, load_torch_payload

        payload = load_torch_payload(init_ckpt_path, map_location="cpu")
        state_dict = extract_state_dict(payload)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Checkpoint payload does not contain a state_dict: {init_ckpt_path}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        bottleneck = getattr(model, "bottleneck", None)
        data_initialized = getattr(bottleneck, "_data_initialized", None)
        if data_initialized is not None and hasattr(data_initialized, "fill_"):
            data_initialized.fill_(True)
        missing = len(getattr(incompatible, "missing_keys", ()) or ())
        unexpected = len(getattr(incompatible, "unexpected_keys", ()) or ())
        print(
            "Initialized stage-1 model weights "
            f"from {init_ckpt_path} (missing={missing}, unexpected={unexpected})."
        )
    trainer_gradient_clip_val = float(getattr(cfg.train, "gradient_clip_val", 0.0) or 0.0)
    uses_manual_opt = not bool(getattr(model, "automatic_optimization", True))
    trainer_accumulate_grad_batches = accumulate_grad_batches
    if uses_manual_opt:
        model.manual_gradient_clip_val = trainer_gradient_clip_val
        model.manual_accumulate_grad_batches = accumulate_grad_batches
        trainer_gradient_clip_val = 0.0
        trainer_accumulate_grad_batches = 1
        print(
            "Manual optimization active (adversarial); applying gradient clipping "
            f"inside the model with value {model.manual_gradient_clip_val} and "
            f"accumulating {model.manual_accumulate_grad_batches} batch(es) per optimizer step."
        )

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
    run_tags = _dedupe_tags([*(getattr(cfg.wandb, "tags", []) or []), *_stage1_wandb_tags(cfg)])
    wandb_id = str(getattr(cfg.wandb, "id", "") or "").strip() or None
    wandb_resume = str(getattr(cfg.wandb, "resume", "") or "").strip() or None
    wandb_kwargs = {}
    if wandb_resume:
        wandb_kwargs["resume"] = wandb_resume
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
        id=wandb_id,
        log_model=False,
        **wandb_kwargs,
    )
    run_metadata = {
        "training_stage": "stage1",
        "stage_role": "autoencoder_training",
        "training_mode": STAGE1_MODE,
        "model_type": cfg.model.type,
        "audio_backbone": str(getattr(cfg.model, "audio_backbone", "") or ""),
        "dataset": cfg.data.dataset,
        "input_channels": resolved_in_channels,
    }
    if bool(getattr(cfg.model, "audio_visqol_paper_audio_mode", False)):
        eval_batch_size = int(
            getattr(cfg.data, "eval_batch_size", None)
            or getattr(cfg.data, "batch_size", 1)
            or 1
        )
        sample_rate = int(getattr(cfg.data, "sample_rate", 24_000) or 24_000)
        samples_per_clip = int(
            getattr(cfg.data, "audio_num_samples", sample_rate) or sample_rate
        )
        run_metadata.update(
            {
                "visqol_version": "3.3.3",
                "visqol_checkpoint_mode": (
                    "audio" if monitor_key == "val/audio_visqol_audio48k" else "speech"
                ),
                "visqol_checkpoint_sample_rate": (
                    48_000 if monitor_key == "val/audio_visqol_audio48k" else 16_000
                ),
                "visqol_paper_metric": "val/audio_visqol_audio48k",
                "visqol_paper_mode": "audio",
                "visqol_paper_sample_rate": 48_000,
                "visqol_comparison_seconds": (
                    eval_batch_size * samples_per_clip / float(sample_rate)
                ),
                "visqol_split": (
                    str(getattr(cfg.data, "audio_split_protocol", "")) or
                    ("speaker-disjoint" if bool(
                        getattr(cfg.data, "audio_split_by_speaker", False)
                    ) else "item-random")
                ),
            }
        )
    # Preserve the entire reproducible recipe, including data and checkpoint
    # settings, alongside Lightning's flat model hyperparameters.
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    wandb_logger.log_hyperparams(run_metadata)

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_ckpt_dir,
        filename=filename_template,
        save_top_k=checkpoint_save_top_k,
        monitor=monitor_key,
        mode=monitor_mode,
        save_last=checkpoint_save_last,
        every_n_epochs=checkpoint_every_n_epochs,
    )
    callbacks = [checkpoint_callback]
    if checkpoint_upload_to_wandb:
        if checkpoint_upload_mode == "files":
            SelectedCheckpointFileCallback = _make_selected_checkpoint_file_callback(Callback)
            callbacks.append(
                SelectedCheckpointFileCallback(
                    checkpoint_callback,
                    upload_dir=Path(cfg.output_dir) / "wandb_checkpoints",
                    every_n_epochs=checkpoint_upload_every_n_epochs,
                )
            )
        else:
            SelectedCheckpointArtifactCallback = _make_selected_checkpoint_artifact_callback(Callback)
            callbacks.append(
                SelectedCheckpointArtifactCallback(
                    checkpoint_callback,
                    artifact_prefix="model",
                    every_n_epochs=checkpoint_upload_every_n_epochs,
                )
            )
    callbacks.extend([
        LearningRateMonitor(logging_interval='step'),
        progress_bar,
    ])
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
    if num_devices <= 1 and strat_lower.startswith("ddp"):
        strategy_cfg = "auto"
        strat_lower = "auto"
    # The rank-zero alternating dictionary update changes a DDP buffer
    # between iterations.  Enable PyTorch's collective-order wrapper for
    # this mode: it performs a CPU-side sequence check before NCCL work and
    # prevents faster ranks from issuing a later buffer broadcast while a
    # peer is still reducing the preceding dynamic adversarial graph.
    dictionary_collective_backend = str(
        getattr(cfg.model, "dictionary_collective_backend", "") or ""
    ).strip().lower()
    if (
        num_devices > 1
        and strat_lower.startswith("ddp")
        and dictionary_collective_backend == "ddp_buffer"
    ):
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        torch.distributed.set_debug_level(torch.distributed.DebugLevel.DETAIL)
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
        accumulate_grad_batches=trainer_accumulate_grad_batches,
        gradient_clip_val=trainer_gradient_clip_val,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=max(
            1,
            int(getattr(cfg.train, "check_val_every_n_epoch", 1) or 1),
        ),
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
    # PyTorch 2.6 defaults torch.load to weights_only=True. A Lightning
    # training checkpoint also contains trusted local optimizer, loop, RNG,
    # callback, and OmegaConf state, so an actual resume must explicitly
    # request the full checkpoint payload. Fresh fits retain Lightning's
    # default behavior.
    fit_kwargs = _stage1_fit_checkpoint_kwargs(ckpt_path)
    trainer.fit(model, datamodule=datamodule, **fit_kwargs)
    print("\nAutoencoder training complete.")

    final_ckpt_path = os.path.join(run_ckpt_dir, "final.ckpt")
    # In DDP, Lightning's checkpoint path may involve strategy collectives. All
    # ranks need to enter the call; Lightning handles rank-zero-only file writes.
    trainer.save_checkpoint(final_ckpt_path)
    if trainer.is_global_zero:
        print(f"Saved final stage-1 checkpoint: {final_ckpt_path}")

    def _run_post_fit_rfid_if_requested() -> None:
        if not trainer.is_global_zero or not bool(getattr(cfg.train, "compute_rfid_after_fit", False)):
            return
        try:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"Warning: could not move model to CPU before post-fit rFID: {exc}")
        _run_post_fit_rfid(final_ckpt_path, cfg, resolved_in_channels)

    if not bool(getattr(cfg.train, "run_test_after_fit", False)):
        _run_post_fit_rfid_if_requested()
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
        _run_post_fit_rfid_if_requested()
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
    _run_post_fit_rfid_if_requested()

