"""Stage-1 training setup: data defaults, datamodules, and model builders."""

from __future__ import annotations

from dataclasses import MISSING, fields, replace
from typing import Any

from src.data.celeba import CelebADataModule
from src.data.cifar10 import CIFAR10DataModule
from src.data.coco import COCODataModule
from src.data.config import DataConfig
from src.data.image_folder import (
    ImageFolderDataModule,
    PAPER_IMAGE_FOLDER_DATASETS,
    normalize_image_folder_dataset_name,
)
from src.data.imagenette2 import Imagenette2DataModule
from src.data.stl10 import STL10DataModule
from src.data.vctk import VCTKDataModule
from src.models.laser import LASER
from src.models.vqvae import VQVAE


_DATA_CONFIG_DEFAULTS = {
    field.name: field.default
    for field in fields(DataConfig)
    if field.default is not MISSING
}

# These defaults are used by CLI tools such as cache extraction, where a Hydra
# dataset config may not be loaded. Full training still reads configs/data/*.yaml.
_DATASET_DEFAULTS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "image_size": 32,
        "data_dir": "../data",
    },
    "celeba": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "image_size": 128,
        "data_dir": "../data/celeba",
    },
    "celebahq": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "image_size": 256,
        "data_dir": "../data/celebahq",
    },
    "coco": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "image_size": 512,
        "data_dir": "../data/coco",
    },
    "imagenette2": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "image_size": 224,
        "data_dir": "../data/imagenette2",
    },
    "stl10": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "image_size": 96,
        "data_dir": "../data/stl10",
    },
    "vctk": {
        "mean": (0.5,),
        "std": (0.5,),
        "image_size": 128,
        "data_dir": "../data/vctk",
        "audio_representation": "spectrogram",
        "sample_rate": 16000,
        "audio_num_samples": 32768,
        "stft_n_fft": 1024,
        "stft_hop_length": 256,
        "stft_win_length": 1024,
        "stft_power": 2.0,
        "stft_log_offset": 1.0e-5,
    },
    "maestro": {
        "mean": (0.0,),
        "std": (1.0,),
        "image_size": 128,
        "data_dir": "../data/maestro-v3.0.0",
        "audio_representation": "waveform",
        "sample_rate": 22050,
        "audio_num_samples": 65536,
        "stft_n_fft": 1024,
        "stft_hop_length": 256,
        "stft_win_length": 1024,
        "stft_power": 2.0,
        "stft_log_offset": 1.0e-5,
        "audio_dc_remove": True,
        "audio_peak_normalize": True,
        "audio_target_peak": 0.95,
        "audio_rms_normalize": True,
        "audio_target_rms": 0.12,
        "audio_max_gain": 8.0,
        "audio_min_crop_rms": 0.03,
        "audio_crop_attempts": 64,
        "audio_fade_samples": 1024,
    },
}

_DATAMODULES = {
    "cifar10": CIFAR10DataModule,
    "coco": COCODataModule,
    "imagenette2": Imagenette2DataModule,
    "celeba": CelebADataModule,
    "celebahq": CelebADataModule,
    "stl10": STL10DataModule,
    "vctk": VCTKDataModule,
    "maestro": VCTKDataModule,
}


def _cfg_get(section: Any, key: str, default: Any = None) -> Any:
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _image_resolution(image_size) -> int:
    if isinstance(image_size, (list, tuple)):
        return int(image_size[0])
    return int(image_size)


def infer_data_channels(data_cfg: Any) -> int:
    mean = _cfg_get(data_cfg, "mean", None)
    if mean is not None:
        try:
            if len(mean) > 0:
                return int(len(mean))
        except TypeError:
            pass
    if str(_cfg_get(data_cfg, "dataset", "")).strip().lower() in {"vctk", "maestro"}:
        return 1
    return 3


def dataset_defaults(dataset: str) -> dict:
    dataset = normalize_image_folder_dataset_name(dataset)
    if dataset in _DATASET_DEFAULTS:
        return dict(_DATASET_DEFAULTS[dataset])
    if dataset in PAPER_IMAGE_FOLDER_DATASETS:
        return {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "image_size": 256,
            "data_dir": f"../data/{dataset}",
        }
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def data_config_from_overrides(
    dataset: str,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    image_size: int | tuple[int, int] | None = None,
    seed: int | None = None,
    augment: bool | None = None,
    **overrides,
) -> DataConfig:
    dataset = normalize_image_folder_dataset_name(dataset)
    values = dict(_DATA_CONFIG_DEFAULTS)
    values.update(dataset_defaults(dataset))
    values["dataset"] = dataset
    explicit = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "image_size": image_size,
        "seed": seed,
        "augment": augment,
        **overrides,
    }
    for key, value in explicit.items():
        if value is not None:
            values[key] = value
    return DataConfig.from_dict(values)


def data_config_from_section(section: Any) -> DataConfig:
    if isinstance(section, DataConfig):
        return section
    if hasattr(section, "items"):
        raw = dict(section.items())
    else:
        raw = {
            key: getattr(section, key)
            for key in DataConfig.__annotations__
            if hasattr(section, key)
        }
    return DataConfig.from_dict(raw)


def build_stage1_datamodule(config: DataConfig):
    dataset_name = normalize_image_folder_dataset_name(config.dataset)
    config = replace(config, dataset=dataset_name)
    if dataset_name in _DATAMODULES:
        return _DATAMODULES[dataset_name](config)
    if dataset_name in PAPER_IMAGE_FOLDER_DATASETS:
        return ImageFolderDataModule(config)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def _audio_model_kwargs(model_cfg: Any) -> dict:
    return {
        "audio_energy_loss_weight": float(_cfg_get(model_cfg, "audio_energy_loss_weight", 0.0)),
        "audio_backbone": _cfg_get(model_cfg, "audio_backbone", "spectrogram"),
        "audio_downsample_rates": tuple(_cfg_get(model_cfg, "audio_downsample_rates", (4, 4, 4))),
        "audio_dilation_cycle": tuple(_cfg_get(model_cfg, "audio_dilation_cycle", (1, 3, 9))),
        "audio_multires_stft_loss_weight": float(_cfg_get(model_cfg, "audio_multires_stft_loss_weight", 0.0)),
        "audio_multires_stft_fft_sizes": tuple(_cfg_get(model_cfg, "audio_multires_stft_fft_sizes", (512, 1024, 2048))),
        "audio_waveform_l1_weight": float(_cfg_get(model_cfg, "audio_waveform_l1_weight", 0.0)),
    }


def laser_model_kwargs(model_cfg: Any, train_cfg: Any, *, in_channels: int, image_size) -> dict:
    # Keep this boundary explicit: it is easier to audit config-to-constructor
    # wiring here than through a second abstraction layer over the model classes.
    return {
        "in_channels": int(in_channels),
        "num_hiddens": _cfg_get(model_cfg, "num_hiddens"),
        "num_embeddings": _cfg_get(model_cfg, "num_embeddings"),
        "embedding_dim": _cfg_get(model_cfg, "embedding_dim"),
        "num_residual_blocks": _cfg_get(model_cfg, "num_residual_blocks"),
        "num_residual_hiddens": _cfg_get(model_cfg, "num_residual_hiddens"),
        "backbone": _cfg_get(model_cfg, "backbone", "simple"),
        "resolution": _image_resolution(image_size),
        "num_downsamples": int(_cfg_get(model_cfg, "num_downsamples", 2)),
        "attn_resolutions": tuple(_cfg_get(model_cfg, "attn_resolutions", ())),
        "dropout": float(_cfg_get(model_cfg, "dropout", 0.0)),
        "channel_multipliers": _cfg_get(model_cfg, "channel_multipliers", None),
        "backbone_latent_channels": _cfg_get(model_cfg, "backbone_latent_channels", None),
        "max_ch_mult": int(_cfg_get(model_cfg, "max_ch_mult", 2)),
        "decoder_extra_residual_layers": int(_cfg_get(model_cfg, "decoder_extra_residual_layers", 1)),
        "use_mid_attention": bool(_cfg_get(model_cfg, "use_mid_attention", True)),
        "out_tanh": bool(_cfg_get(model_cfg, "out_tanh", True)),
        "commitment_cost": _cfg_get(model_cfg, "commitment_cost"),
        "recon_mse_weight": float(_cfg_get(model_cfg, "recon_mse_weight", 1.0)),
        "recon_l1_weight": float(_cfg_get(model_cfg, "recon_l1_weight", 0.0)),
        "recon_edge_weight": float(_cfg_get(model_cfg, "recon_edge_weight", 0.0)),
        "perceptual_weight": _cfg_get(model_cfg, "perceptual_weight"),
        "perceptual_start_step": int(_cfg_get(model_cfg, "perceptual_start_step", 0)),
        "perceptual_warmup_steps": int(_cfg_get(model_cfg, "perceptual_warmup_steps", 0)),
        "learning_rate": _cfg_get(train_cfg, "learning_rate"),
        "beta": _cfg_get(train_cfg, "beta"),
        "compute_fid": _cfg_get(model_cfg, "compute_fid"),
        "fid_feature": _cfg_get(model_cfg, "fid_feature", 2048),
        "bottleneck_loss_weight": _cfg_get(model_cfg, "bottleneck_loss_weight", 0.5),
        "sparsity_level": _cfg_get(model_cfg, "sparsity_level"),
        "dict_learning_rate": _cfg_get(model_cfg, "dict_learning_rate", None),
        "patch_based": _cfg_get(model_cfg, "patch_based", True),
        "patch_size": _cfg_get(model_cfg, "patch_size", 4),
        "patch_stride": _cfg_get(model_cfg, "patch_stride", 2),
        "patch_reconstruction": _cfg_get(model_cfg, "patch_reconstruction", "hann"),
        "coef_max": _cfg_get(model_cfg, "coef_max", None),
        "bounded_omp_refine_steps": _cfg_get(model_cfg, "bounded_omp_refine_steps", 8),
        "dictionary_usage_ema_decay": float(_cfg_get(model_cfg, "dictionary_usage_ema_decay", 0.99)),
        "dictionary_usage_grad_scale": float(_cfg_get(model_cfg, "dictionary_usage_grad_scale", 0.0)),
        "dictionary_usage_grad_min": float(_cfg_get(model_cfg, "dictionary_usage_grad_min", 0.1)),
        "dictionary_usage_grad_max": float(_cfg_get(model_cfg, "dictionary_usage_grad_max", 10.0)),
        "variational_coeffs": bool(_cfg_get(model_cfg, "variational_coeffs", False)),
        "variational_coeff_refine_weight": float(_cfg_get(model_cfg, "variational_coeff_refine_weight", 0.0)),
        "variational_coeff_target_std": float(_cfg_get(model_cfg, "variational_coeff_target_std", 0.25)),
        "variational_coeff_min_std": float(_cfg_get(model_cfg, "variational_coeff_min_std", 0.01)),
        "dictionary_through_decoder": bool(_cfg_get(model_cfg, "dictionary_through_decoder", False)),
        "dictionary_update_mode": str(_cfg_get(model_cfg, "dictionary_update_mode", "gradient")),
        "dictionary_ksvd_lr": float(_cfg_get(model_cfg, "dictionary_ksvd_lr", 0.2)),
        "dictionary_ksvd_update_every": int(_cfg_get(model_cfg, "dictionary_ksvd_update_every", 1)),
        "dictionary_ksvd_min_usage": int(_cfg_get(model_cfg, "dictionary_ksvd_min_usage", 1)),
        "dictionary_ksvd_max_atoms_per_step": int(_cfg_get(model_cfg, "dictionary_ksvd_max_atoms_per_step", 512)),
        "sparsity_reg_weight": _cfg_get(model_cfg, "sparsity_reg_weight", 0.01),
        "coherence_weight": _cfg_get(model_cfg, "coherence_weight", 0.0),
        "audio_multires_loss_weight": float(_cfg_get(model_cfg, "audio_multires_loss_weight", 0.0)),
        "audio_multires_scales": tuple(_cfg_get(model_cfg, "audio_multires_scales", (1, 2, 4, 8))),
        **_audio_model_kwargs(model_cfg),
        "log_images_every_n_steps": _cfg_get(model_cfg, "log_images_every_n_steps", 100),
        "diag_log_interval": _cfg_get(model_cfg, "diag_log_interval", 0),
        "enable_val_latent_visuals": _cfg_get(model_cfg, "enable_val_latent_visuals", False),
        "codebook_visual_max_vectors": int(_cfg_get(model_cfg, "codebook_visual_max_vectors", 1024)),
        "bypass_bottleneck": bool(_cfg_get(model_cfg, "bypass_bottleneck", False)),
        "warmup_steps": int(_cfg_get(train_cfg, "warmup_steps", 0)),
        "min_lr_ratio": float(_cfg_get(train_cfg, "min_lr_ratio", 0.01)),
    }


def vqvae_model_kwargs(model_cfg: Any, train_cfg: Any, *, in_channels: int) -> dict:
    # VQ-VAE has fewer knobs than LASER, but the same explicit mapping keeps the
    # training script small and makes config defaults visible in one place.
    return {
        "in_channels": int(in_channels),
        "num_hiddens": _cfg_get(model_cfg, "num_hiddens"),
        "num_embeddings": _cfg_get(model_cfg, "num_embeddings"),
        "embedding_dim": _cfg_get(model_cfg, "embedding_dim"),
        "num_residual_blocks": _cfg_get(model_cfg, "num_residual_blocks"),
        "num_residual_hiddens": _cfg_get(model_cfg, "num_residual_hiddens"),
        "num_downsamples": int(_cfg_get(model_cfg, "num_downsamples", 2)),
        "commitment_cost": _cfg_get(model_cfg, "commitment_cost"),
        "decay": _cfg_get(model_cfg, "decay"),
        "perceptual_weight": _cfg_get(model_cfg, "perceptual_weight"),
        "learning_rate": _cfg_get(train_cfg, "learning_rate"),
        "beta": _cfg_get(train_cfg, "beta"),
        "compute_fid": _cfg_get(model_cfg, "compute_fid"),
        "fid_feature": _cfg_get(model_cfg, "fid_feature", 2048),
        **_audio_model_kwargs(model_cfg),
        "codebook_init": bool(_cfg_get(model_cfg, "codebook_init", False)),
        "dead_code_threshold": float(_cfg_get(model_cfg, "dead_code_threshold", 0.0)),
        "out_tanh": bool(_cfg_get(model_cfg, "out_tanh", False)),
        "enable_codebook_visuals": bool(_cfg_get(model_cfg, "enable_codebook_visuals", False)),
        "codebook_visual_max_vectors": int(_cfg_get(model_cfg, "codebook_visual_max_vectors", 1024)),
    }


def build_stage1_model(model_cfg: Any, train_cfg: Any, data_cfg: Any):
    in_channels = infer_data_channels(data_cfg)
    model_type = str(_cfg_get(model_cfg, "type", "")).strip().lower()
    if model_type == "laser":
        return LASER(
            **laser_model_kwargs(
                model_cfg,
                train_cfg,
                in_channels=in_channels,
                image_size=_cfg_get(data_cfg, "image_size"),
            )
        )
    if model_type == "vqvae":
        return VQVAE(**vqvae_model_kwargs(model_cfg, train_cfg, in_channels=in_channels))
    raise ValueError(f"Unsupported model type: {_cfg_get(model_cfg, 'type', None)}")
