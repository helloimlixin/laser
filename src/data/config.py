from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class DataConfig:
    """Common configuration for all datasets."""
    dataset: str
    data_dir: str
    batch_size: int = 128
    eval_batch_size: Optional[int] = None
    num_workers: int = 8
    image_size: Union[int, Tuple[int, int]] = 32
    train_crop_size: Optional[Union[int, Tuple[int, int]]] = None
    seed: int = 42
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)  # CIFAR10 default
    std: Tuple[float, ...] = (0.2470, 0.2435, 0.2616)   # CIFAR10 default
    augment: bool = True
    sample_rate: int = 16000
    audio_num_samples: int = 32768
    audio_representation: str = "spectrogram"
    stft_n_fft: int = 1024
    stft_hop_length: int = 256
    stft_win_length: Optional[int] = None
    stft_power: float = 2.0
    stft_log_offset: float = 1e-5
    griffin_lim_iters: int = 16
    audio_dc_remove: bool = False
    audio_peak_normalize: bool = False
    audio_target_peak: float = 0.95
    audio_rms_normalize: bool = False
    audio_target_rms: float = 0.12
    audio_max_gain: float = 8.0
    audio_min_crop_rms: float = 0.0
    audio_crop_attempts: int = 1
    audio_fade_samples: int = 0
    stl10_include_unlabeled: bool = True
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create a DataConfig instance from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
