from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

@dataclass
class DataConfig:
    """Common configuration for all datasets."""
    dataset: str
    data_dir: str
    batch_size: int = 128
    num_workers: int = 4
    image_size: Union[int, Tuple[int, int]] = 32
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)  # CIFAR10 default
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)   # CIFAR10 default
    augment: bool = True
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create a DataConfig instance from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
