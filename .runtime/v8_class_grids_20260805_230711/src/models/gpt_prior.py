"""
Short canonical import path for the simple quantized GPT prior.
"""

from .mingpt_prior import (
    MinGPTQuantizedPrior,
    MinGPTQuantizedPriorConfig,
    build_mingpt_quantized_prior_config,
)

GPTPrior = MinGPTQuantizedPrior
GPTPriorConfig = MinGPTQuantizedPriorConfig
build_gpt_prior_config = build_mingpt_quantized_prior_config

__all__ = [
    "GPTPrior",
    "GPTPriorConfig",
    "build_gpt_prior_config",
    "MinGPTQuantizedPrior",
    "MinGPTQuantizedPriorConfig",
    "build_mingpt_quantized_prior_config",
]
