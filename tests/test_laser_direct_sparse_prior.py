import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


scratch_laser_transformer = _load_module(
    "scratch_laser_transformer_direct_sparse_prior_test_module",
    "scratch/laser_transformer.py",
)
sys.modules["laser_transformer"] = scratch_laser_transformer
scratch_proto = _load_module(
    "scratch_proto_direct_sparse_prior_test_module",
    "scratch/proto.py",
)
sys.modules["proto"] = scratch_proto
scratch_direct_sparse_prior = _load_module(
    "scratch_direct_sparse_prior_test_module",
    "scratch/laser_direct_sparse_prior.py",
)


build_config = scratch_direct_sparse_prior._build_spatial_depth_prior_config


class _FakeQuantizedBottleneck:
    content_vocab_size = 21
    num_embeddings = 16
    n_bins = 5
    coef_max = 7.5

    @staticmethod
    def _dequantize_coeff(bins: torch.Tensor) -> torch.Tensor:
        bins = bins.to(torch.float32)
        return bins * 0.5 - 1.0


class _FakeRealValuedBottleneck:
    content_vocab_size = 13
    num_embeddings = 13
    coef_max = 4.25


def test_build_spatial_depth_prior_config_quantized_uses_full_sparse_depth_and_bins():
    cfg = build_config(
        _FakeQuantizedBottleneck(),
        H=3,
        W=4,
        D=8,
        tf_d_model=64,
        tf_heads=8,
        tf_layers=6,
        tf_ff=128,
        tf_dropout=0.05,
        tf_global_tokens=2,
        real_valued_coeffs=False,
        coeff_max_fallback=99.0,
    )

    assert cfg.H == 3
    assert cfg.W == 4
    assert cfg.D == 8
    assert cfg.vocab_size == 21
    assert cfg.atom_vocab_size == 16
    assert cfg.coeff_vocab_size == 5
    assert cfg.real_valued_coeffs is False
    assert cfg.n_spatial_layers == 6
    assert cfg.n_depth_layers == 3
    assert cfg.n_global_spatial_tokens == 2
    assert torch.equal(
        cfg.coeff_bin_values,
        torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32),
    )
    assert cfg.coeff_max == 7.5


def test_build_spatial_depth_prior_config_real_valued_disables_coeff_vocab():
    cfg = build_config(
        _FakeRealValuedBottleneck(),
        H=2,
        W=2,
        D=6,
        tf_d_model=48,
        tf_heads=6,
        tf_layers=5,
        tf_ff=96,
        tf_dropout=0.1,
        tf_global_tokens=0,
        real_valued_coeffs=True,
        coeff_max_fallback=9.0,
    )

    assert cfg.H == 2
    assert cfg.W == 2
    assert cfg.D == 6
    assert cfg.vocab_size == 13
    assert cfg.atom_vocab_size == 13
    assert cfg.coeff_vocab_size is None
    assert cfg.coeff_bin_values is None
    assert cfg.real_valued_coeffs is True
    assert cfg.n_spatial_layers == 5
    assert cfg.n_depth_layers == 2
    assert cfg.coeff_max == 4.25
