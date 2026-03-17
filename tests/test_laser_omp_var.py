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
    "scratch_laser_transformer_omp_var_test_module",
    "scratch/laser_transformer.py",
)
sys.modules["laser_transformer"] = scratch_laser_transformer
scratch_proto = _load_module(
    "scratch_proto_omp_var_test_module",
    "scratch/proto.py",
)
sys.modules["proto"] = scratch_proto
scratch_omp_var = _load_module(
    "scratch_laser_omp_var_test_module",
    "scratch/laser_omp_var.py",
)

OmpStageVAR = scratch_omp_var.OmpStageVAR
OmpVARConfig = scratch_omp_var.OmpVARConfig


def test_omp_stage_var_forward_shapes_and_duplicate_masking():
    cfg = OmpVARConfig(
        H=1,
        W=2,
        num_stages=2,
        atom_vocab_size=4,
        coeff_vocab_size=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    model = OmpStageVAR(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    tokens = torch.tensor([[[1, 4, 2, 5], [0, 6, 3, 4]]], dtype=torch.long)
    atom_logits, coeff_logits = model(tokens)

    assert atom_logits.shape == (1, cfg.H * cfg.W, cfg.num_stages, cfg.atom_vocab_size)
    assert coeff_logits.shape == (1, cfg.H * cfg.W, cfg.num_stages, cfg.coeff_vocab_size)
    assert not torch.isfinite(atom_logits[0, 0, 1, 1])
    assert torch.isfinite(atom_logits[0, 0, 1, 0])
    assert torch.isfinite(atom_logits[0, 0, 1, 2])


def test_omp_stage_var_generation_preserves_atom_uniqueness_and_parity():
    torch.manual_seed(0)

    cfg = OmpVARConfig(
        H=2,
        W=2,
        num_stages=3,
        atom_vocab_size=5,
        coeff_vocab_size=4,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )
    model = OmpStageVAR(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    tokens = model.generate(batch_size=8, show_progress=False)

    assert tokens.shape == (8, cfg.H * cfg.W, 2 * cfg.num_stages)
    atom_tokens = tokens[:, :, 0::2]
    coeff_tokens = tokens[:, :, 1::2]
    assert torch.all((atom_tokens >= 0) & (atom_tokens < cfg.atom_vocab_size))
    assert torch.all((coeff_tokens >= cfg.atom_vocab_size) & (coeff_tokens < cfg.atom_vocab_size + cfg.coeff_vocab_size))
    for sample in atom_tokens.reshape(-1, cfg.num_stages):
        assert torch.unique(sample).numel() == cfg.num_stages
