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
    "scratch_laser_transformer_sparse_grid_var_test_module",
    "scratch/laser_transformer.py",
)
sys.modules["laser_transformer"] = scratch_laser_transformer
scratch_proto = _load_module(
    "scratch_proto_sparse_grid_var_test_module",
    "scratch/proto.py",
)
sys.modules["proto"] = scratch_proto
scratch_laser_omp_var = _load_module(
    "scratch_laser_omp_var_sparse_grid_var_test_module",
    "scratch/laser_omp_var.py",
)
sys.modules["laser_omp_var"] = scratch_laser_omp_var
scratch_sparse_grid_var = _load_module(
    "scratch_sparse_grid_var_test_module",
    "scratch/laser_sparse_grid_var.py",
)


SparseGridVAR = scratch_sparse_grid_var.SparseGridVAR
SparseGridVARConfig = scratch_sparse_grid_var.SparseGridVARConfig
build_layout = scratch_sparse_grid_var._build_var_sequence_layout
encode_quantized_tokens_from_latent = scratch_sparse_grid_var._encode_quantized_tokens_from_latent
force_greedy_omp_slot_order = scratch_sparse_grid_var._force_greedy_omp_slot_order
spatial_var_cache_expected_meta = scratch_sparse_grid_var._spatial_var_cache_expected_meta
spatial_var_cache_is_compatible = scratch_sparse_grid_var._spatial_var_cache_is_compatible


def test_build_var_sequence_layout_tracks_scale_slices_and_prev_atoms():
    layout = build_layout((1, 2), 6)

    assert layout.seq_len == (1 * 1 + 2 * 2) * 6
    assert layout.scale_slices["1"] == (0, 6)
    assert layout.scale_slices["2"] == (6, 30)
    assert layout.type_ids[:6].tolist() == [0, 1, 0, 1, 0, 1]
    assert layout.prev_atom_positions[0].tolist() == [-1, -1]
    assert layout.prev_atom_positions[2].tolist() == [0, -1]
    assert layout.prev_atom_positions[4].tolist() == [0, 2]
    assert layout.prev_atom_positions[10].tolist() == [6, 8]


def test_sparse_grid_var_generation_preserves_token_types_and_unique_atoms_per_site():
    torch.manual_seed(0)

    cfg = SparseGridVARConfig(
        scales=(1, 2),
        token_depth=4,
        atom_vocab_size=5,
        coeff_vocab_size=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    model = SparseGridVAR(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    seq = model.generate(batch_size=32, show_progress=False)

    assert seq.shape == (32, 20)
    for scale in cfg.scales:
        scale_tokens = model.extract_scale_tokens(seq, scale).view(32, scale * scale, cfg.token_depth)
        atom_tokens = scale_tokens[..., 0::2]
        coeff_tokens = scale_tokens[..., 1::2]
        assert torch.all((atom_tokens >= 0) & (atom_tokens < cfg.atom_vocab_size))
        assert torch.all(
            (coeff_tokens >= cfg.atom_vocab_size)
            & (coeff_tokens < cfg.atom_vocab_size + cfg.coeff_vocab_size)
        )
        for batch_atoms in atom_tokens:
            for cell_atoms in batch_atoms:
                assert torch.unique(cell_atoms).numel() == cell_atoms.numel()


def test_sparse_grid_var_masked_logits_block_duplicate_atoms_within_a_site():
    cfg = SparseGridVARConfig(
        scales=(1,),
        token_depth=6,
        atom_vocab_size=5,
        coeff_vocab_size=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    model = SparseGridVAR(cfg).eval()
    raw_logits = torch.zeros(1, model.layout.seq_len, model.total_vocab_size)
    targets = torch.tensor([[1, 5, 2, 6, 3, 7]], dtype=torch.long)

    masked = model.masked_logits_for_targets(raw_logits, targets)

    assert torch.isfinite(masked[0, 0, 0])
    assert not torch.isfinite(masked[0, 0, cfg.atom_vocab_size])

    assert not torch.isfinite(masked[0, 2, 1])
    assert torch.isfinite(masked[0, 2, 2])
    assert torch.isfinite(masked[0, 2, 4])

    assert not torch.isfinite(masked[0, 4, 1])
    assert not torch.isfinite(masked[0, 4, 2])
    assert torch.isfinite(masked[0, 4, 3])
    assert not torch.isfinite(masked[0, 1, 0])
    assert torch.isfinite(masked[0, 1, cfg.atom_vocab_size])


def test_force_greedy_omp_slot_order_disables_canonicalization():
    class DummyBottleneck:
        def __init__(self):
            self.canonicalize_sparse_slots = True

    class DummyAE:
        def __init__(self):
            self.bottleneck = DummyBottleneck()

    ae = DummyAE()
    force_greedy_omp_slot_order(ae)

    assert ae.bottleneck.canonicalize_sparse_slots is False


def test_encode_quantized_tokens_from_latent_preserves_greedy_omp_slot_order():
    bottleneck = scratch_proto.DictionaryLearningTokenized(
        num_embeddings=8,
        embedding_dim=2,
        sparsity_level=2,
        n_bins=4,
        coef_max=3.0,
        quantize_sparse_coeffs=True,
        canonicalize_sparse_slots=False,
    )
    support = torch.tensor([[3, 1]], dtype=torch.long)
    coeffs = torch.tensor([[0.25, 2.5]], dtype=torch.float32)
    bottleneck.batch_omp_with_support = (
        lambda signals, dictionary: (
            support.expand(signals.size(1), -1),
            coeffs.expand(signals.size(1), -1),
        )
    )

    z = torch.zeros(1, bottleneck.embedding_dim, 1, 1)
    tokens = encode_quantized_tokens_from_latent(bottleneck, z)

    assert tokens[0, 0, 0, 0::2].tolist() == [3, 1]


def test_spatial_var_cache_compatibility_tracks_stage1_and_slot_order_metadata(tmp_path):
    ckpt = tmp_path / "ae.pt"
    ckpt.write_bytes(b"checkpoint")

    class DummyDataset:
        def __init__(self, root: Path):
            self.root = root

        def __len__(self):
            return 7

        def __getitem__(self, idx):
            return torch.zeros(1)

    class DummyBottleneck:
        def __init__(self):
            self.quantize_sparse_coeffs = True
            self.canonicalize_sparse_slots = False
            self.num_embeddings = 16
            self.embedding_dim = 4
            self.sparsity_level = 4
            self.n_bins = 8
            self.coef_max = 3.0
            self.coef_quantization = "uniform"
            self.coef_mu = 0.0
            self.token_depth = 8

    class DummyAE:
        def __init__(self):
            self.bottleneck = DummyBottleneck()

    expected = spatial_var_cache_expected_meta(
        ae=DummyAE(),
        dataset=DummyDataset(tmp_path / "data"),
        stage1_ckpt=ckpt,
        max_items=5,
    )
    cache = dict(expected)

    assert expected["omp_slot_order"] == "greedy"
    assert spatial_var_cache_is_compatible(cache, expected)

    stale_checkpoint = dict(cache)
    stale_checkpoint["stage1_ckpt_size"] = int(stale_checkpoint["stage1_ckpt_size"]) + 1
    assert not spatial_var_cache_is_compatible(stale_checkpoint, expected)

    stale_slot_order = dict(cache)
    stale_slot_order["canonicalize_sparse_slots"] = True
    assert not spatial_var_cache_is_compatible(stale_slot_order, expected)
