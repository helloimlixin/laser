import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


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
    "scratch_laser_transformer_scale_var_test_module",
    "scratch/laser_transformer.py",
)
sys.modules["laser_transformer"] = scratch_laser_transformer
scratch_proto = _load_module(
    "scratch_proto_multiscale_scale_var_test_module",
    "scratch/proto.py",
)
sys.modules["proto"] = scratch_proto
scratch_omp_var = _load_module(
    "scratch_laser_omp_var_scale_var_test_module",
    "scratch/laser_omp_var.py",
)
sys.modules["laser_omp_var"] = scratch_omp_var
scratch_scale_var = _load_module(
    "scratch_scale_var_test_module",
    "scratch/laser_multiscale_omp_var.py",
)


MultiScaleOmpVAR = scratch_scale_var.MultiScaleOmpVAR
MultiScaleOmpVARConfig = scratch_scale_var.MultiScaleOmpVARConfig
ResidualScalePrior = scratch_scale_var.ResidualScalePrior
ResidualScalePriorConfig = scratch_scale_var.ResidualScalePriorConfig
_default_scales_for_latent_size = scratch_scale_var._default_scales_for_latent_size
_multiscale_cache_expected_meta = scratch_scale_var._multiscale_cache_expected_meta
_multiscale_cache_uses_current_formulation = scratch_scale_var._multiscale_cache_uses_current_formulation
_precompute_multiscale_tokens = scratch_scale_var._precompute_multiscale_tokens
_reduce_scale_losses = scratch_scale_var._reduce_scale_losses


class DummyBottleneck:
    def __init__(
        self,
        token_depth: int,
        channels: int,
        num_embeddings: int = 5,
        n_bins: int = 4,
        sparsity_level: int = 2,
    ):
        self.token_depth = int(token_depth)
        self.channels = int(channels)
        self.num_embeddings = int(num_embeddings)
        self.n_bins = int(n_bins)
        self.sparsity_level = int(sparsity_level)
        self.coeff_token_offset = self.num_embeddings
        dictionary = torch.arange(1, self.num_embeddings * self.channels + 1, dtype=torch.float32)
        self.dictionary = dictionary.view(self.num_embeddings, self.channels) / 10.0

    def _dequantize_coeff(self, bin_idx: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(-1.0, 1.0, steps=self.n_bins, dtype=torch.float32, device=bin_idx.device)
        return centers[bin_idx]

    def _reconstruct_sparse(self, support: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        batch, height, width, depth = support.shape
        atoms = self.dictionary.to(device=support.device, dtype=coeffs.dtype)[support.reshape(-1, depth)]
        recon = (atoms * coeffs.reshape(-1, depth).unsqueeze(-1)).sum(dim=1)
        return recon.view(batch, height, width, self.channels).permute(0, 3, 1, 2).contiguous()


def test_default_scales_for_latent_size():
    assert _default_scales_for_latent_size(8) == (1, 2, 4, 8)
    assert _default_scales_for_latent_size(6) == (1, 2, 4, 6)


def test_residual_scale_prior_uses_conditioning():
    cfg = ResidualScalePriorConfig(
        H=2,
        W=2,
        num_stages=2,
        atom_vocab_size=4,
        coeff_vocab_size=3,
        cond_channels=1,
        d_model=4,
        n_heads=1,
        n_layers=0,
        d_ff=8,
        dropout=0.0,
        n_global_tokens=0,
    )
    model = ResidualScalePrior(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.cond_proj.weight[0, 0] = 1.0
        model.norm.weight.fill_(1.0)
        model.atom_head.weight[0, 0] = 1.0

    tokens = torch.tensor(
        [[[0, 4, 1, 5], [1, 4, 2, 5], [2, 4, 3, 5], [3, 4, 0, 5]]],
        dtype=torch.long,
    )
    cond_zero = torch.zeros(1, 1, 2, 2)
    cond_one = torch.ones(1, 1, 2, 2)

    atom_zero, _ = model(tokens, cond_zero)
    atom_one, _ = model(tokens, cond_one)

    assert not torch.allclose(atom_zero[:, :, 0, :], atom_one[:, :, 0, :])


def test_residual_scale_prior_atom_logits_are_parallel_over_spatial_sites():
    cfg = ResidualScalePriorConfig(
        H=1,
        W=2,
        num_stages=1,
        atom_vocab_size=4,
        coeff_vocab_size=3,
        cond_channels=0,
        d_model=4,
        n_heads=1,
        n_layers=0,
        d_ff=8,
        dropout=0.0,
        n_global_tokens=0,
    )
    model = ResidualScalePrior(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        eye = torch.eye(cfg.d_model)
        model.state_proj[0].weight.copy_(eye)
        model.state_proj[2].weight.copy_(eye)
        model.atom_emb.weight[1, 0] = 1.0
        model.atom_emb.weight[2, 0] = 2.0
        model.norm.weight.fill_(1.0)
        model.atom_head.weight[0, 0] = 1.0

    tokens_a = torch.tensor([[[1, 4], [0, 4]]], dtype=torch.long)
    tokens_b = torch.tensor([[[2, 4], [0, 4]]], dtype=torch.long)

    atom_a, _ = model(tokens_a, None)
    atom_b, _ = model(tokens_b, None)

    assert torch.allclose(atom_a[:, 0, 0, :], atom_b[:, 0, 0, :])
    assert torch.allclose(atom_a[:, 1, 0, :], atom_b[:, 1, 0, :])


def test_multiscale_scale_var_generation_shapes_and_token_ranges():
    torch.manual_seed(0)
    cfg = MultiScaleOmpVARConfig(
        scales=(1, 2),
        H=2,
        W=2,
        num_stages=2,
        atom_vocab_size=5,
        coeff_vocab_size=4,
        latent_channels=3,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        n_global_tokens=0,
    )
    model = MultiScaleOmpVAR(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    bottleneck = DummyBottleneck(
        token_depth=2 * cfg.num_stages,
        channels=cfg.latent_channels,
        sparsity_level=cfg.num_stages,
    )
    tokens_by_scale, latents = model.generate(batch_size=4, bottleneck=bottleneck, show_progress=False)

    assert tuple(tokens_by_scale["1"].shape) == (4, 1, 1, 2 * cfg.num_stages)
    assert tuple(tokens_by_scale["2"].shape) == (4, 2, 2, 2 * cfg.num_stages)
    assert tuple(latents.shape) == (4, cfg.latent_channels, 2, 2)

    for scale in cfg.scales:
        scale_tokens = tokens_by_scale[str(scale)]
        atom_tokens = scale_tokens[..., 0::2]
        coeff_tokens = scale_tokens[..., 1::2]
        assert torch.all((atom_tokens >= 0) & (atom_tokens < cfg.atom_vocab_size))
        assert torch.all(
            (coeff_tokens >= cfg.atom_vocab_size)
            & (coeff_tokens < cfg.atom_vocab_size + cfg.coeff_vocab_size)
        )
        for sample in atom_tokens.view(-1, cfg.num_stages):
            assert torch.unique(sample).numel() == cfg.num_stages


def test_multiscale_forward_conditions_finer_scales_on_coarser_latents():
    cfg = MultiScaleOmpVARConfig(
        scales=(1, 2),
        H=2,
        W=2,
        num_stages=1,
        atom_vocab_size=4,
        coeff_vocab_size=3,
        latent_channels=1,
        d_model=4,
        n_heads=1,
        n_layers=0,
        d_ff=8,
        dropout=0.0,
        n_global_tokens=0,
    )
    model = MultiScaleOmpVAR(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        prior = model.scale_priors["2"]
        prior.cond_proj.weight[0, 0] = 1.0
        prior.norm.weight.fill_(1.0)
        prior.atom_head.weight[0, 0] = 1.0

    tokens_a = {
        "1": torch.tensor([[[[1, 4]]]], dtype=torch.long),
        "2": torch.tensor([[[[0, 4], [0, 4]], [[0, 4], [0, 4]]]], dtype=torch.long),
    }
    tokens_b = {
        "1": torch.tensor([[[[2, 5]]]], dtype=torch.long),
        "2": torch.tensor([[[[0, 4], [0, 4]], [[0, 4], [0, 4]]]], dtype=torch.long),
    }

    atom_a, _ = model(
        tokens_a,
        DummyBottleneck(token_depth=2, channels=cfg.latent_channels, sparsity_level=1),
        cond_latents_by_scale=None,
    )
    atom_b, _ = model(
        tokens_b,
        DummyBottleneck(token_depth=2, channels=cfg.latent_channels, sparsity_level=1),
        cond_latents_by_scale=None,
    )

    assert not torch.allclose(atom_a["2"], atom_b["2"])


def test_multiscale_teacher_forced_argmax_outputs_valid_tokens():
    torch.manual_seed(0)
    cfg = MultiScaleOmpVARConfig(
        scales=(1, 2),
        H=2,
        W=2,
        num_stages=2,
        atom_vocab_size=5,
        coeff_vocab_size=4,
        latent_channels=3,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        n_global_tokens=0,
    )
    model = MultiScaleOmpVAR(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    tokens_by_scale = {
        "1": torch.tensor([[[[0, 5, 1, 6]]]], dtype=torch.long),
        "2": torch.tensor(
            [[
                [[0, 5, 1, 6], [1, 5, 2, 6]],
                [[2, 5, 3, 6], [3, 5, 4, 6]],
            ]],
            dtype=torch.long,
        ),
    }
    pred_tokens = model.teacher_forced_argmax(
        tokens_by_scale,
        DummyBottleneck(token_depth=4, channels=cfg.latent_channels, sparsity_level=2),
    )

    for scale in cfg.scales:
        scale_tokens = pred_tokens[str(scale)]
        atom_tokens = scale_tokens[..., 0::2]
        coeff_tokens = scale_tokens[..., 1::2]
        assert torch.all((atom_tokens >= 0) & (atom_tokens < cfg.atom_vocab_size))
        assert torch.all(
            (coeff_tokens >= cfg.atom_vocab_size)
            & (coeff_tokens < cfg.atom_vocab_size + cfg.coeff_vocab_size)
        )
        for sample in atom_tokens.view(-1, cfg.num_stages):
            assert torch.unique(sample).numel() == cfg.num_stages


def test_reduce_scale_losses_can_weight_finer_scales_more_heavily():
    losses = [torch.tensor(8.0), torch.tensor(2.0)]

    uniform = _reduce_scale_losses(losses, [1.0, 1.0])
    token_count = _reduce_scale_losses(losses, [1.0, 4.0])

    assert torch.isclose(uniform, torch.tensor(5.0))
    assert torch.isclose(token_count, torch.tensor(3.2))


def test_multiscale_cache_compatibility_tracks_stage1_and_slot_order_metadata(tmp_path):
    ckpt = tmp_path / "ae.pt"
    ckpt.write_bytes(b"checkpoint")

    class DummyDataset:
        def __init__(self, root: Path):
            self.root = root

        def __len__(self):
            return 7

        def __getitem__(self, idx):
            return torch.zeros(1)

    class DummyMetaBottleneck:
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
            self.bottleneck = DummyMetaBottleneck()

    expected = _multiscale_cache_expected_meta(
        ae=DummyAE(),
        dataset=DummyDataset(tmp_path / "data"),
        stage1_ckpt=ckpt,
        max_items=5,
    )
    cache = dict(expected)

    assert expected["scale_semantics"] == "latent_resolution"
    assert _multiscale_cache_uses_current_formulation(cache, expected)

    stale_checkpoint = dict(cache)
    stale_checkpoint["stage1_ckpt_size"] = int(stale_checkpoint["stage1_ckpt_size"]) + 1
    assert not _multiscale_cache_uses_current_formulation(stale_checkpoint, expected)

    stale_slot_order = dict(cache)
    stale_slot_order["canonicalize_sparse_slots"] = True
    assert not _multiscale_cache_uses_current_formulation(stale_slot_order, expected)


def test_precompute_multiscale_tokens_builds_spatial_pyramid_and_conditioning(tmp_path):
    class FakeBottleneck:
        def __init__(self):
            self.quantize_sparse_coeffs = True
            self.num_embeddings = 8
            self.embedding_dim = 1
            self.n_bins = 16
            self.sparsity_level = 2
            self.coef_max = 15.0
            self.coef_quantization = "uniform"
            self.coef_mu = 0.0
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = 2 * self.sparsity_level
            self.canonicalize_sparse_slots = False

        def _encode_sparse_codes(self, z_e):
            batch, _, height, width = z_e.shape
            support = torch.empty(batch, height, width, self.sparsity_level, dtype=torch.long, device=z_e.device)
            coeffs = torch.empty(batch, height, width, self.sparsity_level, dtype=torch.float32, device=z_e.device)
            support[..., 0] = int(height)
            support[..., 1] = int(height) + 1
            coeffs[..., 0] = float(height)
            coeffs[..., 1] = float(height + 2)
            return support, coeffs

        def _quantize_coeff(self, coeff):
            bin_idx = coeff.round().to(torch.long).clamp(0, self.n_bins - 1)
            return bin_idx, bin_idx.to(torch.float32)

        def _pack_quantized_tokens(self, support, bin_idx):
            tokens = torch.empty(*support.shape[:-1], 2 * support.shape[-1], dtype=torch.long, device=support.device)
            tokens[..., 0::2] = support.to(torch.long)
            tokens[..., 1::2] = bin_idx.to(torch.long) + self.coeff_token_offset
            return tokens

        def _dequantize_coeff(self, coeff_bin):
            return coeff_bin.to(torch.float32)

        def _reconstruct_sparse(self, support, coeffs):
            latent = (coeffs[..., 0] + 10.0 * coeffs[..., 1]).unsqueeze(1)
            return latent.contiguous()

    class FakeAE:
        def __init__(self):
            self.bottleneck = FakeBottleneck()

        def encoder(self, x):
            return torch.zeros(x.size(0), 1, 4, 4, dtype=torch.float32, device=x.device)

    dataset = TensorDataset(torch.zeros(1, 1, 1, 1))
    stage1_ckpt = tmp_path / "fake_stage1.pt"
    stage1_ckpt.write_bytes(b"stage1")
    cache_meta = _multiscale_cache_expected_meta(
        ae=FakeAE(),
        dataset=dataset,
        stage1_ckpt=stage1_ckpt,
        max_items=None,
    )
    cache = _precompute_multiscale_tokens(
        ae=FakeAE(),
        dataset=dataset,
        device=torch.device("cpu"),
        scales=(1, 2, 4),
        cache_path=tmp_path / "cache.pt",
        batch_size=1,
        num_workers=0,
        max_items=None,
        cache_meta=cache_meta,
    )

    assert _multiscale_cache_uses_current_formulation(cache, cache_meta)
    assert cache["tokens_s1"].shape == (1, 4)
    assert cache["tokens_s2"].shape == (1, 16)
    assert cache["tokens_s4"].shape == (1, 64)
    assert cache["tokens_s1"][0].tolist() == [1, 9, 2, 11]
    assert torch.allclose(cache["cond_s2"].float(), torch.full((1, 1, 2, 2), 31.0))
    assert torch.allclose(cache["cond_s4"].float(), torch.full((1, 1, 4, 4), 42.0))
