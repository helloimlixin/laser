from pathlib import Path

import torch

import src.s2 as s2
from src.stage2_compat import Stage1DecodeBundle
from src.stage2_compat import decode_stage2_tokens
from src.stage2_compat import _infer_rq_stage1_config
from src.stage2_compat import reconstruct_stage2_sparse_latent


class _FakeDecoder(torch.nn.Module):
    def __init__(self, *, patch_based=False, patch_stride=1, expose_patch_flag=True, patch_size=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.latent_hw_calls = []
        attrs = {"patch_stride": int(patch_stride)}
        if expose_patch_flag:
            attrs["patch_based"] = bool(patch_based)
        if patch_size is not None:
            attrs["patch_size"] = int(patch_size)
        self.bottleneck = type("BottleneckStub", (), attrs)()

    def decode_from_tokens(self, tokens, latent_hw=None, coeff_vocab_size=None, coeff_bin_values=None):
        self.latent_hw_calls.append(latent_hw)
        vals = tokens.to(torch.float32).sum(dim=(1, 2, 3), keepdim=True)
        return vals.view(-1, 1, 1, 1).expand(-1, 3, 2, 2)

    def decode_from_atoms_and_coeffs(self, atoms, coeffs, latent_hw=None):
        self.latent_hw_calls.append(latent_hw)
        vals = (atoms.to(torch.float32) + coeffs).sum(dim=(1, 2, 3), keepdim=True)
        return vals.view(-1, 1, 1, 1).expand(-1, 3, 2, 2)

    def reconstruct_latent_from_atoms_and_coeffs(self, atoms, coeffs, latent_hw=None):
        self.latent_hw_calls.append(latent_hw)
        return torch.zeros(
            atoms.size(0),
            1,
            int(latent_hw[0]),
            int(latent_hw[1]),
            dtype=torch.float32,
        )


class _FakeTokNet:
    def __init__(self):
        self.device = torch.device("cpu")
        self.prior = type("Prior", (), {"real_valued_coeffs": False})()

    def generate_sparse_codes(self, n, **kwargs):
        return torch.arange(n * 2, dtype=torch.long).view(n, 1, 2)


class _FakeRealNet:
    def __init__(self):
        self.device = torch.device("cpu")
        self.prior = type("Prior", (), {"real_valued_coeffs": True})()

    def generate_sparse_codes(self, n, **kwargs):
        atoms = torch.arange(n * 2, dtype=torch.long).view(n, 1, 2) % 7
        coeffs = torch.linspace(-1.0, 1.0, steps=n * 2, dtype=torch.float32).view(n, 1, 2)
        return atoms, coeffs


class _StubNet:
    def eval(self):
        return self

    def to(self, dev):
        return self


class _StubCropNet(_StubNet):
    def __init__(self):
        self.prior = type("Prior", (), {"cfg": type("Cfg", (), {"H": 2, "W": 2, "D": 4})()})()


class _FakeGPTPrior:
    def __init__(self, *, h=2, w=2, d=2, fill=7):
        self.cfg = type("Cfg", (), {"H": h, "W": w, "D": d})()
        self.real_valued_coeffs = False
        self.fill = int(fill)

    def generate(
        self,
        batch_size,
        *,
        prompt_tokens=None,
        prompt_mask=None,
        **kwargs,
    ):
        assert prompt_tokens is not None
        out = prompt_tokens.clone().view(batch_size, self.cfg.H * self.cfg.W, self.cfg.D)
        mask = prompt_mask.view(batch_size, self.cfg.H * self.cfg.W, self.cfg.D)
        out = out.masked_fill(~mask, self.fill)
        return out


class _FakeSlideNet:
    def __init__(self):
        self.device = torch.device("cpu")
        self.prior = _FakeGPTPrior()


def _bundle(
    *,
    latent_hw=(1, 1),
    patch_based=False,
    patch_stride=1,
    kind="lightning",
    expose_patch_flag=True,
    patch_size=None,
) -> Stage1DecodeBundle:
    return Stage1DecodeBundle(
        kind=str(kind),
        model=_FakeDecoder(
            patch_based=patch_based,
            patch_stride=patch_stride,
            expose_patch_flag=expose_patch_flag,
            patch_size=patch_size,
        ),
        checkpoint_path=Path("/tmp/s1.ckpt"),
        latent_hw=latent_hw,
        coeff_vocab_size=4,
        coeff_bin_values=torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32),
    )


def test_sample_batches_tokens():
    out = s2.sample(
        _FakeTokNet(),
        _bundle(),
        (1, 1, 2),
        n=5,
        bs=2,
        top_k=0,
        dev="cpu",
    )

    assert out.imgs.shape == (5, 3, 2, 2)
    assert out.toks.shape == (5, 1, 1, 2)
    assert out.atoms is None
    assert out.coeffs is None


def test_sample_batches_real_vals():
    out = s2.sample(
        _FakeRealNet(),
        _bundle(),
        (1, 1, 2),
        n=5,
        bs=3,
        top_k=0,
        dev="cpu",
    )

    assert out.imgs.shape == (5, 3, 2, 2)
    assert out.toks is None
    assert out.atoms.shape == (5, 1, 1, 2)
    assert out.coeffs.shape == (5, 1, 1, 2)


def test_sample_slide_generates_larger_latent_grid():
    bundle = _bundle(latent_hw=(8, 8), patch_based=True, patch_stride=4)
    out = s2.sample_slide(
        _FakeSlideNet(),
        bundle,
        (2, 2, 2),
        out_h=3,
        out_w=3,
        n=2,
        bs=1,
        dev="cpu",
    )

    assert out.imgs.shape == (2, 3, 2, 2)
    assert out.toks.shape == (2, 3, 3, 2)
    assert torch.all(out.toks == 7)
    assert bundle.model.latent_hw_calls[-1] == (12, 12)


def test_sample_patch_based_uses_stage1_latent_hw_not_token_grid():
    bundle = _bundle(latent_hw=(4, 4), patch_based=True, patch_stride=4)
    out = s2.sample(
        _FakeTokNet(),
        bundle,
        (1, 1, 2),
        n=2,
        bs=2,
        dev="cpu",
    )

    assert out.imgs.shape == (2, 3, 2, 2)
    assert bundle.model.latent_hw_calls[-1] == (4, 4)


def test_reconstruct_patch_based_uses_full_latent_hw_not_token_grid():
    bundle = _bundle(latent_hw=(8, 8), patch_based=True, patch_stride=4)
    reconstruct_stage2_sparse_latent(
        bundle,
        torch.zeros(1, 2, 2, 2, dtype=torch.long),
        torch.zeros(1, 2, 2, 2, dtype=torch.float32),
        device="cpu",
    )

    assert bundle.model.latent_hw_calls[-1] == (8, 8)


def test_decode_stage2_tokens_detects_rq_patch_bundles_without_patch_flag():
    bundle = _bundle(
        latent_hw=(8, 8),
        patch_stride=4,
        kind="rq",
        expose_patch_flag=False,
        patch_size=8,
    )

    decode_stage2_tokens(
        bundle,
        torch.zeros(1, 2, 2, 2, dtype=torch.long),
        device="cpu",
    )

    assert bundle.model.latent_hw_calls[-1] == (8, 8)


def test_pack_dump_keeps_core_paths():
    run = s2.Run(
        ckpt=Path("/tmp/s2.ckpt"),
        cache_pt=Path("/tmp/cache.pt"),
        cache={},
        hps={},
        net=None,
        s1=_bundle(),
        shape=(1, 1, 2),
        dev=torch.device("cpu"),
    )
    out = s2.sample(_FakeRealNet(), _bundle(), (1, 1, 2), n=2, bs=2, dev="cpu")
    dump = s2.pack_dump(out, run)

    assert dump["shape"] == (1, 1, 2)
    assert dump["stage1_checkpoint"] == "/tmp/s1.ckpt"
    assert dump["stage2_checkpoint"] == "/tmp/s2.ckpt"
    assert dump["token_cache"] == "/tmp/cache.pt"
    assert "atom_ids" in dump and "coeffs" in dump
    assert s2.pick_dev("cpu").type == "cpu"
    assert s2.pick_nrow(9, None) == 3


def test_load_run_falls_back_when_saved_cache_path_is_stale():
    tmp = Path("/tmp")
    got = {}
    old = {
        "load_torch_payload": s2.load_torch_payload,
        "infer_latest_token_cache": s2.infer_latest_token_cache,
        "load_token_cache": s2.load_token_cache,
        "ensure_stage2_cache_metadata": s2.ensure_stage2_cache_metadata,
        "_make_net": s2._make_net,
        "load_stage1_decoder_bundle": s2.load_stage1_decoder_bundle,
    }
    try:
        s2.load_torch_payload = lambda path: {"hyper_parameters": {"token_cache_path": "/stale/cache.pt"}}
        s2.infer_latest_token_cache = lambda ar_output_dir=None: tmp / "fresh_cache.pt"
        def _load_cache(path):
            got["cache_pt"] = Path(path)
            return {"shape": (1, 1, 2), "meta": {}}

        s2.load_token_cache = _load_cache
        s2.ensure_stage2_cache_metadata = lambda cache, **kwargs: {"shape": (1, 1, 2), "meta": {}}
        s2._make_net = lambda *args, **kwargs: (_StubNet(), {}, {"shape": (1, 1, 2), "meta": {}})
        s2.load_stage1_decoder_bundle = lambda *args, **kwargs: _bundle()

        run = s2.load_run(ckpt=tmp / "s2.ckpt", ar_dir=tmp / "ar")

        assert got["cache_pt"] == tmp / "fresh_cache.pt"
        assert run.cache_pt == tmp / "fresh_cache.pt"
    finally:
        for key, val in old.items():
            setattr(s2, key, val)


def test_load_run_clears_stage1_meta_before_override_enrichment():
    tmp = Path("/tmp")
    seen = {}
    old = {
        "load_torch_payload": s2.load_torch_payload,
        "load_token_cache": s2.load_token_cache,
        "ensure_stage2_cache_metadata": s2.ensure_stage2_cache_metadata,
        "_make_net": s2._make_net,
        "load_stage1_decoder_bundle": s2.load_stage1_decoder_bundle,
    }
    try:
        (tmp / "cache.pt").write_bytes(b"")
        (tmp / "new_s1.ckpt").write_bytes(b"")
        s2.load_torch_payload = lambda path: {"hyper_parameters": {}}
        s2.load_token_cache = lambda path: {
            "shape": (1, 1, 2),
            "meta": {
                "stage1_checkpoint": "/old/s1.ckpt",
                "image_size": 256,
                "coeff_bin_values": torch.tensor([-1.0, 0.0, 1.0]),
                "latent_hw": (8, 8),
            },
        }

        def _ensure(cache, **kwargs):
            seen["meta"] = dict(cache.get("meta", {}) or {})
            return {"shape": (1, 1, 2), "meta": seen["meta"]}

        s2.ensure_stage2_cache_metadata = _ensure
        s2._make_net = lambda *args, **kwargs: (_StubNet(), {}, {"shape": (1, 1, 2), "meta": seen["meta"]})
        s2.load_stage1_decoder_bundle = lambda *args, **kwargs: _bundle()

        s2.load_run(
            ckpt=tmp / "s2.ckpt",
            cache_pt=tmp / "cache.pt",
            s1_ckpt=tmp / "new_s1.ckpt",
        )

        assert seen["meta"]["stage1_checkpoint"] == str((tmp / "new_s1.ckpt").resolve())
        assert "image_size" not in seen["meta"]
        assert "coeff_bin_values" not in seen["meta"]
        assert "latent_hw" not in seen["meta"]
    finally:
        for key, val in old.items():
            setattr(s2, key, val)


def test_load_run_prefers_prior_cfg_shape_for_crop_trained_checkpoints():
    tmp = Path("/tmp")
    old = {
        "load_torch_payload": s2.load_torch_payload,
        "load_token_cache": s2.load_token_cache,
        "ensure_stage2_cache_metadata": s2.ensure_stage2_cache_metadata,
        "_make_net": s2._make_net,
        "load_stage1_decoder_bundle": s2.load_stage1_decoder_bundle,
    }
    try:
        (tmp / "cache.pt").write_bytes(b"")
        s2.load_torch_payload = lambda path: {"hyper_parameters": {}}
        s2.load_token_cache = lambda path: {"shape": (4, 4, 4), "meta": {}}
        s2.ensure_stage2_cache_metadata = lambda cache, **kwargs: {"shape": (4, 4, 4), "meta": {}}
        s2._make_net = lambda *args, **kwargs: (_StubCropNet(), {}, {"shape": (4, 4, 4), "meta": {}})
        s2.load_stage1_decoder_bundle = lambda *args, **kwargs: _bundle()

        run = s2.load_run(
            ckpt=tmp / "s2.ckpt",
            cache_pt=tmp / "cache.pt",
        )

        assert run.shape == (2, 2, 4)
    finally:
        for key, val in old.items():
            setattr(s2, key, val)


def test_pick_arch_normalizes_gpt_aliases():
    state = {"prior.pos_emb": torch.zeros(1, 4, 8)}

    assert s2._pick_arch(state, {}, "gpt") == "gpt"
    assert s2._pick_arch(state, {"prior_architecture": "mingpt"}, "auto") == "gpt"


def test_legacy_hps_recovers_missing_prior_shape_from_state():
    payload = {
        "state_dict": {
            "prior.token_emb.weight": torch.zeros(10, 32),
            "prior.token_head.weight": torch.zeros(19, 32),
            "prior.spatial_blocks.0.ffn.0.weight": torch.zeros(64, 32),
            "prior.spatial_blocks.1.ffn.0.weight": torch.zeros(64, 32),
            "prior.depth_blocks.0.ffn.0.weight": torch.zeros(64, 32),
            "prior.global_spatial_tokens": torch.zeros(1, 3, 32),
        }
    }
    cache = {
        "tokens_flat": torch.zeros(4, 8, dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {
            "num_atoms": 11,
            "coeff_vocab_size": 8,
            "n_bins": 8,
            "coeff_bin_values": torch.linspace(-1.0, 1.0, steps=8),
            "coeff_max": 5.0,
        },
    }

    hps = s2._legacy_hps(payload, cache)

    assert hps["prior_architecture"] == "spatial_depth"
    assert hps["prior_d_model"] == 32
    assert hps["prior_d_ff"] == 64
    assert hps["prior_n_spatial_layers"] == 2
    assert hps["prior_n_depth_layers"] == 1
    assert hps["prior_n_global_spatial_tokens"] == 3
    assert hps["resolved_total_vocab_size"] == 19


def test_legacy_hps_recovers_gpt_shape_from_state():
    payload = {
        "state_dict": {
            "prior.token_emb.weight": torch.zeros(10, 32),
            "prior.token_head.weight": torch.zeros(19, 32),
            "prior.blocks.0.ffn.0.weight": torch.zeros(64, 32),
            "prior.blocks.1.ffn.0.weight": torch.zeros(64, 32),
            "prior.pos_emb": torch.zeros(1, 8, 32),
        }
    }
    cache = {
        "tokens_flat": torch.zeros(4, 8, dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {
            "num_atoms": 11,
            "coeff_vocab_size": 8,
            "n_bins": 8,
            "coeff_bin_values": torch.linspace(-1.0, 1.0, steps=8),
        },
    }

    hps = s2._legacy_hps(payload, cache, arch="gpt")

    assert hps["prior_architecture"] == "gpt"
    assert hps["prior_d_model"] == 32
    assert hps["prior_d_ff"] == 64
    assert hps["prior_n_layers"] == 2
    assert hps["prior_n_global_spatial_tokens"] == 0


def test_legacy_hps_recovers_gpt_global_prefix_count_from_state():
    payload = {
        "state_dict": {
            "prior.token_emb.weight": torch.zeros(10, 32),
            "prior.token_head.weight": torch.zeros(19, 32),
            "prior.blocks.0.ffn.0.weight": torch.zeros(64, 32),
            "prior.blocks.1.ffn.0.weight": torch.zeros(64, 32),
            "prior.pos_emb": torch.zeros(1, 10, 32),
            "prior.global_spatial_tokens": torch.zeros(1, 2, 32),
        }
    }
    cache = {
        "tokens_flat": torch.zeros(4, 8, dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {
            "num_atoms": 11,
            "coeff_vocab_size": 8,
            "n_bins": 8,
            "coeff_bin_values": torch.linspace(-1.0, 1.0, steps=8),
        },
    }

    hps = s2._legacy_hps(payload, cache, arch="gpt")

    assert hps["prior_architecture"] == "gpt"
    assert hps["prior_n_layers"] == 2
    assert hps["prior_n_global_spatial_tokens"] == 2


def _rq_state(*, embedding_dim=16, dict_rows=None, num_atoms=2048):
    dict_rows = int(dict_rows or embedding_dim)
    return {
        "bottleneck.dictionary": torch.zeros(dict_rows, num_atoms),
        "encoder.conv_in.weight": torch.zeros(64, 3, 3, 3),
        "encoder.down.0.block.0.conv1.weight": torch.zeros(64, 64, 3, 3),
        "encoder.down.1.block.0.conv1.weight": torch.zeros(128, 64, 3, 3),
        "encoder.down.2.block.0.conv1.weight": torch.zeros(128, 128, 3, 3),
        "encoder.norm_out.weight": torch.zeros(128),
        "encoder.conv_out.weight": torch.zeros(embedding_dim, 128, 3, 3),
    }


def test_infer_rq_stage1_config_recovers_patch_layout_from_checkpoint_shape():
    state = _rq_state(dict_rows=16 * 8 * 8)
    cache = {
        "shape": (16, 16, 32),
        "meta": {"image_size": 256},
    }

    cfg = _infer_rq_stage1_config(state, cache)

    assert cfg["patch_based"] is True
    assert cfg["patch_size"] == 8
    assert cfg["patch_stride"] == 4
    assert cfg["patch_reconstruction"] == "center_crop"
    assert cfg["sparsity_level"] == 16


def test_infer_rq_stage1_config_halves_quantized_token_depth_for_sparsity():
    state = _rq_state(dict_rows=16)
    cache = {
        "shape": (64, 64, 16),
        "meta": {"image_size": 256},
    }

    cfg = _infer_rq_stage1_config(state, cache)

    assert cfg["patch_based"] is False
    assert cfg["sparsity_level"] == 8
