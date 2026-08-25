import torch
from omegaconf import OmegaConf

import scripts.train_official_rqtransformer_laser_stage2 as stage2
from src.models.rqtransformer.configs import RQTransformerConfig


def tiny_scalar_config(depth=4):
    return RQTransformerConfig.create(OmegaConf.create({
        "type": "rq-transformer",
        "block_size": [1, 1, depth],
        "embed_dim": 12,
        "input_embed_dim": 4,
        "shared_tok_emb": True,
        "shared_cls_emb": True,
        "input_emb_vqvae": True,
        "head_emb_vqvae": True,
        "cumsum_depth_ctx": True,
        "vocab_size": 7,
        "vocab_size_cond": 1,
        "block_size_cond": 1,
        "body": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0.0}},
        "head": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0.0}},
    }))


class TinyAux:
    sparsity_level = 2
    num_atoms = 4
    coeff_vocab_size = 3


def test_scalar_sampler_generates_all_alternating_slots(monkeypatch):
    model = stage2.LaserRQTransformer(
        tiny_scalar_config(), num_atoms=TinyAux.num_atoms
    ).eval()
    visited = []
    settings = []

    def fake_cached_forward(_partial, _aux, *, sample_loc, **_kwargs):
        visited.append(sample_loc)
        depth = sample_loc[2]
        logits = torch.full((2, 7), -float("inf"))
        token = (depth // 2 + 1) % 4 if depth % 2 == 0 else 4 + depth // 2
        logits[:, token] = 0.0
        return logits

    def fake_sample(logits, *, temperature, top_k, top_p):
        settings.append((temperature, top_k, top_p))
        return logits.argmax(dim=-1)

    model.cached_forward = fake_cached_forward
    monkeypatch.setattr(stage2, "sample_from_logits", fake_sample)
    tokens = model.sample_sparse(
        2,
        TinyAux(),
        atom_temperature=0.7,
        atom_top_k=2,
        atom_top_p=0.8,
        coeff_temperature=0.9,
        coeff_top_k=1,
        coeff_top_p=0.6,
        amp=False,
    )

    assert tokens.shape == (2, 1, 1, 4)
    assert visited == [(0, 0, depth) for depth in range(4)]
    assert torch.all(tokens[..., 0::2] < TinyAux.num_atoms)
    assert torch.all(tokens[..., 1::2] >= TinyAux.num_atoms)
    assert torch.all(tokens[..., 1::2] < 7)
    assert settings == [
        (0.7, 2, 0.8),
        (0.9, 1, 0.6),
        (0.7, 2, 0.8),
        (0.9, 1, 0.6),
    ]


def test_scalar_sampler_rejects_compound_depth():
    model = stage2.LaserRQTransformer(
        tiny_scalar_config(depth=2), num_atoms=TinyAux.num_atoms
    ).eval()

    try:
        model.sample_sparse(1, TinyAux(), amp=False)
    except ValueError as error:
        assert "two slots per sparse component" in str(error)
    else:
        raise AssertionError("scalar sampler accepted a compound-depth model")
