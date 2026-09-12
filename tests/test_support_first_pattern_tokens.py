from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.models.rqtransformer.configs import RQTransformerConfig
from src.training.rqtransformer import (
    CoefficientPatternCacheDataset,
    SupportFirstLaserRQTransformer,
    support_first_objective,
)


def tiny_config(depth=3):
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
        "body": {
            "n_layer": 1,
            "block": {"n_head": 3, "resid_pdrop": 0.0},
        },
        "head": {
            "n_layer": 1,
            "block": {"n_head": 3, "resid_pdrop": 0.0},
        },
    }))


def tiny_aux(depth=3, pattern_vocab=5):
    dictionary = torch.randn(4, 7)
    dictionary = torch.nn.functional.normalize(dictionary, dim=0)
    patterns = torch.randn(pattern_vocab, depth)

    def coefficient_pattern_latents(atoms, pattern_ids):
        vectors = dictionary.t()[atoms.long()]
        coefficients = patterns[pattern_ids.long()]
        return (vectors * coefficients.unsqueeze(-1)).sum(dim=-2)

    return SimpleNamespace(
        dictionary=dictionary,
        coefficient_patterns=patterns,
        coefficient_pattern_latents=coefficient_pattern_latents,
    )


def test_coefficient_pattern_cache_returns_one_code_per_site(tmp_path):
    target = tmp_path / "patterns.pt"
    torch.save({
        "atoms": torch.zeros(2, 1, 1, 3, dtype=torch.int16),
        "coefficient_pattern_ids": torch.ones(2, 1, 1, dtype=torch.int16),
        "coefficient_patterns": torch.randn(5, 3),
        "labels": torch.zeros(2, dtype=torch.int16),
        "meta": {"format": "laser_coefficient_patterns_v1"},
    }, target)

    cache = CoefficientPatternCacheDataset(target)

    atoms, pattern_ids, label = cache[0]
    assert atoms.shape == (1, 1, 3)
    assert pattern_ids.shape == (1, 1)
    assert label.ndim == 0
    assert cache.coefficient_patterns.shape == (5, 3)


def test_support_first_transport_round_trip_and_site_embedding():
    torch.manual_seed(0)
    model = SupportFirstLaserRQTransformer(
        tiny_config(), num_atoms=7, coefficient_pattern_vocab_size=5
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    pattern_ids = torch.tensor([[[3]]])

    packed = model.pack(atoms, pattern_ids)
    unpacked_atoms, unpacked_patterns = model.unpack(packed)
    embeddings = model.embed_with_model_aux(packed, aux)

    assert torch.equal(unpacked_atoms, atoms)
    assert torch.equal(unpacked_patterns, pattern_ids)
    assert torch.allclose(
        embeddings.sum(dim=-2),
        aux.coefficient_pattern_latents(atoms, pattern_ids),
    )


def test_support_first_teacher_forcing_never_leaks_pattern_or_future_atom():
    torch.manual_seed(1)
    model = SupportFirstLaserRQTransformer(
        tiny_config(), num_atoms=7, coefficient_pattern_vocab_size=5
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    other_final_atom = torch.tensor([[[[2, 4, 5]]]])

    with torch.no_grad():
        low_pattern = model(model.pack(atoms, torch.tensor([[[0]]])), aux)
        high_pattern = model(model.pack(atoms, torch.tensor([[[4]]])), aux)
        changed_final = model(
            model.pack(other_final_atom, torch.tensor([[[0]]])), aux
        )

    # The target pattern is never an input to this site's depth head.
    assert torch.equal(low_pattern["atom_logits"], high_pattern["atom_logits"])
    assert torch.equal(
        low_pattern["pattern_logits"], high_pattern["pattern_logits"]
    )
    # Atom 2 is predicted before atom 2 is known, while the subsequent pattern
    # distribution is explicitly conditioned on the complete selected support.
    assert torch.equal(
        low_pattern["atom_logits"][..., 2, :],
        changed_final["atom_logits"][..., 2, :],
    )
    assert not torch.equal(
        low_pattern["pattern_logits"], changed_final["pattern_logits"]
    )


def test_support_first_cached_heads_match_teacher_forcing():
    torch.manual_seed(2)
    model = SupportFirstLaserRQTransformer(
        tiny_config(), num_atoms=7, coefficient_pattern_vocab_size=5
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    packed = model.pack(atoms, torch.tensor([[[3]]]))

    with torch.no_grad():
        teacher = model(packed, model_aux=aux)
        model.init_cache()
        cached_atoms = []
        final_hidden = None
        for depth_index in range(3):
            final_hidden = model.cached_head_output(
                packed, aux, None, (0, 0, depth_index), amp=False
            )
            atom_logits = model.classifier(final_hidden)
            if depth_index:
                atom_logits = atom_logits.clone()
                atom_logits.scatter_(
                    1,
                    atoms[..., :depth_index].reshape(1, depth_index),
                    -float("inf"),
                )
            cached_atoms.append(atom_logits)
        cached_pattern = model.coefficient_pattern_logits(
            final_hidden, aux.dictionary.t()[atoms.reshape(1, 3)]
        )
        model.init_cache()

    cached_atom_logits = torch.stack(cached_atoms, dim=1).reshape(
        1, 1, 1, 3, 7
    )
    assert torch.allclose(
        cached_atom_logits, teacher["atom_logits"], atol=2e-7, rtol=1e-6
    )
    assert torch.allclose(
        cached_pattern.reshape_as(teacher["pattern_logits"]),
        teacher["pattern_logits"],
        atol=2e-7,
        rtol=1e-6,
    )


def test_support_first_objective_weights_four_atoms_and_one_pattern():
    atom_logits = torch.tensor(
        [[[[[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0], [-1.0, 2.0]]]]],
        requires_grad=True,
    )
    pattern_logits = torch.tensor(
        [[[[2.0, -1.0, -2.0]]]], requires_grad=True
    )
    target_atoms = torch.tensor([[[[0, 1, 0, 1]]]])
    target_pattern_ids = torch.tensor([[[0]]])

    loss, values = support_first_objective(
        atom_logits,
        pattern_logits,
        target_atoms,
        target_pattern_ids,
        atom_weight=2.0,
        accumulation=1,
    )
    expected = (
        2.0 * values["atom_nll"].sum(dim=-1) + values["pattern_nll"]
    ).mean() / 9.0
    loss.backward()

    assert torch.allclose(loss, expected)
    assert atom_logits.grad is not None
    assert pattern_logits.grad is not None
    assert torch.isfinite(atom_logits.grad).all()
    assert torch.isfinite(pattern_logits.grad).all()


def test_support_first_sampling_produces_distinct_support_and_valid_pattern():
    torch.manual_seed(3)
    model = SupportFirstLaserRQTransformer(
        tiny_config(), num_atoms=7, coefficient_pattern_vocab_size=5
    ).eval()
    aux = tiny_aux()

    atoms, pattern_ids = model.sample_compound(
        4,
        aux,
        atom_top_k=7,
        atom_top_p=1.0,
        coeff_top_k=5,
        coeff_top_p=1.0,
        amp=False,
    )

    assert atoms.shape == (4, 1, 1, 3)
    assert pattern_ids.shape == (4, 1, 1)
    assert (pattern_ids >= 0).all() and (pattern_ids < 5).all()
    assert all(len(set(row)) == 3 for row in atoms[:, 0, 0].tolist())
