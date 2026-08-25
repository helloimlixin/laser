from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.models.rqtransformer.configs import RQTransformerConfig
from scripts.train_official_rqtransformer_laser_stage2 import (
    CausalPrefixPatternCacheDataset,
    CausalPrefixPatternLaserRQTransformer,
    CompoundLaserRQTransformer,
    causal_prefix_pattern_objective,
    initialize_causal_prefix_pattern_from_compound,
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


class TinyPrefixAux(SimpleNamespace):
    def prefix_pattern_coefficients(self, pattern_ids):
        return torch.stack([
            self.prefix_patterns[index, pattern_ids[..., index].long()]
            for index in range(pattern_ids.shape[-1])
        ], dim=-2)

    def prefix_pattern_latents(self, atoms, pattern_ids):
        coefficients = self.prefix_pattern_coefficients(pattern_ids)
        vectors = self.dictionary.t()[atoms.long()]
        return torch.einsum("...di,...ic->...dc", coefficients, vectors)

    def compound_embeddings(self, atoms, coeff_ids):
        vectors = self.dictionary.t()[atoms.long()]
        coefficients = self.coeff_bins[coeff_ids.long()]
        coefficients = coefficients * self.coeff_scales.view(
            *([1] * (coefficients.ndim - 1)), -1
        )
        return vectors * coefficients.unsqueeze(-1)


def tiny_aux(depth=3, vocab_sizes=(3, 4, 5), coeff_vocab=5):
    dictionary = torch.randn(4, 7)
    dictionary = torch.nn.functional.normalize(dictionary, dim=0)
    patterns = torch.zeros(depth, max(vocab_sizes), depth)
    for depth_index, vocab_size in enumerate(vocab_sizes):
        patterns[depth_index, :vocab_size, : depth_index + 1] = torch.randn(
            vocab_size, depth_index + 1
        )
    return TinyPrefixAux(
        dictionary=dictionary,
        prefix_patterns=patterns,
        prefix_pattern_vocab_sizes=torch.tensor(vocab_sizes),
        coeff_bins=torch.linspace(-2, 2, coeff_vocab),
        coeff_scales=torch.ones(depth),
    )


def tiny_model(depth=3, vocab_sizes=(3, 4, 5), micro_layers=1):
    return CausalPrefixPatternLaserRQTransformer(
        tiny_config(depth),
        num_atoms=7,
        coeff_vocab_size=5,
        prefix_pattern_vocab_sizes=vocab_sizes,
        micro_transformer_layers=micro_layers,
    ).eval()


def test_causal_prefix_pattern_cache_returns_one_code_per_depth(tmp_path):
    target = tmp_path / "prefix_patterns.pt"
    torch.save({
        "atoms": torch.zeros(2, 1, 1, 3, dtype=torch.int16),
        "prefix_pattern_ids": torch.ones(2, 1, 1, 3, dtype=torch.int16),
        "prefix_patterns": torch.zeros(3, 5, 3),
        "prefix_pattern_vocab_sizes": torch.tensor([3, 4, 5]),
        "labels": torch.zeros(2, dtype=torch.int16),
        "meta": {"format": "laser_causal_prefix_patterns_v1"},
    }, target)

    cache = CausalPrefixPatternCacheDataset(target)

    atoms, pattern_ids, label = cache[0]
    assert atoms.shape == pattern_ids.shape == (1, 1, 3)
    assert label.ndim == 0
    assert cache.prefix_patterns.shape == (3, 5, 3)
    assert cache.pattern_vocab_sizes.tolist() == [3, 4, 5]


def test_prefix_pattern_transport_and_cumulative_state():
    torch.manual_seed(0)
    model = tiny_model()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    pattern_ids = torch.tensor([[[[1, 2, 3]]]])

    packed = model.pack(atoms, pattern_ids)
    unpacked_atoms, unpacked_patterns = model.unpack(packed)
    deltas = model.embed_depth_with_model_aux(packed, aux)
    states = torch.cumsum(deltas, dim=-2)
    prefix = aux.prefix_pattern_latents(atoms, pattern_ids)
    vectors = aux.dictionary.t()[atoms]
    expected = prefix + model.causal_depth_adapter(
        torch.cat((vectors, prefix), dim=-1)
    )

    assert torch.equal(unpacked_atoms, atoms)
    assert torch.equal(unpacked_patterns, pattern_ids)
    assert torch.allclose(states, expected, atol=1e-7, rtol=1e-6)


def test_teacher_forcing_is_causal_and_masks_invalid_pattern_classes():
    torch.manual_seed(1)
    model = tiny_model()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    base_patterns = torch.tensor([[[[1, 2, 3]]]])
    changed_future_patterns = torch.tensor([[[[1, 2, 4]]]])
    changed_final_atom = torch.tensor([[[[2, 4, 5]]]])

    with torch.no_grad():
        baseline = model(model.pack(atoms, base_patterns), aux)
        future_pattern = model(
            model.pack(atoms, changed_future_patterns), aux
        )
        future_atom = model(
            model.pack(changed_final_atom, base_patterns), aux
        )

    # Neither the current prefix token nor current atom is visible when atom 2
    # is predicted.  The pattern head then explicitly conditions on atom 2.
    assert torch.equal(
        baseline["atom_logits"][..., 2, :],
        future_pattern["atom_logits"][..., 2, :],
    )
    assert torch.equal(
        baseline["atom_logits"][..., 2, :],
        future_atom["atom_logits"][..., 2, :],
    )
    assert not torch.equal(
        baseline["pattern_logits"][..., 2, :],
        future_atom["pattern_logits"][..., 2, :],
    )
    assert torch.isneginf(baseline["pattern_logits"][..., 0, 3:]).all()
    assert torch.isneginf(baseline["pattern_logits"][..., 1, 4:]).all()
    assert torch.isfinite(baseline["pattern_logits"][..., 2, :5]).all()


def test_cached_prefix_pattern_heads_match_teacher_forcing():
    torch.manual_seed(2)
    model = tiny_model()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    pattern_ids = torch.tensor([[[[1, 2, 3]]]])
    packed = model.pack(atoms, pattern_ids)

    with torch.no_grad():
        teacher = model(packed, model_aux=aux)
        model.init_cache()
        cached_atoms = []
        cached_patterns = []
        for depth_index in range(3):
            hidden = model.cached_head_output(
                packed, aux, None, (0, 0, depth_index), amp=False
            )
            atom_logits = model.classifier(hidden)
            if depth_index:
                atom_logits = atom_logits.clone()
                atom_logits.scatter_(
                    1,
                    atoms[..., :depth_index].reshape(1, depth_index),
                    -float("inf"),
                )
            cached_atoms.append(atom_logits)
            atom_vector = aux.dictionary.t()[
                atoms[..., depth_index].reshape(-1)
            ]
            local = model.prefix_pattern_logits(
                hidden, atom_vector, depth_index
            )
            padded = local.new_full((1, 5), -float("inf"))
            padded[:, : local.shape[-1]] = local
            cached_patterns.append(padded)
        model.init_cache()

    cached_atom_logits = torch.stack(cached_atoms, dim=1).reshape(
        1, 1, 1, 3, 7
    )
    cached_pattern_logits = torch.stack(cached_patterns, dim=1).reshape(
        1, 1, 1, 3, 5
    )
    assert torch.allclose(
        cached_atom_logits, teacher["atom_logits"], atol=2e-7, rtol=1e-6
    )
    assert torch.allclose(
        cached_pattern_logits,
        teacher["pattern_logits"],
        atol=2e-7,
        rtol=1e-6,
    )


def test_prefix_pattern_objective_weights_each_causal_event():
    atom_logits = torch.tensor(
        [[[[[2.0, -1.0], [-1.0, 2.0]]]]], requires_grad=True
    )
    pattern_logits = torch.tensor(
        [[[[[2.0, -1.0, -2.0], [-1.0, 2.0, -2.0]]]]],
        requires_grad=True,
    )
    atoms = torch.tensor([[[[0, 1]]]])
    patterns = torch.tensor([[[[0, 1]]]])

    loss, values = causal_prefix_pattern_objective(
        atom_logits,
        pattern_logits,
        atoms,
        patterns,
        atom_weight=2.0,
        pattern_weight=3.0,
    )
    expected = (
        2.0 * values["atom_nll"].sum(dim=-1)
        + 3.0 * values["pattern_nll"].sum(dim=-1)
    ).mean() / 10.0
    loss.backward()

    assert torch.allclose(loss, expected)
    assert torch.isfinite(atom_logits.grad).all()
    assert torch.isfinite(pattern_logits.grad).all()


def test_pairhard_backbone_transfer_leaves_only_new_classifiers():
    torch.manual_seed(3)
    source = CompoundLaserRQTransformer(
        tiny_config(),
        num_atoms=7,
        coeff_vocab_size=5,
        micro_transformer_layers=1,
        depth_specific_coeff_heads=True,
        causal_prefix_state=True,
    )
    target = tiny_model()

    report = initialize_causal_prefix_pattern_from_compound(
        target, source.state_dict()
    )

    target_state = target.state_dict()
    source_state = source.state_dict()
    for key, value in target_state.items():
        if key.startswith("prefix_pattern_classifier."):
            continue
        assert torch.equal(value, source_state[key]), key
    assert report["copied_parameters"] < report["total_parameters"]
    assert report["new_tensors"]
    assert all(
        key.startswith("prefix_pattern_classifier.")
        for key in report["new_tensors"]
    )


def test_prefix_pattern_sampling_produces_valid_causal_codes():
    torch.manual_seed(4)
    model = tiny_model()
    aux = tiny_aux()

    atoms, pattern_ids = model.sample_compound(
        4,
        aux,
        atom_top_k=7,
        atom_top_p=1.0,
        coeff_top_k=0,
        coeff_top_p=1.0,
        amp=False,
    )

    assert atoms.shape == pattern_ids.shape == (4, 1, 1, 3)
    assert all(len(set(row)) == 3 for row in atoms[:, 0, 0].tolist())
    for depth_index, vocab_size in enumerate((3, 4, 5)):
        local = pattern_ids[..., depth_index]
        assert (local >= 0).all() and (local < vocab_size).all()
