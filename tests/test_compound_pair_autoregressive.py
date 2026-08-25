from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from scripts.train_official_rqtransformer_laser_stage2 import (
    CompoundLaserRQTransformer,
    build_model,
)
from src.models.rqtransformer.configs import RQTransformerConfig


def tiny_config(depth=2):
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


def tiny_aux(depth=2):
    dictionary = torch.tensor([
        [1.0, 0.0, -1.0, 0.5, 0.2, -0.4, 0.7],
        [0.0, 1.0, 0.5, -1.0, 0.3, 0.8, -0.2],
        [0.5, -0.5, 1.0, 0.0, -0.7, 0.1, 0.6],
        [-0.5, 0.5, 0.0, 1.0, 0.9, -0.3, 0.4],
    ])
    coeff_bins = torch.linspace(-1.0, 1.0, 5)
    coeff_scales = torch.arange(1, depth + 1, dtype=torch.float32)

    def compound_embeddings(atoms, coeff_ids):
        vectors = dictionary.t()[atoms.long()]
        coefficients = coeff_bins[coeff_ids.long()] * coeff_scales
        return vectors * coefficients[..., None]

    return SimpleNamespace(
        dictionary=dictionary,
        coeff_bins=coeff_bins,
        coeff_scales=coeff_scales,
        compound_embeddings=compound_embeddings,
    )


def test_full_pair_chain_uses_past_coefficient_but_not_current_coefficient():
    torch.manual_seed(10)
    model = CompoundLaserRQTransformer(
        tiny_config(),
        num_atoms=7,
        coeff_vocab_size=5,
        pair_autoregressive=True,
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4]]]])
    baseline = atoms * 5 + torch.tensor([[[[0, 2]]]])
    changed_past = atoms * 5 + torch.tensor([[[[4, 2]]]])
    changed_current = atoms * 5 + torch.tensor([[[[0, 4]]]])

    with torch.no_grad():
        base_outputs = model(baseline, model_aux=aux)
        past_outputs = model(changed_past, model_aux=aux)
        current_outputs = model(changed_current, model_aux=aux)

    # Event zero has no pair history, so neither its own nor future
    # coefficient token can influence either prediction made at that event.
    assert torch.equal(
        base_outputs["atom_logits"][..., 0, :],
        past_outputs["atom_logits"][..., 0, :],
    )
    assert torch.equal(
        base_outputs["coeff_logits"][..., 0, :],
        past_outputs["coeff_logits"][..., 0, :],
    )
    # Event one consumes the completed event-zero pair, including c_0.
    assert not torch.equal(
        base_outputs["atom_logits"][..., 1, :],
        past_outputs["atom_logits"][..., 1, :],
    )
    assert not torch.equal(
        base_outputs["coeff_logits"][..., 1, :],
        past_outputs["coeff_logits"][..., 1, :],
    )
    # Its own target coefficient is shifted out and therefore cannot leak.
    assert torch.equal(
        base_outputs["atom_logits"][..., 1, :],
        current_outputs["atom_logits"][..., 1, :],
    )
    assert torch.equal(
        base_outputs["coeff_logits"][..., 1, :],
        current_outputs["coeff_logits"][..., 1, :],
    )


def test_full_pair_cached_sampling_matches_parallel_teacher_forcing():
    torch.manual_seed(11)
    model = CompoundLaserRQTransformer(
        tiny_config(),
        num_atoms=7,
        coeff_vocab_size=5,
        pair_autoregressive=True,
        micro_transformer_layers=1,
        depth_specific_coeff_heads=True,
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4]]]])
    tokens = atoms * 5 + torch.tensor([[[[1, 3]]]])

    with torch.no_grad():
        teacher = model(tokens, model_aux=aux)
        model.init_cache()
        cached_atoms = []
        cached_coefficients = []
        for depth_index in range(2):
            hidden = model.cached_head_output(
                tokens, aux, None, (0, 0, depth_index), amp=False
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
            cached_coefficients.append(model.coefficient_logits(
                hidden, atom_vector, depth_index=depth_index
            ))
        model.init_cache()

    cached_atom_logits = torch.stack(cached_atoms, dim=1).reshape(
        1, 1, 1, 2, 7
    )
    cached_coefficient_logits = torch.stack(
        cached_coefficients, dim=1
    ).reshape(1, 1, 1, 2, 5)
    assert torch.allclose(
        cached_atom_logits, teacher["atom_logits"], atol=2e-7, rtol=1e-6
    )
    assert torch.allclose(
        cached_coefficient_logits,
        teacher["coeff_logits"],
        atol=2e-7,
        rtol=1e-6,
    )


def test_full_pair_cached_next_event_reads_generated_coefficient():
    torch.manual_seed(12)
    model = CompoundLaserRQTransformer(
        tiny_config(),
        num_atoms=7,
        coeff_vocab_size=5,
        pair_autoregressive=True,
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4]]]])
    low = atoms * 5 + torch.tensor([[[[0, 2]]]])
    high = atoms * 5 + torch.tensor([[[[4, 2]]]])
    changed_future = atoms * 5 + torch.tensor([[[[0, 4]]]])

    def event_one_hidden(tokens):
        model.init_cache()
        model.cached_head_output(tokens, aux, None, (0, 0, 0), amp=False)
        return model.cached_head_output(
            tokens, aux, None, (0, 0, 1), amp=False
        )

    with torch.no_grad():
        low_hidden = event_one_hidden(low)
        high_hidden = event_one_hidden(high)
        future_hidden = event_one_hidden(changed_future)
        model.init_cache()

    assert not torch.equal(low_hidden, high_hidden)
    assert torch.equal(low_hidden, future_hidden)


def test_builder_enables_full_pair_ar_without_changing_depth():
    with torch.device("meta"):
        model = build_model(
            18_432,
            16_384,
            compound=True,
            compound_pair_autoregressive=True,
            coeff_vocab_size=2_048,
            sparsity_level=4,
            model_preset="lsun-church-350m",
        )

    assert model.pair_autoregressive is True
    assert model.causal_prefix_state is False
    assert tuple(model.block_size) == (8, 8, 4)


def test_full_pair_ar_rejects_learned_prefix_surrogate():
    try:
        CompoundLaserRQTransformer(
            tiny_config(),
            num_atoms=7,
            coeff_vocab_size=5,
            pair_autoregressive=True,
            causal_prefix_state=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("incompatible pair states were accepted")
