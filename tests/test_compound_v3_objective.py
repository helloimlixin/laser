import torch
from omegaconf import OmegaConf

from src.models.rqtransformer.configs import RQTransformerConfig
from scripts.train_official_rqtransformer_laser_stage2 import (
    CompoundLaserRQTransformer,
    compound_objective,
    scheduled_geometry_weight,
)


def test_compound_objective_weights_atoms_and_zeroes_exact_geometry():
    atom_logits = torch.tensor([[[[[2.0, -1.0], [-1.0, 2.0]]]]])
    coeff_logits = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    target_atoms = torch.tensor([[[[0, 1]]]])
    target_coeff_probs = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    physical = torch.randn(1, 1, 1, 2, 4)

    loss, values = compound_objective(
        atom_logits,
        coeff_logits,
        physical.clone(),
        target_atoms,
        target_coeff_probs,
        physical,
        atom_weight=2.0,
        geometry_weight=0.25,
        accumulation=1,
    )

    expected = (
        2.0 * values["atom_nll"].sum(dim=-1)
        + values["coeff_cross_entropy"].sum(dim=-1)
    ).mean() / 6.0
    assert torch.allclose(values["classification"], expected)
    assert values["geometry"] == 0
    assert torch.allclose(loss, expected)


def test_compound_objective_penalizes_pair_and_spatial_geometry():
    atom_logits = torch.zeros(1, 1, 1, 2, 2)
    coeff_logits = torch.zeros(1, 1, 1, 2, 2)
    target_atoms = torch.zeros(1, 1, 1, 2, dtype=torch.long)
    target_coeff_probs = torch.full((1, 1, 1, 2, 2), 0.5)
    target = torch.ones(1, 1, 1, 2, 4)
    prediction = torch.zeros_like(target)

    _, values = compound_objective(
        atom_logits,
        coeff_logits,
        prediction,
        target_atoms,
        target_coeff_probs,
        target,
        atom_weight=2.0,
        geometry_weight=0.25,
        accumulation=1,
    )

    assert values["geometry_pair_mse"] > 0
    assert values["geometry_spatial_mse"] > 0
    assert values["geometry"] > 0


def test_distribution_geometry_uses_sampling_logits_and_backpropagates():
    atom_logits = torch.tensor(
        [[[[[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]]]]],
        requires_grad=True,
    )
    coeff_logits = torch.tensor(
        [[[[[-20.0, 20.0, -20.0], [-20.0, -20.0, 20.0]]]]],
        requires_grad=True,
    )
    target_atoms = torch.tensor([[[[0, 1]]]])
    target_coeff_probs = torch.tensor(
        [[[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]]]
    )
    dictionary = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    coeff_bins = torch.tensor([-1.0, 1.0, 2.0])
    scales = torch.tensor([2.0, 3.0])
    target = torch.tensor([[[[[2.0, 0.0], [0.0, 6.0]]]]])

    loss, values = compound_objective(
        atom_logits,
        coeff_logits,
        None,
        target_atoms,
        target_coeff_probs,
        target,
        atom_weight=1.5,
        geometry_weight=0.25,
        accumulation=1,
        distribution_geometry=True,
        geometry_dictionary=dictionary,
        geometry_coeff_bins=coeff_bins,
        geometry_coeff_scales=scales,
        geometry_top_k=2,
    )
    loss.backward()

    assert values["geometry"] < 1e-6
    assert atom_logits.grad is not None
    assert coeff_logits.grad is not None
    assert torch.isfinite(atom_logits.grad).all()
    assert torch.isfinite(coeff_logits.grad).all()


def test_micro_transformer_and_depth_specific_coeff_heads():
    raw = OmegaConf.create({
        "type": "rq-transformer",
        "block_size": [1, 1, 2],
        "embed_dim": 12,
        "input_embed_dim": 4,
        "shared_tok_emb": True,
        "shared_cls_emb": True,
        "input_emb_vqvae": True,
        "head_emb_vqvae": True,
        "cumsum_depth_ctx": True,
        "vocab_size": 7,
        "vocab_size_cond": 3,
        "block_size_cond": 1,
        "body": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0.0}},
        "head": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0.0}},
    })
    model = CompoundLaserRQTransformer(
        RQTransformerConfig.create(raw),
        num_atoms=7,
        coeff_vocab_size=5,
        micro_transformer_layers=2,
        depth_specific_coeff_heads=True,
    )
    hidden = torch.randn(2, 1, 1, 2, 12, requires_grad=True)
    atom_vectors = torch.randn(2, 1, 1, 2, 4)
    logits = model.coefficient_logits(hidden, atom_vectors)

    assert logits.shape == (2, 1, 1, 2, 5)
    assert model.coeff_classifier[0] is not model.coeff_classifier[1]
    logits.square().mean().backward()
    assert hidden.grad is not None
    assert all(head[1].weight.grad is not None for head in model.coeff_classifier)


def test_geometry_weight_waits_then_warms_up_to_target():
    assert scheduled_geometry_weight(0.05, 1.99, 2.0, 3.0) == 0.0
    assert scheduled_geometry_weight(0.05, 2.0, 2.0, 3.0) == 0.0
    assert abs(scheduled_geometry_weight(0.05, 3.5, 2.0, 3.0) - 0.025) < 1e-12
    assert abs(scheduled_geometry_weight(0.05, 5.0, 2.0, 3.0) - 0.05) < 1e-12
    assert abs(scheduled_geometry_weight(0.05, 20.0, 2.0, 3.0) - 0.05) < 1e-12
