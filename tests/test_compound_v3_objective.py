import torch
from types import SimpleNamespace
from omegaconf import OmegaConf

from src.models.rqtransformer.configs import RQTransformerConfig
from scripts.train_official_rqtransformer_laser_stage2 import (
    CompoundLaserRQTransformer,
    LaserAux,
    SparseTokenCacheDataset,
    build_model,
    compound_objective,
    scheduled_geometry_weight,
)


def tiny_compound_config(depth=2):
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


def tiny_compound_aux(depth=2):
    dictionary = torch.randn(4, 7)
    coeff_bins = torch.linspace(-1.0, 1.0, 5)
    scales = torch.ones(depth)

    def compound_embeddings(atoms, coeff_ids):
        vectors = dictionary.t()[atoms.long()]
        coefficients = coeff_bins[coeff_ids.long()] * scales
        return vectors * coefficients[..., None]

    return SimpleNamespace(
        dictionary=dictionary,
        coeff_bins=coeff_bins,
        coeff_scales=scales,
        compound_embeddings=compound_embeddings,
    )


def test_hard_compound_targets_use_nearest_custom_coefficient_center():
    aux = LaserAux.__new__(LaserAux)
    torch.nn.Module.__init__(aux)
    aux.coeff_vocab_size = 4
    aux.register_buffer("coeff_bins", torch.tensor([-3.0, -0.25, 0.5, 3.0]))
    coefficients = torch.tensor([[[[-0.2, 2.8]]]])

    ids, probabilities = aux.compound_coeff_ids(
        coefficients, stochastic=False, hard=True
    )

    assert torch.equal(ids, torch.tensor([[[[1, 3]]]]))
    assert torch.equal(
        probabilities,
        torch.tensor([[[[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]]]),
    )


def test_compound_soft_targets_measure_distance_in_physical_coefficient_space():
    aux = LaserAux.__new__(LaserAux)
    torch.nn.Module.__init__(aux)
    aux.coeff_vocab_size = 3
    aux.sparsity_level = 2
    aux.soft_target_physical = True
    aux.register_buffer("coeff_bins", torch.tensor([-1.0, 0.0, 1.0]))
    aux.register_buffer("coeff_scales", torch.tensor([10.0, 1.0]))
    coefficients = torch.zeros(1, 1, 1, 2)

    _, probabilities = aux.compound_coeff_ids(
        coefficients, stochastic=False, temp=0.5
    )

    # Equal normalized offsets have different physical distances.  The large
    # first-depth scale must produce a much sharper target around zero.
    assert probabilities[0, 0, 0, 0, 1] > 0.999
    assert probabilities[0, 0, 0, 1, 1] < 0.8


def test_compound_depth_history_does_not_leak_final_omp_coefficients():
    model = CompoundLaserRQTransformer(
        tiny_compound_config(), num_atoms=7, coeff_vocab_size=5
    ).eval()
    aux = tiny_compound_aux()
    atoms = torch.tensor([[[[2, 4]]]])
    first = atoms * 5 + torch.tensor([[[[0, 2]]]])
    second = atoms * 5 + torch.tensor([[[[4, 2]]]])

    with torch.no_grad():
        first_logits = model(first, model_aux=aux)["atom_logits"]
        second_logits = model(second, model_aux=aux)["atom_logits"]

    # At a single spatial site, depth 1 may depend on atom 0 but not on atom
    # 0's final OMP coefficient, which was only determined after atom 1.
    assert torch.equal(first_logits[..., 1, :], second_logits[..., 1, :])


def test_encode_sparse_components_returns_every_causal_omp_prefix():
    aux = LaserAux.__new__(LaserAux)
    torch.nn.Module.__init__(aux)
    aux.encoder = torch.nn.Identity()
    aux.quant_conv = torch.nn.Identity()
    aux.sparsity_level = 2
    aux.coeff_max = 10.0
    aux.clamp_coeffs = False
    aux.register_buffer("dictionary", torch.eye(2))
    aux.register_buffer("coeff_scales", torch.ones(2))
    images = torch.tensor([[[[3.0]], [[2.0]]]])

    atoms, final_coeffs, prefixes = aux.encode_sparse_components(
        images, return_prefix_coeffs=True
    )

    assert atoms.tolist() == [[[[0, 1]]]]
    assert torch.allclose(final_coeffs, torch.tensor([[[[3.0, 2.0]]]]))
    assert torch.allclose(
        prefixes,
        torch.tensor([[[[[3.0, 0.0], [3.0, 2.0]]]]]),
    )
    reconstructed = aux.causal_prefix_reconstructions(atoms, prefixes)
    assert torch.allclose(
        reconstructed,
        torch.tensor([[[[[3.0, 0.0], [3.0, 2.0]]]]]),
    )


def test_sparse_cache_requires_and_returns_causal_prefixes(tmp_path):
    target = tmp_path / "cache.pt"
    payload = {
        "atoms": torch.zeros(2, 1, 1, 2, dtype=torch.int16),
        "coeffs": torch.zeros(2, 1, 1, 2, dtype=torch.float16),
        "prefix_coeffs": torch.zeros(2, 1, 1, 2, 2, dtype=torch.float16),
        "labels": torch.zeros(2, dtype=torch.int16),
        "meta": {"format": "laser_compound_causal_prefix_v2"},
    }
    torch.save(payload, target)

    cache = SparseTokenCacheDataset(target, include_prefix_coeffs=True)

    assert len(cache[0]) == 4
    assert cache[0][2].shape == (1, 1, 2, 2)


def test_causal_prefix_depth_context_uses_past_but_not_future_state():
    torch.manual_seed(0)
    model = CompoundLaserRQTransformer(
        tiny_compound_config(), num_atoms=7, coeff_vocab_size=5,
        causal_prefix_state=True,
    ).eval()
    aux = tiny_compound_aux()
    tokens = torch.tensor([[[[2 * 5 + 1, 4 * 5 + 3]]]])
    base = torch.zeros(1, 1, 1, 2, 4)
    changed_future = base.clone()
    changed_future[..., 1, :] = 100.0
    changed_past = base.clone()
    changed_past[..., 0, :] = 100.0

    with torch.no_grad():
        baseline = model(
            tokens, model_aux=aux, causal_prefix_reconstructions=base
        )
        future = model(
            tokens, model_aux=aux, causal_prefix_reconstructions=changed_future
        )
        past = model(
            tokens, model_aux=aux, causal_prefix_reconstructions=changed_past
        )

    # Atom 1 is conditioned on prefix 0. Prefix 1 is a target produced only
    # after atom 1, so changing it cannot affect the atom-1 logits.
    assert torch.equal(
        baseline["atom_logits"][..., 1, :], future["atom_logits"][..., 1, :]
    )
    assert not torch.equal(
        baseline["atom_logits"][..., 1, :], past["atom_logits"][..., 1, :]
    )
    assert baseline["causal_prefix_prediction"].shape == (1, 1, 1, 2, 4)


def test_compound_objective_trains_causal_prefix_prediction():
    atom_logits = torch.zeros(1, 1, 1, 2, 2)
    coeff_logits = torch.zeros(1, 1, 1, 2, 2)
    target_atoms = torch.zeros(1, 1, 1, 2, dtype=torch.long)
    target_coeff_probs = torch.full((1, 1, 1, 2, 2), 0.5)
    target_prefix = torch.ones(1, 1, 1, 2, 4)
    predicted_prefix = torch.zeros_like(target_prefix, requires_grad=True)

    loss, values = compound_objective(
        atom_logits,
        coeff_logits,
        None,
        target_atoms,
        target_coeff_probs,
        None,
        atom_weight=1.0,
        causal_prefix_prediction=predicted_prefix,
        target_causal_prefix=target_prefix,
        causal_prefix_weight=0.25,
        geometry_weight=0.0,
        accumulation=1,
    )
    loss.backward()

    assert torch.allclose(values["causal_prefix"], torch.tensor(1.0))
    assert predicted_prefix.grad is not None
    assert torch.isfinite(predicted_prefix.grad).all()


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


def test_compound_objective_directly_regresses_expected_coefficient():
    atom_logits = torch.zeros(1, 1, 1, 1, 2)
    coeff_logits = torch.tensor([[[[[-5.0, 0.0, 5.0]]]]], requires_grad=True)
    target_atoms = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    target_coeff_probs = torch.tensor([[[[[1.0, 0.0, 0.0]]]]])
    bins = torch.tensor([-1.0, 0.0, 1.0])

    loss, values = compound_objective(
        atom_logits,
        coeff_logits,
        None,
        target_atoms,
        target_coeff_probs,
        None,
        atom_weight=1.0,
        coeff_regression_weight=2.0,
        geometry_weight=0.0,
        accumulation=1,
        coefficient_bins=bins,
    )
    loss.backward()

    assert values["coefficient_regression"] > 0
    assert values["predicted_coefficients"] > values["target_coefficients"]
    assert coeff_logits.grad is not None
    assert torch.isfinite(coeff_logits.grad).all()


def test_compound_objective_crps_respects_coefficient_bin_distance():
    atom_logits = torch.zeros(1, 1, 1, 1, 2)
    target_atoms = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    target_coeff_probs = torch.tensor([[[[[1.0, 0.0, 0.0]]]]])
    bins = torch.tensor([-1.0, 0.0, 1.0])

    def crps_for(logits):
        _, values = compound_objective(
            atom_logits,
            torch.tensor([[[[[*logits]]]]], requires_grad=True),
            None,
            target_atoms,
            target_coeff_probs,
            None,
            atom_weight=1.0,
            coeff_crps_weight=1.0,
            geometry_weight=0.0,
            accumulation=1,
            coefficient_bins=bins,
        )
        return values["coefficient_crps"]

    adjacent = crps_for([-10.0, 10.0, -10.0])
    distant = crps_for([-10.0, -10.0, 10.0])

    assert adjacent > 0
    assert distant > adjacent


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


def test_ffhq_compound_preset_preserves_original_350m_geometry():
    with torch.device("meta"):
        model = build_model(
            3072,
            2048,
            compound=True,
            coeff_vocab_size=1024,
            compound_micro_transformer_layers=2,
            compound_depth_specific_coeff_heads=True,
            model_preset="ffhq-350m",
        )

    assert tuple(model.block_size) == (8, 8, 2)
    assert model.config.embed_dim == 1024
    assert model.config.vocab_size_cond == 1
    assert len(model.body_transformer.blocks) == 24
    assert len(model.head_transformer.blocks) == 4
    assert model.body_transformer.blocks[0].attn.n_head == 16
    assert sum(parameter.numel() for parameter in model.parameters()) == 383_477_760
