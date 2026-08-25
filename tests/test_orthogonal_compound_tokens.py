import torch
from omegaconf import OmegaConf

from scripts.train_official_rqtransformer_laser_stage2 import (
    CompoundLaserRQTransformer,
    LaserAux,
    OrthogonalCompoundLaserRQTransformer,
    compound_objective,
    initialize_orthogonal_from_compound,
)
from src.models.rqtransformer.configs import RQTransformerConfig


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
        "body": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0.0}},
        "head": {"n_layer": 1, "block": {"n_head": 3, "resid_pdrop": 0.0}},
    }))


def tiny_aux(depth=3):
    generator = torch.Generator().manual_seed(20260821)
    aux = LaserAux.__new__(LaserAux)
    torch.nn.Module.__init__(aux)
    aux.sparsity_level = depth
    aux.num_atoms = 7
    aux.coeff_vocab_size = 5
    aux.register_buffer(
        "dictionary",
        torch.nn.functional.normalize(
            torch.randn(4, 7, generator=generator), dim=0
        ),
    )
    aux.register_buffer("coeff_bins", torch.linspace(-1, 1, 5))
    aux.register_buffer("coeff_scales", torch.ones(depth))
    return aux


def test_orthogonal_model_can_strictly_load_compound_weights():
    torch.manual_seed(0)
    source = CompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5
    )
    target = OrthogonalCompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5
    )
    target.load_state_dict(source.state_dict(), strict=True)


def test_causal_orthogonal_model_transfers_every_pairhard_weight():
    torch.manual_seed(10)
    source = CompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5,
        causal_prefix_state=True,
    )
    target = OrthogonalCompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5,
        causal_prefix_state=True,
    )

    transfer = initialize_orthogonal_from_compound(
        target, source.state_dict()
    )

    assert transfer["copied_parameters"] == transfer["total_parameters"]
    assert transfer["new_tensors"] == []
    assert transfer["ignored_source_tensors"] == []


def test_causal_orthogonal_atom_path_matches_transferred_pairhard_path():
    torch.manual_seed(11)
    source = CompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5,
        causal_prefix_state=True,
    ).eval()
    target = OrthogonalCompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5,
        causal_prefix_state=True,
    ).eval()
    target.load_state_dict(source.state_dict(), strict=True)
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    coeff_ids = torch.tensor([[[[1, 2, 3]]]])
    packed = atoms * 5 + coeff_ids
    prefixes = aux.orthogonal_embeddings(atoms, coeff_ids).cumsum(dim=-2)

    with torch.no_grad():
        source_atom_logits = source(
            packed,
            model_aux=aux,
            causal_prefix_reconstructions=prefixes,
        )["atom_logits"]
        target_atom_logits = target(
            packed,
            model_aux=aux,
            causal_prefix_reconstructions=prefixes,
        )["atom_logits"]

    torch.testing.assert_close(
        target_atom_logits, source_atom_logits, atol=0, rtol=0
    )


def test_orthogonal_depth_context_uses_past_gamma_but_not_future_gamma():
    torch.manual_seed(1)
    model = OrthogonalCompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    baseline = atoms * 5 + torch.tensor([[[[1, 2, 3]]]])
    changed_past = atoms * 5 + torch.tensor([[[[4, 2, 3]]]])
    changed_future = atoms * 5 + torch.tensor([[[[1, 2, 4]]]])

    with torch.no_grad():
        base_logits = model(baseline, model_aux=aux)["atom_logits"]
        past_logits = model(changed_past, model_aux=aux)["atom_logits"]
        future_logits = model(changed_future, model_aux=aux)["atom_logits"]

    assert not torch.equal(
        base_logits[..., 1, :], past_logits[..., 1, :]
    )
    assert torch.equal(
        base_logits[..., 1, :], future_logits[..., 1, :]
    )


def test_orthogonal_cached_heads_match_teacher_forcing():
    torch.manual_seed(2)
    model = OrthogonalCompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    coeff_ids = torch.tensor([[[[1, 2, 3]]]])
    packed = atoms * 5 + coeff_ids

    with torch.no_grad():
        teacher = model(packed, model_aux=aux)
        model.init_cache()
        cached_atoms = []
        cached_coefficients = []
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
            basis, _ = aux.orthogonal_basis(atoms[..., : depth_index + 1])
            refined = model.refine_coefficient_hidden(hidden, basis[:, 0, 0, -1])
            cached_atoms.append(atom_logits)
            cached_coefficients.append(
                model.classify_coefficients(refined, depth_index=depth_index)
            )
        model.init_cache()

    cached_atom_logits = torch.stack(cached_atoms, dim=1).reshape(1, 1, 1, 3, 7)
    cached_coeff_logits = torch.stack(cached_coefficients, dim=1).reshape(
        1, 1, 1, 3, 5
    )
    torch.testing.assert_close(
        cached_atom_logits, teacher["atom_logits"], atol=2e-7, rtol=1e-6
    )
    torch.testing.assert_close(
        cached_coeff_logits, teacher["coeff_logits"], atol=2e-7, rtol=1e-6
    )


def test_causal_orthogonal_sampling_accumulates_exact_generated_prefixes():
    torch.manual_seed(12)
    model = OrthogonalCompoundLaserRQTransformer(
        tiny_config(), num_atoms=7, coeff_vocab_size=5,
        causal_prefix_state=True,
    ).eval()
    aux = tiny_aux()
    observed_prefixes = []
    cached_head_output = model.cached_head_output

    def capture_prefix(packed, model_aux, cond, sample_loc, amp=True):
        observed_prefixes.append(
            model._active_prefix_reconstructions.detach().clone()
        )
        return cached_head_output(
            packed, model_aux, cond, sample_loc, amp=amp
        )

    model.cached_head_output = capture_prefix
    with torch.no_grad():
        atoms, coeff_ids = model.sample_compound(
            1,
            aux,
            atom_top_k=1,
            coeff_top_k=1,
            atom_top_p=1.0,
            coeff_top_p=1.0,
            amp=False,
        )

    contributions = aux.orthogonal_embeddings(atoms, coeff_ids)
    expected_prefixes = contributions.cumsum(dim=-2)
    assert torch.count_nonzero(observed_prefixes[0]) == 0
    torch.testing.assert_close(
        observed_prefixes[1][..., 0, :],
        expected_prefixes[..., 0, :],
    )
    torch.testing.assert_close(
        observed_prefixes[2][..., 1, :],
        expected_prefixes[..., 1, :],
    )
    assert model._active_prefix_reconstructions is None


def test_distribution_geometry_uses_orthogonal_candidate_directions():
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    coeff_ids = torch.tensor([[[[3, 3, 3]]]])
    atom_logits = torch.full((1, 1, 1, 3, 7), -100.0)
    atom_logits.scatter_(-1, atoms.unsqueeze(-1), 100.0)
    coeff_logits = torch.full((1, 1, 1, 3, 5), -100.0)
    coeff_logits.scatter_(-1, coeff_ids.unsqueeze(-1), 100.0)
    coeff_targets = torch.nn.functional.one_hot(
        coeff_ids, num_classes=5
    ).float()
    physical = aux.orthogonal_embeddings(atoms, coeff_ids)

    _, orthogonal = compound_objective(
        atom_logits,
        coeff_logits,
        None,
        atoms,
        coeff_targets,
        physical,
        atom_weight=1.5,
        geometry_weight=0.05,
        accumulation=1,
        distribution_geometry=True,
        geometry_dictionary=aux.dictionary,
        geometry_coeff_bins=aux.coeff_bins,
        geometry_coeff_scales=aux.coeff_scales,
        geometry_top_k=1,
        geometry_orthogonal=True,
    )
    _, raw_dictionary = compound_objective(
        atom_logits,
        coeff_logits,
        None,
        atoms,
        coeff_targets,
        physical,
        atom_weight=1.5,
        geometry_weight=0.05,
        accumulation=1,
        distribution_geometry=True,
        geometry_dictionary=aux.dictionary,
        geometry_coeff_bins=aux.coeff_bins,
        geometry_coeff_scales=aux.coeff_scales,
        geometry_top_k=1,
        geometry_orthogonal=False,
    )

    assert orthogonal["geometry"].item() < 1e-10
    assert raw_dictionary["geometry"].item() > 1e-4


def test_closed_loop_coefficient_controls_the_next_atom_prediction():
    torch.manual_seed(13)
    low = OrthogonalCompoundLaserRQTransformer(
        tiny_config(),
        num_atoms=7,
        coeff_vocab_size=5,
        causal_prefix_state=True,
        closed_loop_coefficients=True,
        closed_loop_coeff_state="expected",
    ).eval()
    high = OrthogonalCompoundLaserRQTransformer(
        tiny_config(),
        num_atoms=7,
        coeff_vocab_size=5,
        causal_prefix_state=True,
        closed_loop_coefficients=True,
        closed_loop_coeff_state="expected",
    ).eval()
    high.load_state_dict(low.state_dict(), strict=True)
    with torch.no_grad():
        low.coeff_classifier[1].weight.zero_()
        high.coeff_classifier[1].weight.zero_()
        low.coeff_classifier[1].bias.fill_(-100.0)
        high.coeff_classifier[1].bias.fill_(-100.0)
        low.coeff_classifier[1].bias[0] = 100.0
        high.coeff_classifier[1].bias[-1] = 100.0

    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    coeff_ids = torch.tensor([[[[1, 2, 3]]]])
    packed = atoms * 5 + coeff_ids
    target_prefix = aux.orthogonal_embeddings(
        atoms, coeff_ids
    ).cumsum(dim=-2)

    with torch.no_grad():
        low_logits = low(
            packed,
            model_aux=aux,
            causal_prefix_reconstructions=target_prefix,
        )["atom_logits"]
        high_logits = high(
            packed,
            model_aux=aux,
            causal_prefix_reconstructions=target_prefix,
        )["atom_logits"]

    torch.testing.assert_close(
        low_logits[..., 0, :], high_logits[..., 0, :], atol=0, rtol=0
    )
    assert not torch.equal(
        low_logits[..., 1, :], high_logits[..., 1, :]
    )


def test_future_atom_loss_backpropagates_into_past_coefficient_head():
    torch.manual_seed(14)
    model = OrthogonalCompoundLaserRQTransformer(
        tiny_config(),
        num_atoms=7,
        coeff_vocab_size=5,
        causal_prefix_state=True,
        closed_loop_coefficients=True,
        closed_loop_coeff_state="expected",
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    coeff_ids = torch.tensor([[[[1, 2, 3]]]])
    packed = atoms * 5 + coeff_ids
    target_prefix = aux.orthogonal_embeddings(
        atoms, coeff_ids
    ).cumsum(dim=-2)

    outputs = model(
        packed,
        model_aux=aux,
        causal_prefix_reconstructions=target_prefix,
    )
    future_atom_loss = outputs["atom_logits"][..., 1, :].square().mean()
    gradient = torch.autograd.grad(
        future_atom_loss,
        model.coeff_classifier[1].weight,
    )[0]

    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0


def test_closed_loop_prefix_is_the_predicted_coefficient_accumulation():
    torch.manual_seed(15)
    model = OrthogonalCompoundLaserRQTransformer(
        tiny_config(),
        num_atoms=7,
        coeff_vocab_size=5,
        causal_prefix_state=True,
        closed_loop_coefficients=True,
        closed_loop_coeff_state="expected",
    ).eval()
    aux = tiny_aux()
    atoms = torch.tensor([[[[2, 4, 6]]]])
    coeff_ids = torch.tensor([[[[1, 2, 3]]]])
    packed = atoms * 5 + coeff_ids
    target_prefix = aux.orthogonal_embeddings(
        atoms, coeff_ids
    ).cumsum(dim=-2)

    with torch.no_grad():
        outputs = model(
            packed,
            model_aux=aux,
            causal_prefix_reconstructions=target_prefix,
        )
        normalized = (
            outputs["coeff_logits"].float().softmax(dim=-1)
            * aux.coeff_bins
        ).sum(dim=-1)
        basis, _ = aux.orthogonal_basis(atoms)
        expected = (
            basis
            * (normalized * aux.coeff_scales).unsqueeze(-1)
        ).cumsum(dim=-2)

    torch.testing.assert_close(
        outputs["closed_loop_prefix"], expected, atol=2e-7, rtol=1e-6
    )
