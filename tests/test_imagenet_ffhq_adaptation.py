from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src import ffhq_v4_archived as archive
from src.models.rqtransformer.configs import RQTransformerConfig
from src.training.imagenet_ffhq_adapter import (
    ARCHIVE_SHA256, ImageNetFFHQCompound, build_model, compound_objective, verify_archive,
)
from src.training.rqtransformer import LaserAux, scheduled_geometry_weight


def tiny():
    config = RQTransformerConfig.create(OmegaConf.create(dict(
        type='rq-transformer', block_size=[2, 2, 4], embed_dim=12, input_embed_dim=4,
        shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=7,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)))))
    dictionary = torch.randn(4, 7)
    bins, scales = torch.linspace(-3, 3, 5), torch.tensor([8., 4., 3., 2.])
    aux = SimpleNamespace(dictionary=dictionary, coeff_bins=bins, coeff_scales=scales,
        coeff_vocab_size=5, num_atoms=7, soft_target_physical=False, sparsity_level=4)
    aux.compound_embeddings = lambda atoms, ids: dictionary.T[atoms] * (bins[ids] * scales)[..., None]
    return config, aux


def test_archive_identity_and_imagenet_architecture_are_verified_without_weights():
    assert verify_archive() == ARCHIVE_SHA256
    with torch.device('meta'):
        model = build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
            sparsity_level=4, compound_micro_transformer_layers=2,
            compound_depth_specific_coeff_heads=True, compound_pair_autoregressive=True,
            physical_pair_context=False, model_preset='imagenet-1400m')
    assert tuple(model.block_size) == (8, 8, 4)
    assert model.config.vocab_size_cond == 1000
    assert model.config.embed_dim == 1536
    assert len(model.body_transformer.blocks) == 42
    assert len(model.head_transformer.blocks) == 6
    assert len(model.coeff_micro_transformer.blocks) == 2
    assert len(model.coeff_classifier) == 4
    assert model.contribution_head is None


@pytest.mark.parametrize('geometry_weight', [0., .025, .05])
def test_adapted_forward_objective_and_gradients_match_actual_ffhq_archive(geometry_weight):
    torch.manual_seed(71)
    config, aux = tiny()
    original = archive.CompoundLaserRQTransformer(config, 7, 5,
        micro_transformer_layers=2, depth_specific_coeff_heads=True).eval()
    adapted = ImageNetFFHQCompound(config, 7, 5,
        micro_transformer_layers=2, depth_specific_coeff_heads=True).eval()
    adapted.load_state_dict(original.state_dict(), strict=True)
    atoms = torch.arange(32).reshape(2, 2, 2, 4) % 7
    ids = torch.arange(32).reshape_as(atoms) % 5
    tokens, labels = atoms * 5 + ids, torch.tensor([5, 999])
    old = original(tokens, aux, cond=labels)
    new = adapted(tokens, aux, cond=labels)
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    assert torch.isfinite(new['atom_logits']).all()  # Archived training has no seen-atom mask.
    changed = adapted(tokens, aux, cond=torch.tensor([6, 998]))
    assert not torch.equal(new['atom_logits'], changed['atom_logits'])
    probabilities = torch.randn(2, 2, 2, 4, 5).softmax(-1)
    clean_coefficients = torch.randn(2, 2, 2, 4)
    physical = aux.dictionary.T[atoms] * (clean_coefficients * aux.coeff_scales)[..., None]
    settings = dict(atom_weight=1.5, geometry_weight=geometry_weight, accumulation=4,
        distribution_geometry=True, geometry_dictionary=aux.dictionary,
        geometry_coeff_bins=aux.coeff_bins, geometry_coeff_scales=aux.coeff_scales,
        geometry_top_k=4)
    left, old_values = archive.compound_objective(old['atom_logits'], old['coeff_logits'],
        None, atoms, probabilities, physical, **settings)
    right, new_values = compound_objective(new['atom_logits'], new['coeff_logits'],
        None, atoms, probabilities, physical, **settings, coeff_regression_weight=0.,
        coeff_crps_weight=0., coefficient_bins=aux.coeff_bins, geometry_candidate_coeff_logits=None,
        target_atom_ids=None, target_atom_weights=None, target_atom_probabilities=None,
        causal_prefix_prediction=None, target_causal_prefix=None, causal_prefix_weight=0.)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key in old_values:
        torch.testing.assert_close(old_values[key], new_values[key], rtol=0, atol=0)
    left.backward()
    right.backward()
    for (name, p), (other, q) in zip(original.named_parameters(), adapted.named_parameters()):
        assert name == other
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            assert torch.isfinite(q.grad).all()


def test_maintained_evaluation_api_preserves_archived_sampling_and_rng():
    torch.manual_seed(72)
    config, aux = tiny()
    model = ImageNetFFHQCompound(config, 7, 5,
        micro_transformer_layers=2, depth_specific_coeff_heads=True).eval()
    settings = dict(cond=torch.tensor([5, 999]), atom_top_k=7, atom_top_p=.9,
        coeff_top_p=.85, atom_temperature=.9, coeff_temperature=1., amp=False)
    torch.manual_seed(73)
    expected = archive.CompoundLaserRQTransformer.sample_compound(model, 2, aux, **settings)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(73)
    actual = model.sample_compound(2, aux, **dict(settings, atom_top_k=0),
        coeff_top_k=5, causal_prefix_sampling='predicted')
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert bool((actual[0].sort(-1).values.diff(dim=-1) > 0).all())


def test_ffhq_normalized_targets_retain_the_requested_200_bin_kernel():
    _, aux = tiny()
    aux.coeff_vocab_size = 2048
    aux.coeff_bins = torch.linspace(-3, 3, 2048)
    coefficients = torch.zeros(1, 1, 1, 4)
    temperature = 2 * (200 * 6 / 2047)**2
    torch.manual_seed(74)
    expected = archive.LaserAux.compound_coeff_ids(aux, coefficients, temp=temperature)
    torch.manual_seed(74)
    actual = LaserAux.compound_coeff_ids(aux, coefficients, temp=temperature)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    probability = actual[1][0, 0, 0, 0].double()
    centers = aux.coeff_bins.double()
    mean = (probability * centers).sum()
    sigma = (probability * (centers - mean).square()).sum().sqrt()
    assert float(sigma / (6 / 2047)) == pytest.approx(200, rel=1e-4)


def test_geometry_warmup_matches_ffhq_and_nonarchived_objectives_are_rejected():
    for epoch in [0., 1.99, 2., 2.5, 3.5, 5., 20.]:
        assert scheduled_geometry_weight(.05, epoch, 2., 3.) == archive.scheduled_geometry_weight(.05, epoch, 2., 3.)
    with pytest.raises(ValueError, match='coeff_crps_weight'):
        compound_objective(coeff_crps_weight=.1)
