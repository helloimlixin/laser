"""Causality, physical-codec and incremental decoding gates for the DC adapter."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from src.models.compound_dc_transformer import CompoundDCTransformer, sample_categorical


def setup(block_size=(2, 3, 3), *, active=True, overlap=2):
    torch.manual_seed(219)
    model = CompoundDCTransformer(num_atoms=13, coeff_vocab_size=7,
        block_size=block_size, dictionary_dim=5, width=24, heads=3,
        encoder_ffn_layers=(1, 1), atom_ffn_layers=(1, 2),
        coefficient_ffn_layers=(1, 1), overlap=overlap, dropout=0.).eval()
    aux = SimpleNamespace(dictionary=F.normalize(torch.randn(5, 13), dim=0),
        coeff_bins=torch.tensor([-3., -2., -1., 0., 1., 2., 3.]),
        coeff_scales=torch.tensor([2. + 3. * d for d in range(block_size[-1])]))
    sites = block_size[0] * block_size[1]
    atoms = (torch.arange(sites)[:, None] * 3 + torch.arange(block_size[-1])) % 13
    atoms = atoms.reshape(1, *block_size)
    coefficients = torch.arange(atoms.numel()).reshape_as(atoms) % 7
    if active:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'gate' in name:
                    parameter.fill_(.23)
    return model, aux, atoms, coefficients


def target_loss(output, atoms, coefficients):
    return (F.cross_entropy(output['atom_logits'].reshape(-1, 13), atoms.reshape(-1))
            + F.cross_entropy(output['coeff_logits'].reshape(-1, 7), coefficients.reshape(-1))) / 2


def test_physical_pairs_keep_sign_and_exact_depth_scale_including_overlap():
    model, aux, atoms, coefficients = setup()
    flat_a, flat_c = model.depth_major(atoms), model.depth_major(coefficients)
    for chunk in range(model.depths):
        expected = torch.zeros(1, model.sites, 5)
        for depth in range(chunk):
            a = atoms[..., depth].reshape(1, model.sites)
            c = coefficients[..., depth].reshape_as(a)
            expected += aux.dictionary.T[a] * (aux.coeff_bins[c] * aux.coeff_scales[depth])[..., None]
        torch.testing.assert_close(model.prefix_latent(atoms, coefficients, aux, chunk), expected)
    previous = model.sites - 1
    actual = model.pair_physical(flat_a[:, previous], flat_c[:, previous], 0, aux)
    expected = aux.dictionary.T[flat_a[:, previous]] * (aux.coeff_bins[flat_c[:, previous]] * aux.coeff_scales[0])[:, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    wrong = aux.dictionary.T[flat_a[:, previous]] * (aux.coeff_bins[flat_c[:, previous]] * aux.coeff_scales[1])[:, None]
    assert not torch.equal(actual, wrong)


def test_depth_major_round_trip_and_all_real_pairs_predicted_once():
    model, aux, atoms, coefficients = setup()
    packed = atoms * 7 + coefficients
    assert torch.equal(model.raster_depth(model.depth_major(packed)), packed)
    with torch.no_grad():
        output = model(packed, model_aux=aux)
    assert output['atom_logits'].shape == (*packed.shape, 13)
    assert output['coeff_logits'].shape == (*packed.shape, 7)


def test_no_current_coefficient_or_future_event_leakage_with_active_attention():
    model, aux, atoms, coefficients = setup()
    packed = atoms * 7 + coefficients
    with torch.no_grad():
        reference = model(packed, model_aux=aux)
        for event in range(model.events):
            changed = model.depth_major(packed).clone()
            changed[:, event] = (changed[:, event] // 7) * 7 + (changed[:, event] % 7 + 2) % 7
            changed[:, event + 1:] = torch.randint(0, 13 * 7, changed[:, event + 1:].shape)
            actual = model(model.raster_depth(changed), model_aux=aux)
            for key in reference:
                torch.testing.assert_close(model.depth_major(actual[key])[:, :event + 1],
                    model.depth_major(reference[key])[:, :event + 1], rtol=0, atol=0)


def test_current_atom_conditions_only_its_coefficient_and_later_events():
    model, aux, atoms, coefficients = setup()
    packed = atoms * 7 + coefficients
    event = model.sites + 2
    changed = model.depth_major(packed).clone()
    changed[:, event] = ((changed[:, event] // 7 + 5) % 13) * 7 + changed[:, event] % 7
    with torch.no_grad():
        before = model(packed, model_aux=aux)
        after = model(model.raster_depth(changed), model_aux=aux)
    torch.testing.assert_close(model.depth_major(before['atom_logits'])[:, :event + 1],
                               model.depth_major(after['atom_logits'])[:, :event + 1], rtol=0, atol=0)
    torch.testing.assert_close(model.depth_major(before['coeff_logits'])[:, :event],
                               model.depth_major(after['coeff_logits'])[:, :event], rtol=0, atol=0)
    assert not torch.equal(model.depth_major(before['coeff_logits'])[:, event],
                           model.depth_major(after['coeff_logits'])[:, event])


@pytest.mark.parametrize(('block_size', 'overlap'), [((2, 3, 3), 0), ((2, 3, 3), 2), ((8, 8, 4), 8)])
def test_cached_logits_match_dense_through_all_plane_boundaries(block_size, overlap):
    model, aux, atoms, coefficients = setup(block_size, overlap=overlap)
    with torch.no_grad():
        dense = model(atoms * 7 + coefficients, model_aux=aux)
        dense = {key: model.depth_major(value) for key, value in dense.items()}
        flat_atoms = model.depth_major(atoms)
        model.init_cache()
        for event in range(model.events):
            atom = model.cached_atom_logits(atoms, coefficients, aux, event)
            coefficient = model.cached_coefficient_logits(atoms, coefficients, flat_atoms[:, event], aux, event)
            torch.testing.assert_close(atom, dense['atom_logits'][:, event], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(coefficient, dense['coeff_logits'][:, event], rtol=2e-5, atol=2e-6)
        assert all(block._length == model.sites + (overlap if model.depths > 1 else 0)
                   for block in [*model.atom_decoder, *model.coefficient_decoder])
        assert all(block._cross_keys is not None for block in [*model.atom_decoder, *model.coefficient_decoder])
        model.init_cache()
        assert all(block._keys is None and block._cross_keys is None and block._length == 0
                   for block in [*model.atom_decoder, *model.coefficient_decoder])


def test_completed_plane_context_changes_next_plane_but_not_earlier_predictions():
    model, aux, atoms, coefficients = setup()
    changed = coefficients.clone()
    changed[:, 0, 0, 0] = (changed[:, 0, 0, 0] + 4) % 7
    with torch.no_grad():
        before = model(atoms * 7 + coefficients, model_aux=aux)
        after = model(atoms * 7 + changed, model_aux=aux)
    for key in before:
        first = model.depth_major(before[key]); second = model.depth_major(after[key])
        torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
        # Position zero of plane one has explicit physical memory of plane zero.
        finite = torch.isfinite(first[:, model.sites])
        assert (first[:, model.sites][finite] - second[:, model.sites][finite]).abs().max() > 1e-5


def test_rezero_initialization_and_two_updates_activate_branch_gradients():
    model, aux, atoms, coefficients = setup(active=False)
    gates = [p for name, p in model.named_parameters() if 'gate' in name]
    assert gates and all(torch.count_nonzero(p) == 0 for p in gates)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    output = model(atoms * 7 + coefficients, model_aux=aux)
    target_loss(output, atoms, coefficients).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.count_nonzero(block.self_qkv.weight.grad) == 0
               for block in [*model.prefix_encoder, *model.atom_decoder, *model.coefficient_decoder])
    assert all(torch.count_nonzero(p.grad) > 0 for p in gates)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    target_loss(model(atoms * 7 + coefficients, model_aux=aux), atoms, coefficients).backward()
    assert all(torch.count_nonzero(block.self_qkv.weight.grad) > 0
               for block in [*model.prefix_encoder, *model.atom_decoder, *model.coefficient_decoder])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_current_atom_remains_visible_with_large_history_activation():
    model, aux, _, _ = setup(active=False)
    torch.manual_seed(17)
    hidden = torch.randn(2, 1, model.width) * 1e5
    local = torch.randn_like(hidden)
    first = torch.tensor([[1], [2]])
    second = torch.tensor([[5], [8]])
    with torch.no_grad():
        a = model.coefficient_classifier(model.coefficient_inputs(hidden, first, local, aux))
        b = model.coefficient_classifier(model.coefficient_inputs(hidden, second, local, aux))
    assert (a - b).abs().max() > .01


def test_sampling_preserves_unique_atom_per_site_and_resets_caches():
    model, aux, _, _ = setup()
    with torch.no_grad():
        atoms, coefficients = model.sample_compound(4, aux, atom_top_k=0, atom_top_p=None,
            coeff_top_k=0, coeff_top_p=None, amp=False)
    assert tuple(atoms.shape) == (4, *model.block_size)
    assert atoms.min() >= 0 and atoms.max() < 13 and coefficients.min() >= 0 and coefficients.max() < 7
    for depth in range(1, model.depths):
        assert not (atoms[..., depth, None] == atoms[..., :depth]).any()
    assert model._next_atom == model._next_coefficient == 0
    assert all(block._keys is None for block in [*model.atom_decoder, *model.coefficient_decoder])


def test_masked_atoms_are_excluded_from_dense_likelihood():
    model, aux, atoms, coefficients = setup()
    with torch.no_grad():
        logits = model(atoms * 7 + coefficients, model_aux=aux)['atom_logits']
    for depth in range(1, model.depths):
        used_logits = logits[..., depth, :].gather(-1, atoms[..., :depth])
        assert torch.isneginf(used_logits).all()
    assert torch.isfinite(logits.gather(-1, atoms[..., None])).all()


def test_sampling_filter_validation_and_top_one():
    logits = torch.tensor([[1., -torch.inf, 3., 0.]])
    for _ in range(5):
        assert sample_categorical(logits, top_k=1, top_p=None).item() == 2
    for kwargs in [dict(temperature=0), dict(top_k=-1), dict(top_p=0), dict(top_p=1.1)]:
        with pytest.raises(ValueError):
            sample_categorical(logits, **kwargs)
