from types import SimpleNamespace

import pytest
import torch

from src.training.fresh_paired_teacher import (
    FreshPairedTeacher, fresh_paired_targets, loss_parts,
)


def fixture(dtype=torch.float64, sites=9):
    generator = torch.Generator().manual_seed(17)
    dictionary = torch.randn(4, 7, generator=generator, dtype=dtype)
    # Deliberately nonunit columns, negative pairs, unequal depth grids.
    dictionary *= torch.tensor([.3, 1.7, .8, 2., .6, 1.1, .4], dtype=dtype)
    atoms = torch.tensor([0, 2, 4]).expand(sites, -1).clone()
    coefficients = torch.randn(sites, 3, generator=generator, dtype=dtype) * .4
    bins = torch.tensor([-.9, -.3, .3, .9], dtype=dtype)
    scales = torch.tensor([1.4, .7, 1.1], dtype=dtype)
    return atoms, coefficients, dictionary, scales[:, None] * bins


def run(inputs, **kwargs):
    return fresh_paired_targets(*inputs, temperature=.8,
        generator=torch.Generator().manual_seed(91), **kwargs)


def test_full_atom_and_coefficient_distributions_match_explicit_distances():
    inputs = fixture()
    atoms, coefficients, dictionary, grids = inputs
    out = run(inputs, warmup_sweeps=1, site_chunk_size=3, return_trace=True)
    z0 = (dictionary.T[atoms] * coefficients[..., None]).sum(-2)
    torch.testing.assert_close(out['target_vectors'], z0)
    trace = out['trace']
    for d in range(3):
        before_atoms = trace['atoms_before'][:, d]
        before_ids = trace['coefficient_ids_before'][:, d]
        before_values = grids[torch.arange(3), before_ids]
        before_vectors = dictionary.T[before_atoms] * before_values[..., None]
        other = [j for j in range(3) if j != d]
        residual = z0 - before_vectors[:, other].sum(-2)
        torch.testing.assert_close(trace['residuals'][:, d], residual)
        magnitude = before_values[:, d].abs()
        signed_values = magnitude[:, None] * torch.tensor([-1., 1.], dtype=grids.dtype)
        book = dictionary.T[None, :, None, :] * signed_values[:, None, :, None]
        logits = -(residual[:, None, None, :] - book).square().sum(-1) / .8
        logits.scatter_(1, before_atoms[:, other, None].expand(-1, -1, 2), -torch.inf)
        qa = logits.flatten(1).softmax(-1).reshape(len(atoms), 7, 2).sum(-1)
        torch.testing.assert_close(out['atom_probs'][:, d], qa, rtol=1e-11, atol=1e-12)
        selected = dictionary.T[out['atoms'][:, d]]
        coef_book = selected[:, None, :] * grids[d, :, None]
        qc = (-(residual[:, None, :] - coef_book).square().sum(-1) / .8).softmax(-1)
        torch.testing.assert_close(out['coefficient_probs'][:, d], qc, rtol=1e-11, atol=1e-12)
        # Soft labels were stored with exactly the returned autoregressive prefix.
        assert torch.equal(before_atoms[:, :d], out['atoms'][:, :d])
        assert torch.equal(before_ids[:, :d], out['coefficient_ids'][:, :d])
        assert bool((out['atom_probs'][:, d].gather(1, before_atoms[:, other]) == 0).all())
    actual = (dictionary.T[out['atoms']] * grids[torch.arange(3), out['coefficient_ids']][..., None]).sum(-2)
    torch.testing.assert_close(out['reconstruction'], actual)
    torch.testing.assert_close(out['sampled_distortion'], (z0-actual).square().sum(-1))
    assert not out['atom_probs'].requires_grad


def test_pairs_distinct_full_support_and_no_forced_atom_change():
    out = run(fixture(sites=31), return_trace=True)
    for d in range(3):
        before = out['trace']['atoms_before'][:, d]
        current_mass = out['atom_probs'][:, d].gather(1, before[:, d:d+1])
        assert bool((current_mass > 0).all())
        eligible = torch.ones(31, 7, dtype=torch.bool)
        eligible.scatter_(1, before[:, [j for j in range(3) if j != d]], False)
        assert bool((out['atom_probs'][:, d][eligible] > 0).all())
    assert bool((out['atoms'].sort(-1).values.diff(dim=-1) > 0).all())
    assert bool((out['coefficient_probs'] > 0).all())


def test_bank_gathers_whole_pairs_and_outputs_can_leave_saved_supports():
    anchors, coefficients, dictionary, grids = fixture(torch.float32, sites=32)
    bank_atoms = torch.stack((anchors, anchors.roll(1, -1)), -2)
    bank_coefficients = torch.stack((coefficients, coefficients.roll(1, -1)), -2)
    bank_before = bank_coefficients.clone()
    aux = SimpleNamespace(dictionary=dictionary, coeff_scales=torch.tensor([1.4, .7, 1.1]),
                          coeff_bins=torch.tensor([-.9, -.3, .3, .9]))
    teacher = FreshPairedTeacher(aux, .8, site_chunk_size=7)
    choices = torch.arange(32) % 2
    out = teacher(bank_atoms, bank_coefficients, bank_choices=choices,
                  generator=torch.Generator().manual_seed(41))
    selected_atoms = bank_atoms[torch.arange(32), choices]
    selected_coefficients = bank_coefficients[torch.arange(32), choices]
    direct = teacher.from_anchors(selected_atoms, selected_coefficients,
        reference_bank_atoms=bank_atoms, generator=torch.Generator().manual_seed(41))
    for key in ('packed', 'atoms', 'coefficient_ids', 'atom_probs', 'coefficient_probs', 'target_vectors'):
        assert torch.equal(out[key], direct[key])
    assert bool(out['atom_outside_bank'].any())
    assert float(out['diagnostics']['support_outside_bank_fraction']) > 0
    assert torch.equal(bank_coefficients, bank_before)


def test_replay_and_chunking_are_valid_without_claiming_equal_draw_order():
    inputs = fixture(torch.float32, sites=8)
    outputs = []
    for chunk in (1, 3, 8):
        first = run(inputs, site_chunk_size=chunk, warmup_sweeps=0)
        second = run(inputs, site_chunk_size=chunk, warmup_sweeps=0)
        assert torch.equal(first['packed'], second['packed'])
        assert torch.equal(first['atom_probs'], second['atom_probs'])
        torch.testing.assert_close(first['atom_probs'].sum(-1), torch.ones(8, 3))
        torch.testing.assert_close(first['coefficient_probs'].sum(-1), torch.ones(8, 3))
        outputs.append(first)
    # The first distribution precedes all random draws and must agree across chunks.
    torch.testing.assert_close(outputs[0]['atom_probs'][:, 0], outputs[-1]['atom_probs'][:, 0])
    g = torch.Generator().manual_seed(31)
    first = fresh_paired_targets(*inputs, temperature=.8, generator=g)
    second = fresh_paired_targets(*inputs, temperature=.8, generator=g)
    assert not torch.equal(first['packed'], second['packed'])


def test_spatial_bank_shapes_and_packed_pair_identity():
    atoms, coefficients, dictionary, grids = fixture(torch.float32, sites=12)
    bank_atoms = torch.stack((atoms, atoms.roll(1, -1)), -2).reshape(2, 2, 3, 2, 3)
    bank_coefficients = torch.stack((coefficients, coefficients.roll(1, -1)), -2).reshape_as(bank_atoms)
    aux = SimpleNamespace(dictionary=dictionary, coeff_scales=torch.tensor([1.4, .7, 1.1]),
                          coeff_bins=torch.tensor([-.9, -.3, .3, .9]))
    out = FreshPairedTeacher(aux, .8, site_chunk_size=5)(bank_atoms, bank_coefficients,
        generator=torch.Generator().manual_seed(39))
    assert out['packed'].shape == (2, 2, 3, 3)
    assert out['target_vectors'].shape == (2, 2, 3, 4)
    assert out['atom_probs'].shape == (2, 2, 3, 3, 7)
    assert out['coefficient_probs'].shape == (2, 2, 3, 3, 4)
    assert torch.equal(out['packed'] // 4, out['atoms'])
    assert torch.equal(out['packed'] % 4, out['coefficient_ids'])


@pytest.mark.parametrize('weight', [1.5, 1., 2/3, 7/3])
def test_normalized_loss_gradients_and_prefix_only_model_mask(weight):
    out = run(fixture(sites=4))
    generator = torch.Generator().manual_seed(47)
    atom_logits = torch.randn(4, 3, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    coefficient_logits = torch.randn(4, 3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    loss, parts = loss_parts({'atom_logits':atom_logits, 'coeff_logits':coefficient_logits}, out, weight)
    # Independent reference normalizes each model distribution over explicitly
    # enumerated prefix-eligible atom IDs, without a dense masking operation.
    reference_atoms = []
    for site in range(4):
        for d in range(3):
            eligible = torch.tensor([a for a in range(7) if a not in out['atoms'][site, :d]])
            reference_atoms.append(-(out['atom_probs'][site, d, eligible]
                * atom_logits[site, d, eligible].log_softmax(-1)).sum())
    reference_coefficient = -(out['coefficient_probs'] * coefficient_logits.log_softmax(-1)).sum(-1).mean()
    expected = (weight*torch.stack(reference_atoms).mean()+reference_coefficient)/(weight+1)
    torch.testing.assert_close(loss, expected)
    expected_gradients = torch.autograd.grad(expected, (atom_logits, coefficient_logits), retain_graph=True)
    loss.backward()
    torch.testing.assert_close(atom_logits.grad, expected_gradients[0])
    torch.testing.assert_close(coefficient_logits.grad, expected_gradients[1])
    assert torch.isfinite(atom_logits.grad).all() and torch.isfinite(coefficient_logits.grad).all()
    for d in range(1, 3):
        assert bool((atom_logits.grad[:, d].gather(1, out['atoms'][:, :d]) == 0).all())
    # Excluded teacher alternatives still remain in the model softmax at depth0.
    zero_target = out['atom_probs'][:, 0] == 0
    assert bool((atom_logits.grad[:, 0][zero_target] > 0).all())


def test_no_solves_and_caller_precision_restored_even_on_failure(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('continuous solve/refitting is forbidden')
    for name in ('solve', 'lstsq', 'cholesky'):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    monkeypatch.setattr(torch, 'cholesky_solve', forbidden)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = run(fixture(torch.float32))
        assert out['atom_probs'].dtype == torch.float32
        assert torch.backends.cuda.matmul.allow_tf32 is True
        monkeypatch.setattr(torch, 'multinomial', forbidden)
        with pytest.raises(AssertionError, match='forbidden'):
            run(fixture(torch.float32))
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_bad_temperatures_rejected(bad):
    with pytest.raises(ValueError):
        fresh_paired_targets(*fixture(), temperature=bad)


def test_asymmetric_bins_and_duplicate_support_rejected():
    atoms, coefficients, dictionary, grids = fixture()
    asymmetric = grids.clone(); asymmetric[0, 0] -= .01
    with pytest.raises(ValueError, match='symmetric'):
        fresh_paired_targets(atoms, coefficients, dictionary, asymmetric, temperature=.8)
    atoms[:, 1] = atoms[:, 0]
    with pytest.raises(ValueError, match='distinct'):
        fresh_paired_targets(atoms, coefficients, dictionary, grids, temperature=.8)
