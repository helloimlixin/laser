import itertools

import pytest
import torch

from src.training.omp_joint_targets import omp_bank_joint_targets, omp_joint_cross_entropy


def small_bank():
    atoms = torch.tensor([[[0, 1, 2], [0, 2, 1], [1, 0, 2]]])
    centers = torch.tensor([[[-.7, .3, .8], [.6, -.4, -.5], [.2, .7, -.1]]],
                           dtype=torch.float64)
    bins = torch.tensor([-1., 0., 1.], dtype=torch.float64)
    return atoms, centers, bins


def enumerate_joint(atoms, centers, bins, temperature):
    rows = []
    for variant in range(atoms.shape[-2]):
        kernels = torch.exp(-(centers[0, variant, :, None] - bins).square() / temperature)
        kernels /= kernels.sum(-1, keepdim=True)
        for ids in itertools.product(range(len(bins)), repeat=atoms.shape[-1]):
            mass = torch.prod(kernels[torch.arange(len(ids)), torch.tensor(ids)]) / atoms.shape[-2]
            rows.append((atoms[0, variant], torch.tensor(ids), mass))
    return rows


def test_both_conditionals_equal_exhaustively_enumerated_joint():
    atoms, centers, bins = small_bank()
    observed_atoms = atoms[:, 0]
    observed_coefficients = torch.tensor([[2, 0, 1]])
    out = omp_bank_joint_targets(atoms, centers, observed_atoms, observed_coefficients,
                                 bins, temperature=.6)
    rows = enumerate_joint(atoms, centers, bins, .6)
    for d in range(3):
        qa, qc = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
        for a, c, mass in rows:
            if torch.equal(a[:d], observed_atoms[0, :d]) and torch.equal(c[:d], observed_coefficients[0, :d]):
                qa[a[d]] += mass
                if a[d] == observed_atoms[0, d]:
                    qc[c[d]] += mass
        qa /= qa.sum()
        qc /= qc.sum()
        actual = torch.zeros_like(qa).scatter_add(0, out.atom_ids[0, d], out.atom_weights[0, d])
        torch.testing.assert_close(actual, qa, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(out.coefficient_probabilities[0, d], qc, rtol=1e-12, atol=1e-12)


def test_current_and_future_coefficients_do_not_affect_current_targets():
    atoms, centers, bins = small_bank()
    first = torch.tensor([[0, 0, 1]])
    changed = torch.tensor([[2, 1, 0]])
    a = omp_bank_joint_targets(atoms, centers, atoms[:, 0], first, bins, temperature=.6)
    b = omp_bank_joint_targets(atoms, centers, atoms[:, 0], changed, bins, temperature=.6)
    torch.testing.assert_close(a.atom_weights[:, 0], b.atom_weights[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(a.coefficient_probabilities[:, 0], b.coefficient_probabilities[:, 0], atol=0, rtol=0)
    assert not torch.equal(a.atom_weights[:, 1], b.atom_weights[:, 1])


def test_coefficient_target_conditions_on_current_atom_and_previous_coefficients():
    atoms, centers, bins = small_bank()
    ids = torch.tensor([[1, 0, 2]])
    a = omp_bank_joint_targets(atoms, centers, atoms[:, 0], ids, bins, temperature=.6)
    b = omp_bank_joint_targets(atoms, centers, atoms[:, 2], ids, bins, temperature=.6)
    torch.testing.assert_close(a.atom_weights[:, 0], b.atom_weights[:, 0])
    assert not torch.equal(a.coefficient_probabilities[:, 0], b.coefficient_probabilities[:, 0])
    # Given atom 1 at depth zero, only the third trajectory is possible.
    expected = (-(centers[0, 2, 0] - bins).square() / .6).softmax(-1)
    torch.testing.assert_close(b.coefficient_probabilities[0, 0], expected)


def test_duplicate_trajectory_multiplicity_and_chunk_invariance():
    atoms, centers, bins = small_bank()
    atoms = atoms[:, [0, 0, 2]].expand(5, -1, -1)
    centers = centers[:, [0, 0, 2]].expand(5, -1, -1)
    ids = torch.ones(5, 3, dtype=torch.long)
    a = omp_bank_joint_targets(atoms, centers, atoms[:, 0], ids, bins, temperature=.6, site_chunk_size=1)
    b = omp_bank_joint_targets(atoms, centers, atoms[:, 0], ids, bins, temperature=.6, site_chunk_size=4)
    torch.testing.assert_close(a.atom_weights, b.atom_weights, rtol=0, atol=0)
    torch.testing.assert_close(a.coefficient_probabilities, b.coefficient_probabilities, rtol=0, atol=0)
    assert float(a.atom_weights[0, 0, :2].sum()) == pytest.approx(2/3)


def test_sparse_joint_loss_and_gradients_match_dense_unweighted_ce():
    atoms, centers, bins = small_bank()
    out = omp_bank_joint_targets(atoms, centers, atoms[:, 0], torch.tensor([[2, 1, 0]]), bins, temperature=.6)
    torch.manual_seed(31)
    a = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)
    c = torch.randn_like(a, requires_grad=True)
    logits = a.masked_fill(torch.tensor([[[False, False, False], [True, False, False], [True, True, False]]]), -torch.inf)
    actual = omp_joint_cross_entropy(logits, c, out)
    dense = torch.zeros_like(a).scatter_add(-1, out.atom_ids, out.atom_weights)
    safe = torch.where(dense > 0, logits.log_softmax(-1), 0.)
    expected = -(dense * safe).sum(-1).mean() - (out.coefficient_probabilities * c.log_softmax(-1)).sum(-1).mean()
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual, (a, c), retain_graph=True)
    refs = torch.autograd.grad(expected, (a, c))
    for got, want in zip(grads, refs):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want)


def test_absent_support_and_invalid_temperature_rejected():
    atoms, centers, bins = small_bank()
    with pytest.raises(ValueError, match='absent'):
        omp_bank_joint_targets(atoms, centers, torch.tensor([[2, 1, 0]]), torch.zeros(1, 3, dtype=torch.long), bins, temperature=.6)
    for temperature in [0., -1., float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='temperature'):
            omp_bank_joint_targets(atoms, centers, atoms[:, 0], torch.zeros(1, 3, dtype=torch.long), bins, temperature=temperature)
