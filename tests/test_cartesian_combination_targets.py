import itertools

import pytest
import torch

from src.training.cartesian_combination_targets import (
    cartesian_combination_cross_entropy, cartesian_combination_targets,
)


def fixture(n=5):
    g = torch.Generator().manual_seed(812)
    signals = torch.randn(n, 3, generator=g, dtype=torch.float64)
    dictionary = torch.randn(3, 6, generator=g, dtype=torch.float64)
    atoms = torch.tensor([[[0, 1, 0], [2, 1, 3], [4, 2, 5]]]).expand(n, -1, -1).clone()
    ids = torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 1]]]).expand_as(atoms).clone()
    grids = torch.tensor([[-1., -.2, .4, .9], [-.8, -.1, .3, 1.],
                          [-.7, -.3, .2, .8]], dtype=torch.float64)
    return signals, dictionary, atoms, ids, grids


def brute(signals, dictionary, atoms, ids, grids, temperature, log_base=None,
          forbid_repeated_atoms=True, duplicate_policy="deduplicate"):
    n, depth, choices = atoms.shape
    states = torch.tensor(list(itertools.product(range(choices), repeat=depth)))
    base = torch.zeros_like(atoms, dtype=signals.dtype) if log_base is None else log_base.clone()
    if duplicate_policy == "deduplicate":
        for site in range(n):
            for d in range(depth):
                seen = set()
                for j in range(choices):
                    if not torch.isfinite(base[site, d, j]):
                        continue
                    key = (int(atoms[site, d, j]), int(ids[site, d, j]))
                    if key in seen:
                        base[site, d, j] = -torch.inf
                    seen.add(key)
    energies, masses, complete_atoms, complete_ids = [], [], [], []
    for state in states:
        a = torch.stack([atoms[:, d, state[d]] for d in range(depth)], -1)
        c = torch.stack([ids[:, d, state[d]] for d in range(depth)], -1)
        values = torch.stack([grids[d, c[:, d]] for d in range(depth)], -1)
        reconstruction = (dictionary.T[a] * values[..., None]).sum(-2)
        mass = torch.stack([base[:, d, state[d]] for d in range(depth)], -1).sum(-1)
        if forbid_repeated_atoms:
            mass[(a.sort(-1).values.diff(dim=-1) == 0).any(-1)] = -torch.inf
        energies.append((signals - reconstruction).square().sum(-1))
        masses.append(mass)
        complete_atoms.append(a)
        complete_ids.append(c)
    energy = torch.stack(energies, -1)
    mass = torch.stack(masses, -1)
    probability = (mass - energy / temperature).softmax(-1)
    return probability, energy, torch.stack(complete_atoms, 1), torch.stack(complete_ids, 1)


def assert_marginals(out, expected, complete_atoms, complete_ids, atom_count, bin_count):
    n, _, depth = complete_atoms.shape
    posterior = expected.clone()
    for d in range(depth):
        posterior /= posterior.sum(-1, keepdim=True)
        want_atom = torch.zeros(n, atom_count, dtype=expected.dtype).scatter_add_(
            -1, complete_atoms[:, :, d], posterior)
        got_atom = torch.zeros_like(want_atom).scatter_add_(
            -1, out.atom_target_ids[:, d], out.atom_target_weights[:, d])
        torch.testing.assert_close(got_atom, want_atom)
        atom_mask = complete_atoms[:, :, d] == out.atoms[:, d, None]
        conditioned = posterior * atom_mask
        conditioned /= conditioned.sum(-1, keepdim=True)
        want_coefficient = torch.zeros(n, bin_count, dtype=expected.dtype).scatter_add_(
            -1, complete_ids[:, :, d], conditioned)
        got_coefficient = torch.zeros_like(want_coefficient).scatter_add_(
            -1, out.coefficient_target_ids[:, d], out.coefficient_target_weights[:, d])
        torch.testing.assert_close(got_coefficient, want_coefficient)
        torch.testing.assert_close(out.atom_entropy[:, d],
                                   -(want_atom * want_atom.clamp_min(1e-300).log()).sum(-1))
        torch.testing.assert_close(out.coefficient_entropy[:, d],
                                   -(want_coefficient * want_coefficient.clamp_min(1e-300).log()).sum(-1))
        posterior = conditioned * (complete_ids[:, :, d] == out.coefficient_ids[:, d, None])


def test_exact_full_vector_distances_and_sampled_prefix_marginals():
    args = fixture()
    log_base = torch.randn(args[2].shape, generator=torch.Generator().manual_seed(77),
                           dtype=torch.float64) * .3
    out = cartesian_combination_targets(*args, temperature=.73,
        candidate_log_base_mass=log_base, site_chunk_size=2,
        generator=torch.Generator().manual_seed(88))
    q, energy, atoms, ids = brute(*args, .73, log_base)
    assert_marginals(out, q, atoms, ids, args[1].shape[1], args[4].shape[1])
    torch.testing.assert_close(out.expected_distortion, (q * energy).sum(-1))
    torch.testing.assert_close(out.joint_entropy, -(q * q.clamp_min(1e-300).log()).sum(-1))
    torch.testing.assert_close(out.sampled_distortion,
                               (args[0] - out.reconstruction).square().sum(-1))
    assert not out.reconstruction.requires_grad


def test_cross_terms_change_atom_probabilities_from_independent_pair_scores():
    signals = torch.tensor([[1.3, .2]], dtype=torch.float64)
    dictionary = torch.tensor([[1., 1., .4], [0., 1., -.6]], dtype=torch.float64)
    atoms = torch.tensor([[[0, 1], [1, 2]]])
    ids = torch.tensor([[[1, 2], [1, 2]]])
    grids = torch.tensor([[-1., .5, 1.2], [-1., .5, 1.2]], dtype=torch.float64)
    out = cartesian_combination_targets(signals, dictionary, atoms, ids, grids,
                                        temperature=.8, forbid_repeated_atoms=False)
    q, energy, all_atoms, all_ids = brute(signals, dictionary, atoms, ids, grids,
                                        .8, forbid_repeated_atoms=False)
    assert_marginals(out, q, all_atoms, all_ids, 3, 3)
    vectors = dictionary.T[atoms] * grids[torch.arange(2)[None, :, None], ids][..., None]
    unary = vectors.square().sum(-1) - 2 * (signals[:, None, None] * vectors).sum(-1)
    independent_first = (-unary[:, 0] / .8).softmax(-1)
    assert (out.atom_target_weights[:, 0] - independent_first).abs().max() > .01
    torch.testing.assert_close(out.expected_distortion, (q * energy).sum(-1))


@pytest.mark.parametrize("duplicate_policy", ["deduplicate", "sum"])
def test_aliases_condition_on_token_prefix_not_candidate_index(duplicate_policy):
    x, dictionary, atoms, ids, grids = fixture(12)
    atoms[:, 0, 2] = atoms[:, 0, 0]
    ids[:, 0, 2] = ids[:, 0, 0]
    atoms[:, 1, 2] = atoms[:, 1, 0]
    ids[:, 1, 2] = ids[:, 1, 0]
    base = torch.zeros_like(atoms, dtype=x.dtype)
    base[:, 0, 2] = .7
    # A zero-mass earlier duplicate must not suppress its usable later alias.
    base[0, 0, 0] = -torch.inf
    out = cartesian_combination_targets(x, dictionary, atoms, ids, grids,
        temperature=2., candidate_log_base_mass=base, duplicate_policy=duplicate_policy,
        generator=torch.Generator().manual_seed(15))
    q, _, all_atoms, all_ids = brute(x, dictionary, atoms, ids, grids, 2., base,
                                    duplicate_policy=duplicate_policy)
    assert_marginals(out, q, all_atoms, all_ids, dictionary.shape[1], grids.shape[1])
    # Explicit sequence entropy sums duplicate leaves before taking entropy.
    expected_entropies = []
    for site in range(len(x)):
        masses = {}
        for j in range(q.shape[1]):
            key = tuple(zip(all_atoms[site, j].tolist(), all_ids[site, j].tolist()))
            masses[key] = masses.get(key, 0.) + q[site, j]
        masses = torch.stack(list(masses.values()))
        expected_entropies.append(-(masses * masses.clamp_min(1e-300).log()).sum())
    torch.testing.assert_close(out.joint_entropy, torch.stack(expected_entropies))


def test_stochastic_draws_follow_the_same_joint_and_do_not_collapse_suffixes():
    n = 6000
    x = torch.tensor([[.25, -.4]], dtype=torch.float64).expand(n, -1)
    dictionary = torch.tensor([[1., 0., .5, -.2], [0., 1., -.2, .6]], dtype=torch.float64)
    atoms = torch.tensor([[[0, 1], [2, 3]]]).expand(n, -1, -1)
    ids = torch.tensor([[[0, 1], [0, 1]]]).expand_as(atoms)
    grids = torch.tensor([[-.5, .5], [-.5, .5]], dtype=torch.float64)
    out = cartesian_combination_targets(x, dictionary, atoms, ids, grids,
        temperature=.9, site_chunk_size=500, generator=torch.Generator().manual_seed(143))
    q, _, _, _ = brute(x[:1], dictionary, atoms[:1], ids[:1], grids, .9)
    observed = out.atoms[:, 0] * 2 + out.atoms[:, 1] - 2
    frequency = torch.bincount(observed, minlength=4).double() / n
    torch.testing.assert_close(frequency, q[0], atol=.025, rtol=0)
    # A first coefficient identifies its first-slot branch, but later choices
    # remain stochastic and carry positive soft-target entropy.
    assert (out.atom_entropy[:, 1] > .1).all()


def test_no_refitting_input_mutation_and_rng_replay(monkeypatch):
    args = fixture()
    originals = [a.clone() for a in args]
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a linear solver would violate this teacher's no-refit contract")
    for name in ("solve", "lstsq", "pinv", "inv", "cholesky"):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    g = torch.Generator().manual_seed(20)
    state = g.get_state()
    first = cartesian_combination_targets(*args, temperature=1.3, generator=g)
    g.set_state(state)
    replay = cartesian_combination_targets(*args, temperature=1.3, generator=g)
    for before, after in zip(originals, args):
        assert torch.equal(before, after)
    assert torch.equal(first.atoms, replay.atoms)
    assert torch.equal(first.coefficient_ids, replay.coefficient_ids)
    values = args[4][torch.arange(3)[None], first.coefficient_ids]
    torch.testing.assert_close(first.physical_coefficients, values)
    torch.testing.assert_close(first.reconstruction,
                               (args[1].T[first.atoms] * values[..., None]).sum(-2))


def test_loss_and_gradients_equal_dense_conditional_cross_entropy():
    args = fixture(2)
    target = cartesian_combination_targets(*args, temperature=.9)
    g = torch.Generator().manual_seed(31)
    a = torch.randn(2, 3, 6, generator=g, dtype=torch.float64, requires_grad=True)
    c = torch.randn(2, 3, 4, generator=g, dtype=torch.float64, requires_grad=True)
    got, _, _ = cartesian_combination_cross_entropy(a, c, target)
    qa = torch.zeros_like(a).scatter_add_(-1, target.atom_target_ids, target.atom_target_weights)
    qc = torch.zeros_like(c).scatter_add_(-1, target.coefficient_target_ids, target.coefficient_target_weights)
    want = (-(qa * a.log_softmax(-1)).sum(-1) - (qc * c.log_softmax(-1)).sum(-1)).mean() / 2
    torch.testing.assert_close(got, want)
    for actual, expected in zip(torch.autograd.grad(got, (a, c), retain_graph=True),
                                torch.autograd.grad(want, (a, c))):
        torch.testing.assert_close(actual, expected)


def test_spatial_leading_dimensions_and_fp32():
    x, dictionary, atoms, ids, grids = fixture(6)
    out = cartesian_combination_targets(x.float().reshape(2, 3, 3), dictionary.float(),
        atoms.reshape(2, 3, 3, 3), ids.reshape(2, 3, 3, 3), grids.float(),
        temperature=.5, site_chunk_size=2)
    assert out.atoms.shape == (2, 3, 3)
    assert out.expected_distortion.shape == (2, 3)
    torch.testing.assert_close(out.atom_target_weights.sum(-1), torch.ones(2, 3, 3))
    torch.testing.assert_close(out.coefficient_target_weights.sum(-1), torch.ones(2, 3, 3))


def test_repeated_atom_constraint_and_empty_support_fail_explicitly():
    x, dictionary, atoms, ids, grids = fixture(1)
    atoms.fill_(0)
    with pytest.raises(ValueError, match="no finite-mass valid complete combination"):
        cartesian_combination_targets(x, dictionary, atoms, ids, grids, temperature=1.)
    out = cartesian_combination_targets(x, dictionary, atoms, ids, grids,
                                        temperature=1., forbid_repeated_atoms=False)
    assert (out.atoms == 0).all()
    with pytest.raises(ValueError, match="no finite-mass valid complete combination"):
        cartesian_combination_targets(x, dictionary, atoms, ids, grids,
            temperature=1., forbid_repeated_atoms=False,
            candidate_log_base_mass=torch.full_like(atoms, -torch.inf, dtype=x.dtype))


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf"), True])
def test_bad_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        cartesian_combination_targets(*fixture(), temperature=temperature)


def test_invalid_coefficients_and_base_masses_rejected():
    args = list(fixture())
    args[4][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        cartesian_combination_targets(*args, temperature=1.)
    args = fixture()
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="log base masses"):
            cartesian_combination_targets(*args, temperature=1.,
                candidate_log_base_mass=torch.full_like(args[2], value, dtype=args[0].dtype))
