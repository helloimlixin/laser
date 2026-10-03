import itertools

import pytest
import torch

from src.training.combination_search import search_combinations, wide_pair_pool


def fixture(sites=3):
    g = torch.Generator().manual_seed(1701)
    dictionary = torch.randn(5, 8, generator=g, dtype=torch.float64)
    atoms = torch.tensor([[[0, 1], [2, 3], [4, 5]]]).expand(sites, -1, -1).clone()
    ids = torch.tensor([[[0, 1], [1, 0], [0, 1]]]).expand_as(atoms).clone()
    grids = torch.tensor([[-.7, .9], [-.4, 1.2], [-.3, .5]], dtype=torch.float64)
    signal = torch.randn(sites, 5, generator=g, dtype=torch.float64)
    return signal, dictionary, atoms, ids, grids


def brute(args, temperature):
    signal, dictionary, atoms, ids, grids = args
    states = list(itertools.product(range(atoms.shape[-1]), repeat=atoms.shape[-2]))
    vectors, all_atoms, all_ids = [], [], []
    for state in states:
        a = torch.stack([atoms[:, d, j] for d, j in enumerate(state)], -1)
        b = torch.stack([ids[:, d, j] for d, j in enumerate(state)], -1)
        c = grids[torch.arange(atoms.shape[-2]), b]
        vectors.append((dictionary.T[a] * c[..., None]).sum(-2))
        all_atoms.append(a)
        all_ids.append(b)
    vectors = torch.stack(vectors, 1)
    error = (vectors - signal[:, None]).square().sum(-1)
    probability = (-error / temperature).softmax(-1)
    return error, probability, torch.stack(all_atoms, 1), torch.stack(all_ids, 1)


def test_wide_beam_recovers_complete_finite_law_with_cross_terms():
    args = fixture()
    bank = search_combinations(*args, beam_width=8, site_chunk_size=2)
    expected_error, expected_p, atoms, ids = brute(args, .7)
    assert bank.valid.sum(-1).tolist() == [8, 8, 8]
    for site in range(3):
        for j in range(8):
            matches = ((bank.atoms[site] == atoms[site, j])
                       & (bank.coefficient_ids[site] == ids[site, j])).all(-1) & bank.valid[site]
            assert matches.sum() == 1
            torch.testing.assert_close(bank.errors[site, matches], expected_error[site, j:j+1])
            torch.testing.assert_close(bank.probabilities(.7)[site, matches], expected_p[site, j:j+1])


def test_compensating_two_pair_change_survives_narrow_search():
    dictionary = torch.tensor([[1., 0., .6, .8], [0., 1., .8, .6]], dtype=torch.float64)
    args = (torch.tensor([[1., 1.]], dtype=torch.float64), dictionary,
            torch.tensor([[[0, 2], [1, 3]]]), torch.tensor([[[1, 0], [1, 0]]]),
            torch.tensor([[5./7., 1.], [5./7., 1.]], dtype=torch.float64))
    bank = search_combinations(*args, beam_width=2)
    assert bank.atoms.tolist() == [[[0, 1], [2, 3]]]
    torch.testing.assert_close(bank.errors, torch.zeros_like(bank.errors), atol=1e-25, rtol=0)
    torch.testing.assert_close(bank.probabilities(.3), torch.full((1, 2), .5, dtype=torch.float64))
    a, b = bank.sample(.3, draws=2000, generator=torch.Generator().manual_seed(56))
    assert .46 < (a[..., 0] == 2).double().mean() < .54
    assert torch.equal((a[..., 0] == 2), (b[..., 0] == 0))


def test_duplicates_and_repeated_support_never_receive_extra_mass():
    args = list(fixture(1))
    args[2] = torch.tensor([[[0, 0, 1], [2, 2, 0], [4, 4, 2]]])
    args[3] = torch.tensor([[[0, 0, 1], [1, 1, 0], [0, 0, 1]]])
    bank = search_combinations(*args, beam_width=32, sweeps=2)
    keys = [tuple(zip(a.tolist(), b.tolist())) for a, b in
            zip(bank.atoms[0, bank.valid[0]], bank.coefficient_ids[0, bank.valid[0]])]
    assert len(keys) == len(set(keys))
    assert all(len({a for a, _ in key}) == 3 for key in keys)
    assert (bank.probabilities(.5)[~bank.valid] == 0).all()
    assert (bank.atoms[:, 0] == torch.tensor([[0, 2, 4]])).all()


def test_alias_token_sequences_preserved_and_single_beam_keeps_anchor():
    signal = torch.tensor([[1., 1.]], dtype=torch.float64)
    dictionary = torch.tensor([[1., -1., 0., 0.], [0., 0., 1., -1.]], dtype=torch.float64)
    atoms = torch.tensor([[[0, 1], [2, 3]]])
    ids = torch.tensor([[[1, 0], [1, 0]]])
    grids = torch.tensor([[-1., 1.], [-1., 1.]], dtype=torch.float64)
    bank = search_combinations(signal, dictionary, atoms, ids, grids, beam_width=8)
    assert bank.valid.sum() == 4
    torch.testing.assert_close(bank.probabilities(.5)[bank.valid], torch.full((4,), .25, dtype=torch.float64))
    one = search_combinations(signal, dictionary, atoms, ids, grids, beam_width=1)
    assert one.atoms.tolist() == [[[0, 2]]]
    assert one.valid.tolist() == [[True]]


def test_search_chunking_units_and_inputs_are_consistent():
    args = fixture(5)
    before = [v.clone() for v in args]
    bank = search_combinations(*args, beam_width=8, site_chunk_size=1)
    other = search_combinations(*args, beam_width=8, site_chunk_size=5)
    for field in ('atoms', 'coefficient_ids', 'valid', 'errors'):
        torch.testing.assert_close(getattr(bank, field), getattr(other, field))
    scaled = search_combinations(args[0] * 3, args[1] * 3, *args[2:], beam_width=8)
    torch.testing.assert_close(scaled.probabilities(.7 * 9), bank.probabilities(.7))
    for old, new in zip(before, args):
        assert torch.equal(old, new)


def test_proposals_use_actual_bins_rng_replay_and_no_linear_solver(monkeypatch):
    _, dictionary, atoms, ids, grids = fixture(2)
    anchors = atoms[..., 0]
    coefficients = grids[torch.arange(3), ids[..., 0]]
    def forbidden(*args, **kwargs):
        raise AssertionError('coefficient fitting is forbidden')
    for name in ('solve', 'lstsq', 'pinv', 'inv', 'cholesky'):
        monkeypatch.setattr(torch.linalg, name, forbidden)
    kwargs = dict(temperature=.3, alternatives_per_depth=4, site_chunk_size=1)
    pool = wide_pair_pool(anchors, coefficients, dictionary, grids,
                         generator=torch.Generator().manual_seed(100), **kwargs)
    replay = wide_pair_pool(anchors, coefficients, dictionary, grids,
                           generator=torch.Generator().manual_seed(100), **kwargs)
    assert pool.atoms.shape == (2, 3, 15)
    assert torch.equal(pool.atoms, replay.atoms)
    assert torch.equal(pool.coefficient_ids, replay.coefficient_ids)
    assert torch.equal(pool.atoms[..., 0], anchors)
    bank = search_combinations(pool.target_vectors, dictionary, pool.atoms,
                               pool.coefficient_ids, grids, beam_width=16)
    a, b = bank.sample(.3, generator=torch.Generator().manual_seed(212))
    assert ((a.sort(-1).values.diff(dim=-1)) != 0).all()
    assert (b >= 0).all() and (b < grids.shape[1]).all()


def test_search_restores_precision_and_validates_inputs():
    args = fixture(1)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.autocast('cpu', dtype=torch.bfloat16):
            bank = search_combinations(*args)
            assert torch.is_autocast_enabled('cpu')
            assert bank.errors.dtype == torch.float64
            assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    for width in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            search_combinations(*args, beam_width=width)
    bad = list(args)
    bad[2] = bad[2].clone()
    bad[2][:, 1, 0] = bad[2][:, 0, 0]
    with pytest.raises(ValueError, match='distinct'):
        search_combinations(*bad)
    with pytest.raises(ValueError):
        bank.probabilities(0)


def test_support_quota_keeps_alternative_atoms_when_coefficient_variants_dominate():
    signal = torch.tensor([[1., 1.]], dtype=torch.float64)
    dictionary = torch.tensor([[1., 0., .8, .4], [0., 1., .6, .916515]], dtype=torch.float64)
    atoms = torch.tensor([[[0, 0, 0, 2], [1, 1, 1, 3]]])
    ids = torch.tensor([[[1, 0, 2, 1], [1, 0, 2, 1]]])
    grids = torch.tensor([[.999, 1., 1.001], [.999, 1., 1.001]], dtype=torch.float64)
    plain = search_combinations(signal, dictionary, atoms, ids, grids, beam_width=4, support_quota=0)
    diverse = search_combinations(signal, dictionary, atoms, ids, grids, beam_width=4, support_quota=2)
    assert ((plain.atoms == torch.tensor([0, 1])).all(-1) | ~plain.valid).all()
    assert (((diverse.atoms != torch.tensor([0, 1])).any(-1)) & diverse.valid).sum() >= 2
    physical = grids[torch.arange(2), diverse.coefficient_ids]
    explicit = (signal[:, None]-(dictionary.T[diverse.atoms]*physical[..., None]).sum(-2)).square().sum(-1)
    torch.testing.assert_close(diverse.errors, explicit)


def test_novel_support_proposals_are_distinct_and_missing_support_is_padding():
    args = fixture(2)
    bank = search_combinations(*args, beam_width=8)
    anchor = args[2][..., 0]
    a, b, usable = bank.sample_novel_supports(anchor, .5, draws=4,
                                             generator=torch.Generator().manual_seed(111))
    assert usable.all()
    for site in range(2):
        supports = [tuple(v) for v in a[site].sort(-1).values.tolist()]
        assert len(set(supports)) == 4
        assert tuple(anchor[site].sort().values.tolist()) not in supports
    one = search_combinations(*args, beam_width=1)
    padded_a, padded_b, valid = one.sample_novel_supports(anchor, .5, draws=4)
    assert not valid.any()
    assert torch.equal(padded_a, one.atoms.expand(-1, 4, -1))
