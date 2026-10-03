import pytest
import torch

from src.training.cartesian_pair_candidates import (
    cartesian_pair_candidates, dictionary_neighbor_ids,
    nearest_physical_coefficient_ids,
)


def fixture():
    dictionary = torch.nn.functional.normalize(torch.tensor([
        [1., .99, .9, .85, .7, .65],
        [.1, .2, .3, .4, .6, .7],
        [.2, .18, .3, .2, .4, .3],
    ]), dim=0)
    atoms = torch.tensor([[0, 2, 4, 5], [1, 3, 4, 0]])
    grids = torch.stack([torch.linspace(-2, 2, 17) * s for s in [2., 1.5, 1., .5]])
    coefficients = torch.tensor([[.8, -.6, .7, -.2], [3., -2., 1.9, -.99]])
    neighbors, _ = dictionary_neighbor_ids(dictionary, count=2, chunk_size=2)
    return atoms, coefficients, dictionary, grids, neighbors


def test_anchor_exact_reconstruction_without_refit_or_input_mutation(monkeypatch):
    args = fixture()
    originals = [x.clone() for x in args]
    def fail(*args, **kwargs):
        raise AssertionError('No linear solves or refitting allowed')
    for name in ['lstsq', 'solve', 'solve_triangular', 'pinv', 'inv', 'cholesky']:
        monkeypatch.setattr(torch.linalg, name, fail)
    pool = cartesian_pair_candidates(*args, max_bin_offset=[1, 2, 3, 4],
        generator=torch.Generator().manual_seed(19))
    atoms, coefficients, dictionary, grids, _ = args
    brute = (coefficients[..., None]-grids).abs().argmin(-1)
    assert torch.equal(pool.atoms[..., 0], atoms)
    assert torch.equal(pool.coefficient_ids[..., 0], brute)
    physical = grids[torch.arange(4), brute]
    expected = (dictionary.T[atoms] * physical[..., None]).sum(-2)
    torch.testing.assert_close(pool.anchor_vectors, expected, atol=0, rtol=0)
    # Alternative atoms use the exact same coefficient choices as anchors.
    assert torch.equal(pool.coefficient_ids[..., :3], pool.coefficient_ids[..., 3:])
    for before, after in zip(originals, args):
        assert torch.equal(before, after)


def test_fresh_pools_replay_and_local_branching():
    args=fixture()
    one=cartesian_pair_candidates(*args, max_bin_offset=5, generator=torch.Generator().manual_seed(12))
    replay=cartesian_pair_candidates(*args, max_bin_offset=5, generator=torch.Generator().manual_seed(12))
    other=cartesian_pair_candidates(*args, max_bin_offset=5, generator=torch.Generator().manual_seed(13))
    assert torch.equal(one.atoms,replay.atoms)
    assert torch.equal(one.coefficient_ids,replay.coefficient_ids)
    assert not (torch.equal(one.atoms,other.atoms) and torch.equal(one.coefficient_ids,other.coefficient_ids))
    assert (one.atoms[...,0] != one.atoms[...,3]).all()
    assert one.atoms.shape == (2,4,6)
    assert (one.bin_offsets >= 1).all() and (one.bin_offsets <= 5).all()


def test_neighbors_match_dense_cosine_and_never_self():
    _, _, dictionary, _, _ = fixture()
    ids, values = dictionary_neighbor_ids(dictionary, count=3, chunk_size=2)
    full = torch.nn.functional.normalize(dictionary,dim=0).T @ torch.nn.functional.normalize(dictionary,dim=0)
    full.fill_diagonal_(-torch.inf)
    expected_values, expected_ids = full.topk(3,dim=-1)
    assert torch.equal(ids,expected_ids)
    torch.testing.assert_close(values,expected_values)
    assert (values > 0).all()
    assert (ids != torch.arange(len(ids))[:,None]).all()


def test_bin_edges_preserve_anchor_and_expose_duplicates_for_teacher_dedup():
    atoms, coefficients, dictionary, grids, neighbors=fixture()
    coefficients=grids[:,0].expand_as(coefficients).clone()
    pool=cartesian_pair_candidates(atoms,coefficients,dictionary,grids,neighbors,max_bin_offset=100)
    assert (pool.anchor_coefficient_ids==0).all()
    assert torch.equal(pool.coefficient_ids[...,0],pool.coefficient_ids[...,1])
    assert (pool.coefficient_ids>=0).all() and (pool.coefficient_ids<17).all()


def test_arbitrary_leading_dimensions_and_nonuniform_grids():
    values=torch.tensor([[-2.,-.2,0.,2.],[-4.,0.,1.,3.]])
    coefficients=torch.tensor([[[[-.1,.4],[1.2,2.7]]]])
    result=nearest_physical_coefficient_ids(coefficients,values)
    expected=(coefficients[...,None]-values).abs().argmin(-1)
    assert torch.equal(result,expected)
    args=fixture()
    pool=cartesian_pair_candidates(args[0][None,:,None],args[1][None,:,None],*args[2:],max_bin_offset=2)
    assert pool.atoms.shape==(1,2,1,4,6)
    assert pool.anchor_vectors.shape==(1,2,1,3)


@pytest.mark.parametrize('offset', [0, -1, 2.5, [1,2], [1,0,2,3]])
def test_invalid_offset_rejected(offset):
    with pytest.raises(ValueError):
        cartesian_pair_candidates(*fixture(),max_bin_offset=offset)
