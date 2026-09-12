import pytest
import torch

from src.complete_sparse_codec import (
    pack_exact, unpack_exact, sparse_latents, decode_site_ids, fit_complete_codebook,
)
from src.coefficient_pattern_codec import assign_coefficient_patterns


def test_exact_complete_site_exceeds_int64_and_round_trips():
    atoms, coefficients = [16383, 12, 301, 15200], [0, 2047, 1023, 91]
    value = pack_exact(atoms, coefficients)
    assert value > 2 ** 63 and value < 2 ** 100
    assert unpack_exact(value) == (atoms, coefficients)
    assert pack_exact([16383] * 4, [2047] * 4) == 2 ** 100 - 1
    for bad in (-1, 2 ** 100):
        with pytest.raises(ValueError):
            unpack_exact(bad)
    with pytest.raises(ValueError):
        pack_exact([0] * 4, [2048] * 4)
    with pytest.raises(TypeError):
        pack_exact([.5] * 4, [0] * 4)


def test_codebook_id_recovers_atoms_and_signed_coefficients_without_source_support():
    codebook = {'atoms': torch.tensor([[0, 1], [1, 2]]),
                'coefficient_ids': torch.tensor([[0, 2], [2, 1]])}
    dictionary, bins, scales = torch.eye(3), torch.tensor([-1., 0., 1.]), torch.tensor([2., 3.])
    ids = torch.tensor([[1, 0]])
    atoms, coefficients = decode_site_ids(ids, codebook)
    z = sparse_latents(atoms, coefficients, dictionary, bins, scales)
    torch.testing.assert_close(z, torch.tensor([[[0., 2., 0.], [-2., 3., 0.]]]))
    with pytest.raises(ValueError):
        decode_site_ids(torch.tensor([2]), codebook)


def test_fitted_prototypes_are_complete_training_codes_and_assignment_is_geometric():
    dictionary = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    atoms = torch.tensor([[0], [2], [0], [1], [1], [1]])
    ids = torch.tensor([[2], [0], [2], [1], [2], [1]])
    bins, scales = torch.tensor([-1., .9, 1.]), torch.ones(1)
    book = fit_complete_codebook(atoms, ids, dictionary, bins, scales, num_codes=2,
                                iterations=5, seed=3, chunk_size=2)
    chosen = book['fit_representative_indices']
    assert torch.equal(book['atoms'], atoms[chosen])
    assert torch.equal(book['coefficient_ids'], ids[chosen])
    z = sparse_latents(atoms, ids, dictionary, bins, scales)
    labels, errors = assign_coefficient_patterns(z, book['latents'], chunk_size=2)
    actual_atoms, actual_ids = decode_site_ids(labels, book)
    actual_z = sparse_latents(actual_atoms, actual_ids, dictionary, bins, scales)
    torch.testing.assert_close(errors, (z - actual_z).square().sum(-1), atol=1e-6, rtol=1e-6)
    assert labels[0] == labels[1]  # Equivalent latents need not have equal atom IDs.
