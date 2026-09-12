import pytest
import torch

from src.complete_sparse_codec import sparse_latents
from src.support_pattern_integer_codec import (
    pack_support_pattern, unpack_support_pattern, decode_support_pattern_integers,
)


def test_large_complete_integer_round_trip_has_no_int64_truncation():
    atoms = torch.tensor([[[16383, 123, 8292, 0], [16383] * 4]])
    patterns = torch.tensor([[3172, 4095]])
    values = pack_support_pattern(atoms, patterns, num_patterns=4096)
    assert all(value > 2 ** 63 for value in values)
    assert values[1] == 2 ** 68 - 1
    recovered_atoms, recovered_patterns = unpack_support_pattern(values, num_patterns=4096)
    assert torch.equal(recovered_atoms.reshape_as(atoms), atoms)
    assert torch.equal(recovered_patterns.reshape_as(patterns), patterns)
    with pytest.raises(ValueError):
        unpack_support_pattern([2 ** 68], num_patterns=4096)
    with pytest.raises(ValueError):
        pack_support_pattern(atoms, patterns.float(), num_patterns=4096)
    with pytest.raises(ValueError):
        pack_support_pattern(atoms, torch.tensor([[4096, 0]]), num_patterns=4096)


def test_integer_alone_recovers_exact_support_and_signed_coefficients():
    atoms, patterns = torch.tensor([[2, 0], [1, 2]]), torch.tensor([1, 0])
    table = torch.tensor([[0, 2], [2, 1]])
    integers = pack_support_pattern(atoms, patterns, num_patterns=2, num_atoms=3)
    recovered, coefficients = decode_support_pattern_integers(integers, table, num_atoms=3)
    actual = sparse_latents(recovered, coefficients, torch.eye(3), torch.tensor([-1., 0., 1.]), torch.tensor([2., 3.]))
    assert torch.equal(actual, torch.tensor([[0., 0., 2.], [0., -2., 3.]]))
    assert torch.equal(recovered, atoms)
