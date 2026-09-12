import pytest
import torch

from src.complete_sparse_codec import sparse_latents
from src.learned_sparse_site_codec import SparseSiteProjector
from src.residual_site_integer_codec import (
    pack_fields, unpack_fields, fit_residual_codebooks, encode_residual_fields, decode_residual_ids,
)


def test_integer_round_trip_includes_largest_signed_int64_and_mixed_fields():
    fields = torch.tensor([[511] * 7, [0, 1, 2, 3, 4, 5, 6]])
    packed = pack_fields(fields, [9] * 7)
    assert packed[0].item() == 2 ** 63 - 1
    assert torch.equal(unpack_fields(packed, [9] * 7), fields)
    mixed = torch.tensor([[[16383, 0, 123, 300, 127]]])
    assert torch.equal(unpack_fields(pack_fields(mixed, [14] * 4 + [7]), [14] * 4 + [7]), mixed)
    for values, widths in [(torch.tensor([[-1]]), [4]), (torch.tensor([[16]]), [4]),
                           (torch.tensor([[1.]]), [4]), (torch.tensor([[1]]), [64])]:
        with pytest.raises(ValueError):
            pack_fields(values, widths)
    with pytest.raises(ValueError):
        unpack_fields(torch.tensor([16]), [4])


def test_residual_integer_decodes_full_sparse_code_without_source_atoms_or_coefficients():
    torch.manual_seed(23)
    vectors = torch.randn(64, 8)
    books = fit_residual_codebooks(vectors, stages=3, vocabulary=8, iterations=3, chunk_size=16)
    source = torch.randn(2, 3, 8)
    fields = encode_residual_fields(source, books, chunk_size=4)
    previous = source.square().sum(-1)
    projector = SparseSiteProjector(torch.eye(8), torch.linspace(-3, 3, 127), torch.ones(4))
    for depth in range(1, 4):
        ids = pack_fields(fields[..., :depth], [3] * depth)
        dense = decode_residual_ids(ids, books[:depth])
        error = (source - dense).square().sum(-1)
        assert (error <= previous + 1e-5).all()
        previous = error
        recovered = projector(dense)
        z = sparse_latents(recovered['atoms'], recovered['coefficient_ids'],
                           projector.dictionary, projector.bins, projector.scales)
        assert torch.equal(z, recovered['latents'])
        repeated = projector(decode_residual_ids(ids.clone(), books[:depth]))
        assert torch.equal(repeated['atoms'], recovered['atoms'])
        assert torch.equal(repeated['coefficient_ids'], recovered['coefficient_ids'])
