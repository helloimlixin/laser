import pytest
import torch
from torch.nn import functional as F

from src.models.laser_depth_integer_tokenizer import DepthIntegerCodebooks
from src.training.laser_depth_codebook_fit import initialize_depth_books, refit_depth_books


def test_same_id_uses_its_depths_dictionary_and_prefix():
    books = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    q = DepthIntegerCodebooks(books, [1., 2.], coefficient_bins=5, coefficient_max=1.)
    atoms = torch.tensor([[0, 0], [1, 1]])
    codes = q.pack(atoms, torch.full_like(atoms, 4))
    assert torch.equal(q.unpack(codes)[0], atoms)
    emb = q.get_code_emb_with_depth(codes)[0]
    torch.testing.assert_close(emb, torch.tensor([[[1., 0.], [0., 2.]], [[0., 1.], [2., 0.]]]))
    torch.testing.assert_close(q.get_code_emb_with_depth(codes[:, :1])[0], emb[:, :1])
    for invalid in [torch.tensor([[-1, 0]]), torch.tensor([[10, 0]])]:
        with pytest.raises(ValueError): q.unpack(invalid)


def test_grouped_omp_refits_selected_values_and_is_deterministic():
    books = torch.stack([torch.eye(3)[:, :2], F.normalize(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]), dim=0)])
    q = DepthIntegerCodebooks(books, [2., 2.], coefficient_bins=512)
    x = torch.tensor([[2., 1., 0.], [0., 2., 3.], [-2., -1., 0.]])
    a, c = q.encode_components(x, chunk_size=1)
    a2, c2 = q.encode_components(x, chunk_size=8)
    assert torch.equal(a, a2)
    torch.testing.assert_close(c, c2)
    assert float((x-q.reconstruct(a, c)).square().max()) < 1.e-9
    result = q.quantize(x)
    torch.testing.assert_close(result['quantized'], q.get_code_emb_with_depth(result['codes'])[0].sum(-2))
    assert result['codes'].shape == (3, 2)
    assert q.vocab_size == 1024


def test_signed_ray_initialization_and_coordinate_update_reduce_fixed_code_error():
    torch.manual_seed(22)
    source = F.normalize(torch.randn(6, 12), dim=0)
    energy = torch.rand(3, 12)
    books = initialize_depth_books(source, energy, 4, iterations=3)
    again = initialize_depth_books(source, energy, 4, iterations=3)
    torch.testing.assert_close(books, again, rtol=0, atol=0)
    assert books.shape == (3, 6, 4)
    q = DepthIntegerCodebooks(books, [1.] * 3)
    x = torch.randn(50, 6)
    atoms, values = q.encode_components(x, chunk_size=7)
    result = refit_depth_books(q, x, atoms, values, chunk_size=9)
    assert result['fixed_code_mse_after'] <= result['fixed_code_mse_before']
    torch.testing.assert_close(q.dictionaries.norm(dim=1), torch.ones(3, 4))
