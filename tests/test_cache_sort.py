import torch

from src.cache_sort import canonicalize_cache, sort_sparse_pairs, sort_token_pairs


def test_sort_sparse_pairs_orders_atoms_and_coeffs_together():
    atom_ids = torch.tensor([[[[7, 2, 5, 1]]]], dtype=torch.long)
    coeffs = torch.tensor([[[[0.7, 0.2, 0.5, 0.1]]]], dtype=torch.float32)

    got_atoms, got_coeffs = sort_sparse_pairs(atom_ids, coeffs)

    assert torch.equal(got_atoms, torch.tensor([[[[1, 2, 5, 7]]]], dtype=torch.long))
    assert torch.allclose(got_coeffs, torch.tensor([[[[0.1, 0.2, 0.5, 0.7]]]], dtype=torch.float32))


def test_sort_sparse_pairs_keeps_duplicate_atoms_stable():
    atom_ids = torch.tensor([[[[3, 1, 3, 1]]]], dtype=torch.long)
    coeffs = torch.tensor([[[[0.3, 0.1, 0.4, 0.2]]]], dtype=torch.float32)

    got_atoms, got_coeffs = sort_sparse_pairs(atom_ids, coeffs)

    assert torch.equal(got_atoms, torch.tensor([[[[1, 1, 3, 3]]]], dtype=torch.long))
    assert torch.allclose(got_coeffs, torch.tensor([[[[0.1, 0.2, 0.3, 0.4]]]], dtype=torch.float32))


def test_sort_token_pairs_orders_interleaved_pairs():
    toks = torch.tensor([[[[7, 70, 2, 20, 5, 50, 1, 10]]]], dtype=torch.int32)

    got = sort_token_pairs(toks)

    want = torch.tensor([[[[1, 10, 2, 20, 5, 50, 7, 70]]]], dtype=torch.int32)
    assert torch.equal(got, want)


def test_sort_token_pairs_rejects_odd_depth():
    toks = torch.tensor([1, 2, 3], dtype=torch.int32)

    try:
        sort_token_pairs(toks)
    except ValueError as exc:
        assert "even token depth" in str(exc)
    else:
        raise AssertionError("Expected ValueError for odd token depth")


def test_canonicalize_cache_sorts_legacy_real_valued_rows_and_marks_meta():
    cache = {
        "tokens_flat": torch.tensor([[7, 2, 5, 1]], dtype=torch.int32),
        "coeffs_flat": torch.tensor([[0.7, 0.2, 0.5, 0.1]], dtype=torch.float32),
        "shape": (1, 2, 2),
        "meta": {"dataset": "toy"},
    }

    got = canonicalize_cache(cache)

    assert torch.equal(got["tokens_flat"], torch.tensor([[2, 7, 1, 5]], dtype=torch.int32))
    assert torch.allclose(got["coeffs_flat"], torch.tensor([[0.2, 0.7, 0.1, 0.5]], dtype=torch.float32))
    assert got["meta"]["support_order"] == "atom_id"


def test_canonicalize_cache_sorts_legacy_quantized_rows_per_site():
    cache = {
        "tokens_flat": torch.tensor([[7, 70, 2, 20, 5, 50, 1, 10]], dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {"dataset": "toy"},
    }

    got = canonicalize_cache(cache)

    want = torch.tensor([[2, 20, 7, 70, 1, 10, 5, 50]], dtype=torch.int32)
    assert torch.equal(got["tokens_flat"], want)
    assert got["meta"]["support_order"] == "atom_id"
