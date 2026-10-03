import torch


def test_orthogonal_coefficients_match_each_least_squares_prefix_of_source():
    from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients
    torch.manual_seed(89)
    support = torch.nn.functional.normalize(torch.randn(20, 4, 16), dim=-1)
    source = torch.randn(20, 16)
    coefficients = torch.linalg.lstsq(support.transpose(-1, -2), source[..., None]).solution[..., 0]
    gamma, basis = dictionary_to_orthogonal_coefficients(support, coefficients)
    torch.testing.assert_close(gamma, (basis * source[:, None]).sum(-1), atol=2e-6, rtol=2e-5)
    for depth in range(1, 5):
        prefix = support[:, :depth]
        c = torch.linalg.lstsq(prefix.transpose(-1, -2), source[..., None]).solution[..., 0]
        g, q = dictionary_to_orthogonal_coefficients(prefix, c)
        torch.testing.assert_close(g, gamma[:, :depth], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close((g[..., None] * q).sum(-2), (c[..., None] * prefix).sum(-2), atol=2e-6, rtol=2e-5)


def test_orthogonal_codec_ignores_bfloat16_autocast():
    from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients
    torch.manual_seed(90)
    support = torch.randn(8, 4, 16)
    coefficients = torch.randn(8, 4)
    expected = dictionary_to_orthogonal_coefficients(support, coefficients)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = dictionary_to_orthogonal_coefficients(support, coefficients)
    for a, b in zip(actual, expected):
        assert a.dtype == torch.float32
        torch.testing.assert_close(a, b, rtol=0, atol=0)

from src.orthogonal_sparse_codec import (
    dictionary_to_orthogonal_coefficients,
    ordered_support_basis,
    orthogonal_to_dictionary_coefficients,
    reconstruct_from_orthogonal,
)


def _normalized_support(batch=5, depth=4, channels=16):
    generator = torch.Generator().manual_seed(20260821)
    vectors = torch.randn(batch, depth, channels, generator=generator)
    return torch.nn.functional.normalize(vectors, dim=-1)


def test_orthogonal_coordinates_preserve_reconstruction_and_coefficients():
    support = _normalized_support()
    coefficients = torch.randn(5, 4, generator=torch.Generator().manual_seed(7))
    gamma, basis = dictionary_to_orthogonal_coefficients(support, coefficients)

    expected = torch.einsum("...k,...kc->...c", coefficients, support)
    actual = reconstruct_from_orthogonal(support, gamma)
    recovered, recovered_basis = orthogonal_to_dictionary_coefficients(
        support, gamma
    )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(recovered, coefficients, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(recovered_basis, basis)
    gram = torch.einsum("...ic,...jc->...ij", basis, basis)
    identity = torch.eye(4).expand_as(gram)
    torch.testing.assert_close(gram, identity, atol=2e-6, rtol=2e-6)


def test_basis_and_gamma_are_prefix_stable():
    support = _normalized_support(batch=3)
    coefficients = torch.randn(3, 4, generator=torch.Generator().manual_seed(8))
    gamma, full_basis = dictionary_to_orthogonal_coefficients(
        support, coefficients
    )
    full_reconstruction = torch.einsum(
        "...k,...kc->...c", coefficients, support
    )

    for depth in range(1, 5):
        prefix_basis, _ = ordered_support_basis(support[:, :depth])
        torch.testing.assert_close(
            prefix_basis, full_basis[:, :depth], atol=2e-6, rtol=2e-6
        )
        projected = torch.einsum(
            "...kc,...c->...k", prefix_basis, full_reconstruction
        )
        torch.testing.assert_close(
            projected, gamma[:, :depth], atol=2e-6, rtol=2e-6
        )


def test_degenerate_generated_support_stays_finite():
    support = _normalized_support(batch=2)
    support[:, 2] = support[:, 1]
    basis, lower = ordered_support_basis(support)
    assert torch.isfinite(basis).all()
    assert torch.isfinite(lower).all()
    assert (lower.diagonal(dim1=-2, dim2=-1) > 0).all()
