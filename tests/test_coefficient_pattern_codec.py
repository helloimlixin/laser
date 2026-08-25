import torch

from src.coefficient_pattern_codec import (
    assign_coefficient_patterns,
    decode_coefficient_patterns,
    fit_coefficient_patterns,
    reconstruct_pattern_latents,
    selected_support_grams,
)


def test_support_gram_distance_matches_direct_latent_distance():
    dictionary = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    atoms = torch.tensor([[0, 1], [2, 3]])
    coefficients = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    patterns = torch.tensor([[1.5, -0.5], [0.0, 2.5], [2.5, -1.0]])
    grams = selected_support_grams(atoms, dictionary, chunk_size=1)

    assignments, distances = assign_coefficient_patterns(
        coefficients, patterns, grams=grams, chunk_size=1
    )

    support_vectors = dictionary[atoms]
    direct = []
    for row in range(len(coefficients)):
        errors = coefficients[row] - patterns
        latent_errors = (support_vectors[row].unsqueeze(0) * errors[..., None]).sum(1)
        direct.append(latent_errors.square().sum(1))
    direct = torch.stack(direct)
    expected_distances, expected_assignments = direct.min(dim=1)
    assert torch.equal(assignments, expected_assignments)
    assert torch.allclose(distances, expected_distances, atol=1e-6)


def test_geometry_aware_lloyd_fit_recovers_separated_patterns():
    coefficients = torch.tensor(
        [
            [-2.1, 0.9],
            [-2.0, 1.0],
            [-1.9, 1.1],
            [2.9, -0.9],
            [3.0, -1.0],
            [3.1, -1.1],
        ]
    )
    grams = torch.eye(2).expand(len(coefficients), 2, 2).clone()
    initial = coefficients[[0, 1]].clone()

    patterns = fit_coefficient_patterns(
        coefficients,
        num_patterns=2,
        grams=grams,
        iterations=5,
        initial_patterns=initial,
        chunk_size=2,
    )

    patterns = patterns[patterns[:, 0].argsort()]
    assert torch.allclose(patterns, torch.tensor([[-2.0, 1.0], [3.0, -1.0]]))


def test_pattern_decode_reconstructs_sparse_latents():
    dictionary = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0**-0.5, 2.0**-0.5],
        ]
    )
    atoms = torch.tensor([[[0, 1], [2, 0]]])
    pattern_ids = torch.tensor([[0, 1]])
    patterns = torch.tensor([[2.0, -1.0], [0.5, 3.0]])

    coefficients = decode_coefficient_patterns(pattern_ids, patterns)
    reconstructed = reconstruct_pattern_latents(
        atoms, pattern_ids, patterns, dictionary
    )

    expected = (dictionary[atoms] * coefficients[..., None]).sum(dim=-2)
    assert torch.equal(coefficients, torch.tensor([[[2.0, -1.0], [0.5, 3.0]]]))
    assert torch.allclose(reconstructed, expected)
