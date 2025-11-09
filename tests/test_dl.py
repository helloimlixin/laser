import torch
import pytest
import sys
import os

# Add the src directory to the path (match tests/test_vq.py style)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.bottleneck import DictionaryLearning


def test_dictionary_learning_shapes_and_loss():
    torch.manual_seed(0)
    B, D, H, W = 2, 32, 4, 4
    K, S = 64, 4

    dl = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=D,
        sparsity_level=S,
        commitment_cost=0.25,
        decay=0.99,
        tolerance=1e-7,
        omp_debug=False,
    )
    z = torch.randn(B, D, H, W, requires_grad=True)
    z_dl, loss, coeffs = dl(z)

    assert z_dl.shape == z.shape
    assert coeffs.shape == (K, B * H * W)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_dictionary_learning_sparsity_and_norms():
    torch.manual_seed(1)
    B, D, H, W = 1, 16, 4, 4
    K, S = 32, 3

    dl = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=D,
        sparsity_level=S,
        commitment_cost=0.25,
        decay=0.99,
        tolerance=1e-7,
        omp_debug=False,
    )
    z = torch.randn(B, D, H, W)
    _, _, coeffs = dl(z)

    nonzeros_per_col = (coeffs.abs() > 1e-8).sum(dim=0)
    assert int(nonzeros_per_col.max().item()) <= S

    with torch.no_grad():
        norms = torch.linalg.norm(dl.dictionary, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_dictionary_learning_backward_grad():
    torch.manual_seed(2)
    B, D, H, W = 2, 16, 4, 4
    K, S = 32, 3

    dl = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=D,
        sparsity_level=S,
        commitment_cost=0.25,
        decay=0.99,
        tolerance=1e-7,
        omp_debug=False,
    )
    z = torch.randn(B, D, H, W, requires_grad=True)
    z_dl, loss, _ = dl(z)
    total = torch.nn.functional.mse_loss(z_dl, z) + loss
    total.backward()

    assert dl.dictionary.grad is not None
    assert torch.isfinite(dl.dictionary.grad).all()
    assert z.grad is not None and torch.isfinite(z.grad).all()


