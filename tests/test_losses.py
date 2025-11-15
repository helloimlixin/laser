import torch

from src.models.losses import (
    multi_resolution_dct_loss,
    multi_resolution_gradient_loss,
)
from src.models.bottleneck import DictionaryLearning


def test_multi_resolution_dct_loss_zero_for_identical_inputs():
    x = torch.randn(2, 3, 16, 16)
    y = x.clone()
    loss = multi_resolution_dct_loss(x, y, num_levels=3)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_multi_resolution_dct_loss_detects_difference():
    x = torch.zeros(1, 3, 16, 16)
    y = torch.zeros(1, 3, 16, 16)
    y[:, :, 0, 0] = 1.0
    loss = multi_resolution_dct_loss(x, y, num_levels=2)
    assert loss > 0


def test_multi_resolution_gradient_loss_zero_for_identical_inputs():
    x = torch.randn(2, 3, 32, 32)
    y = x.clone()
    loss = multi_resolution_gradient_loss(x, y, num_levels=3)
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_multi_resolution_gradient_loss_detects_edge_difference():
    x = torch.zeros(1, 3, 16, 16)
    y = torch.zeros(1, 3, 16, 16)
    y[:, :, :, 8:] = 1.0  # introduce vertical edge
    loss = multi_resolution_gradient_loss(x, y, num_levels=2)
    assert loss > 0


def test_dictionary_orthogonality_loss_zero_for_orthonormal_atoms():
    bottleneck = DictionaryLearning(num_embeddings=2, embedding_dim=2, patch_size=1, normalize_atoms=False)
    with torch.no_grad():
        bottleneck.dictionary.copy_(torch.eye(2))
    loss = bottleneck.orthogonality_loss()
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_dictionary_orthogonality_loss_detects_correlated_atoms():
    bottleneck = DictionaryLearning(num_embeddings=2, embedding_dim=2, patch_size=1, normalize_atoms=False)
    twin_atom = torch.tensor([[1.0, 1.0],
                              [0.0, 0.0]])
    with torch.no_grad():
        bottleneck.dictionary.copy_(twin_atom)
    loss = bottleneck.orthogonality_loss()
    assert loss > 0
