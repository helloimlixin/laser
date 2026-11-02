import pytest
import torch
from torch import nn

from src.training.density_gradient import install_density_gradient_scale, scale_gradient


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(3, 4)
        self.sign = nn.Linear(4, 2)
        self.gmm_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 6))])


@pytest.mark.parametrize('scale', [0., .1, 1.])
def test_only_density_gradient_into_shared_features_changes(scale):
    torch.manual_seed(9)
    model = Heads().double()
    x = torch.randn(5, 3, dtype=torch.float64)
    shared = model.shared(x)
    reference = model.gmm_heads[0](shared)
    sign = model.sign(shared).square().mean()
    density = reference.square().mean()
    gs = torch.autograd.grad(sign, model.shared.weight, retain_graph=True)[0]
    gd = torch.autograd.grad(density, model.shared.weight, retain_graph=True)[0]
    head_parameters = list(model.gmm_heads.parameters())
    head_gradients = torch.autograd.grad(density, head_parameters)
    keys = list(model.state_dict())
    handles = install_density_gradient_scale(model, scale)
    shared = model.shared(x)
    actual = model.gmm_heads[0](shared)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    (actual.square().mean() + model.sign(shared).square().mean()).backward()
    torch.testing.assert_close(model.shared.weight.grad, gs + scale * gd)
    for parameter, expected in zip(head_parameters, head_gradients):
        torch.testing.assert_close(parameter.grad, expected)
    assert list(model.state_dict()) == keys
    with pytest.raises(ValueError, match='already installed'):
        install_density_gradient_scale(model, scale)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize('scale', [-1, 1.01, float('nan'), float('inf')])
def test_invalid_policy_rejected(scale):
    with pytest.raises(ValueError):
        scale_gradient(torch.ones(2), scale)
