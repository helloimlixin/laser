import pytest
import torch
from torch import nn
from src.training.innovation_depth_context import install_innovation_depth_context


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_mlp = nn.Linear(5, 7)

    def cached_head_output(self, pairs, aux, cond, sample_loc, amp=True):
        d = sample_loc[-1]
        if d == 0:
            return None
        return self.head_mlp(pairs[:, :d].sum(1))


def test_training_and_cached_depth_tokens_match_and_reset_between_sites():
    torch.manual_seed(7)
    model = Model()
    original = model.head_mlp
    keys = list(model.state_dict())
    install_innovation_depth_context(model)
    pairs = torch.randn(2, 3, 4, 5, requires_grad=True)
    full = model.head_mlp(pairs.cumsum(-2))
    expected = torch.nn.functional.linear(pairs, original.weight, original.bias)
    torch.testing.assert_close(full, expected)
    full.square().mean().backward()
    assert torch.isfinite(pairs.grad).all()
    for site in range(3):
        for d in range(4):
            cached = model.cached_head_output(pairs[:, site], None, None, (0, site, d))
            if d:
                torch.testing.assert_close(cached, full[:, site, d-1])
    assert list(model.state_dict()) == keys
    with pytest.raises(ValueError, match='already installed'):
        install_innovation_depth_context(model)
