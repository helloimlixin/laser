"""Guard the sample weighting of short accumulation groups in the new driver."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.original_rq_training import accumulated_update


class ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)

    def forward(self, codes, model_aux=None, amp=False):
        return self.linear(codes)

    def compute_loss(self, logits, targets, use_soft_target=True):
        return -(targets * logits.log_softmax(-1)).sum(-1).mean()


class OneRankDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def no_sync(self):
        return nullcontext()


def test_unequal_microbatches_match_the_mean_over_actual_images():
    torch.manual_seed(731)
    model = ToyPredictor()
    reference = ToyPredictor()
    reference.load_state_dict(model.state_dict())
    xs = torch.randn(11, 3)
    targets = torch.tensor([.1, .9]).expand(11, 2)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=.03)
    reference_loss = reference.compute_loss(reference(xs), targets)
    reference_loss.backward()
    ref_opt.step()
    quantizer = SimpleNamespace(get_soft_codes=lambda z, **kwargs: (targets[:len(z)], z))
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    with patch('src.original_rq_training.dist.all_reduce'), \
         patch('src.original_rq_training.dist.get_world_size', return_value=1):
        result = accumulated_update(OneRankDDP(model), SimpleNamespace(quantizer=quantizer),
            optimizer, torch.amp.GradScaler('cuda', enabled=False), [xs[:8], xs[8:]], max_gn=100)
    assert result['global_images'] == 11
    assert abs(result['loss'] - reference_loss.item()) < 1e-6
    torch.testing.assert_close(model.linear.weight, reference.linear.weight, atol=1e-7, rtol=1e-6)


def test_partial_final_accumulation_group_takes_a_full_mean_gradient():
    torch.manual_seed(932)
    model = ToyPredictor()
    initial = model.linear.weight.detach().clone()
    xs = torch.randn(2, 3)
    targets = torch.tensor([[1., 0.], [0., 1.]])
    expected_gradient = torch.autograd.grad(model.compute_loss(model(xs), targets), model.linear.weight)[0]
    quantizer = SimpleNamespace(get_soft_codes=lambda z, **kwargs: (targets, z))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    with patch('src.original_rq_training.dist.all_reduce'), \
         patch('src.original_rq_training.dist.get_world_size', return_value=1):
        result = accumulated_update(OneRankDDP(model), SimpleNamespace(quantizer=quantizer),
            optimizer, torch.amp.GradScaler('cuda', enabled=False), [xs], max_gn=100)
    assert result['optimizer_updated']
    torch.testing.assert_close(model.linear.weight, initial - .1 * expected_gradient)
