import pytest
import torch
from src.training.parameter_ema import ParameterEMA


def test_ema_matches_explicit_weights_and_resumes_exactly():
    m = torch.nn.Linear(3, 2)
    ema = ParameterEMA(m, .9)
    expected = {k: p.detach().clone() for k, p in m.named_parameters()}
    for _ in range(4):
        with torch.no_grad():
            for k, p in m.named_parameters():
                p.add_(torch.randn_like(p))
                expected[k] = .9 * expected[k] + .1 * p
        ema.update(m)
    restored = ParameterEMA(m, .9, ema.state_dict())
    assert restored.updates == 4
    for k in expected:
        torch.testing.assert_close(ema.values[k], expected[k])
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1)
    ema.update(m)
    restored.update(m)
    for k in expected:
        torch.testing.assert_close(ema.values[k], restored.values[k], rtol=0, atol=0)


def test_evaluation_restores_parameters_and_optimizer_references_on_failure():
    m = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(m.parameters())
    ema = ParameterEMA(m, .5)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1)
    original = {k: p.detach().clone() for k, p in m.named_parameters()}
    parameter_ids = [id(p) for p in m.parameters()]
    with pytest.raises(RuntimeError, match='evaluation failed'):
        with ema.apply(m):
            for k, p in m.named_parameters():
                torch.testing.assert_close(p, ema.values[k], rtol=0, atol=0)
            raise RuntimeError('evaluation failed')
    for k, p in m.named_parameters():
        torch.testing.assert_close(p, original[k], rtol=0, atol=0)
    assert [id(p) for p in optimizer.param_groups[0]['params']] == parameter_ids
