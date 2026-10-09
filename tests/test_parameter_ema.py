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


def test_first_update_is_averaged_after_adam_and_keeps_its_moments():
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(2, 3, dtype=torch.float64)).square().sum().backward()
    optimizer.step(); optimizer.zero_grad()
    ema = ParameterEMA(model, .999)
    original = {k: p.detach().clone() for k, p in model.named_parameters()}
    model(torch.ones(2, 3, dtype=torch.float64)).square().sum().backward()
    optimizer.step(); optimizer.zero_grad()
    ema.update(model)
    moments = {p: {k: t.clone() for k, t in s.items()} for p, s in optimizer.state.items()}
    for k, p in model.named_parameters():
        torch.testing.assert_close(ema.values[k], original[k].lerp(p, .001), rtol=0, atol=1e-15)
    with ema.apply(model):
        assert any(not torch.equal(p, ema.values[k]) for k, p in original.items())
    for p, fields in moments.items():
        for k, t in fields.items():
            torch.testing.assert_close(optimizer.state[p][k], t, rtol=0, atol=0)


def test_live_checkpoint_view_can_be_frozen_without_mutating_saved_ema():
    model = torch.nn.Linear(3, 2)
    ema = ParameterEMA(model)
    live = ema.state_dict(cpu=False)
    frozen = {k: t.clone() for k, t in live['values'].items()}
    with torch.no_grad():
        for p in model.parameters():p.add_(1)
    ema.update(model)
    assert all(live['values'][k].data_ptr() == t.data_ptr() for k, t in ema.values.items())
    assert all(not torch.equal(frozen[k], t) for k, t in ema.values.items())
