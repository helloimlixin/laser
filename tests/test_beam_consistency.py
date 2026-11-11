from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.training.beam_consistency import (
    BeamConsistencyHeads, corrupt_code_context, extend_adam_payload,
    masked_cosine_loss, remap_data_cursor, trial_schedule,
)


def test_shifted_current_and_next_targets_and_hook_cleanup():
    heads = BeamConsistencyHeads(3, 3, 4, shallow_layer=0, deep_layer=1)
    heads.current_head = nn.Identity()
    heads.next_head = nn.Identity()
    clean = torch.eye(3).unsqueeze(0)
    shifted = torch.cat([torch.zeros(1, 1, 3), clean[:, :-1]], 1)
    class Replace(nn.Module):
        def forward(self, _):
            return clean.clone()
    blocks = nn.Sequential(nn.Identity(), Replace())
    model = SimpleNamespace(body_transformer=SimpleNamespace(blocks=blocks))
    with heads.capture(model, clean):
        result = blocks(shifted)
    assert torch.equal(result, clean)
    assert float(heads.losses['current']) == pytest.approx(0.)
    assert float(heads.losses['next']) == pytest.approx(0.)
    assert not any(block._forward_hooks for block in blocks)
    with pytest.raises(RuntimeError):
        with heads.capture(model, clean):
            raise RuntimeError('failed forward')
    assert not any(block._forward_hooks for block in blocks)


def test_auxiliary_gradients_reach_body_and_heads_but_not_targets():
    torch.manual_seed(4)
    blocks = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
    model = SimpleNamespace(body_transformer=SimpleNamespace(blocks=blocks))
    heads = BeamConsistencyHeads(5, 3, 7, shallow_layer=0, deep_layer=1)
    inputs = torch.randn(2, 4, 5)
    target = torch.randn(2, 4, 3, requires_grad=True)
    expected = blocks(inputs)
    with heads.capture(model, target):
        actual = blocks(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sum(heads.losses.values()).backward()
    assert target.grad is None
    for parameter in list(blocks.parameters()) + list(heads.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_zero_vectors_do_not_create_nan_or_spurious_loss():
    prediction = torch.randn(2, 4, 3, requires_grad=True)
    loss = masked_cosine_loss(prediction, torch.zeros_like(prediction))
    assert float(loss.detach()) == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_noise_preserves_targets_and_global_rng_and_replays():
    codes = torch.arange(4*8*8*4).reshape(4, 8, 8, 4) % 17
    before = codes.clone()
    global_rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(12)
    state = generator.get_state()
    noisy, fraction = corrupt_code_context(codes, 17, .5, generator)
    assert torch.equal(codes, before) and torch.equal(torch.get_rng_state(), global_rng)
    assert 0 < float(fraction) < .5 and int(noisy.min()) >= 0 and int(noisy.max()) < 17
    generator.set_state(state)
    repeated, _ = corrupt_code_context(codes, 17, .5, generator)
    assert torch.equal(noisy, repeated)
    assert corrupt_code_context(codes, 17, 0, generator)[0] is codes


def test_existing_adam_moments_and_age_survive_new_heads():
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
    saved = optimizer.state_dict()
    head = nn.Linear(2, 1)
    parameters = list(model.parameters()) + list(head.parameters())
    restored = torch.optim.AdamW(parameters, lr=.001)
    restored.load_state_dict(extend_adam_payload(saved, 2))
    for old_id, parameter in zip(saved['param_groups'][0]['params'], model.parameters()):
        for key in ['step', 'exp_avg', 'exp_avg_sq']:
            torch.testing.assert_close(restored.state[parameter][key], saved['state'][old_id][key], rtol=0, atol=0)
    assert all(parameter not in restored.state for parameter in head.parameters())
    restored.zero_grad()
    head(model(torch.ones(1, 3))).square().sum().backward()
    restored.step()
    assert {int(restored.state[p]['step']) for p in model.parameters()} == {3}
    assert {int(restored.state[p]['step']) for p in head.parameters()} == {1}


def test_sample_cursor_and_noise_schedule_boundaries():
    assert remap_data_cursor(17, 108*8, 10750, 8, 64) == (17, 108*64)
    assert remap_data_cursor(18, 0, 11268, 8, 64) == (18, 0)
    with pytest.raises(ValueError):
        remap_data_cursor(17, 1, 10750, 8, 64)
    with pytest.raises(ValueError):
        remap_data_cursor(17, 108*8, 10751, 8, 64)
    assert trial_schedule(0, 1252)['auxiliary_ramp'] == 1/64
    assert trial_schedule(63, 1252)['noise_cap'] == pytest.approx(.02)
    assert trial_schedule(1251, 1252)['noise_cap'] == 0
