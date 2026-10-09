import copy

import pytest
import torch

from scripts.tools.imagenet_epoch64_lower_lr import INITIAL_LR, lower_rewind_lr
from scripts.tools.imagenet_zero_floor_lr import create_scheduler


def trained_source():
    parameter = torch.nn.Parameter(torch.tensor([.4, -.2], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=2.876958645048721e-6,
                                  betas=(.9, .95), weight_decay=1e-4)
    for _ in range(3):
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    state = dict(kind='fid-adaptive-cosine-v1',
        policy=dict(initial_lr=1e-4, min_lr=0., total_steps=27544,
                    baseline_fid=15.764706963349738, patience=1, min_delta=.02,
                    factor=.5, cooldown=0, decay_start_step=5008, decay_steps=22536),
        last_epoch=5008, multiplier=.02876958645048721,
        best=15.556776106764232, bad_epochs=0, cooldown_remaining=0,
        reductions=3, last_observation_step=5008, last_fid=15.554077729782705)
    return parameter, optimizer.state_dict(), state


def schedule(optimizer, state):
    return create_scheduler(optimizer, initial_lr=1e-4, min_lr=0.,
                            total_steps=27544, completed_steps=state['last_epoch'],
                            state_dict=state)


def advance(parameter, optimizer, scheduler):
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


def test_preserves_trained_adam_and_controller_history():
    _, optimizer, state = trained_source()
    original = copy.deepcopy(state)
    revised_optimizer, revised_state = lower_rewind_lr(optimizer, state)
    assert revised_optimizer['state'] is optimizer['state']
    assert revised_optimizer['param_groups'][0]['lr'] == pytest.approx(INITIAL_LR)
    assert optimizer['param_groups'][0]['lr'] > INITIAL_LR
    assert state == original
    assert revised_state['multiplier'] < state['multiplier']
    assert {k: v for k, v in revised_state.items() if k != 'multiplier'} == {
        k: v for k, v in state.items() if k != 'multiplier'}
    for key in optimizer['param_groups'][0]:
        if key != 'lr':
            assert revised_optimizer['param_groups'][0][key] == optimizer['param_groups'][0][key]


def test_full_resume_keeps_exact_next_updates_after_official_reduction(tmp_path):
    parameter, source, source_schedule = trained_source()
    revised, revised_schedule = lower_rewind_lr(source, source_schedule)
    optimizer = torch.optim.AdamW([parameter])
    optimizer.load_state_dict(revised)
    controller = schedule(optimizer, revised_schedule)
    for _ in range(4):
        advance(parameter, optimizer, controller)
    decision = controller.observe(15.7)
    assert decision['lr_after'] == decision['lr_before'] * .5
    path = tmp_path / 'resume.pt'
    torch.save(dict(parameter=parameter.detach(), optimizer=optimizer.state_dict(),
                    scheduler=controller.state_dict()), path)
    raw = torch.load(path, weights_only=False)
    restored_parameter = torch.nn.Parameter(raw['parameter'].clone())
    restored_optimizer = torch.optim.AdamW([restored_parameter])
    restored_optimizer.load_state_dict(raw['optimizer'])
    restored_controller = schedule(restored_optimizer, raw['scheduler'])
    assert restored_controller.state_dict() == controller.state_dict()
    for _ in range(5):
        advance(parameter, optimizer, controller)
        advance(restored_parameter, restored_optimizer, restored_controller)
    assert torch.equal(parameter, restored_parameter)
    assert controller.state_dict() == restored_controller.state_dict()
    assert optimizer.param_groups[0]['lr'] == restored_optimizer.param_groups[0]['lr']
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(optimizer.state[parameter][key],
                           restored_optimizer.state[restored_parameter][key])


def test_keeps_zero_endpoint_and_rejects_invalid_source_or_rate():
    _, optimizer, state = trained_source()
    for rate in (0., -1e-7, float('nan'), float('inf'), 2.876958645048721e-6):
        with pytest.raises(ValueError):
            lower_rewind_lr(optimizer, state, initial_lr=rate)
    bad_optimizer = copy.deepcopy(optimizer)
    bad_optimizer['param_groups'][0]['lr'] *= 2
    with pytest.raises(ValueError):
        lower_rewind_lr(bad_optimizer, state)
    for section, key, value in ((None, 'last_epoch', 5009),
                               ('policy', 'min_lr', 3e-8),
                               ('policy', 'decay_steps', 3756)):
        invalid = copy.deepcopy(state)
        (invalid if section is None else invalid[section])[key] = value
        with pytest.raises(ValueError):
            lower_rewind_lr(optimizer, invalid)
    revised, revised_state = lower_rewind_lr(optimizer, state)
    parameter = torch.nn.Parameter(torch.zeros(1))
    restored = torch.optim.AdamW([parameter], lr=revised['param_groups'][0]['lr'])
    controller = schedule(restored, revised_state)
    for _ in range(22536):
        controller.step()
    assert restored.param_groups[0]['lr'] == 0.
    decision = controller.observe(15.8)
    assert decision['decision'] == 'at_floor'
    assert controller.reductions == revised_state['reductions']
