import copy

import pytest
import torch

from scripts.tools.imagenet_scale_lr import scale_saved_learning_rate
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule


def trained_source():
    parameter = torch.nn.Parameter(torch.tensor([.2, -.4], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=1e-6, betas=(.9, .95), weight_decay=1e-4)
    for _ in range(3):
        parameter.square().sum().backward(); optimizer.step(); optimizer.zero_grad()
    schedule = FidAdaptiveSchedule(optimizer, initial_lr=1e-4, min_lr=0., total_steps=100,
                                  baseline_fid=15.5)
    for _ in range(17):
        schedule.step()
    return parameter, optimizer.state_dict(), schedule.state_dict()


def restore(parameter, optimizer_state, state):
    optimizer = torch.optim.AdamW([parameter])
    optimizer.load_state_dict(optimizer_state)
    scheduler = FidAdaptiveSchedule(optimizer, **state['policy'], completed_steps=state['last_epoch'], state_dict=state)
    return optimizer, scheduler


def test_1000_times_preserves_moments_groups_clock_and_history():
    _, optimizer, state = trained_source()
    original = copy.deepcopy(state)
    revised, schedule = scale_saved_learning_rate(optimizer, state, 1000.)
    assert revised['state'] is optimizer['state']
    assert state == original
    assert schedule['multiplier'] == state['multiplier'] * 1000
    assert revised['param_groups'][0]['lr'] == pytest.approx(optimizer['param_groups'][0]['lr'] * 1000)
    assert {k: v for k, v in schedule.items() if k != 'multiplier'} == {k: v for k, v in state.items() if k != 'multiplier'}
    for key, value in optimizer['param_groups'][0].items():
        if key != 'lr':
            assert revised['param_groups'][0][key] == value


def test_scaled_mid_epoch_resume_retains_exact_next_updates(tmp_path):
    parameter, optimizer, state = trained_source()
    revised, schedule_state = scale_saved_learning_rate(optimizer, state, 1000.)
    optimizer, scheduler = restore(parameter, revised, schedule_state)
    torch.save(dict(parameter=parameter.detach(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), tmp_path/'resume.pt')
    raw = torch.load(tmp_path/'resume.pt', weights_only=False)
    restored_parameter = torch.nn.Parameter(raw['parameter'].clone())
    restored_optimizer, restored_schedule = restore(restored_parameter, raw['optimizer'], raw['scheduler'])
    for _ in range(5):
        for p, o, s in ((parameter, optimizer, scheduler), (restored_parameter, restored_optimizer, restored_schedule)):
            p.square().sum().backward(); o.step(); o.zero_grad(); s.step()
    assert torch.equal(parameter, restored_parameter)
    assert scheduler.state_dict() == restored_schedule.state_dict()
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(optimizer.state[parameter][key], restored_optimizer.state[restored_parameter][key])


def test_scaling_rejects_invalid_factor_and_mismatched_lr():
    _, optimizer, state = trained_source()
    for factor in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            scale_saved_learning_rate(optimizer, state, factor)
    bad = copy.deepcopy(optimizer); bad['param_groups'][0]['lr'] *= 2
    with pytest.raises(ValueError):
        scale_saved_learning_rate(bad, state, 1000.)
    bad_state = copy.deepcopy(state); bad_state['policy']['min_lr'] = 1e-8
    with pytest.raises(ValueError):
        scale_saved_learning_rate(optimizer, bad_state, 1000.)
