import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.tools.imagenet_zero_floor_lr import create_scheduler, zero_floor_state
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule


def source():
    p = torch.nn.Parameter(torch.tensor([.25, -.5]))
    opt = torch.optim.AdamW([p], lr=1e-4, betas=(.9, .95), weight_decay=1e-4)
    for _ in range(17):
        p.grad = torch.tensor([.125, -.75]); opt.step()
    state = dict(kind='fid-adaptive-cosine-v1',
        policy=dict(initial_lr=1e-4, min_lr=3e-7, total_steps=27544,
                    baseline_fid=15.764706963349738, patience=1,
                    min_delta=.02, factor=.5, cooldown=0,
                    decay_start_step=5008, decay_steps=3756),
        last_epoch=5008, multiplier=0.028769586450487206,
        best=15.556776106764232, bad_epochs=0, cooldown_remaining=0,
        reductions=3, last_observation_step=5008,
        last_fid=15.554077729782705)
    opt.param_groups[0]['lr'] = FidAdaptiveSchedule.lr_at_step(state['policy'], 5008, state['multiplier'])
    return p, opt, state


def revised():
    p, opt, state = source()
    optimizer, saved = zero_floor_state(opt.state_dict(), state)
    opt.load_state_dict(optimizer)
    schedule = create_scheduler(opt, initial_lr=1e-4, min_lr=0.,
        total_steps=27544, completed_steps=5008, state_dict=saved)
    return p, opt, schedule


def test_zero_floor_preserves_adam_clock_and_controller_history():
    p, opt, state = source()
    before_optimizer, before_schedule = copy.deepcopy(opt.state_dict()), copy.deepcopy(state)
    optimizer, saved = zero_floor_state(opt.state_dict(), state)
    opt.load_state_dict(optimizer)
    assert state == before_schedule
    for key, value in before_optimizer['state'][0].items():
        assert torch.equal(value, opt.state[p][key])
    for key in FidAdaptiveSchedule._fields():
        assert saved[key] == before_schedule[key]
    assert saved['policy']['min_lr'] == 0.
    assert saved['policy']['decay_steps'] == 36 * 626
    assert opt.param_groups[0]['lr'] == pytest.approx(2.8769586450487206e-6)


def test_rate_is_positive_until_final_epoch_and_reaches_exact_zero():
    _, opt, schedule = revised()
    previous = opt.param_groups[0]['lr']
    for _ in range(36 * 626 - 1):
        schedule.step()
        current = opt.param_groups[0]['lr']
        assert 0. < current <= previous
        previous = current
        if schedule.last_epoch == 8764:
            assert current > 0.
    assert schedule.last_epoch == 27543
    schedule.step()
    assert opt.param_groups[0]['lr'] == 0.


def test_stalled_fid_halves_actual_lr_without_a_positive_floor():
    _, opt, schedule = revised()
    for _ in range(626):schedule.step()
    before = opt.param_groups[0]['lr']
    decision = schedule.observe(15.60)
    assert decision['decision'] == 'reduced'
    assert opt.param_groups[0]['lr'] == before / 2
    assert schedule.reductions == 4


def test_final_update_and_zero_floor_state_resume_exactly():
    p, opt, schedule = revised()
    for _ in range(36 * 626 - 1):schedule.step()
    q = torch.nn.Parameter(p.detach().clone())
    other = torch.optim.AdamW([q], lr=1.)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored = create_scheduler(other, initial_lr=1e-4, min_lr=0.,
        total_steps=27544, completed_steps=27543,
        state_dict=copy.deepcopy(schedule.state_dict()))
    for parameter, optimizer, scheduler in ((p, opt, schedule), (q, other, restored)):
        parameter.grad = torch.tensor([.5, -.125]); optimizer.step(); scheduler.step()
    event = schedule.observe(15.60)
    assert event == restored.observe(15.60)
    assert event['decision'] == 'at_floor'
    assert schedule.state_dict() == restored.state_dict()
    assert torch.equal(p, q)
    for key, value in opt.state[p].items():
        assert torch.equal(value, other.state[q][key])


def test_zero_floor_still_requires_positive_peak_and_rejects_negative_floor():
    for peak, floor in ((0., 0.), (1e-4, -1e-8)):
        with pytest.raises(ValueError, match='Invalid adaptive'):
            FidAdaptiveSchedule(SimpleNamespace(param_groups=[{}]),
                initial_lr=peak, min_lr=floor, total_steps=10, baseline_fid=15.)
