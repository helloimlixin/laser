import copy

import pytest
import torch

from scripts.tools.imagenet_lower_floor_lr import create_scheduler, lower_floor_state
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule


def source(step=5008):
    p = torch.nn.Parameter(torch.tensor([.25, -.5]))
    opt = torch.optim.AdamW([p], lr=1e-4, betas=(.9, .95), weight_decay=1e-4)
    for _ in range(17):
        p.grad = torch.tensor([.125, -.75]); opt.step()
    state = dict(kind='fid-adaptive-cosine-v1',
        policy=dict(initial_lr=1e-4, min_lr=3e-7, total_steps=27544,
                    baseline_fid=15.764706963349738, patience=1,
                    min_delta=.02, factor=.5, cooldown=0,
                    decay_start_step=5008, decay_steps=3756),
        last_epoch=step, multiplier=0.028769586450487206,
        best=15.556776106764232, bad_epochs=0, cooldown_remaining=0,
        reductions=3, last_observation_step=5008,
        last_fid=15.554077729782705)
    opt.param_groups[0]['lr'] = FidAdaptiveSchedule.lr_at_step(state['policy'], step, state['multiplier'])
    return p, opt, state


def revised(step=5008):
    p, opt, state = source(step)
    optimizer, saved = lower_floor_state(opt.state_dict(), state)
    opt.load_state_dict(optimizer)
    schedule = create_scheduler(opt, initial_lr=1e-4, min_lr=3e-8,
        total_steps=27544, completed_steps=step, state_dict=saved)
    return p, opt, schedule


def test_only_floor_and_group_lr_change_with_adam_and_history_preserved():
    p, opt, state = source()
    before_optimizer, before_schedule = copy.deepcopy(opt.state_dict()), copy.deepcopy(state)
    optimizer, saved = lower_floor_state(opt.state_dict(), state)
    opt.load_state_dict(optimizer)
    for key, value in before_optimizer['state'][0].items():
        assert torch.equal(value, opt.state[p][key])
    assert state == before_schedule
    restored_policy = dict(saved['policy'], min_lr=3e-7)
    assert restored_policy == before_schedule['policy']
    for key in FidAdaptiveSchedule._fields():
        assert saved[key] == before_schedule[key]
    assert opt.param_groups[0]['betas'] == before_optimizer['param_groups'][0]['betas']
    assert opt.param_groups[0]['lr'] == pytest.approx(2.9060955574552057e-6)


def test_completed_cosine_uses_new_floor_and_reports_no_false_reduction():
    _, opt, schedule = revised(11000)
    assert opt.param_groups[0]['lr'] == 3e-8
    multiplier, count = schedule.multiplier, schedule.reductions
    for _ in range(2):
        schedule.step()
        decision = schedule.observe(15.60)
        assert decision['decision'] == 'at_floor'
        assert decision['lr_before'] == decision['lr_after'] == 3e-8
        assert schedule.multiplier == multiplier and schedule.reductions == count
    assert schedule.last_epoch == schedule.last_observation_step == 11002
    assert schedule.last_fid == 15.60


def test_stalls_above_floor_still_reduce_actual_lr():
    _, opt, schedule = revised()
    for _ in range(626):
        schedule.step()
    before = opt.param_groups[0]['lr']
    decision = schedule.observe(15.60)
    assert decision['decision'] == 'reduced'
    assert opt.param_groups[0]['lr'] == pytest.approx(3e-8 + (before - 3e-8) / 2)
    assert schedule.reductions == 4


def test_resume_replays_adam_update_and_floor_controller_state():
    p, opt, schedule = revised(11000)
    p.grad = torch.tensor([-.25, .375]); opt.step(); schedule.step(); schedule.observe(15.60)
    q = torch.nn.Parameter(p.detach().clone())
    other = torch.optim.AdamW([q], lr=1.)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored = create_scheduler(other, initial_lr=1e-4, min_lr=3e-8,
        total_steps=27544, completed_steps=11001,
        state_dict=copy.deepcopy(schedule.state_dict()))
    for fid in (15.52, 15.60):
        for parameter, optimizer, scheduler in ((p, opt, schedule), (q, other, restored)):
            parameter.grad = torch.tensor([.5, -.125]); optimizer.step(); scheduler.step()
        assert schedule.observe(fid) == restored.observe(fid)
        assert schedule.state_dict() == restored.state_dict()
        assert torch.equal(p, q)
        for key, value in opt.state[p].items():
            assert torch.equal(value, other.state[q][key])


def test_invalid_floor_and_inconsistent_source_lr_are_rejected():
    _, opt, state = source()
    for floor in (-3e-8, 3e-7, 1e-6):
        with pytest.raises(ValueError, match='new floor'):
            lower_floor_state(opt.state_dict(), state, min_lr=floor)
    opt.param_groups[0]['lr'] = .1
    with pytest.raises(ValueError, match='Source optimizer LR'):
        lower_floor_state(opt.state_dict(), state)
    assert opt.param_groups[0]['lr'] == .1
