import copy

import pytest
import torch

from scripts.tools.imagenet_fid15554_aggressive_lr import (
    create_scheduler, revise_continuation_state,
)
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule


def source():
    p = torch.nn.Parameter(torch.tensor([.25, -.5]))
    opt = torch.optim.AdamW([p], lr=1e-4, betas=(.9, .95), weight_decay=1e-4)
    for _ in range(17):
        p.grad = torch.tensor([.125, -.75])
        opt.step()
    state = dict(kind='fid-adaptive-cosine-v1',
        policy=dict(initial_lr=1e-4, min_lr=3e-7, total_steps=27544,
                    baseline_fid=15.764706963349738, patience=2,
                    min_delta=.05, factor=.5, cooldown=1),
        last_epoch=5008, multiplier=.25, best=15.556776106764232,
        bad_epochs=1, cooldown_remaining=0, reductions=2,
        last_observation_step=5008, last_fid=15.554077729782705)
    opt.param_groups[0]['lr'] = FidAdaptiveSchedule.lr_at_step(state['policy'], 5008, .25)
    return p, opt, state


def revised():
    p, opt, state = source()
    optimizer, schedule = revise_continuation_state(opt.state_dict(), state)
    opt.load_state_dict(optimizer)
    scheduler = create_scheduler(opt, initial_lr=1e-4, min_lr=3e-7,
        total_steps=27544, completed_steps=5008, state_dict=schedule)
    return p, opt, scheduler


def test_revision_preserves_adam_and_scored_clock_with_smaller_raw_best():
    p, opt, source_schedule = source()
    original = copy.deepcopy(opt.state_dict())
    before_schedule = copy.deepcopy(source_schedule)
    optimizer, schedule = revise_continuation_state(opt.state_dict(), source_schedule)
    opt.load_state_dict(optimizer)
    for key, tensor in original['state'][0].items():
        assert torch.equal(tensor, opt.state[p][key])
    assert source_schedule == before_schedule
    assert schedule['last_epoch'] == schedule['last_observation_step'] == 5008
    assert schedule['best'] == source_schedule['best']
    assert schedule['last_fid'] == source_schedule['last_fid'] < schedule['best']
    assert opt.param_groups[0]['lr'] == pytest.approx(3.168327769113575e-6)
    assert schedule['cooldown_remaining'] == 0


def test_cosine_reaches_floor_after_six_epochs_and_stays_there():
    _, opt, schedule = revised()
    previous = opt.param_groups[0]['lr']
    for _ in range(6 * 626):
        schedule.step()
        current = opt.param_groups[0]['lr']
        assert 3e-7 <= current <= previous
        previous = current
    assert schedule.last_epoch == 8764
    assert opt.param_groups[0]['lr'] == 3e-7
    for _ in range(626):
        schedule.step()
        assert opt.param_groups[0]['lr'] == 3e-7


def test_first_and_consecutive_stalls_reduce_without_cooldown():
    _, opt, schedule = revised()
    for _ in range(2):
        for _ in range(626):
            schedule.step()
        before = opt.param_groups[0]['lr']
        decision = schedule.observe(15.60)
        assert decision['decision'] == 'reduced'
        assert schedule.cooldown_remaining == 0
        assert opt.param_groups[0]['lr'] == pytest.approx(3e-7 + (before - 3e-7) / 2)


def test_resume_replays_next_adam_update_and_controller_decision():
    p, opt, schedule = revised()
    p.grad = torch.tensor([-.25, .375])
    opt.step(); schedule.step(); schedule.observe(15.60)
    q = torch.nn.Parameter(p.detach().clone())
    other = torch.optim.AdamW([q], lr=1.)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored = create_scheduler(other, initial_lr=1e-4, min_lr=3e-7,
        total_steps=27544, completed_steps=5009,
        state_dict=copy.deepcopy(schedule.state_dict()))
    for fid in (15.52, 15.60):
        for parameter, optimizer, scheduler in ((p, opt, schedule), (q, other, restored)):
            parameter.grad = torch.tensor([.5, -.125])
            optimizer.step(); scheduler.step()
        assert schedule.observe(fid) == restored.observe(fid)
        assert schedule.state_dict() == restored.state_dict()
        assert torch.equal(p, q)
        for key, tensor in opt.state[p].items():
            assert torch.equal(tensor, other.state[q][key])


def test_inconsistent_source_and_resume_lr_are_rejected_without_mutation():
    _, opt, state = source()
    opt.param_groups[0]['lr'] = .1
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='Source optimizer LR'):
        revise_continuation_state(opt.state_dict(), state)
    assert state == before and opt.param_groups[0]['lr'] == .1
    _, opt, schedule = revised()
    saved = schedule.state_dict()
    opt.param_groups[0]['lr'] = .1
    with pytest.raises(ValueError, match='Saved optimizer LR'):
        create_scheduler(opt, initial_lr=1e-4, min_lr=3e-7, total_steps=27544,
                         completed_steps=5008, state_dict=saved)
    assert opt.param_groups[0]['lr'] == .1
