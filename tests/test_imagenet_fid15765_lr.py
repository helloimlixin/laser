import copy

import pytest
import torch

from scripts.tools.imagenet_fid16379_lr import create_scheduler as old_scheduler, halve_continuation_state
from scripts.tools.imagenet_fid15765_lr import create_scheduler, BASELINE_FID


def revised_source():
    p = torch.nn.Parameter(torch.tensor([.25, -.5]))
    opt = torch.optim.AdamW([p], lr=1e-4, betas=(.9, .95), weight_decay=1e-4)
    schedule = old_scheduler(opt, initial_lr=1e-4, min_lr=3e-7, total_steps=27544)
    for _ in range(17):
        p.grad = torch.tensor([.125, -.75]); opt.step(); schedule.step()
    schedule.observe(BASELINE_FID)
    original = copy.deepcopy(opt.state_dict())
    original_schedule = copy.deepcopy(schedule.state_dict())
    reduced, state = halve_continuation_state(original, original_schedule)
    state['policy'].update(baseline_fid=BASELINE_FID, patience=2)
    opt.load_state_dict(reduced)
    revised = create_scheduler(opt, initial_lr=1e-4, min_lr=3e-7,
        total_steps=27544, completed_steps=17, state_dict=state)
    return p, opt, revised, original, original_schedule


def test_revision_retains_adam_and_clock_and_reduces_after_two_stalls():
    p, opt, schedule, original, before = revised_source()
    for key, value in original['state'][0].items():
        assert torch.equal(value, opt.state[p][key])
    assert schedule.last_epoch == before['last_epoch'] == 17
    assert schedule.best == before['best'] == BASELINE_FID
    assert opt.param_groups[0]['lr'] == pytest.approx(3e-7 + (original['param_groups'][0]['lr'] - 3e-7) / 2)
    decisions = []
    rates = [opt.param_groups[0]['lr']]
    for _ in range(3):
        p.grad = torch.tensor([-.25, .375]); opt.step(); schedule.step()
        decisions.append(schedule.observe(16.1)['decision'])
        rates.append(opt.param_groups[0]['lr'])
    assert decisions == ['cooldown', 'watch', 'reduced']
    assert schedule.multiplier == .25
    assert all(a >= b >= 3e-7 for a, b in zip(rates, rates[1:]))


def test_resume_replays_next_update_and_fid_controller_history():
    p, opt, schedule, _, _ = revised_source()
    for _ in range(3):
        p.grad = torch.tensor([-.25, .375]); opt.step(); schedule.step(); schedule.observe(16.1)
    q = torch.nn.Parameter(p.detach().clone())
    restored_opt = torch.optim.AdamW([q], lr=1.)
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored = create_scheduler(restored_opt, initial_lr=1e-4, min_lr=3e-7,
        total_steps=27544, completed_steps=20, state_dict=copy.deepcopy(schedule.state_dict()))
    for fid in (15.7, 15.8, 15.9):
        for parameter, optimizer, scheduler in ((p, opt, schedule), (q, restored_opt, restored)):
            parameter.grad = torch.tensor([.5, -.125]); optimizer.step(); scheduler.step()
        assert schedule.observe(fid) == restored.observe(fid)
        assert schedule.state_dict() == restored.state_dict()
        assert torch.equal(p, q)
        for key, value in opt.state[p].items():
            assert torch.equal(value, restored_opt.state[q][key])


def test_changed_policy_and_inconsistent_lr_are_rejected_without_mutation():
    _, opt, schedule, _, _ = revised_source()
    state = copy.deepcopy(schedule.state_dict()); state['policy']['patience'] = 3
    with pytest.raises(ValueError, match='configuration changed'):
        create_scheduler(opt, initial_lr=1e-4, min_lr=3e-7, total_steps=27544,
                         completed_steps=17, state_dict=state)
    state = copy.deepcopy(schedule.state_dict()); opt.param_groups[0]['lr'] = .1
    with pytest.raises(ValueError, match='Saved optimizer LR'):
        create_scheduler(opt, initial_lr=1e-4, min_lr=3e-7, total_steps=27544,
                         completed_steps=17, state_dict=state)
    assert opt.param_groups[0]['lr'] == .1
