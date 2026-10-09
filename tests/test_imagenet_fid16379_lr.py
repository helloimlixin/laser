import copy

import pytest
import torch

from scripts.tools.imagenet_fid16379_lr import create_scheduler, halve_continuation_state


def setup():
    p = torch.nn.Parameter(torch.tensor([.25, -.5]))
    opt = torch.optim.AdamW([p], lr=3e-4, betas=(.9, .95), weight_decay=1e-4)
    for _ in range(626):
        p.grad = torch.tensor([.125, -.75])
        opt.step()
    before = copy.deepcopy(opt.state_dict())
    schedule = create_scheduler(opt, initial_lr=1e-4, min_lr=3e-7, total_steps=27544)
    return p, opt, schedule, before


def test_lr_migration_preserves_trained_adam_and_monotonic_decay():
    p, opt, schedule, before = setup()
    assert int(opt.state[p]['step']) == 626
    for key, value in before['state'][0].items():
        assert torch.equal(value, opt.state[p][key])
    assert opt.param_groups[0]['lr'] == 1e-4
    rates = [opt.param_groups[0]['lr']]
    for step in range(1, 27545):
        schedule.step()
        if step % 1252 == 0:
            schedule.observe(16.5)
        rates.append(opt.param_groups[0]['lr'])
    assert rates[-1] == 3e-7 and schedule.reductions > 0
    assert all(a >= b >= 3e-7 for a, b in zip(rates, rates[1:]))


def test_reload_replays_next_adam_update_and_plateau_decisions():
    p, opt, schedule, _ = setup()
    for _ in range(3):
        p.grad = torch.tensor([-.25, .375])
        opt.step(); schedule.step(); schedule.observe(16.5)
    assert schedule.reductions == 1
    restored = torch.nn.Parameter(p.detach().clone())
    restored_opt = torch.optim.AdamW([restored], lr=1.)
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored_schedule = create_scheduler(restored_opt, initial_lr=1e-4, min_lr=3e-7,
        total_steps=27544, completed_steps=3, state_dict=copy.deepcopy(schedule.state_dict()))
    for fid in (16.4, 16.2, 16.4, 16.4, 16.4):
        for param, optimizer, scheduler in ((p, opt, schedule), (restored, restored_opt, restored_schedule)):
            param.grad = torch.tensor([.5, -.125])
            optimizer.step(); scheduler.step()
        assert schedule.observe(fid) == restored_schedule.observe(fid)
        assert torch.equal(p, restored)
        assert schedule.state_dict() == restored_schedule.state_dict()
        for key, value in opt.state[p].items():
            assert torch.equal(value, restored_opt.state[restored][key])


def test_resume_rejects_changed_policy_lr_and_cursor_without_mutation():
    _, opt, schedule, _ = setup()
    state = copy.deepcopy(schedule.state_dict())
    opt.param_groups[0]['lr'] = 3e-4
    before = copy.deepcopy([{k: v for k, v in g.items() if k != 'params'}
                            for g in opt.param_groups])
    with pytest.raises(ValueError, match='Saved optimizer LR'):
        create_scheduler(opt, initial_lr=1e-4, min_lr=3e-7, total_steps=27544, state_dict=state)
    assert [{k: v for k, v in g.items() if k != 'params'} for g in opt.param_groups] == before
    with pytest.raises(ValueError, match='configuration changed'):
        create_scheduler(opt, initial_lr=3e-4, min_lr=3e-7, total_steps=27544, state_dict=state)
    with pytest.raises(ValueError, match='cursor'):
        create_scheduler(opt, initial_lr=1e-4, min_lr=3e-7, total_steps=27544,
                         completed_steps=1, state_dict=state)


def test_scored_checkpoint_lr_reduction_preserves_adam_schedule_clock_and_history():
    p, opt, schedule, _ = setup()
    for _ in range(5):
        p.grad = torch.tensor([-.25, .375]); opt.step(); schedule.step()
    schedule.observe(16.24268000164875)
    original = copy.deepcopy(opt.state_dict())
    old_schedule = copy.deepcopy(schedule.state_dict())
    reduced_opt, reduced_schedule = halve_continuation_state(original, old_schedule)
    assert reduced_opt['state'] is original['state']
    assert original['param_groups'][0]['lr'] == opt.param_groups[0]['lr']
    assert old_schedule == schedule.state_dict()
    assert reduced_schedule['last_epoch'] == reduced_schedule['last_observation_step'] == 5
    assert reduced_schedule['best'] == old_schedule['best']
    assert reduced_schedule['policy'] == old_schedule['policy']
    assert reduced_schedule['multiplier'] == .5 and reduced_schedule['reductions'] == 1
    expected_lr = 3e-7 + (opt.param_groups[0]['lr'] - 3e-7) / 2
    assert reduced_opt['param_groups'][0]['lr'] == pytest.approx(expected_lr)
    restored = torch.nn.Parameter(p.detach().clone())
    restored_opt = torch.optim.AdamW([restored], lr=1.)
    restored_opt.load_state_dict(reduced_opt)
    restored_schedule = create_scheduler(restored_opt, initial_lr=1e-4, min_lr=3e-7,
        total_steps=27544, completed_steps=5, state_dict=reduced_schedule)
    assert int(restored_opt.state[restored]['step']) == 631
    for key, value in opt.state[p].items():
        assert torch.equal(value, restored_opt.state[restored][key])
    restored.grad = torch.tensor([.5, -.125])
    restored_opt.step(); restored_schedule.step()
    assert int(restored_opt.state[restored]['step']) == 632
    assert restored_schedule.last_epoch == 6
    assert restored_opt.param_groups[0]['lr'] < expected_lr


def test_lr_reduction_rejects_inconsistent_source_without_mutating_it():
    _, opt, schedule, _ = setup()
    original = copy.deepcopy(opt.state_dict())
    old_schedule = copy.deepcopy(schedule.state_dict())
    original['param_groups'][0]['lr'] = 1.
    with pytest.raises(ValueError, match='Source optimizer LR'):
        halve_continuation_state(original, old_schedule)
    assert original['param_groups'][0]['lr'] == 1.
    assert old_schedule == schedule.state_dict()
