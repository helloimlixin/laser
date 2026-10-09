import copy
import pytest
import torch
from scripts.tools.imagenet_fid17_lr import create_scheduler


def setup():
    parameter = torch.nn.Parameter(torch.tensor([.25, -.5]))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4, betas=(.9, .95), weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, initial_lr=3e-4, min_lr=3e-7, total_steps=28170)
    return parameter, optimizer, scheduler


def test_epoch55_restart_has_bounded_warmup_and_decay():
    _, optimizer, scheduler = setup()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(3e-5)
    assert scheduler.lr_at(626) == pytest.approx(3e-4)
    assert scheduler.lr_at(28170) == pytest.approx(3e-7)
    rates = [scheduler.lr_at(step) for step in range(28171)]
    assert all(a <= b for a, b in zip(rates[:626], rates[1:627]))
    assert all(a >= b for a, b in zip(rates[626:], rates[627:]))


def test_reload_retains_adam_moments_and_identical_next_update():
    parameter, optimizer, scheduler = setup()
    for _ in range(17):
        parameter.grad = torch.tensor([.125, -.75])
        optimizer.step()
        scheduler.step()
    state = copy.deepcopy(optimizer.state_dict())
    schedule_state = copy.deepcopy(scheduler.state_dict())
    restored = torch.nn.Parameter(parameter.detach().clone())
    restored_optimizer = torch.optim.AdamW([restored], lr=1e-3)
    restored_optimizer.load_state_dict(state)
    restored_schedule = create_scheduler(restored_optimizer, initial_lr=3e-4, min_lr=3e-7,
        total_steps=28170, completed_steps=17, state_dict=schedule_state)
    for tensor, opt, sched in ((parameter, optimizer, scheduler),
                              (restored, restored_optimizer, restored_schedule)):
        tensor.grad = torch.tensor([-.25, .375])
        opt.step()
        sched.step()
    assert torch.equal(parameter, restored)
    assert optimizer.param_groups[0]['lr'] == restored_optimizer.param_groups[0]['lr']
    for key, tensor in optimizer.state[parameter].items():
        assert torch.equal(tensor, restored_optimizer.state[restored][key])


def test_inconsistent_resume_lr_and_changed_schedule_are_rejected():
    _, optimizer, scheduler = setup()
    state = copy.deepcopy(scheduler.state_dict())
    optimizer.param_groups[0]['lr'] = 4.5e-4
    with pytest.raises(ValueError, match='Saved optimizer LR'):
        create_scheduler(optimizer, initial_lr=3e-4, min_lr=3e-7,
                         total_steps=28170, state_dict=state)
    assert optimizer.param_groups[0]['lr'] == 4.5e-4
    with pytest.raises(ValueError, match='configuration changed'):
        create_scheduler(optimizer, initial_lr=4.5e-4, min_lr=3e-7,
                         total_steps=28170, state_dict=state)
    assert optimizer.param_groups[0]['lr'] == 4.5e-4
