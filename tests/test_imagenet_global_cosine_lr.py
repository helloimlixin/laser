import copy
import math

import pytest
import torch

from scripts.tools.imagenet_global_cosine_lr import (
    GlobalCosineSchedule, create_scheduler, migrate_global_cosine, peak_for_resume_lr,
)


def trained_source():
    parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
    optimizer = torch.optim.AdamW([parameter], lr=7.452648490495602e-6)
    for _ in range(3):
        parameter.square().sum().backward(); optimizer.step(); optimizer.zero_grad()
    return optimizer.state_dict(), {'kind':'source-custom', 'last_epoch':13146}


def migrated(peak=1e-5):
    optimizer, source = trained_source()
    updated, state = migrate_global_cosine(optimizer, source, global_step=48202,
        steps_per_epoch=626, peak_lr=peak, baseline_fid=15.24941539209243)
    live = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=updated['param_groups'][0]['lr'])
    live.load_state_dict(updated)
    schedule = create_scheduler(live, initial_lr=peak, min_lr=0., total_steps=62600,
        completed_steps=48202, state_dict=state)
    return optimizer, updated, live, schedule, source


def test_epoch77_rate_and_original_published_peak():
    _, _, live, schedule, _ = migrated()
    assert live.param_groups[0]['lr'] == pytest.approx(1.2494446518477022e-6)
    assert schedule.last_epoch == 77 * 626
    assert schedule.lr_at_step(schedule.policy, 78 * 626) == pytest.approx(1.1474337861210543e-6)
    _, _, original, _, _ = migrated(5e-4)
    assert original.param_groups[0]['lr'] == pytest.approx(6.24722325923851e-5)


def test_migration_preserves_all_trained_adam_tensors_and_source_history():
    old, updated, _, schedule, source = migrated()
    assert updated['state'] is old['state']
    for key, fields in old['state'].items():
        assert all(updated['state'][key][field] is value for field, value in fields.items())
    assert old['param_groups'][0]['lr'] == 7.452648490495602e-6
    assert schedule.source_controller == source and schedule.last_epoch == 48202


def test_curve_matches_pytorch_original_cosine_and_reaches_exact_zero():
    _, _, live, schedule, _ = migrated()
    # This PyTorch release keeps the loaded current LR at a nonzero initial
    # cursor; seed it exactly as a complete optimizer checkpoint would.
    comparison = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))],
                                 lr=live.param_groups[0]['lr'])
    comparison.param_groups[0]['initial_lr'] = 1e-5
    original = torch.optim.lr_scheduler.CosineAnnealingLR(comparison,
        T_max=62600, eta_min=0., last_epoch=48201)
    for _ in range(626):
        live.step(); schedule.step(); comparison.step(); original.step()
        assert live.param_groups[0]['lr'] == pytest.approx(comparison.param_groups[0]['lr'], rel=1e-11)
    assert schedule.lr_at_step(schedule.policy, 62600) == 0.
    assert schedule.lr_at_step(schedule.policy, 0) == 1e-5


def test_reload_preserves_next_update_and_fid_cannot_reduce_the_lr():
    _, _, live, schedule, _ = migrated()
    for _ in range(20): live.step(); schedule.step()
    saved = copy.deepcopy(schedule.state_dict())
    replacement = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=live.param_groups[0]['lr'])
    resumed = create_scheduler(replacement, initial_lr=1e-5, min_lr=0., total_steps=62600,
        completed_steps=48222, state_dict=saved)
    assert resumed.state_dict() == saved
    old_lr = replacement.param_groups[0]['lr']
    decision = resumed.observe(50.)
    assert decision['decision'] == 'monitor' and replacement.param_groups[0]['lr'] == old_lr
    assert resumed.multiplier == 1. and resumed.reductions == 0
    live.step(); schedule.step(); replacement.step(); resumed.step()
    assert replacement.param_groups[0]['lr'] == live.param_groups[0]['lr']


def test_wrong_optimizer_lr_and_reset_cursor_are_rejected():
    _, _, live, schedule, _ = migrated()
    state = schedule.state_dict()
    live.param_groups[0]['lr'] *= 10
    with pytest.raises(ValueError):GlobalCosineSchedule(live,state)
    with pytest.raises(ValueError):create_scheduler(live,initial_lr=1e-5,min_lr=0.,
        total_steps=62600,completed_steps=0,state_dict=state)


def test_requested_lower_resume_rate_decays_through_all_remaining_epochs():
    peak = peak_for_resume_lr(1e-6, global_step=48202, total_steps=62600)
    _, _, live, schedule, _ = migrated(peak)
    assert live.param_groups[0]['lr'] == pytest.approx(1e-6, rel=1e-12)
    previous = live.param_groups[0]['lr']
    for step in range(48203, 62601):
        live.step(); schedule.step()
        rate = live.param_groups[0]['lr']
        assert 0 <= rate <= previous
        if step % 626 == 0:
            schedule.observe(15.5)
            assert schedule.multiplier == 1. and schedule.reductions == 0
        previous = rate
    assert schedule.last_epoch == 62600 and previous == 0.
    with pytest.raises(ValueError):schedule.step()


@pytest.mark.parametrize('rate,step', [(0.,48202),(-1e-6,48202),(float('nan'),48202),(1e-6,62600)])
def test_invalid_requested_resume_rate_is_rejected(rate, step):
    with pytest.raises(ValueError):peak_for_resume_lr(rate,global_step=step,total_steps=62600)
