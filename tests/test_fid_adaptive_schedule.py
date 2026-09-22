import copy
from types import SimpleNamespace

import pytest

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule


def make(**kwargs):
    optimizer = SimpleNamespace(param_groups=[{'lr': 99.}])
    return FidAdaptiveSchedule(optimizer, initial_lr=3e-5, min_lr=1e-6,
                               total_steps=30, baseline_fid=10.688463, **kwargs)


def test_plateau_reduces_and_resume_replays_identically():
    schedule = make()
    for _ in range(2):
        schedule.step()
        assert schedule.observe(10.7)['decision'] == 'watch'
    state = copy.deepcopy(schedule.state_dict())
    restored = make(completed_steps=2, state_dict=state)
    for fid in [10.71, 10.6, 10.59, 10.58, 10.7, 10.72, 10.73]:
        schedule.step()
        restored.step()
        assert schedule.observe(fid) == restored.observe(fid)
        assert schedule.state_dict() == restored.state_dict()
        assert schedule.optimizer.param_groups == restored.optimizer.param_groups
    assert schedule.reductions >= 1


def test_improvement_resets_patience_and_duplicate_is_idempotent():
    schedule = make()
    for fid in [10.8, 10.8, 10.5]:
        schedule.step()
        event = schedule.observe(fid)
    assert event['decision'] == 'improved'
    assert schedule.bad_epochs == schedule.reductions == 0
    assert schedule.observe(10.5) is None
    with pytest.raises(ValueError, match='new optimizer'):
        schedule.observe(10.51)


def test_monotonic_curve_floor_and_schedule_contract():
    schedule = make()
    lrs = [schedule.optimizer.param_groups[0]['lr']]
    for _ in range(30):
        schedule.step()
        schedule.observe(10.8)
        lrs.append(schedule.optimizer.param_groups[0]['lr'])
    assert lrs[0] == 3e-5 and lrs[-1] == 1e-6
    assert all(a >= b >= 1e-6 for a, b in zip(lrs, lrs[1:]))
    with pytest.raises(ValueError, match='exhausted'):
        schedule.step()
    with pytest.raises(ValueError, match='mismatch'):
        make(completed_steps=2)
    with pytest.raises(ValueError, match='changed'):
        make(state_dict={'kind': 'old-cosine'})
    with pytest.raises(ValueError, match='finite'):
        make().observe(float('nan'))
