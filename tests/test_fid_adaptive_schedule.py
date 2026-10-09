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


def test_continuation_cosine_anneals_at_saved_step_and_stays_at_floor():
    optimizer = SimpleNamespace(param_groups=[{'lr': 99.}])
    policy = dict(initial_lr=5e-5, min_lr=1e-5, total_steps=100,
        baseline_fid=25.14, decay_start_step=20, decay_steps=10)
    initial = FidAdaptiveSchedule(optimizer, **policy)
    state = initial.state_dict()
    state.update(last_epoch=20, last_observation_step=20)
    schedule = FidAdaptiveSchedule(optimizer, **policy, completed_steps=20, state_dict=state)
    assert optimizer.param_groups[0]['lr'] == 5e-5
    for _ in range(5):schedule.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(3e-5)
    clone = SimpleNamespace(param_groups=[{}])
    restored = FidAdaptiveSchedule(clone, **policy, completed_steps=25,
        state_dict=copy.deepcopy(schedule.state_dict()))
    for _ in range(15):
        schedule.step();restored.step()
        assert optimizer.param_groups == clone.param_groups
    assert optimizer.param_groups[0]['lr'] == 1e-5
def test_fresh_warmup_and_fid_reductions_resume_without_a_previous_run_baseline():
    from types import SimpleNamespace
    import copy
    from src.training.fid_adaptive_schedule import FidAdaptiveSchedule
    policy = dict(initial_lr=.0002, min_lr=.00001, total_steps=40,
        baseline_fid=None, patience=3, min_delta=.1, factor=.5, cooldown=2,
        warmup_steps=4, decay_start_step=4, decay_steps=36)
    optimizer = SimpleNamespace(param_groups=[dict(lr=.0002)])
    scheduler = FidAdaptiveSchedule(optimizer, **policy)
    assert optimizer.param_groups[0]['lr'] == .00005
    scheduler.step(); scheduler.observe(195.)
    for _ in range(3): scheduler.step()
    assert optimizer.param_groups[0]['lr'] == .0002
    scheduler.observe(80.)
    for fid in (40., 41.):
        scheduler.step(); scheduler.observe(fid)
    resumed_optimizer = SimpleNamespace(param_groups=copy.deepcopy(optimizer.param_groups))
    resumed = FidAdaptiveSchedule(resumed_optimizer, **policy,
        completed_steps=scheduler.last_epoch, state_dict=scheduler.state_dict())
    for fid in (42., 41., 40., 39., 38.):
        scheduler.step(); resumed.step()
        assert scheduler.observe(fid) == resumed.observe(fid)
        assert scheduler.state_dict() == resumed.state_dict()
    assert scheduler.reductions == 1
