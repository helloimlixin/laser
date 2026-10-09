"""A checked, checkpointed LR policy for the trained epoch56 optimizer."""
import copy
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule

BASELINE_FID = 16.378750087355286
ACTIVE_SCHEDULE = None


def halve_continuation_state(optimizer_state, scheduler_state):
    """Reduce the cosine amplitude at a scored checkpoint, preserving Adam."""
    if scheduler_state.get('kind') != 'fid-adaptive-cosine-v1':
        raise ValueError('A saved adaptive cosine schedule is required')
    policy = scheduler_state['policy']
    step = scheduler_state['last_epoch']
    if step != scheduler_state['last_observation_step']:
        raise ValueError('LR revision requires the scored checkpoint step')
    expected = FidAdaptiveSchedule.lr_at_step(policy, step, scheduler_state['multiplier'])
    if any(not math.isclose(float(g['lr']), expected, rel_tol=1e-12, abs_tol=1e-15)
           for g in optimizer_state['param_groups']):
        raise ValueError('Source optimizer LR disagrees with its scheduler')
    state = copy.deepcopy(scheduler_state)
    state.update(multiplier=state['multiplier'] * .5, bad_epochs=0,
                 cooldown_remaining=policy['cooldown'], reductions=state['reductions'] + 1)
    reduced = FidAdaptiveSchedule.lr_at_step(policy, step, state['multiplier'])
    optimizer = dict(optimizer_state, param_groups=[dict(g, lr=reduced)
                                                    for g in optimizer_state['param_groups']])
    return optimizer, state


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    global ACTIVE_SCHEDULE
    policy = dict(initial_lr=float(initial_lr), min_lr=float(min_lr),
                  total_steps=int(total_steps), baseline_fid=BASELINE_FID,
                  patience=3, min_delta=.05, factor=.5, cooldown=1)
    if state_dict is None:
        if completed_steps:
            raise ValueError('An in-progress continuation requires scheduler state')
    else:
        if state_dict.get('kind') != 'fid-adaptive-cosine-v1' or state_dict.get('policy') != policy:
            raise ValueError('Scheduler configuration changed on resume')
        if state_dict['last_epoch'] != completed_steps or not 0 <= completed_steps <= total_steps:
            raise ValueError('Scheduler cursor does not match checkpoint progress')
        expected_lr = min_lr + (initial_lr - min_lr) * .5 * (
            1 + math.cos(math.pi * completed_steps / total_steps)) * state_dict['multiplier']
        if any(not math.isclose(float(g['lr']), expected_lr, rel_tol=1e-12, abs_tol=1e-15)
               for g in optimizer.param_groups):
            raise ValueError('Saved optimizer LR does not match its scheduler')
    ACTIVE_SCHEDULE = FidAdaptiveSchedule(optimizer, **policy,
        completed_steps=completed_steps, state_dict=state_dict)
    return ACTIVE_SCHEDULE


def observe_official_fid(fid):
    if ACTIVE_SCHEDULE is None:
        raise RuntimeError('Official FID arrived before scheduler initialization')
    return ACTIVE_SCHEDULE.observe(fid)
