"""A six-epoch continuation cosine, preserving the trained Adam and clock."""
import copy
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule

BASELINE_FID = 15.764706963349738
DECAY_START_STEP = 5008
DECAY_STEPS = 6 * 626
ACTIVE_SCHEDULE = None


def policy_for(initial_lr, min_lr, total_steps):
    return dict(initial_lr=float(initial_lr), min_lr=float(min_lr),
                total_steps=int(total_steps), baseline_fid=BASELINE_FID,
                patience=1, min_delta=.02, factor=.5, cooldown=0,
                decay_start_step=DECAY_START_STEP, decay_steps=DECAY_STEPS)


def revise_continuation_state(optimizer_state, scheduler_state):
    """Cut source amplitude eightfold and finish the decay six epochs later."""
    if scheduler_state.get('kind') != 'fid-adaptive-cosine-v1':
        raise ValueError('A saved adaptive cosine schedule is required')
    old_policy = scheduler_state['policy']
    step = scheduler_state['last_epoch']
    if step != DECAY_START_STEP or step != scheduler_state['last_observation_step']:
        raise ValueError('Revision requires the scored epoch64 checkpoint')
    old_lr = FidAdaptiveSchedule.lr_at_step(old_policy, step, scheduler_state['multiplier'])
    if any(not math.isclose(float(g['lr']), old_lr, rel_tol=1e-12, abs_tol=1e-15)
           for g in optimizer_state['param_groups']):
        raise ValueError('Source optimizer LR disagrees with its scheduler')
    state = copy.deepcopy(scheduler_state)
    policy = policy_for(old_policy['initial_lr'], old_policy['min_lr'], old_policy['total_steps'])
    reduced_lr = policy['min_lr'] + (old_lr - policy['min_lr']) / 8
    multiplier = (reduced_lr - policy['min_lr']) / (policy['initial_lr'] - policy['min_lr'])
    state.update(policy=policy, multiplier=multiplier, bad_epochs=0,
                 cooldown_remaining=0, reductions=state['reductions'] + 1)
    optimizer = dict(optimizer_state, param_groups=[dict(g, lr=reduced_lr)
                                                    for g in optimizer_state['param_groups']])
    return optimizer, state


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    global ACTIVE_SCHEDULE
    policy = policy_for(initial_lr, min_lr, total_steps)
    if state_dict is None:
        raise ValueError('The aggressive continuation requires a full saved scheduler')
    if state_dict.get('kind') != 'fid-adaptive-cosine-v1' or state_dict.get('policy') != policy:
        raise ValueError('Scheduler configuration changed on resume')
    if state_dict['last_epoch'] != completed_steps or not 0 <= completed_steps <= total_steps:
        raise ValueError('Scheduler cursor does not match checkpoint progress')
    expected = FidAdaptiveSchedule.lr_at_step(policy, completed_steps, state_dict['multiplier'])
    if any(not math.isclose(float(g['lr']), expected, rel_tol=1e-12, abs_tol=1e-15)
           for g in optimizer.param_groups):
        raise ValueError('Saved optimizer LR does not match its scheduler')
    ACTIVE_SCHEDULE = FidAdaptiveSchedule(optimizer, **policy,
        completed_steps=completed_steps, state_dict=state_dict)
    return ACTIVE_SCHEDULE


def observe_official_fid(fid):
    if ACTIVE_SCHEDULE is None:
        raise RuntimeError('Official FID arrived before scheduler initialization')
    return ACTIVE_SCHEDULE.observe(fid)
