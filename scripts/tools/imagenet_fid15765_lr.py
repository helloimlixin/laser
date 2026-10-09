"""Resume-checked cosine LR with a two-evaluation official FID patience."""
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule

BASELINE_FID = 15.764706963349738
ACTIVE_SCHEDULE = None


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    global ACTIVE_SCHEDULE
    policy = dict(initial_lr=float(initial_lr), min_lr=float(min_lr),
                  total_steps=int(total_steps), baseline_fid=BASELINE_FID,
                  patience=2, min_delta=.05, factor=.5, cooldown=1)
    if state_dict is None:
        if completed_steps:
            raise ValueError('An in-progress continuation requires scheduler state')
    else:
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
