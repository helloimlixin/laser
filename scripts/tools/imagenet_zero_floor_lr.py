"""Zero-floor continuation through epoch100, with the saved trained Adam."""
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule
try:
    from imagenet_lower_floor_lr import BASELINE_FID, FloorAwareSchedule, lower_floor_state
except ModuleNotFoundError:
    from scripts.tools.imagenet_lower_floor_lr import BASELINE_FID, FloorAwareSchedule, lower_floor_state

DECAY_START_STEP = 5008
ACTIVE_SCHEDULE = None


def zero_floor_state(optimizer_state, scheduler_state):
    if (scheduler_state['last_epoch'] != DECAY_START_STEP
            or scheduler_state['last_observation_step'] != DECAY_START_STEP):
        raise ValueError('Zero-floor rewind requires the scored epoch64 checkpoint')
    optimizer, state = lower_floor_state(optimizer_state, scheduler_state, min_lr=0.)
    state['policy']['decay_steps'] = state['policy']['total_steps'] - DECAY_START_STEP
    expected = FidAdaptiveSchedule.lr_at_step(state['policy'], DECAY_START_STEP, state['multiplier'])
    assert all(math.isclose(g['lr'], expected, rel_tol=1e-12, abs_tol=1e-15)
               for g in optimizer['param_groups'])
    return optimizer, state


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    global ACTIVE_SCHEDULE
    policy = dict(initial_lr=float(initial_lr), min_lr=float(min_lr),
                  total_steps=int(total_steps), baseline_fid=BASELINE_FID,
                  patience=1, min_delta=.02, factor=.5, cooldown=0,
                  decay_start_step=DECAY_START_STEP,
                  decay_steps=int(total_steps) - DECAY_START_STEP)
    if min_lr != 0.:
        raise ValueError('The continuation requires the official zero floor')
    if state_dict is None or state_dict.get('kind') != 'fid-adaptive-cosine-v1' or state_dict.get('policy') != policy:
        raise ValueError('Scheduler configuration changed on resume')
    if state_dict['last_epoch'] != completed_steps or not 0 <= completed_steps <= total_steps:
        raise ValueError('Scheduler cursor does not match checkpoint progress')
    expected = FidAdaptiveSchedule.lr_at_step(policy, completed_steps, state_dict['multiplier'])
    if any(not math.isclose(float(g['lr']), expected, rel_tol=1e-12, abs_tol=1e-15)
           for g in optimizer.param_groups):
        raise ValueError('Saved optimizer LR does not match its scheduler')
    ACTIVE_SCHEDULE = FloorAwareSchedule(optimizer, **policy,
        completed_steps=completed_steps, state_dict=state_dict)
    return ACTIVE_SCHEDULE


def observe_official_fid(fid):
    if ACTIVE_SCHEDULE is None:
        raise RuntimeError('Official FID arrived before scheduler initialization')
    return ACTIVE_SCHEDULE.observe(fid)
