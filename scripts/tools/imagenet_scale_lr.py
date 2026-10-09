"""Scale a saved zero-floor cosine amplitude without resetting trained Adam."""
import copy
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule


def scale_saved_learning_rate(optimizer_state, scheduler_state, factor):
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('LR factor must be finite and positive')
    if scheduler_state.get('kind') != 'fid-adaptive-cosine-v1':
        raise ValueError('An adaptive cosine state is required')
    policy = scheduler_state['policy']
    if policy['min_lr'] != 0.:
        raise ValueError('Amplitude scaling requires the saved zero floor')
    step = scheduler_state['last_epoch']
    if not 0 <= step <= policy['total_steps']:
        raise ValueError('Invalid saved scheduler cursor')
    previous = FidAdaptiveSchedule.lr_at_step(policy, step, scheduler_state['multiplier'])
    groups = optimizer_state['param_groups']
    if not groups or previous <= 0 or any(not math.isclose(float(g['lr']), previous,
            rel_tol=1e-12, abs_tol=0.) for g in groups):
        raise ValueError('A positive optimizer LR matching the saved scheduler is required')
    revised = copy.deepcopy(scheduler_state)
    revised['multiplier'] *= factor
    new_lr = FidAdaptiveSchedule.lr_at_step(policy, step, revised['multiplier'])
    if not math.isfinite(new_lr) or new_lr <= 0:
        raise ValueError('Scaled LR must remain finite and positive')
    assert math.isclose(new_lr, previous * factor, rel_tol=1e-12, abs_tol=0.)
    optimizer = dict(optimizer_state,
                     param_groups=[dict(group, lr=new_lr) for group in groups])
    return optimizer, revised
