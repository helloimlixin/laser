"""Revise only the saved LR amplitude for a full epoch64 zero-floor rewind."""
import copy
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule

# Half the preceding run's actual post-evaluation LR at epoch67.
INITIAL_LR = 1.7674647817084015e-7


def lower_rewind_lr(optimizer_state, scheduler_state, *, initial_lr=INITIAL_LR,
                    scored_epoch64=True, allow_zero=False):
    if (scheduler_state.get('kind') != 'fid-adaptive-cosine-v1'
            or scheduler_state['last_epoch'] != scheduler_state['last_observation_step']
            or (scored_epoch64 and scheduler_state['last_epoch'] != 5008)):
        raise ValueError('A scored adaptive cosine state is required')
    policy = scheduler_state['policy']
    if (policy['min_lr'] != 0. or policy.get('decay_start_step') != 5008
            or policy.get('decay_steps') != policy['total_steps'] - 5008):
        raise ValueError('Preserve the saved zero-floor cosine interval through epoch100')
    previous_lr = FidAdaptiveSchedule.lr_at_step(
        policy, scheduler_state['last_epoch'], scheduler_state['multiplier'])
    groups = optimizer_state['param_groups']
    if not groups or any(not math.isclose(float(g['lr']), previous_lr,
                                         rel_tol=1e-12, abs_tol=0.) for g in groups):
        raise ValueError('Source optimizer LR disagrees with its scheduler')
    if (not math.isfinite(initial_lr)
            or not (0. <= initial_lr if allow_zero else 0. < initial_lr)
            or initial_lr >= previous_lr
            or math.isclose(initial_lr, previous_lr, rel_tol=1e-12, abs_tol=0.)):
        raise ValueError('The starting LR must be finite, positive, and lower')
    state = copy.deepcopy(scheduler_state)
    state['multiplier'] *= initial_lr / previous_lr
    revised_lr = FidAdaptiveSchedule.lr_at_step(
        policy, state['last_epoch'], state['multiplier'])
    assert math.isclose(revised_lr, initial_lr, rel_tol=1e-12, abs_tol=0.)
    # Share all trained moment tensors/counters; copy only group settings.
    optimizer = dict(optimizer_state,
                     param_groups=[dict(group, lr=revised_lr) for group in groups])
    return optimizer, state
