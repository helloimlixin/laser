"""Warm up a trained continuation without resetting its optimizer clock."""
import copy
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule

ACTIVE_SCHEDULE = None
KIND = 'continuation-warmup-cosine-v1'


class ContinuationWarmupSchedule(FidAdaptiveSchedule):
    def __init__(self, optimizer, state):
        if state.get('kind') != KIND:
            raise ValueError('Expected a saved continuation warmup schedule')
        self.optimizer = optimizer
        self.policy = copy.deepcopy(state['policy'])
        self.source_controller = copy.deepcopy(state['source_controller'])
        p = self.policy
        start, duration = p['warmup_start_step'], p['warmup_steps']
        if not (0 <= start < p['total_steps'] and duration > 0
                and start + duration < p['total_steps']
                and 0 < p['warmup_initial_lr'] <= p['initial_lr']
                and p['min_lr'] == 0. and p['adaptive_reductions'] is False
                and p['decay_start_step'] == start + duration
                and p['decay_steps'] == p['total_steps'] - start - duration):
            raise ValueError('Invalid continuation warmup interval')
        for name in self._fields():
            setattr(self, name, copy.deepcopy(state[name]))
        if not start <= self.last_epoch <= p['total_steps']:
            raise ValueError('Scheduler cursor is outside the continuation')
        expected = self.lr_at_step(p, self.last_epoch, self.multiplier)
        if any(not math.isclose(g['lr'], expected, rel_tol=1e-12, abs_tol=1e-20)
               for g in optimizer.param_groups):
            raise ValueError('Optimizer LR differs from its saved scheduler')
        self._apply()

    @staticmethod
    def lr_at_step(policy, step, multiplier=1.):
        if multiplier != 1.:
            raise ValueError('The sustained trial does not use adaptive LR reductions')
        position = step - policy['warmup_start_step']
        if position < 0:
            raise ValueError('Step precedes the continuation')
        if position < policy['warmup_steps']:
            fraction = position / policy['warmup_steps']
            return policy['warmup_initial_lr'] + fraction * (
                policy['initial_lr'] - policy['warmup_initial_lr'])
        position = min(step - policy['decay_start_step'], policy['decay_steps'])
        return policy['initial_lr'] * .5 * (
            1. + math.cos(math.pi * position / policy['decay_steps']))

    def state_dict(self):
        result = super().state_dict()
        result['kind'] = KIND
        result['source_controller'] = copy.deepcopy(self.source_controller)
        return result


def migrate_continuation(optimizer_state, scheduler_state, *, peak_lr=1e-5,
                         start_lr=3e-7, warmup_steps=200):
    if (scheduler_state.get('kind') != 'fid-adaptive-cosine-v1'
            or scheduler_state['policy']['min_lr'] != 0.):
        raise ValueError('Migration requires the trained zero-floor cosine state')
    old_lr = FidAdaptiveSchedule.lr_at_step(scheduler_state['policy'],
        scheduler_state['last_epoch'], scheduler_state['multiplier'])
    if any(not math.isclose(g['lr'], old_lr, rel_tol=1e-12, abs_tol=1e-20)
           for g in optimizer_state['param_groups']):
        raise ValueError('Source optimizer LR differs from its scheduler')
    if not (math.isfinite(peak_lr) and math.isfinite(start_lr)
            and 0 < start_lr <= peak_lr and warmup_steps > 0):
        raise ValueError('Invalid warmup rates or duration')
    state = copy.deepcopy(scheduler_state)
    start, end = state['last_epoch'], state['policy']['total_steps']
    if start + warmup_steps >= end:
        raise ValueError('Warmup must leave a cosine decay interval')
    state['kind'] = KIND
    state['multiplier'] = 1.
    state['source_controller'] = copy.deepcopy(scheduler_state)
    p = state['policy']
    p.pop('warmup_steps', None)
    p.update(initial_lr=float(peak_lr), adaptive_reductions=False,
        warmup_start_step=start, warmup_steps=int(warmup_steps),
        warmup_initial_lr=float(start_lr), decay_start_step=start + warmup_steps,
        decay_steps=end - start - warmup_steps)
    optimizer = dict(optimizer_state, param_groups=[dict(g, lr=start_lr, initial_lr=peak_lr)
                                                   for g in optimizer_state['param_groups']])
    return optimizer, state


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    global ACTIVE_SCHEDULE
    if state_dict is None or state_dict.get('kind') != KIND:
        raise ValueError('A saved continuation schedule is required')
    p = state_dict['policy']
    if (p['initial_lr'], p['min_lr'], p['total_steps'], state_dict['last_epoch']) != (
            initial_lr, min_lr, total_steps, completed_steps):
        raise ValueError('Continuation scheduler configuration or clock changed')
    ACTIVE_SCHEDULE = ContinuationWarmupSchedule(optimizer, state_dict)
    return ACTIVE_SCHEDULE


def scale_saved_continuation(optimizer_state, scheduler_state, factor):
    """Scale a trained cosine's amplitude, retaining Adam and its absolute clock."""
    if scheduler_state.get('kind') != KIND or not math.isfinite(factor) or factor <= 0:
        raise ValueError('Expected a saved continuation and a finite positive LR factor')
    expected = ContinuationWarmupSchedule.lr_at_step(scheduler_state['policy'],
        scheduler_state['last_epoch'], scheduler_state['multiplier'])
    if any(not math.isclose(g['lr'], expected, rel_tol=1e-12, abs_tol=1e-20)
           for g in optimizer_state['param_groups']):
        raise ValueError('Source optimizer LR differs from its saved continuation')
    state = copy.deepcopy(scheduler_state)
    for key in ('initial_lr', 'warmup_initial_lr'):
        state['policy'][key] *= factor
        if not math.isfinite(state['policy'][key]):
            raise ValueError('Scaled LR is not finite')
    current = ContinuationWarmupSchedule.lr_at_step(state['policy'], state['last_epoch'])
    optimizer = dict(optimizer_state, param_groups=[
        dict(g, lr=current, initial_lr=state['policy']['initial_lr'])
        for g in optimizer_state['param_groups']])
    return optimizer, state


def observe_official_fid(fid):
    if ACTIVE_SCHEDULE is None:
        raise RuntimeError('Official FID arrived before scheduler initialization')
    return ACTIVE_SCHEDULE.observe(fid)
