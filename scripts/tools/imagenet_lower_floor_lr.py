"""Lower the saved cosine floor without resetting Adam or its schedule clock."""
import copy
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule

NEW_FLOOR = 3e-8
BASELINE_FID = 15.764706963349738
ACTIVE_SCHEDULE = None


def lower_floor_state(optimizer_state, scheduler_state, *, min_lr=NEW_FLOOR):
    if scheduler_state.get('kind') != 'fid-adaptive-cosine-v1':
        raise ValueError('A saved adaptive cosine schedule is required')
    old_policy = scheduler_state['policy']
    if not 0 <= min_lr < old_policy['min_lr']:
        raise ValueError('The new floor must be nonnegative and lower')
    step = scheduler_state['last_epoch']
    old_lr = FidAdaptiveSchedule.lr_at_step(old_policy, step, scheduler_state['multiplier'])
    if any(not math.isclose(float(g['lr']), old_lr, rel_tol=1e-12, abs_tol=1e-15)
           for g in optimizer_state['param_groups']):
        raise ValueError('Source optimizer LR disagrees with its scheduler')
    state = copy.deepcopy(scheduler_state)
    state['policy']['min_lr'] = float(min_lr)
    new_lr = FidAdaptiveSchedule.lr_at_step(state['policy'], step, state['multiplier'])
    optimizer = dict(optimizer_state, param_groups=[dict(g, lr=new_lr)
                                                    for g in optimizer_state['param_groups']])
    return optimizer, state


class FloorAwareSchedule(FidAdaptiveSchedule):
    def observe(self, fid):
        multiplier, reductions = self.multiplier, self.reductions
        decision = super().observe(fid)
        if (decision is not None and decision['decision'] == 'reduced'
                and decision['lr_before'] == decision['lr_after']):
            self.multiplier, self.reductions = multiplier, reductions
            decision.update(decision='at_floor', multiplier=multiplier, reductions=reductions)
        return decision


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    global ACTIVE_SCHEDULE
    policy = dict(initial_lr=float(initial_lr), min_lr=float(min_lr),
                  total_steps=int(total_steps), baseline_fid=BASELINE_FID,
                  patience=1, min_delta=.02, factor=.5, cooldown=0,
                  decay_start_step=5008, decay_steps=3756)
    if min_lr != NEW_FLOOR:
        raise ValueError('The continuation requires the requested lower floor')
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
