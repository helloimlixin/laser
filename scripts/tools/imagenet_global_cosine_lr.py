"""Original RQ-Transformer zero-floor cosine, indexed by global data progress."""
import copy
import math

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule

KIND = 'original-rqtransformer-global-cosine-v1'
ACTIVE_SCHEDULE = None


def peak_for_resume_lr(resume_lr, *, global_step, total_steps):
    """Scale the original cosine so its current position has the requested LR."""
    if (not math.isfinite(resume_lr) or resume_lr <= 0 or total_steps <= 0
            or not 0 <= global_step < total_steps):
        raise ValueError('A positive resume LR requires a position before cosine end')
    fraction = .5 * (1 + math.cos(math.pi * global_step / total_steps))
    if fraction <= 0:
        raise ValueError('Cosine position is too close to zero to scale safely')
    return float(resume_lr / fraction)


class GlobalCosineSchedule(FidAdaptiveSchedule):
    def __init__(self, optimizer, state):
        if state.get('kind') != KIND or state.get('clock_origin_global_step') != 0:
            raise ValueError('Expected an absolute global-step RQ cosine state')
        policy = state['policy']
        if policy['min_lr'] != 0 or policy.get('adaptive_reductions') is not False:
            raise ValueError('RQ cosine requires zero floor and no metric-triggered reductions')
        expected = self.lr_at_step(policy, state['last_epoch'])
        if any(not math.isclose(group['lr'], expected, rel_tol=1e-12, abs_tol=1e-20)
               for group in optimizer.param_groups):
            raise ValueError('Optimizer rate does not match its saved global cosine position')
        self.source_controller = copy.deepcopy(state['source_controller'])
        self.steps_per_epoch = int(state['steps_per_epoch'])
        restored = dict(state, kind='fid-adaptive-cosine-v1')
        super().__init__(optimizer, **policy, completed_steps=state['last_epoch'],
                         state_dict=restored)

    def state_dict(self):
        state = super().state_dict()
        state.update(kind=KIND, clock_origin_global_step=0,
                     steps_per_epoch=self.steps_per_epoch,
                     source_controller=copy.deepcopy(self.source_controller))
        return state


def migrate_global_cosine(optimizer_state, scheduler_state, *, global_step,
                          steps_per_epoch, peak_lr=1e-5, schedule_epochs=100,
                          baseline_fid):
    """Change the LR policy only; retain every Adam moment and counter tensor."""
    if (not math.isfinite(peak_lr) or peak_lr <= 0 or steps_per_epoch <= 0
            or schedule_epochs <= 0 or not 0 <= global_step <= steps_per_epoch * schedule_epochs
            or not math.isfinite(baseline_fid) or baseline_fid < 0):
        raise ValueError('Invalid global cosine migration')
    policy = dict(initial_lr=float(peak_lr), min_lr=0.,
                  total_steps=int(steps_per_epoch * schedule_epochs),
                  baseline_fid=float(baseline_fid), patience=1, min_delta=0.,
                  factor=.5, cooldown=0, adaptive_reductions=False)
    state = dict(kind=KIND, policy=policy, last_epoch=int(global_step),
                 multiplier=1., best=float(baseline_fid), bad_epochs=0,
                 cooldown_remaining=0, reductions=0,
                 last_observation_step=int(global_step), last_fid=float(baseline_fid),
                 clock_origin_global_step=0, steps_per_epoch=int(steps_per_epoch),
                 source_controller=copy.deepcopy(scheduler_state))
    lr = GlobalCosineSchedule.lr_at_step(policy, global_step)
    optimizer = dict(optimizer_state, param_groups=[
        dict(group, lr=lr, initial_lr=peak_lr) for group in optimizer_state['param_groups']])
    return optimizer, state


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    global ACTIVE_SCHEDULE
    if state_dict is None:
        raise ValueError('Global cosine continuation requires a saved complete state')
    policy = state_dict['policy']
    if (policy['initial_lr'], policy['min_lr'], policy['total_steps'], state_dict['last_epoch']) != (
            initial_lr, min_lr, total_steps, completed_steps):
        raise ValueError('Global cosine configuration or cursor changed on resume')
    ACTIVE_SCHEDULE = GlobalCosineSchedule(optimizer, state_dict)
    return ACTIVE_SCHEDULE


def observe_official_fid(fid):
    if ACTIVE_SCHEDULE is None:
        raise RuntimeError('Official FID arrived before scheduler initialization')
    return ACTIVE_SCHEDULE.observe(fid)
