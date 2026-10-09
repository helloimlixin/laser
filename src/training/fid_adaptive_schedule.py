"""Resume-safe cosine decay with additional reductions driven by matched FID."""
import copy
import math


class FidAdaptiveSchedule:
    def __init__(self, optimizer, *, initial_lr, min_lr, total_steps,
                 baseline_fid, patience=3, min_delta=.02, factor=.5, cooldown=1,
                 completed_steps=0, state_dict=None, decay_start_step=None, decay_steps=None,
                 warmup_steps=0, adaptive_reductions=True):
        if not (0 <= min_lr <= initial_lr and initial_lr > 0 and total_steps > 0
                and patience >= 1 and min_delta >= 0 and 0 < factor < 1
                and cooldown >= 0 and 0 <= warmup_steps < total_steps
                and isinstance(adaptive_reductions, bool)
                and (baseline_fid is None or math.isfinite(baseline_fid))):
            raise ValueError('Invalid adaptive FID schedule')
        self.optimizer = optimizer
        self.policy = dict(initial_lr=initial_lr, min_lr=min_lr,
                           total_steps=total_steps, baseline_fid=baseline_fid,
                           patience=patience, min_delta=min_delta,
                           factor=factor, cooldown=cooldown)
        # Omit the default to retain compatibility with existing full states.
        if not adaptive_reductions:
            self.policy['adaptive_reductions'] = False
        if decay_start_step is not None or decay_steps is not None:
            if (decay_start_step is None or decay_steps is None
                    or not 0 <= decay_start_step < total_steps
                    or not 0 < decay_steps <= total_steps - decay_start_step):
                raise ValueError('Invalid continuation cosine interval')
            self.policy.update(decay_start_step=decay_start_step, decay_steps=decay_steps)
        if warmup_steps:
            if decay_start_step != warmup_steps or decay_steps != total_steps - warmup_steps:
                raise ValueError('Warmup must end at the start of the fresh cosine interval')
            self.policy['warmup_steps'] = warmup_steps
        self.last_epoch = 0
        self.multiplier = 1.
        self.best = math.inf if baseline_fid is None else float(baseline_fid)
        self.bad_epochs = 0
        self.cooldown_remaining = 0
        self.reductions = 0
        self.last_observation_step = 0
        self.last_fid = None if baseline_fid is None else float(baseline_fid)
        if state_dict is not None:
            if state_dict.get('kind') != 'fid-adaptive-cosine-v1' or state_dict['policy'] != self.policy:
                raise ValueError('Adaptive FID schedule changed on resume')
            for name in self._fields():
                setattr(self, name, copy.deepcopy(state_dict[name]))
        if self.last_epoch != completed_steps or not 0 <= completed_steps <= total_steps:
            raise ValueError('Adaptive scheduler/checkpoint step mismatch')
        self._apply()

    @staticmethod
    def _fields():
        return ('last_epoch', 'multiplier', 'best', 'bad_epochs',
                'cooldown_remaining', 'reductions', 'last_observation_step', 'last_fid')

    @staticmethod
    def lr_at_step(policy, step, multiplier=1.):
        warmup = policy.get('warmup_steps', 0)
        if step < warmup:
            return policy['initial_lr'] * (step + 1) / warmup * multiplier
        start = policy.get('decay_start_step', 0)
        duration = policy.get('decay_steps', policy['total_steps'])
        position = min(max(step - start, 0), duration)
        cosine = .5 * (1 + math.cos(math.pi * position / duration))
        return policy['min_lr'] + (policy['initial_lr'] - policy['min_lr']) * cosine * multiplier

    def _apply(self):
        p = self.policy
        lr = self.lr_at_step(p, self.last_epoch, self.multiplier)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
            group['initial_lr'] = p['initial_lr']
        return lr

    def step(self):
        if self.last_epoch >= self.policy['total_steps']:
            raise ValueError('Adaptive schedule exhausted')
        self.last_epoch += 1
        self._apply()

    def observe(self, fid):
        fid = float(fid)
        if not math.isfinite(fid) or fid < 0:
            raise ValueError('FID must be finite and nonnegative')
        if self.last_epoch <= self.last_observation_step:
            if self.last_epoch == self.last_observation_step and fid == self.last_fid:
                return None
            raise ValueError('FID observation must follow new optimizer updates')
        before = self.optimizer.param_groups[0]['lr']
        improved = fid < self.best - self.policy['min_delta']
        if improved:
            self.best = fid
        if self.cooldown_remaining:
            self.cooldown_remaining -= 1
            self.bad_epochs = 0
            decision = 'cooldown'
        elif improved:
            self.bad_epochs = 0
            decision = 'improved'
        else:
            self.bad_epochs += 1
            adaptive = self.policy.get('adaptive_reductions', True)
            decision = 'watch' if adaptive else 'monitor'
            if adaptive and self.bad_epochs >= self.policy['patience']:
                self.multiplier *= self.policy['factor']
                self.bad_epochs = 0
                self.cooldown_remaining = self.policy['cooldown']
                self.reductions += 1
                decision = 'reduced'
        self.last_observation_step = self.last_epoch
        self.last_fid = fid
        return dict(fid=fid, schedule_step=self.last_epoch, decision=decision,
                    lr_before=before, lr_after=self._apply(),
                    multiplier=self.multiplier, reductions=self.reductions,
                    bad_epochs=self.bad_epochs, controller_best=self.best)

    def state_dict(self):
        return dict(kind='fid-adaptive-cosine-v1', policy=copy.deepcopy(self.policy),
                    **{name: copy.deepcopy(getattr(self, name)) for name in self._fields()})
