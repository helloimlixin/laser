"""Resume-safe cosine decay with additional reductions driven by matched FID."""
import copy
import math


class FidAdaptiveSchedule:
    def __init__(self, optimizer, *, initial_lr, min_lr, total_steps,
                 baseline_fid, patience=3, min_delta=.02, factor=.5, cooldown=1,
                 completed_steps=0, state_dict=None):
        if not (0 < min_lr <= initial_lr and total_steps > 0
                and patience >= 1 and min_delta >= 0 and 0 < factor < 1
                and cooldown >= 0 and math.isfinite(baseline_fid)):
            raise ValueError('Invalid adaptive FID schedule')
        self.optimizer = optimizer
        self.policy = dict(initial_lr=initial_lr, min_lr=min_lr,
                           total_steps=total_steps, baseline_fid=baseline_fid,
                           patience=patience, min_delta=min_delta,
                           factor=factor, cooldown=cooldown)
        self.last_epoch = 0
        self.multiplier = 1.
        self.best = float(baseline_fid)
        self.bad_epochs = 0
        self.cooldown_remaining = 0
        self.reductions = 0
        self.last_observation_step = 0
        self.last_fid = float(baseline_fid)
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

    def _apply(self):
        p = self.policy
        cosine = .5 * (1 + math.cos(math.pi * self.last_epoch / p['total_steps']))
        lr = p['min_lr'] + (p['initial_lr'] - p['min_lr']) * cosine * self.multiplier
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
            decision = 'watch'
            if self.bad_epochs >= self.policy['patience']:
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
