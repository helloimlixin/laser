"""Keep LR steady while generation improves; reduce after repeated FID plateaus."""
import math


class FIDPlateauLR:
    kind = 'fid-plateau-lr-v1'

    def __init__(self, optimizer, *, policy, completed_steps, state_dict, revision_from=None):
        self.optimizer = optimizer
        self.policy = dict(policy)
        p = self.policy
        self.last_epoch = int(completed_steps)
        if not (0 <= p['anchor_step'] < p['total_steps']
                and 0 < p['factor'] < 1 and p['patience'] >= 2
                and 0 < p['relative_threshold'] < .1 and p['cooldown_evaluations'] >= 0
                and 0 <= p['min_lr'] <= min(p['anchor_lr'], p['hold_lr'])
                and max(p['anchor_lr'], p['hold_lr']) <= p['initial_lr']
                and p['ramp_steps'] > 0 and p['anchor_step'] + p['ramp_steps'] <= p['total_steps']
                and p['initial_lr'] > 0 and p['initial_best_fid'] > 0
                and 0 <= p['initial_last_evaluation_step'] <= p['anchor_step']
                and all(math.isfinite(p[k]) for k in ('factor','relative_threshold','min_lr','anchor_lr','hold_lr','initial_lr','initial_best_fid'))):
            raise ValueError('Invalid FID plateau LR policy')
        if not p['anchor_step'] <= self.last_epoch <= p['total_steps']:
            raise ValueError('Checkpoint step outside FID plateau policy')
        if state_dict is None or state_dict.get('last_epoch') != self.last_epoch:
            raise ValueError('Scheduler/checkpoint step mismatch')
        if state_dict.get('kind') == self.kind:
            revising = state_dict['policy'] != p
            if revising:
                preserved = ('initial_lr', 'min_lr', 'total_steps', 'factor', 'patience',
                             'relative_threshold', 'cooldown_evaluations')
                if (revision_from != state_dict['policy'] or self.last_epoch != p['anchor_step']
                        or abs(state_dict['current_lr']-p['anchor_lr']) > 1e-12
                        or p['initial_best_fid'] != state_dict['best_fid']
                        or p['initial_last_evaluation_step'] != state_dict['last_evaluation_step']
                        or any(p[k] != revision_from[k] for k in preserved)):
                    raise ValueError('FID plateau policy changed without a verified revision')
                # Validate the complete previous scheduler before accepting the
                # new ramp; Adam moments and the FID decision history stay intact.
                FIDPlateauLR(optimizer, policy=revision_from,
                             completed_steps=completed_steps, state_dict=state_dict)
            self.current_lr = float(state_dict['current_lr'])
            self.best_fid = float(state_dict['best_fid'])
            self.bad_evaluations = int(state_dict['bad_evaluations'])
            self.cooldown_remaining = int(state_dict['cooldown_remaining'])
            self.reductions = int(state_dict['reductions'])
            self.last_evaluation_step = int(state_dict['last_evaluation_step'])
            self.ramp_active = True if revising else state_dict.get('ramp_active', self.reductions == 0)
            if not (p['min_lr'] <= self.current_lr <= max(p['anchor_lr'], p['hold_lr']) and math.isfinite(self.current_lr)
                    and 0 < self.best_fid <= p['initial_best_fid'] and math.isfinite(self.best_fid)
                    and 0 <= self.bad_evaluations < p['patience']
                    and 0 <= self.cooldown_remaining <= p['cooldown_evaluations']
                    and self.reductions >= 0 and 0 <= self.last_evaluation_step <= self.last_epoch):
                raise ValueError('Invalid saved FID plateau state')
            if self.ramp_active and abs(self.current_lr-self.ramp_lr(self.last_epoch)) > 1e-12:
                raise ValueError('Saved warm ramp LR does not match data progress')
        else:
            if (self.last_epoch != p['anchor_step'] or state_dict.get('kind') is not None
                    or state_dict.get('T_max') != p['total_steps']
                    or state_dict.get('eta_min') != p['min_lr']
                    or len(state_dict.get('base_lrs', [])) != len(optimizer.param_groups)
                    or any(abs(x-p['initial_lr'])>1e-12 for x in state_dict['base_lrs'])):
                raise ValueError('Migration requires the verified original cosine checkpoint')
            original = p['min_lr'] + .5*(p['initial_lr']-p['min_lr'])*(1+math.cos(math.pi*self.last_epoch/p['total_steps']))
            if abs(original-p['anchor_lr'])>1e-12:
                raise ValueError('Anchor LR does not match source cosine schedule')
            self.current_lr = p['anchor_lr']
            self.best_fid = p['initial_best_fid']
            self.bad_evaluations = self.cooldown_remaining = self.reductions = 0
            self.last_evaluation_step = p['initial_last_evaluation_step']
            self.ramp_active = True
        if (len(state_dict.get('_last_lr', [])) != len(optimizer.param_groups)
                or any(abs(x-self.current_lr)>1e-12 for x in state_dict['_last_lr'])
                or any(abs(g['lr']-self.current_lr)>1e-12 for g in optimizer.param_groups)):
            raise ValueError('Optimizer/scheduler LR mismatch at resume')
        self.base_lrs = [p['initial_lr']] * len(optimizer.param_groups)
        self._apply()

    def _apply(self):
        for group in self.optimizer.param_groups:
            group['lr'] = self.current_lr
            group['initial_lr'] = self.policy['initial_lr']
        self._last_lr = [self.current_lr] * len(self.optimizer.param_groups)

    def ramp_lr(self,step):
        p = self.policy
        fraction = min(1.,(step-p['anchor_step'])/p['ramp_steps'])
        return p['anchor_lr']+(p['hold_lr']-p['anchor_lr'])*fraction

    def step(self):
        if self.last_epoch >= self.policy['total_steps']:
            raise ValueError('FID plateau training schedule exhausted')
        self.last_epoch += 1
        if self.ramp_active:
            self.current_lr = self.ramp_lr(self.last_epoch)
            self._apply()

    def observe_fid(self, fid):
        if not math.isfinite(fid) or fid < 0:
            raise ValueError('FID observation must be finite and nonnegative')
        if self.last_epoch == self.last_evaluation_step:
            return dict(action='duplicate-evaluation-ignored', fid=fid, lr=self.current_lr)
        self.last_evaluation_step = self.last_epoch
        significant = fid < self.best_fid * (1-self.policy['relative_threshold'])
        action = 'hold'
        if significant:
            self.best_fid = float(fid)
            self.bad_evaluations = 0
        elif self.cooldown_remaining == 0:
            self.bad_evaluations += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.bad_evaluations = 0
        elif self.bad_evaluations >= self.policy['patience']:
            lowered = max(self.policy['min_lr'],self.current_lr*self.policy['factor'])
            if lowered < self.current_lr:
                self.current_lr = lowered
                self.reductions += 1
                self.ramp_active = False
                action = 'reduce'
            self.bad_evaluations = 0
            self.cooldown_remaining = self.policy['cooldown_evaluations']
        self._apply()
        return dict(action=action,fid=fid,best_fid=self.best_fid,bad_evaluations=self.bad_evaluations,
                    cooldown_remaining=self.cooldown_remaining,reductions=self.reductions,lr=self.current_lr)

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return dict(kind=self.kind,policy=dict(self.policy),last_epoch=self.last_epoch,
            current_lr=self.current_lr,best_fid=self.best_fid,bad_evaluations=self.bad_evaluations,
            ramp_active=self.ramp_active,
            cooldown_remaining=self.cooldown_remaining,reductions=self.reductions,
            last_evaluation_step=self.last_evaluation_step,base_lrs=list(self.base_lrs),_last_lr=list(self._last_lr))
