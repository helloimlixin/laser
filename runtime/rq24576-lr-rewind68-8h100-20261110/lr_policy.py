"""Preserve the original cosine through the rewind point, then use the trial LR."""
import math


class LearningRateBranch:
    def __init__(self, original, optimizer, policy, global_step, saved_policy=None):
        self.original=original
        self.optimizer=optimizer
        self.policy=dict(policy)
        self.after_scheduler=original.after_scheduler
        assert self.after_scheduler.last_epoch==global_step
        assert self.after_scheduler.T_max==policy['original_horizon_steps']
        if saved_policy is not None:assert saved_policy==self.policy
        self.change_step=round(policy['change_epoch']*policy['steps_per_epoch'])
        self.ramp_end=round((policy['change_epoch']+policy['warmup_epochs'])*policy['steps_per_epoch'])
        self.hold_end=round(policy['hold_until_epoch']*policy['steps_per_epoch'])
        self.end_step=round(policy['end_epoch']*policy['steps_per_epoch'])
        assert self.change_step<self.ramp_end<=self.hold_end<self.end_step
        assert 0<policy['min_lr']<policy['peak_lr']
        if global_step>self.change_step:
            expected=self.value_at_step(global_step)
            assert all(math.isclose(g['lr'],expected,rel_tol=1e-9,abs_tol=1e-13) for g in optimizer.param_groups)

    def value_at_step(self, step):
        p=self.policy
        start=p['original_base_lr']*.5*(1+math.cos(math.pi*self.change_step/p['original_horizon_steps']))
        if step<=self.change_step:
            return p['original_base_lr']*.5*(1+math.cos(math.pi*step/p['original_horizon_steps']))
        if step<self.ramp_end:
            fraction=(step-self.change_step)/(self.ramp_end-self.change_step)
            return start+fraction*(p['peak_lr']-start)
        if step<=self.hold_end:return p['peak_lr']
        fraction=min(1.,(step-self.hold_end)/(self.end_step-self.hold_end))
        return p['min_lr']+(p['peak_lr']-p['min_lr'])*.5*(1+math.cos(math.pi*fraction))

    def step(self, epoch=None):
        assert epoch is None,'This trial advances one optimizer update at a time'
        self.original.step()
        step=self.after_scheduler.last_epoch
        if step>self.change_step:
            lr=self.value_at_step(step)
            for group in self.optimizer.param_groups:group['lr']=lr
            self.after_scheduler._last_lr=[g['lr'] for g in self.optimizer.param_groups]

    def get_last_lr(self):return self.original.get_last_lr()

    def state_dict(self):
        result=self.original.state_dict()
        result['learning_rate_branch']=self.policy
        return result

    def load_state_dict(self, state):
        assert state.get('learning_rate_branch')==self.policy
        self.original.load_state_dict(state)
