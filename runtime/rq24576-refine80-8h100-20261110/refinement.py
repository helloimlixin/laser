"""Explicit schedules and FP32 EMA for late RQTransformer continuation."""
import copy
import math
import torch


def teacher_temperature(step, policy):
    fraction=max(0.,min(1.,(step-policy['start_step'])/policy['anneal_steps']))
    return policy['initial']+(policy['final']-policy['initial'])*fraction


class ModelEMA:
    def __init__(self, model, decay):
        assert 0.<decay<1.
        self.model=copy.deepcopy(model).requires_grad_(False).eval()
        self.decay=float(decay);self.updates=0

    @torch.no_grad()
    def update(self, model):
        averaged=list(self.model.parameters());current=list(model.parameters())
        assert len(averaged)==len(current)
        # lerp keeps the full averaging operation in place; no full model temporary.
        torch._foreach_lerp_(averaged,current,1.-self.decay)
        for dst,src in zip(self.model.buffers(),model.buffers()):dst.copy_(src)
        self.updates+=1


class LearningRateRefinement:
    def __init__(self, original, optimizer, policy, step, saved_policy=None):
        self.original=original;self.optimizer=optimizer;self.policy=dict(policy)
        self.after_scheduler=original.after_scheduler
        assert self.after_scheduler.last_epoch==step
        assert self.after_scheduler.T_max==policy['original_horizon_steps']
        assert policy['start_step']<policy['ramp_end_step']<policy['end_step']
        if saved_policy is not None:assert saved_policy==self.policy
        expected=self.value_at_step(step)
        assert all(math.isclose(g['lr'],expected,rel_tol=1e-8,abs_tol=1e-12) for g in optimizer.param_groups)

    def value_at_step(self, step):
        p=self.policy
        if step<=p['start_step']:return p['start_lr']
        if step<p['ramp_end_step']:
            fraction=(step-p['start_step'])/(p['ramp_end_step']-p['start_step'])
            return p['start_lr']+(p['peak_lr']-p['start_lr'])*fraction
        fraction=max(0.,min(1.,(step-p['ramp_end_step'])/(p['end_step']-p['ramp_end_step'])))
        return p['min_lr']+(p['peak_lr']-p['min_lr'])*.5*(1.+math.cos(math.pi*fraction))

    def step(self):
        self.original.step()
        lr=self.value_at_step(self.after_scheduler.last_epoch)
        for group in self.optimizer.param_groups:group['lr']=lr
        self.after_scheduler._last_lr=[g['lr'] for g in self.optimizer.param_groups]

    def get_last_lr(self):return self.original.get_last_lr()

    def state_dict(self):
        state=self.original.state_dict();state['late_refinement']=self.policy
        return state

    def load_state_dict(self,state):
        assert state['late_refinement']==self.policy
        self.original.load_state_dict(state)
