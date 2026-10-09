"""A fully serialized warmup/cosine policy for fresh CC3M training."""
import math


class WarmupCosineSchedule:
    kind = 'warmup-cosine-v1'

    def __init__(self, optimizer, *, initial_lr, min_lr, total_steps,
                 warmup_steps, completed_steps=0, state_dict=None):
        if not (0 < initial_lr and 0 <= min_lr <= initial_lr
                and 0 <= warmup_steps < total_steps
                and 0 <= completed_steps <= total_steps):
            raise ValueError('Invalid warmup/cosine policy or progress')
        self.optimizer = optimizer
        self.policy = dict(initial_lr=initial_lr, min_lr=min_lr,
                           total_steps=total_steps, warmup_steps=warmup_steps)
        self.last_epoch = completed_steps
        if state_dict is not None:
            expected = self.lr_at_step(self.policy, completed_steps)
            if (state_dict.get('kind') != self.kind
                    or state_dict['policy'] != self.policy
                    or state_dict['last_epoch'] != completed_steps
                    or state_dict['_last_lr'] != [expected] * len(optimizer.param_groups)
                    or any(group['lr'] != expected for group in optimizer.param_groups)):
                raise ValueError('Warmup/cosine checkpoint and optimizer disagree')
        elif completed_steps:
            raise ValueError('Resuming warmup/cosine requires scheduler state')
        self._apply()

    @staticmethod
    def lr_at_step(policy, completed_steps):
        warmup = policy['warmup_steps']
        if completed_steps < warmup:
            return policy['initial_lr'] * (completed_steps + 1) / warmup
        position = min(1., (completed_steps - warmup) / (policy['total_steps'] - warmup))
        return policy['min_lr'] + .5 * (policy['initial_lr'] - policy['min_lr']) * (
            1 + math.cos(math.pi * position))

    def _apply(self):
        lr = self.lr_at_step(self.policy, self.last_epoch)
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def step(self):
        self.last_epoch += 1
        self._apply()

    def state_dict(self):
        return dict(kind=self.kind, policy=dict(self.policy), last_epoch=self.last_epoch,
                    _last_lr=[group['lr'] for group in self.optimizer.param_groups])
