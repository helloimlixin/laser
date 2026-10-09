"""Checkpointed warmup/cosine schedule for a fresh-Adam epoch55 restart."""
import math
import torch


class WarmupCosineLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, *, peak_lr, min_lr, total_steps,
                 warmup_steps, warmup_start_ratio=.1):
        if not 0 < warmup_steps < total_steps:
            raise ValueError('Warmup must be shorter than the complete schedule')
        if not 0 < min_lr < peak_lr or not 0 < warmup_start_ratio <= 1:
            raise ValueError('Invalid learning-rate bounds')
        self.schedule_version = 'laser-epoch55-warmup-cosine-v1'
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.warmup_start_ratio = float(warmup_start_ratio)
        for group in optimizer.param_groups:
            group['initial_lr'] = self.peak_lr
        super().__init__(optimizer)

    def lr_at(self, step):
        step = min(max(int(step), 0), self.total_steps)
        if step <= self.warmup_steps:
            return self.peak_lr * (self.warmup_start_ratio +
                (1 - self.warmup_start_ratio) * step / self.warmup_steps)
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + .5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return [self.lr_at(self.last_epoch) for _ in self.optimizer.param_groups]


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    saved_lrs = [float(group['lr']) for group in optimizer.param_groups]
    if state_dict is not None:
        expected = dict(schedule_version='laser-epoch55-warmup-cosine-v1', peak_lr=float(initial_lr),
                        min_lr=float(min_lr), total_steps=int(total_steps),
                        warmup_steps=626, warmup_start_ratio=.1)
        for key, value in expected.items():
            if state_dict.get(key) != value:
                raise ValueError('Scheduler configuration changed at ' + key)
        if state_dict['last_epoch'] != completed_steps or not 0 <= completed_steps <= total_steps:
            raise ValueError('Scheduler cursor does not match checkpoint progress')
        if completed_steps <= 626:
            expected_lr = initial_lr * (.1 + .9 * completed_steps / 626)
        else:
            progress = (completed_steps - 626) / (total_steps - 626)
            expected_lr = min_lr + .5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))
        if any(not math.isclose(lr, expected_lr, rel_tol=1e-12, abs_tol=1e-15) for lr in saved_lrs):
            raise ValueError('Saved optimizer LR does not match the checkpointed schedule')
    scheduler = WarmupCosineLR(optimizer, peak_lr=initial_lr, min_lr=min_lr,
                              total_steps=total_steps, warmup_steps=626)
    if state_dict is None:
        if completed_steps:
            raise ValueError('An in-progress warmup/cosine resume requires its scheduler state')
        return scheduler
    scheduler.load_state_dict(state_dict)
    for group, lr in zip(optimizer.param_groups, saved_lrs):
        group['lr'] = lr
    return scheduler
