"""Preserve epoch-40 LR, smoothly adopt a cosine ending at epoch 90."""
import math


def batch_config(plan, late=False):
    if late:
        raise ValueError('This experiment has no batch-size transition')
    return dict(phase='fixed', **{k: plan[k] for k in
        ['global_batch', 'per_gpu_batch', 'microbatch', 'accumulation', 'steps_per_epoch']},
        score_chunk_size=plan['vector_score_chunk_size'])


class RecipeSchedule:
    KIND = 'fixed256-smooth-transition-cosine90-v1'
    FIELDS = ['base_lrs', 'epochs', 'budget_steps', 'T_max', 'steps_per_epoch',
              'eta_min', 'source_epochs', 'source_eta_min', 'transition_start',
              'transition_end', 'phase']

    def __init__(self, optimizer, *, plan):
        self.optimizer = optimizer
        self.base_lrs = [float(plan['lr'])] * len(optimizer.param_groups)
        self.epochs = int(plan['cosine_epochs'])
        self.budget_steps = int(plan['total_steps'])
        self.T_max = int(plan['cosine_total_steps'])
        self.steps_per_epoch = int(plan['steps_per_epoch'])
        self.eta_min = float(plan['min_lr'])
        self.source_epochs = int(plan['schedule_source_cosine_epochs'])
        self.source_eta_min = float(plan['schedule_source_min_lr'])
        self.transition_start = float(plan['anneal_transition_start_epoch'])
        self.transition_end = float(plan['anneal_transition_end_epoch'])
        assert self.T_max == self.epochs * self.steps_per_epoch == self.budget_steps
        assert self.epochs == int(plan['epochs'])
        assert 0 <= self.transition_start < self.transition_end < self.epochs
        assert self.source_epochs > self.epochs
        assert 0 <= self.source_eta_min <= self.eta_min < min(self.base_lrs)
        self.phase = 'fixed'
        self.last_epoch = 0
        self.epoch_progress = 0.
        self._last_lr = self.rate(0.)
        self._apply()

    def _apply(self):
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group['lr'] = lr

    def rate(self, epoch_progress):
        epoch = float(epoch_progress)
        old_progress = min(1., max(0., epoch / self.source_epochs))
        old = [self.source_eta_min + (base-self.source_eta_min)*.5*
               (1+math.cos(math.pi*old_progress)) for base in self.base_lrs]
        if epoch <= self.transition_start:
            return old
        progress = min(1., max(0., epoch / self.epochs))
        target = [self.eta_min + (base-self.eta_min)*.5*
                  (1+math.cos(math.pi*progress)) for base in self.base_lrs]
        if epoch >= self.transition_end:
            return target
        fraction = (epoch-self.transition_start)/(self.transition_end-self.transition_start)
        weight = fraction*fraction*(3-2*fraction)
        return [(1-weight)*left + weight*right for left, right in zip(old, target)]

    def step(self, epoch_progress):
        self.last_epoch += 1
        self.epoch_progress = self.last_epoch / self.steps_per_epoch
        assert self.last_epoch <= self.budget_steps
        assert abs(float(epoch_progress)-self.epoch_progress) < 1e-12
        self._last_lr = self.rate(self.epoch_progress)
        self._apply()

    def state_dict(self):
        return dict(kind=self.KIND, **{k: getattr(self, k) for k in
                    self.FIELDS + ['last_epoch', 'epoch_progress', '_last_lr']})

    def load_state_dict(self, state):
        assert state['kind'] == self.KIND
        for key in self.FIELDS:
            assert state[key] == getattr(self, key), key
        self.last_epoch = int(state['last_epoch'])
        self.epoch_progress = float(state['epoch_progress'])
        assert 0 <= self.last_epoch <= self.budget_steps
        assert self.epoch_progress == self.last_epoch / self.steps_per_epoch
        self._last_lr = self.rate(self.epoch_progress)
        assert self._last_lr == state['_last_lr']
        self._apply()


def final_step(plan):
    return int(plan['epochs'])*int(plan['steps_per_epoch'])


def cursor_for_step(plan, step):
    if step == 0:
        return 0, 0
    steps = int(plan['steps_per_epoch'])
    return (int(step)-1)//steps, ((int(step)-1)%steps+1)*int(plan['global_batch'])
