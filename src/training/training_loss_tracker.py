"""Resume-safe reporting for sample-weighted global training batches."""
import copy
import math


class TrainingLossTracker:
    def __init__(self, start_step, state=None):
        self.state = dict(version=1, start_step=start_step, last_step=start_step,
                          updates=0, samples=0, loss_sum=0., ce_sum=0.,
                          loss_ema=None, ce_ema=None, ema_decay=.95)
        if state is not None:
            self.state = copy.deepcopy(state)
            if self.state['version'] != 1 or self.state['last_step'] != start_step:
                raise ValueError('Loss tracker cursor must match the full checkpoint')

    def update(self, loss, ce, crps, weight, samples, step, *, additional_loss=0.):
        values = (loss, ce, crps, weight, additional_loss)
        if not all(math.isfinite(x) and x >= 0 for x in values) or samples <= 0:
            raise ValueError('Training loss statistics must be finite with positive samples')
        if step != self.state['last_step'] + 1:
            raise ValueError('Training loss statistics require consecutive optimizer steps')
        if not math.isclose(loss, ce + weight * crps + additional_loss, rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError('Global loss disagrees with its declared objective components')
        s = self.state
        s['last_step'] = step
        s['updates'] += 1
        s['samples'] += samples
        s['loss_sum'] += samples * loss
        s['ce_sum'] += samples * ce
        for name, value in (('loss_ema', loss), ('ce_ema', ce)):
            s[name] = value if s[name] is None else s['ema_decay'] * s[name] + (1 - s['ema_decay']) * value
        return {'train/loss': loss, 'train/cross_entropy': ce,
                'train/coefficient_crps': crps, 'train/coefficient_crps_weight': weight,
                'train/loss_ema': s['loss_ema'], 'train/cross_entropy_ema': s['ce_ema'],
                'train/loss_running_mean': s['loss_sum'] / s['samples'],
                'train/cross_entropy_running_mean': s['ce_sum'] / s['samples'],
                'train/loss_global_batch_samples': samples}

    def state_dict(self):
        return copy.deepcopy(self.state)
