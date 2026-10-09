"""Parameter EMA with explicit initialization, recovery, and reversible evaluation."""
from contextlib import contextmanager
import torch


class ParameterEMA:
    def __init__(self, model, decay=0.999, state=None):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0, 1)')
        self.decay = float(decay)
        self.updates = 0
        self.values = {k: p.detach().clone() for k, p in model.named_parameters()}
        if state is not None:
            if state['decay'] != self.decay or set(state['values']) != set(self.values):
                raise ValueError('incompatible EMA checkpoint')
            for k, v in self.values.items():
                if state['values'][k].shape != v.shape:
                    raise ValueError(f'incompatible EMA shape: {k}')
                v.copy_(state['values'][k])
            self.updates = int(state['updates'])
            if self.updates < 0:
                raise ValueError('EMA update count must be nonnegative')

    @torch.no_grad()
    def update(self, model):
        params = dict(model.named_parameters())
        if set(params) != set(self.values):
            raise ValueError('model parameters changed after EMA initialization')
        torch._foreach_lerp_(list(self.values.values()),
                             [params[k].detach() for k in self.values], 1 - self.decay)
        self.updates += 1

    def state_dict(self, *, cpu=True):
        """Return a CPU copy, or live tensors for a synchronous snapshotter."""
        return dict(decay=self.decay, updates=self.updates,
                    values={k: v.detach().to('cpu', copy=True) if cpu else v.detach()
                            for k, v in self.values.items()})

    @contextmanager
    def apply(self, model):
        params = dict(model.named_parameters())
        original = {k: p.detach().to('cpu', copy=True) for k, p in params.items()}
        try:
            with torch.no_grad():
                for k, p in params.items():
                    p.copy_(self.values[k])
            yield model
        finally:
            with torch.no_grad():
                for k, p in params.items():
                    p.copy_(original[k])
