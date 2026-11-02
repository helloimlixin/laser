"""Recover AMP overflows by replaying the same accumulated image batch.

Only a successful optimizer step advances the schedule or data cursor. The
buffer holds CPU DataLoader batches, and RNG restoration reproduces online
stochastic targets and dropout while the scaler reduces the backward scale.
"""
import random

import numpy as np
import torch


class ReplayWindow:
    def __init__(self, device, generators=()):
        self.device = torch.device(device)
        self.generators = tuple(generators)
        self.batches = []
        self.rng = None

    def begin(self):
        self.batches = []
        self.rng = dict(python=random.getstate(), numpy=np.random.get_state(),
                        cpu=torch.get_rng_state())
        self.rng['generators'] = [generator.get_state() for generator in self.generators]
        if self.device.type == 'cuda':
            self.rng['cuda'] = torch.cuda.get_rng_state(self.device)

    def append(self, images, labels, microbatch):
        if self.rng is None or images.device.type != 'cpu' or labels.device.type != 'cpu':
            raise ValueError('Replay requires an initialized window of CPU image batches')
        self.batches.append((images, labels, microbatch))

    def replay(self, backward):
        if not self.batches or self.rng is None:
            raise ValueError('No accumulated batch to replay')
        random.setstate(self.rng['python']);np.random.set_state(self.rng['numpy'])
        torch.set_rng_state(self.rng['cpu'])
        for generator, state in zip(self.generators, self.rng['generators']):
            generator.set_state(state)
        if self.device.type == 'cuda':
            torch.cuda.set_rng_state(self.rng['cuda'], self.device)
        for images, labels, microbatch in self.batches:
            backward(images, labels, microbatch)


def recoverable_amp_step(parameters, optimizer, scaler, *, max_norm,
                         replay_backward, on_overflow=None, max_retries=8):
    """Clip and step once, retrying genuine scaled-gradient overflows only.

    Nonfinite forward losses must still fail in the caller. Finite individual
    gradients with an overflowing aggregate norm also fail without stepping.
    GradScaler's public step/update API skips Adam and resets its growth
    tracker on overflow. Neither optimizer nor scheduler advances on retry.
    """
    parameters = list(parameters)
    retries = 0
    while True:
        scaler.unscale_(optimizer)
        old_scale = scaler.get_scale()
        try:
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm,
                                                  error_if_nonfinite=True)
        except RuntimeError:
            # clip_grad_norm_ raises before modifying any gradient. Confirm
            # the unscale pass saw an actual inf/NaN before asking AMP to skip.
            nonfinite = any(not torch.isfinite(p.grad).all().item()
                            for p in parameters if p.grad is not None)
            if not scaler.is_enabled() or not nonfinite:
                raise
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            if not 0 < new_scale < old_scale:
                raise FloatingPointError('AMP failed to reduce the overflowing loss scale')
            retries += 1
            if on_overflow is not None:
                on_overflow(dict(retry=retries,previous_scale=old_scale,scale=new_scale))
            if retries > max_retries:
                raise FloatingPointError('Nonfinite gradients persisted after AMP batch retries')
            replay_backward()
            continue
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise FloatingPointError('AMP skipped an update despite finite clipped gradients')
        return norm, retries
