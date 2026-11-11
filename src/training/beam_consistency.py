"""Small reAR-inspired regularizers for a frozen beam16 RQ tokenizer.

The spatial transformer receives [class, site0, ..., site62]. Its shallow
features at positions 1: recover the current clean site embeddings; deeper
features at positions :64 predict the next clean site embeddings. Each site
embedding is the sum of its four fixed physical code vectors. Targets never
enter the transformer forward pass, and these heads are unused at inference.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


def masked_cosine_loss(predicted, target, eps=1e-8):
    """Average within each image first, ignoring genuinely zero target vectors."""
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError('Expected matching [batch, sites, channels] tensors')
    target = target.detach().float()
    valid = target.square().sum(-1) > eps**2
    distances = 1 - F.cosine_similarity(predicted.float(), target, dim=-1, eps=eps)
    per_image = (distances * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return per_image.mean()


class BeamConsistencyHeads(nn.Module):
    def __init__(self, hidden_dim=1536, latent_dim=256, projection_dim=512,
                 shallow_layer=0, deep_layer=31):
        super().__init__()
        if shallow_layer < 0 or deep_layer <= shallow_layer:
            raise ValueError('Require a shallow layer before the deeper layer')
        self.shallow_layer, self.deep_layer = shallow_layer, deep_layer
        def head():
            return nn.Sequential(nn.Linear(hidden_dim, projection_dim), nn.GELU(),
                                 nn.Linear(projection_dim, latent_dim))
        self.current_head, self.next_head = head(), head()
        self.losses = {}
        self._active = False

    @contextmanager
    def capture(self, model, clean_site_embeddings):
        if self._active or clean_site_embeddings.ndim != 3:
            raise ValueError('Capture requires one clean [batch, sites, latent] target')
        self._active = True
        self.losses = {}
        clean = clean_site_embeddings.detach()
        def shallow(_module, _inputs, output):
            if output.shape[:2] != clean.shape[:2]:
                raise ValueError('Unexpected class-shifted body sequence length')
            self.losses['current'] = masked_cosine_loss(self.current_head(output[:, 1:]), clean[:, :-1])
        def deep(_module, _inputs, output):
            if output.shape[:2] != clean.shape[:2]:
                raise ValueError('Unexpected body prediction sequence length')
            self.losses['next'] = masked_cosine_loss(self.next_head(output), clean)
        handles = []
        try:
            blocks = model.body_transformer.blocks
            handles.append(blocks[self.shallow_layer].register_forward_hook(shallow))
            handles.append(blocks[self.deep_layer].register_forward_hook(deep))
            yield self
            if set(self.losses) != {'current', 'next'}:
                raise RuntimeError('Both regularized body layers must execute')
        finally:
            for handle in handles:
                handle.remove()
            self._active = False


def corrupt_code_context(codes, vocab_size, maximum_probability, generator):
    """Uniform replacement using a separate RNG; never modify clean target IDs."""
    if codes.dtype != torch.long or codes.ndim != 4 or not 0 <= maximum_probability <= 1:
        raise ValueError('Expected clean [B,H,W,D] integer codes and probability in [0,1]')
    if vocab_size < 2:
        raise ValueError('Vocabulary must contain at least two entries')
    if maximum_probability == 0:
        return codes, codes.new_zeros((), dtype=torch.float32)
    rates = torch.rand((len(codes), 1, 1, 1), device=codes.device, generator=generator) * maximum_probability
    mask = torch.rand(codes.shape, device=codes.device, generator=generator) < rates
    replacements = torch.randint(vocab_size, codes.shape, device=codes.device, generator=generator)
    noisy = torch.where(mask, replacements, codes)
    return noisy, (noisy != codes).float().mean()


def trial_schedule(update, updates, ramp_updates=64, maximum_noise=.02):
    if not 0 <= update < updates or updates < 1 or ramp_updates < 1:
        raise ValueError('Invalid finite trial update or ramp')
    ramp = min(1., (update + 1) / ramp_updates)
    decay = max(0., (updates - update - 1) / max(1, updates - ramp_updates))
    return dict(auxiliary_ramp=ramp, noise_cap=maximum_noise * ramp * min(1., decay))


def extend_adam_payload(saved, extra_parameter_tensors):
    """Retain all existing moments/ages; new heads start with no Adam history."""
    if len(saved['param_groups']) != 1 or extra_parameter_tensors < 0:
        raise ValueError('Expected the source single-group AdamW optimizer')
    group = dict(saved['param_groups'][0])
    ids = list(group['params'])
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Invalid source optimizer parameter IDs')
    group['params'] = ids + list(range(max(ids)+1, max(ids)+1+extra_parameter_tensors))
    return dict(state=saved['state'], param_groups=[group])


def remap_data_cursor(next_epoch, next_microbatch, global_step, saved_accumulation,
                      new_accumulation, steps_per_epoch=626):
    if min(saved_accumulation, new_accumulation, steps_per_epoch) < 1:
        raise ValueError('Accumulation and epoch length must be positive')
    if min(next_epoch, next_microbatch, global_step) < 0 or next_microbatch % saved_accumulation:
        raise ValueError('Resume cursor must be an optimizer boundary')
    offset = next_microbatch // saved_accumulation
    if offset > steps_per_epoch or next_epoch*steps_per_epoch + offset != global_step:
        raise ValueError('Global step and sample cursor disagree')
    return next_epoch, offset * new_accumulation
