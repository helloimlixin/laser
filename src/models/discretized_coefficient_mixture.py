"""Ordered coefficient likelihoods with the existing scalar token vocabulary.

The first and last bins absorb the logistic tails, exactly matching a clamped
nearest-centre quantizer.  The returned values are normalized log probabilities
and can be consumed by the existing categorical loss and sampling code.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def discretized_logistic_log_probs(logits, means, scales, centers):
    """Integrate a logistic mixture over nearest-centre bin intervals.

    Mixture parameters have shape ``[..., mixtures]``. Calculations retain
    float64 for numerical checks and otherwise use float32, including under
    autocast. No subtraction of nearly equal CDF values or probability floor
    is needed in the interior bins.
    """
    if logits.shape != means.shape or means.shape != scales.shape:
        raise ValueError('Mixture weights, means and scales must have equal shapes')
    if centers.ndim != 1 or centers.numel() < 2:
        raise ValueError('At least two coefficient bin centers are required')
    dtype = torch.float64 if means.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=means.device.type, enabled=False):
        logits, means, scales = (x.to(dtype) for x in (logits, means, scales))
        centers = centers.to(device=means.device, dtype=dtype)
        edges = (centers[1:] + centers[:-1]) * .5
        means, scales = means.unsqueeze(-1), scales.unsqueeze(-1)
        first = F.logsigmoid((edges[:1] - means) / scales)
        last = F.logsigmoid((means - edges[-1:]) / scales)
        lower = (edges[:-1] - means) / scales
        upper = (edges[1:] - means) / scales
        width = (edges[1:] - edges[:-1]) / scales
        middle = F.logsigmoid(upper) + F.logsigmoid(-lower) + torch.log(-torch.expm1(-width))
        component = torch.cat((first, middle, last), dim=-1)
        result = torch.logsumexp(F.log_softmax(logits, -1).unsqueeze(-1) + component, dim=-2)
        # Removes floating-point partition error; not a truncation of the tails.
        return result - result.logsumexp(-1, keepdim=True)


class DiscretizedCoefficientMixture(nn.Module):
    """A conditional logistic mixture, evaluated as an exact discrete PMF."""
    def __init__(self, hidden_dim, *, bins=512, coefficient_max=3., components=8,
                 width=256, min_scale=1.e-3):
        super().__init__()
        if min(hidden_dim, width, components) < 1 or bins < 2:
            raise ValueError('Positive dimensions and at least two bins are required')
        if not math.isfinite(coefficient_max) or coefficient_max <= 0:
            raise ValueError('Coefficient range must be positive and finite')
        if not math.isfinite(min_scale) or not 0 < min_scale < coefficient_max:
            raise ValueError('Minimum scale must lie inside the coefficient range')
        self.components = int(components)
        self.coefficient_max = float(coefficient_max)
        self.min_scale = float(min_scale)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden = nn.Linear(hidden_dim, width)
        self.parameters_projection = nn.Linear(width, 3 * components)
        self.register_buffer('bin_centers', torch.linspace(-coefficient_max, coefficient_max, bins))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.hidden.weight, std=.02)
        nn.init.zeros_(self.hidden.bias)
        # Start with a broad symmetric density. Hidden features immediately
        # train the output projection; no random, extremely sharp components.
        nn.init.normal_(self.parameters_projection.weight, std=.001)
        with torch.no_grad():
            weights, means, scales = self.parameters_projection.bias.chunk(3)
            weights.zero_()
            extent = self.coefficient_max * .8
            means.copy_(torch.linspace(-extent, extent, self.components, device=means.device))
            scales.fill_(math.log(math.expm1(max(.35 - self.min_scale, .01))))

    def distribution_parameters(self, hidden):
        features = F.silu(self.hidden(self.norm(hidden)))
        # BF16 rounding of a mean can exceed the width of a coefficient bin.
        # This small final projection and the probability calculations use FP32.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            values = F.linear(features.float(), self.parameters_projection.weight.float(),
                              self.parameters_projection.bias.float())
        weights, means, raw_scales = values.chunk(3, dim=-1)
        scales = F.softplus(raw_scales) + self.min_scale
        return weights, means, scales

    def forward(self, hidden):
        return discretized_logistic_log_probs(*self.distribution_parameters(hidden), self.bin_centers)


def install_coefficient_mixture(model, *, coefficient_max=3., components=8, width=256,
                                min_scale=1.e-3, copy_norm=True):
    """Replace only the final coefficient classifiers, retaining conditioners."""
    if not hasattr(model, 'coeff_classifier') or not hasattr(model, 'coeff_vocab_size'):
        raise ValueError('A compound model with a separate coefficient classifier is required')
    old = model.coeff_classifier
    specific = isinstance(old, nn.ModuleList)
    originals = list(old) if specific else [old]
    if any(not isinstance(head, nn.Sequential) or not isinstance(head[0], nn.LayerNorm)
           for head in originals):
        raise ValueError('Expected the original categorical coefficient classifiers')
    heads = []
    for original in originals:
        parameter = next(original.parameters())
        replacement = DiscretizedCoefficientMixture(int(model.config.embed_dim),
            bins=int(model.coeff_vocab_size), coefficient_max=coefficient_max,
            components=components, width=width, min_scale=min_scale).to(parameter.device)
        if copy_norm:
            replacement.norm = copy.deepcopy(original[0])
        replacement.train(model.training)
        heads.append(replacement)
    model.coeff_classifier = nn.ModuleList(heads) if specific else heads[0]
    return model
