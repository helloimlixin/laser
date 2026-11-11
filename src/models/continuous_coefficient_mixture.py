"""Conditional densities for physical, unquantized sparse coefficients."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ContinuousCoefficientMixture(nn.Module):
    """Gaussian mixture in asinh(c / scale), including the physical Jacobian.

    ``values`` has a final observation axis; parameters have a final mixture
    axis. All likelihood and sampling arithmetic uses FP32 (FP64 for oracles).
    """

    def __init__(self, width, components=8, min_std=.03):
        super().__init__()
        if width < 1 or components < 1 or not 0 < min_std < .5:
            raise ValueError('Positive dimensions and 0 < min_std < .5 required')
        self.components, self.min_std = components, min_std
        self.norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, 256)
        self.projection = nn.Linear(256, components * 3)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.projection.weight, std=.001)
        with torch.no_grad():
            weights, means, scales = self.projection.bias.chunk(3)
            weights.zero_()
            means.copy_(torch.linspace(-1.5, 1.5, self.components, device=means.device))
            scales.fill_(math.log(math.expm1(.5 - self.min_std)))

    def forward(self, features):
        h = F.silu(self.hidden(self.norm(features)))
        with torch.autocast(device_type=h.device.type, enabled=False):
            weights, means, raw_std = F.linear(
                h.float(), self.projection.weight.float(), self.projection.bias.float()).chunk(3, -1)
        return weights, means, F.softplus(raw_std) + self.min_std

    @staticmethod
    def log_prob(parameters, values, scale):
        weights, means, std = parameters
        dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=values.device.type, enabled=False):
            values = values.to(dtype)
            scale = torch.as_tensor(scale, device=values.device, dtype=dtype)
            y = torch.asinh(values / scale)
            component = (-.5 * ((y[..., None] - means[..., None, :]) / std[..., None, :]).square()
                         - std[..., None, :].log() - .5 * math.log(2 * math.pi))
            # hypot avoids squaring extreme but finite physical values.
            return (torch.logsumexp(weights.log_softmax(-1)[..., None, :] + component, -1)
                    - torch.hypot(scale, values).log())

    @staticmethod
    def sample(parameters, scale, generator=None):
        weights, means, std = (p.float() for p in parameters)
        shape = weights.shape[:-1]
        component = torch.multinomial(
            weights.reshape(-1, weights.shape[-1]).softmax(-1), 1,
            generator=generator).reshape(*shape, 1)
        mu = means.gather(-1, component).squeeze(-1)
        sigma = std.gather(-1, component).squeeze(-1)
        normal = torch.randn(shape, device=means.device, dtype=torch.float32, generator=generator)
        with torch.autocast(device_type=means.device.type, enabled=False):
            return scale * torch.sinh(mu + sigma * normal)
