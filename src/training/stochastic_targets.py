"""Stochastic residual targets with calibrated, fixed temperatures per depth.

The same geometry-derived distribution supplies both the sampled code and the
soft label. Later targets condition on the residual from the sampled prefix.
Calibration matches mean entropy at each depth, not entropy of every example.
"""
import math
from numbers import Real
from types import MethodType

import torch


TARGET_POLICY_VERSION = 'compact_residual_depth_temperature_v1'


def depth_temperatures(temperature, depth):
    """Accept a positive scalar or exactly one positive value per depth."""
    if isinstance(temperature, Real) and not isinstance(temperature, bool):
        values = [float(temperature)] * depth
    else:
        if isinstance(temperature, (str, bytes, bool)):
            raise ValueError('Temperature must be a scalar or one value per depth')
        try:
            raw = list(temperature)
            if any(isinstance(value, bool) for value in raw):
                raise ValueError('Boolean temperatures are invalid')
            values = [float(value) for value in raw]
        except (TypeError, ValueError) as error:
            raise ValueError('Temperature must be a scalar or one value per depth') from error
    if len(values) != depth or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Exactly one finite positive temperature per depth is required')
    return values


def _book_at(quantizer, depth):
    return quantizer.codebooks[depth] if hasattr(quantizer, 'codebooks') else quantizer


def compact_scores(quantizer, residual, depth):
    """Negative squared distances, omitting the common residual norm."""
    book = _book_at(quantizer, depth)
    # Match the existing expanded-book arithmetic and atom-major token order.
    dictionary, levels, norms = book.dictionary.float(), book.levels.float(), book.norms.float()
    correlations = residual.float() @ dictionary
    penalty = norms[:, None] * levels.square()
    scores = (2 * correlations[..., None] * levels[None, :, :] - penalty).flatten(1)
    return torch.cat([scores.new_zeros(len(residual), 1), scores], 1)


@torch.no_grad()
def compact_soft_codes(quantizer, x, temp=.5, stochastic=True, chunk_size=128):
    temperatures = depth_temperatures(temp, quantizer.depth)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('Positive integer chunk size required')
    if not torch.is_floating_point(x) or x.shape[-1] != quantizer.dictionary.shape[0]:
        raise ValueError('Floating-point latents with the dictionary dimension required')
    with torch.autocast(device_type=x.device.type, enabled=False):
        shape = x.shape[:-1]
        residual = x.reshape(-1, x.shape[-1]).float().clone()
        targets = residual.new_empty(len(residual), quantizer.depth, quantizer.vocab_size)
        codes = torch.empty(len(residual), quantizer.depth, device=x.device, dtype=torch.long)
        for depth, temperature in enumerate(temperatures):
            book = _book_at(quantizer, depth)
            for start in range(0, len(residual), chunk_size):
                r = residual[start:start + chunk_size]
                scores = compact_scores(quantizer, r, depth)
                probability = (scores / temperature).softmax(-1)
                chosen = (torch.multinomial(probability, 1).squeeze(-1) if stochastic
                          else scores.argmax(-1))
                targets[start:start + len(r), depth] = probability
                codes[start:start + len(r), depth] = chosen
                r.sub_(book.embed(chosen).float())
        return targets.reshape(*shape, quantizer.depth, quantizer.vocab_size), codes.reshape(*shape, quantizer.depth)


def install_compact_target_policy(quantizer):
    """Attach the versioned target method to a frozen runtime, without changing weights."""
    for depth in range(quantizer.depth):
        book = _book_at(quantizer, depth)
        if not all(hasattr(book, name) for name in ['dictionary', 'levels', 'norms', 'embed']):
            raise TypeError('This target policy requires a compact scaled-atom codebook')
    quantizer.get_soft_codes = MethodType(compact_soft_codes, quantizer)


def _mean_entropy(scores, temperature, chunk_size):
    total = scores.new_zeros((), dtype=torch.float64)
    for chunk in scores.split(chunk_size):
        log_probability = (chunk / temperature).log_softmax(-1)
        total += -(log_probability.exp() * log_probability).sum(-1).double().sum()
    return (total / len(scores)).item()


@torch.no_grad()
def calibrate_depth_temperatures(quantizer, latents, target_entropies, *, seed=99200,
                                 min_temperature=1e-4, max_temperature=4.,
                                 tolerance=.002, max_iterations=24, chunk_size=128):
    """Fit temperatures in order, sampling earlier depths before fitting later ones.

    Mean entropy is monotone in temperature for the fixed residual population
    at one depth. The population is updated only after that temperature is fixed.
    RNG state outside the function is preserved. Unreachable targets fail instead
    of silently clipping the policy to a temperature bound.
    """
    goals = list(target_entropies)
    if (len(goals) != quantizer.depth or any(not math.isfinite(h) or not 0 < h < math.log(quantizer.vocab_size) for h in goals)
            or not 0 < min_temperature < max_temperature or not math.isfinite(max_temperature)
            or not math.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1 or chunk_size < 1
            or latents.numel() == 0 or not torch.isfinite(latents).all()):
        raise ValueError('Invalid entropy targets, calibration bounds, or latents')
    device = latents.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    temperatures, trace = [], []
    with torch.random.fork_rng(devices=devices), torch.autocast(device_type=device.type, enabled=False):
        torch.manual_seed(seed)
        residual = latents.reshape(-1, latents.shape[-1]).float().clone()
        for depth, goal in enumerate(goals):
            scores = residual.new_empty(len(residual), quantizer.vocab_size)
            for start in range(0, len(residual), chunk_size):
                scores[start:start + chunk_size] = compact_scores(quantizer, residual[start:start + chunk_size], depth)
            low, high = float(min_temperature), float(max_temperature)
            low_entropy = _mean_entropy(scores, low, chunk_size)
            high_entropy = _mean_entropy(scores, high, chunk_size)
            if goal < low_entropy - tolerance or goal > high_entropy + tolerance:
                raise ValueError(f'Unreachable entropy target at depth {depth}: {goal} outside [{low_entropy}, {high_entropy}]')
            temperature = low if abs(low_entropy-goal) <= tolerance else high
            achieved = low_entropy if temperature == low else high_entropy
            iteration = 0
            while abs(achieved-goal) > tolerance and iteration < max_iterations:
                temperature = math.sqrt(low * high)
                achieved = _mean_entropy(scores, temperature, chunk_size)
                if achieved < goal:
                    low = temperature
                else:
                    high = temperature
                iteration += 1
            if abs(achieved-goal) > tolerance:
                raise RuntimeError(f'Entropy calibration did not converge at depth {depth}')
            temperatures.append(temperature)
            trace.append(dict(depth=depth, target_entropy=goal, achieved_entropy=achieved,
                              temperature=temperature, iterations=iteration,
                              entropy_at_min_temperature=low_entropy, entropy_at_max_temperature=high_entropy))
            for start in range(0, len(residual), chunk_size):
                probability = (scores[start:start + chunk_size] / temperature).softmax(-1)
                chosen = torch.multinomial(probability, 1).squeeze(-1)
                residual[start:start + len(chosen)].sub_(_book_at(quantizer, depth).embed(chosen).float())
            del scores
    return dict(selected_temperature=temperatures, target_entropy_nats_per_depth=goals,
                trace=trace, target_policy=TARGET_POLICY_VERSION)
