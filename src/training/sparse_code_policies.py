"""Explicit coefficient target measures and depth-dependent sparse sampling."""
import math

import torch


def coefficient_targets(coefficients, centers, temperature, *, measure='centers',
                        stochastic=True, hard=False, generator=None):
    """Return coefficient IDs and probabilities in physical units.

    ``centers`` retains the original categorical distance-softmax objective.
    ``cells`` integrates a Gaussian over nearest-center Voronoi cells, so
    nonuniform center density does not act as an implicit coefficient prior.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Coefficient temperature must be finite and positive')
    if centers.ndim != 1 or len(centers) < 2 or not torch.all(centers[1:] > centers[:-1]):
        raise ValueError('Coefficient centers must be strictly increasing')
    if measure not in {'centers', 'cells'}:
        raise ValueError('Unknown coefficient target measure')
    values = coefficients.float()
    centers = centers.to(device=values.device, dtype=torch.float32)
    midpoints = (centers[:-1] + centers[1:]) * .5
    nearest = torch.bucketize(values.contiguous(), midpoints)
    if hard:
        probabilities = torch.nn.functional.one_hot(nearest, len(centers)).float()
    elif measure == 'centers':
        probabilities = (-(values[..., None] - centers).square() / temperature).softmax(-1)
    else:
        edges = torch.cat((centers.new_tensor([-float('inf')]), midpoints,
                           centers.new_tensor([float('inf')])))
        # exp(-distance²/tau) has variance tau/2. erf uses sigma*sqrt(2)=sqrt(tau).
        cdf = .5 * (1 + torch.erf((edges - values[..., None]) / math.sqrt(temperature)))
        probabilities = cdf.diff(dim=-1).clamp_min(0)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    if stochastic and not hard:
        ids = torch.multinomial(probabilities.reshape(-1, len(centers)), 1,
                                generator=generator).reshape(values.shape)
    else:
        ids = nearest
    return ids.long(), probabilities


class DepthAtomSampler:
    """Apply separate cutoffs to atom draws; coefficient draws are unchanged."""
    def __init__(self, sample, num_atoms, cutoffs):
        if not cutoffs or any(not isinstance(k, int) or not 1 <= k <= num_atoms for k in cutoffs):
            raise ValueError('Invalid per-depth atom cutoffs')
        self.sample = sample
        self.num_atoms = num_atoms
        self.cutoffs = tuple(cutoffs)
        self.atom_draw = 0

    def reset(self):
        self.atom_draw = 0

    def __call__(self, logits, **kwargs):
        if logits.shape[-1] == self.num_atoms:
            kwargs['top_k'] = self.cutoffs[self.atom_draw % len(self.cutoffs)]
            self.atom_draw += 1
        return self.sample(logits, **kwargs)
