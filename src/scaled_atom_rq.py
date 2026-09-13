"""Residual quantization with a shared codebook of signed, scaled sparse atoms.

Code 0 is the single zero vector. Code 1+a*B+b denotes levels[b]*dictionary[:,a].
All coefficient levels use physical latent units and are shared across depths.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ScaledAtomRQ(nn.Module):
    def __init__(self, dictionary, levels, depth=4):
        super().__init__()
        dictionary = torch.as_tensor(dictionary).detach().clone()
        levels = torch.as_tensor(levels, dtype=dictionary.dtype, device=dictionary.device).detach().clone()
        if dictionary.ndim != 2 or levels.ndim != 1 or not len(levels):
            raise ValueError('Expected [channels, atoms] dictionary and one-dimensional levels')
        if not torch.isfinite(dictionary).all() or not torch.isfinite(levels).all():
            raise ValueError('Nonfinite codebook')
        if (levels == 0).any() or not (levels[1:] > levels[:-1]).all():
            raise ValueError('Nonzero levels must be strictly increasing; zero has its own token')
        norms = dictionary.square().sum(0)
        if (norms <= 0).any() or depth < 1:
            raise ValueError('Atoms must be nonzero and depth positive')
        self.register_buffer('dictionary', dictionary)
        self.register_buffer('levels', levels)
        self.register_buffer('atom_norms_squared', norms)
        self.depth = int(depth)
        self.num_atoms = dictionary.shape[1]
        self.vocab_size = 1 + self.num_atoms * len(levels)

    def embed(self, codes):
        if (codes < 0).any() or (codes >= self.vocab_size).any():
            raise ValueError('Code outside expanded vocabulary')
        packed = (codes.long() - 1).clamp_min(0)
        atoms = packed // len(self.levels)
        bins = packed % len(self.levels)
        values = self.levels[bins] * (codes != 0)
        return self.dictionary.T[atoms] * values[..., None]

    def get_code_emb_with_depth(self, codes):
        return self.embed(codes), None

    def expanded_codebook(self):
        entries = (self.dictionary.T[:, None] * self.levels[None, :, None]).flatten(0, 1)
        return torch.cat([entries.new_zeros(1, entries.shape[-1]), entries], 0)

    @torch.no_grad()
    def quantize(self, x, return_trajectory=False):
        if x.shape[-1] != self.dictionary.shape[0]:
            raise ValueError('Latent channels do not match dictionary')
        original_shape = x.shape[:-1]
        residual = x.reshape(-1, x.shape[-1]).clone()
        reconstruction = torch.zeros_like(residual)
        tokens, prefixes, projections = [], [], []
        boundaries = (self.levels[:-1] + self.levels[1:]) / 2
        rows = torch.arange(len(residual), device=residual.device)
        for _ in range(self.depth):
            correlations = residual @ self.dictionary
            ideal = correlations / self.atom_norms_squared
            bins = torch.bucketize(ideal.contiguous(), boundaries)
            gains = self.levels[bins]
            # Exact nearest expanded codeword: minimize squared distance by
            # maximizing its reduction from ||residual||². No atom exclusion/refit.
            improvements = 2 * correlations * gains - self.atom_norms_squared * gains.square()
            best_improvement, atoms = improvements.max(-1)
            selected_bins = bins[rows, atoms]
            code = 1 + atoms * len(self.levels) + selected_bins
            code = torch.where(best_improvement > 0, code, 0)
            contribution = self.embed(code)
            residual -= contribution
            reconstruction += contribution
            tokens.append(code)
            if return_trajectory:
                prefixes.append(reconstruction.clone())
                projections.append(ideal[rows, atoms])
        result = {'codes': torch.stack(tokens, -1).reshape(*original_shape, self.depth),
                  'quantized': reconstruction.reshape_as(x)}
        if return_trajectory:
            result['prefixes'] = torch.stack(prefixes, -2).reshape(*original_shape, self.depth, x.shape[-1])
            result['projections'] = torch.stack(projections, -1).reshape(*original_shape, self.depth)
        return result

    @torch.no_grad()
    def soft_codes(self, x, temperature, stochastic=True):
        """Dense reference for small-batch verification/original RQ compatibility."""
        if temperature <= 0:
            raise ValueError('Temperature must be positive in squared latent-distance units')
        book = self.expanded_codebook()
        residual = x.clone()
        probabilities, codes = [], []
        for _ in range(self.depth):
            distance = (residual.square().sum(-1, keepdim=True) + book.square().sum(-1)
                        - 2 * residual @ book.T)
            probability = (-distance / temperature).softmax(-1)
            if stochastic:
                code = torch.multinomial(probability.reshape(-1, len(book)), 1).reshape(x.shape[:-1])
            else:
                code = distance.argmin(-1)
            residual -= self.embed(code)
            probabilities.append(probability)
            codes.append(code)
        return torch.stack(probabilities, -2), torch.stack(codes, -1)


@torch.no_grad()
def continuous_matching_pursuit(x, dictionary, depth=4, return_prefixes=False):
    residual = x.reshape(-1, x.shape[-1]).clone()
    norms = dictionary.square().sum(0)
    rows = torch.arange(len(residual), device=residual.device)
    reconstructed = torch.zeros_like(residual)
    atoms, coefficients, prefixes = [], [], []
    for _ in range(depth):
        correlations = residual @ dictionary
        selected = (correlations.square() / norms).argmax(-1)
        coefficient = correlations[rows, selected] / norms[selected]
        contribution = dictionary.T[selected] * coefficient[:, None]
        residual -= contribution
        reconstructed += contribution
        atoms.append(selected)
        coefficients.append(coefficient)
        if return_prefixes:
            prefixes.append(reconstructed.clone())
    result = {'atoms': torch.stack(atoms, -1).reshape(*x.shape[:-1], depth),
              'coefficients': torch.stack(coefficients, -1).reshape(*x.shape[:-1], depth),
              'quantized': reconstructed.reshape_as(x)}
    if return_prefixes:
        result['prefixes'] = torch.stack(prefixes, -2).reshape(*x.shape[:-1], depth, x.shape[-1])
    return result


@torch.no_grad()
def orthogonal_matching_pursuit(x, dictionary, gram, depth=4):
    """The existing frozen Church OMP control, with its dictionary Gram cached."""
    signals = x.reshape(-1, x.shape[-1])
    correlations = signals @ dictionary
    corr = correlations
    n = len(signals)
    rows = torch.arange(n, device=x.device)
    available = torch.ones_like(corr, dtype=torch.bool)
    support = torch.empty(n, 0, device=x.device, dtype=torch.long)
    chol = signals.new_ones(n, 1, 1)
    for d in range(1, depth + 1):
        selected = corr.abs().masked_fill(~available, -1).argmax(-1)
        available[rows, selected] = False
        if d > 1:
            cross = gram[support, selected[:, None].expand(n, d-1)].unsqueeze(-1)
            solved = torch.linalg.solve_triangular(chol, cross, upper=False).transpose(1, 2)
            bottom = (1 - solved.square().sum(2, keepdim=True)).clamp_min(1e-10).sqrt()
            chol = torch.cat([torch.cat([chol, signals.new_zeros(n, d-1, 1)], 2),
                              torch.cat([solved, bottom], 2)], 1)
        support = torch.cat([support, selected[:, None]], 1)
        selected_corr = correlations[rows[:, None], support]
        coefficients = torch.cholesky_solve(selected_corr.unsqueeze(-1), chol).squeeze(-1)
        corr = correlations - coefficients.unsqueeze(1).bmm(gram[support]).squeeze(1)
    reconstructed = (dictionary.T[support] * coefficients[..., None]).sum(-2)
    return {'atoms': support.reshape(*x.shape[:-1], depth),
            'coefficients': coefficients.reshape(*x.shape[:-1], depth),
            'quantized': reconstructed.reshape_as(x)}


def fit_signed_levels(coefficients, signed_levels, iterations=60):
    """Lloyd calibration of shared physical magnitudes, with one explicit zero."""
    if signed_levels < 2 or signed_levels % 2:
        raise ValueError('Use an even count of nonzero signed levels')
    values = coefficients.detach().float().abs().reshape(-1)
    values = values[values > 0]
    k = signed_levels // 2
    centers = torch.quantile(values, torch.linspace(.1, .99, k, device=values.device))
    if k == 1:
        centers = values.mean().reshape(1)
    for _ in range(iterations):
        # Include the zero decision region, but keep the zero level fixed.
        full = torch.cat([centers.new_zeros(1), centers])
        labels = torch.bucketize(values, (full[:-1] + full[1:]) / 2)
        counts = torch.bincount(labels, minlength=k+1)
        sums = torch.bincount(labels, weights=values, minlength=k+1)
        updated = torch.where(counts[1:] > 0, sums[1:] / counts[1:].clamp_min(1), centers)
        if (updated - centers).abs().max() < 1e-6:
            centers = updated
            break
        centers = updated
    if not (centers > 0).all() or not (centers[1:] > centers[:-1]).all():
        raise ValueError('Collapsed coefficient levels')
    return torch.cat([-centers.flip(0), centers])


class FrozenSparseBackbone(nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        from rqvae.models.rqvae.rqvae import RQVAE
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        state = payload['state_dict']
        dictionary_key = 'quantizer.dictionary' if 'quantizer.dictionary' in state else 'bottleneck.dictionary'
        dictionary = state[dictionary_key].float()
        assert dictionary.shape == (256, 16384)
        reference = RQVAE(embed_dim=256, n_embed=16384, decay=.99, loss_type='mse',
            latent_loss_weight=.25, bottleneck_type='rq', latent_shape=[8,8,256], code_shape=[8,8,4],
            shared_codebook=True, restart_unused_codes=True,
            ddconfig=dict(double_z=False,z_channels=256,resolution=256,in_channels=3,out_ch=3,
                ch=128,ch_mult=[1,1,2,2,4,4],num_res_blocks=2,attn_resolutions=[8],dropout=0.))
        adapted = {}
        for name, value in state.items():
            if name.startswith(('encoder.', 'decoder.', 'quant_conv.', 'post_quant_conv.')):
                adapted[name] = value
            elif name.startswith('pre_bottleneck.'):
                adapted['quant_conv.' + name.removeprefix('pre_bottleneck.')] = value
            elif name.startswith('post_bottleneck.'):
                adapted['post_quant_conv.' + name.removeprefix('post_bottleneck.')] = value
        missing, extra = reference.load_state_dict(adapted, strict=False)
        assert not [name for name in missing if not name.startswith('quantizer.')] and not extra
        self.encoder, self.decoder = reference.encoder, reference.decoder
        self.quant_conv, self.post_quant_conv = reference.quant_conv, reference.post_quant_conv
        self.register_buffer('dictionary', F.normalize(dictionary.clone(), dim=0))
        self.requires_grad_(False).eval()

    def encode(self, xs):
        return self.quant_conv(self.encoder(xs)).permute(0, 2, 3, 1).contiguous()

    def decode(self, latent):
        return self.decoder(self.post_quant_conv(latent.permute(0,3,1,2).contiguous()))
