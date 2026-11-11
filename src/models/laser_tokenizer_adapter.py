"""Frozen LASER checkpoint/dictionary adapter using upstream encoder/decoder."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src import rq_upstream
from rqvae.models.rqvae.rqvae import RQVAE


class FrozenLaserTokenizer(nn.Module):
    def __init__(self, checkpoint, *, num_atoms, coeff_vocab_size, coeff_max,
                 coeff_scales, sparsity_level):
        super().__init__()
        self.num_atoms = int(num_atoms)
        # None selects the continuous codec; no coefficient levels are created.
        self.coeff_vocab_size = 0 if coeff_vocab_size is None else int(coeff_vocab_size)
        self.sparsity_level = int(sparsity_level)
        self.vocab_size = self.num_atoms + self.coeff_vocab_size
        self.clamp_coeffs = False
        self.soft_target_physical = False
        stage1 = RQVAE(embed_dim=256, n_embed=self.num_atoms, decay=.99,
            loss_type='mse', latent_loss_weight=.25, bottleneck_type='rq',
            ddconfig=dict(double_z=False, z_channels=256, resolution=256,
                in_channels=3, out_ch=3, ch=128, ch_mult=[1, 1, 2, 2, 4, 4],
                num_res_blocks=2, attn_resolutions=[8], dropout=0.),
            latent_shape=[8, 8, 256], code_shape=[8, 8, self.sparsity_level],
            shared_codebook=True, restart_unused_codes=True)
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        state = payload['state_dict']
        filtered = {}
        for key, value in state.items():
            if key.startswith(('encoder.', 'decoder.', 'quant_conv.', 'post_quant_conv.')):
                filtered[key] = value
            elif key.startswith('pre_bottleneck.'):
                filtered['quant_conv.' + key.removeprefix('pre_bottleneck.')] = value
            elif key.startswith('post_bottleneck.'):
                filtered['post_quant_conv.' + key.removeprefix('post_bottleneck.')] = value
        missing, unexpected = stage1.load_state_dict(filtered, strict=False)
        if unexpected or any(not key.startswith('quantizer.') for key in missing):
            raise RuntimeError(f'LASER stage-1 mismatch: {missing}, {unexpected}')
        self.encoder, self.decoder = stage1.encoder, stage1.decoder
        self.quant_conv, self.post_quant_conv = stage1.quant_conv, stage1.post_quant_conv
        dictionary_key = 'quantizer.dictionary' if 'quantizer.dictionary' in state else 'bottleneck.dictionary'
        self.register_buffer('dictionary', F.normalize(state[dictionary_key].float(), dim=0))
        self.register_buffer('coeff_bins', torch.empty(0) if coeff_vocab_size is None else
                             torch.linspace(-coeff_max, coeff_max, self.coeff_vocab_size))
        scales = torch.tensor(coeff_scales, dtype=torch.float32)
        if scales.shape != (self.sparsity_level,) or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError('LASER requires a positive finite coefficient scale per pair')
        self.register_buffer('coeff_scales', scales)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def sparse_targets(self, atoms, coeffs, *, temp=.01125, stochastic=True,
                       compact=True, hard=False):
        if self.coeff_vocab_size == 0:
            raise ValueError('Continuous coefficients do not have categorical targets')
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError('Coefficient temperature must be positive and finite')
        atoms, coeffs = atoms.long(), coeffs.float()
        ids = (coeffs[..., None] - self.coeff_bins).abs().argmin(-1)
        probabilities = (-(coeffs[..., None] - self.coeff_bins).square() / temp).softmax(-1)
        if hard:
            probabilities = F.one_hot(ids, self.coeff_vocab_size).float()
        elif stochastic:
            ids = torch.multinomial(probabilities.reshape(-1, self.coeff_vocab_size), 1).reshape_as(ids)
        tokens = torch.empty(*atoms.shape[:-1], 2 * self.sparsity_level,
                             device=atoms.device, dtype=torch.long)
        tokens[..., 0::2], tokens[..., 1::2] = atoms, ids + self.num_atoms
        if compact:
            return tokens, (atoms, probabilities)
        targets = probabilities.new_zeros(*tokens.shape, self.vocab_size)
        targets[..., 0::2, :self.num_atoms].scatter_(-1, atoms[..., None], 1.)
        targets[..., 1::2, self.num_atoms:] = probabilities
        return tokens, targets

    @torch.no_grad()
    def decode_tokens(self, tokens):
        if self.coeff_vocab_size == 0:
            raise ValueError('Use continuous sparse codes to decode continuous coefficients')
        atoms = tokens[..., 0::2].long()
        ids = tokens[..., 1::2].long() - self.num_atoms
        coefficients = self.coeff_bins[ids] * self.coeff_scales
        latent = (self.dictionary.T[atoms] * coefficients[..., None]).sum(-2)
        return self.decoder(self.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())).clamp(-1., 1.)
