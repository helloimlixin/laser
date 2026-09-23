"""Stacked causal atom/value decoders for complete sparse-pair histories.

Adapts DCTransformer's stacked conditional decoder factorization (Nash et al.,
2021, section 3.2) to a fixed raster/depth ordering of LASER sparse events.
Every decoder attends to all earlier events; no spatial pooling or depth reset.
"""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width, heads, dropout, max_events, *, pair_attention=False):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by attention heads')
        self.heads, self.max_events = heads, max_events
        self.pair_attention = pair_attention
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)
        self.reset_cache()

    def reset_cache(self):
        self._keys = self._values = None
        self._length = 0

    def forward(self, x, *, cached=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, width // self.heads,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        if cached:
            if self.training or torch.is_grad_enabled() or length != 1:
                raise ValueError('cached decoding requires one event in eval without gradients')
            if self._length >= self.max_events:
                raise ValueError('decoder context exhausted')
            if self._keys is None:
                shape = (batch, self.heads, self.max_events, width // self.heads)
                self._keys, self._values = k.new_empty(shape), v.new_empty(shape)
            if self._keys.shape[0] != batch or self._keys.dtype != k.dtype:
                raise ValueError('reset the cache before changing batch size or precision')
            self._keys[:, :, self._length:self._length+1] = k
            self._values[:, :, self._length:self._length+1] = v
            self._length += 1
            k, v = self._keys[:, :, :self._length], self._values[:, :, :self._length]
        if self.pair_attention:
            if cached or length != 2:
                raise ValueError('field fusion requires exactly two uncached tokens')
            y = causal_pair_attention(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=not cached)
        y = y.transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.projection(y))
        return x + self.dropout(self.mlp(self.norm2(x)))


def causal_pair_attention(q, k, v):
    """Exact two-position causal softmax attention, without a fused kernel.

    Position zero has one visible value. At position one, the two-way softmax
    is a sigmoid of the score difference. FP32 score/weight arithmetic avoids
    the large-batch BF16 fused-attention backward failure on the target runtime.
    """
    score_difference = (q[..., 1, :].float() * (
        k[..., 0, :].float() - k[..., 1, :].float())).sum(-1) / math.sqrt(q.shape[-1])
    first_weight = score_difference.sigmoid().unsqueeze(-1)
    second = (first_weight * v[..., 0, :].float()
              + (1. - first_weight) * v[..., 1, :].float()).to(v.dtype)
    return torch.stack((v[..., 0, :], second), dim=-2)


class FullHistoryCompoundTransformer(nn.Module):
    def __init__(self, *, num_atoms=16384, coeff_vocab_size=2048,
                 block_size=(8, 8, 4), width=1024, heads=16,
                 atom_layers=24, coefficient_layers=8, dropout=.1,
                 normalize_coefficient_inputs=False,
                 dictionary_atom_conditioning=False):
        super().__init__()
        if min(atom_layers, coefficient_layers) < 1:
            raise ValueError('both fields require a causal decoder')
        self.num_atoms, self.coeff_vocab_size = num_atoms, coeff_vocab_size
        self.block_size = tuple(block_size)
        self.events, self.width = math.prod(block_size), width
        self.normalize_coefficient_inputs = normalize_coefficient_inputs
        self.dictionary_atom_conditioning = dictionary_atom_conditioning
        if dictionary_atom_conditioning and not normalize_coefficient_inputs:
            raise ValueError('dictionary attention requires normalized conditioning fields')
        self.atom_embedding = nn.Embedding(num_atoms, width)
        self.coefficient_embedding = nn.Embedding(coeff_vocab_size, width)
        self.physical_projection = nn.Linear(256, width, bias=False)
        self.prefix_projection = nn.Linear(256, width, bias=False)
        self.position = nn.Parameter(torch.empty(1, self.events, width))
        self.start = nn.Parameter(torch.empty(1, 1, width))
        self.atom_decoder = nn.ModuleList([
            CausalBlock(width, heads, dropout, self.events) for _ in range(atom_layers)])
        self.coefficient_decoder = nn.ModuleList([
            CausalBlock(width, heads, dropout, self.events) for _ in range(coefficient_layers)])
        self.atom_classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_atoms))
        self.coefficient_classifiers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, coeff_vocab_size))
            for _ in range(self.block_size[-1])])
        if normalize_coefficient_inputs:
            # Normalize each conditioning field before fusion. A deep residual
            # stream can otherwise overwhelm the small learned atom embedding.
            # These are hidden-feature norms; physical coefficients stay raw.
            self.coefficient_input_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        if dictionary_atom_conditioning:
            self.coefficient_atom_projection = nn.Linear(256, width, bias=False)
            self.coefficient_input_norms.append(nn.LayerNorm(width))
            # Each event has its own [history, current atom] attention pair.
            # This local fusion precedes, and does not replace, full-history
            # coefficient attention. No current coefficient is an input.
            self.coefficient_field_position = nn.Parameter(torch.empty(1, 2, width))
            self.coefficient_field_decoder = nn.ModuleList([
                CausalBlock(width, heads, dropout, 2, pair_attention=True) for _ in range(2)])
        self.apply(self._initialize)
        nn.init.normal_(self.position, std=.02)
        nn.init.normal_(self.start, std=.02)
        if dictionary_atom_conditioning:
            nn.init.normal_(self.coefficient_field_position, std=.02)
        self.init_cache()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def init_cache(self):
        for block in [*self.atom_decoder, *self.coefficient_decoder]:
            block.reset_cache()
        self._next_atom = self._next_coefficient = 0
        self._current_hidden = self._previous_pair = None

    def pair_embedding(self, atoms, coefficients, aux):
        # Coefficient values are in raw units, including in the physical input.
        physical = aux.dictionary.T[atoms] * aux.coeff_bins[coefficients][..., None]
        return (self.atom_embedding(atoms) + self.coefficient_embedding(coefficients)
                + self.physical_projection(physical))

    def _context(self, amp, device):
        return torch.autocast('cuda', dtype=torch.bfloat16) if amp and device.type == 'cuda' else nullcontext()

    def current_atom_features(self, current_atom, aux=None):
        identity = self.atom_embedding(current_atom)
        geometry = None
        if self.dictionary_atom_conditioning:
            geometry = self.coefficient_atom_projection(aux.dictionary.T[current_atom])
        return identity, geometry

    def coefficient_inputs(self, hidden, current_atom, previous, prefix, aux=None):
        identity, geometry = self.current_atom_features(current_atom, aux)
        fields = (hidden, identity, previous, self.prefix_projection(prefix))
        if self.dictionary_atom_conditioning:
            history, identity, previous, prefix, geometry = [
                norm(value) for norm, value in zip(self.coefficient_input_norms, (*fields, geometry))]
            context = (history + previous + prefix) / math.sqrt(3.)
            atom = (identity + geometry) / math.sqrt(2.)
            pair = torch.stack((context, atom), dim=-2).reshape(-1, 2, self.width)
            pair = pair + self.coefficient_field_position
            for block in self.coefficient_field_decoder:
                pair = block(pair)
            return (context + pair[:, -1].reshape_as(context)) / math.sqrt(2.)
        if self.normalize_coefficient_inputs:
            return sum(norm(value) for norm, value in zip(self.coefficient_input_norms, fields)) * .5
        return sum(fields)

    @staticmethod
    def _mask_used(logits, atoms):
        logits = logits.clone()
        for depth in range(1, atoms.shape[-1]):
            logits[..., depth, :].scatter_(-1, atoms[..., :depth], -torch.inf)
        return logits

    def forward(self, packed, model_aux=None, cond=None, amp=False):
        if tuple(packed.shape[1:]) != self.block_size:
            raise ValueError('full raster/depth sequence is required')
        if cond is not None and torch.count_nonzero(cond):
            raise ValueError('this Church model is unconditional')
        batch = packed.shape[0]
        atoms, coefficients = packed.long() // self.coeff_vocab_size, packed.long() % self.coeff_vocab_size
        with self._context(amp, packed.device):
            pairs = self.pair_embedding(atoms, coefficients, model_aux).reshape(batch, self.events, self.width)
            previous = torch.cat((self.start.expand(batch, -1, -1), pairs[:, :-1]), dim=1)
            hidden = previous + self.position
            for block in self.atom_decoder:
                hidden = block(hidden)
            atom_logits = self.atom_classifier(hidden).reshape(*packed.shape, self.num_atoms)
            atom_logits = self._mask_used(atom_logits, atoms)
            physical = model_aux.dictionary.T[atoms] * model_aux.coeff_bins[coefficients][..., None]
            shifted = torch.cat((torch.zeros_like(physical[..., :1, :]), physical[..., :-1, :]), dim=-2)
            prefix = shifted.cumsum(-2).reshape(batch, self.events, 256)
            values = self.coefficient_inputs(hidden, atoms.reshape(batch, self.events), previous, prefix, model_aux)
            for block in self.coefficient_decoder:
                values = block(values)
            values = values.reshape(*packed.shape, self.width)
            coefficient_logits = torch.stack([
                classifier(values[..., depth, :])
                for depth, classifier in enumerate(self.coefficient_classifiers)], dim=-2)
        return {'atom_logits': atom_logits, 'coeff_logits': coefficient_logits}

    @torch.no_grad()
    def cached_atom_logits(self, atoms, coefficients, aux, event, *, amp=False):
        if self.training or event != self._next_atom or self._next_atom != self._next_coefficient:
            raise ValueError('atom decoder must consume completed pairs in order in eval')
        if not 0 <= event < self.events:
            raise ValueError('event outside sequence')
        batch = atoms.shape[0]
        with self._context(amp, atoms.device):
            if event:
                a, c = atoms.reshape(batch, -1)[:, event-1], coefficients.reshape(batch, -1)[:, event-1]
                previous = self.pair_embedding(a, c, aux)[:, None]
            else:
                previous = self.start.expand(batch, -1, -1)
            hidden = previous + self.position[:, event:event+1]
            for block in self.atom_decoder:
                hidden = block(hidden, cached=True)
            logits = self.atom_classifier(hidden[:, 0])
        site, depth = divmod(event, self.block_size[-1])
        if depth:
            logits.scatter_(-1, atoms.reshape(batch, -1, self.block_size[-1])[:, site, :depth], -torch.inf)
        self._current_hidden, self._previous_pair = hidden, previous
        self._next_atom += 1
        return logits

    @torch.no_grad()
    def cached_coefficient_logits(self, atoms, coefficients, current_atom, aux, event, *, amp=False):
        if self.training or event != self._next_coefficient or self._next_atom != event+1:
            raise ValueError('current atom must be predicted before its coefficient')
        batch, depths = atoms.shape[0], self.block_size[-1]
        site, depth = divmod(event, depths)
        if depth:
            a = atoms.reshape(batch, -1, depths)[:, site, :depth]
            c = coefficients.reshape(batch, -1, depths)[:, site, :depth]
            prefix = (aux.dictionary.T[a] * aux.coeff_bins[c][..., None]).cumsum(1)[:, -1]
        else:
            prefix = aux.dictionary.new_zeros(batch, 256)
        with self._context(amp, atoms.device):
            values = self.coefficient_inputs(self._current_hidden, current_atom[:, None],
                                             self._previous_pair, prefix[:, None], aux)
            for block in self.coefficient_decoder:
                values = block(values, cached=True)
            logits = self.coefficient_classifiers[depth](values[:, 0])
        self._next_coefficient += 1
        self._current_hidden = self._previous_pair = None
        return logits

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, atom_temperature=1.,
                        atom_top_k=250, atom_top_p=1., coeff_temperature=1.,
                        coeff_top_k=0, coeff_top_p=1., amp=True):
        from src.models.rqtransformer.transformers import sample_from_logits
        if self.training:
            raise ValueError('sampling requires eval mode')
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, *self.block_size, device=device, dtype=torch.long)
        coefficients = torch.zeros_like(atoms)
        self.init_cache()
        try:
            for event in range(self.events):
                logits = self.cached_atom_logits(atoms, coefficients, model_aux, event, amp=amp)
                atom = sample_from_logits(logits, temperature=atom_temperature,
                    top_k=min(atom_top_k or self.num_atoms, self.num_atoms), top_p=atom_top_p)
                logits = self.cached_coefficient_logits(atoms, coefficients, atom, model_aux, event, amp=amp)
                coefficient = sample_from_logits(logits, temperature=coeff_temperature,
                    top_k=coeff_top_k or self.coeff_vocab_size, top_p=coeff_top_p)
                atoms.reshape(batch_size, -1)[:, event] = atom
                coefficients.reshape(batch_size, -1)[:, event] = coefficient
        finally:
            self.init_cache()
        return atoms, coefficients
