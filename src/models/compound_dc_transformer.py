"""DCTransformer architecture adapted to frozen LASER atom/coefficient pairs.

The 8x8 spatial positions are known, so there is no position predictor. Four
depth-major planes form four target chunks. Each chunk has a bidirectional
encoder of the physical latent reconstructed from *earlier planes only*, and
stacked causal atom/value decoders with cross-attention into that encoder.
Completed pairs from the preceding chunk provide a short causal overlap.

This is an adaptation of Nash et al. (2021), not their DCT representation or
an exact reproduction. In particular, OMP depth is not DCT frequency order.
"""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F


class ReZeroPARBlock(nn.Module):
    """Pre-LN self-attention, optional cross-attention, then several FFNs.

    Each residual branch has its own zero-initialized learned scalar. Branch
    weights consequently have zero (but present) gradients at initialization;
    their gradients become active after the corresponding gate moves.
    """

    def __init__(self, width, heads, feedforward_layers, dropout, max_events,
                 *, cross_attention=False, causal=True):
        super().__init__()
        if width % heads or feedforward_layers < 1:
            raise ValueError('invalid attention width/heads or FFN count')
        self.width, self.heads, self.max_events = width, heads, max_events
        self.causal, self.cross_attention = causal, cross_attention
        self.self_norm = nn.LayerNorm(width)
        self.self_qkv = nn.Linear(width, 3 * width)
        self.self_projection = nn.Linear(width, width)
        self.self_gate = nn.Parameter(torch.zeros(()))
        if cross_attention:
            self.cross_query_norm = nn.LayerNorm(width)
            self.cross_memory_norm = nn.LayerNorm(width)
            self.cross_query = nn.Linear(width, width)
            self.cross_key_value = nn.Linear(width, 2 * width)
            self.cross_projection = nn.Linear(width, width)
            self.cross_gate = nn.Parameter(torch.zeros(()))
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(feedforward_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
            for _ in range(feedforward_layers)])
        self.ffn_gates = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in self.ffns])
        self.dropout = nn.Dropout(dropout)
        self.reset_cache()

    def reset_cache(self):
        self._keys = self._values = self._cross_keys = self._cross_values = None
        self._length = 0

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.width // self.heads).transpose(1, 2)

    def _merge(self, x):
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.width)

    def forward(self, x, memory=None, *, cached=False):
        if cached and (self.training or torch.is_grad_enabled() or x.shape[1] != 1 or not self.causal):
            raise ValueError('cache requires one causal event in eval without gradients')
        q, k, v = [self._heads(y) for y in self.self_qkv(self.self_norm(x)).chunk(3, -1)]
        if cached:
            if self._length >= self.max_events:
                raise ValueError('chunk context exhausted')
            if self._keys is None:
                shape = (x.shape[0], self.heads, self.max_events, self.width // self.heads)
                self._keys, self._values = k.new_empty(shape), v.new_empty(shape)
            if self._keys.shape[0] != x.shape[0] or self._keys.dtype != k.dtype:
                raise ValueError('reset cache before changing batch size or precision')
            self._keys[:, :, self._length:self._length + 1] = k
            self._values[:, :, self._length:self._length + 1] = v
            self._length += 1
            k, v = self._keys[:, :, :self._length], self._values[:, :, :self._length]
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and not cached)
        x = x + self.self_gate * self.dropout(self.self_projection(self._merge(attended)))
        if self.cross_attention:
            if memory is None:
                raise ValueError('cross-attention requires encoded prefix memory')
            q = self._heads(self.cross_query(self.cross_query_norm(x)))
            if cached and self._cross_keys is not None:
                k, v = self._cross_keys, self._cross_values
            else:
                k, v = [self._heads(y) for y in self.cross_key_value(self.cross_memory_norm(memory)).chunk(2, -1)]
                if cached:
                    self._cross_keys, self._cross_values = k, v
            attended = F.scaled_dot_product_attention(q, k, v)
            x = x + self.cross_gate * self.dropout(self.cross_projection(self._merge(attended)))
        for norm, ffn, gate in zip(self.ffn_norms, self.ffns, self.ffn_gates):
            x = x + gate * self.dropout(ffn(norm(x)))
        return x


def sample_categorical(logits, *, temperature=1., top_k=0, top_p=None):
    """FP32 categorical sampling; None disables nucleus filtering entirely."""
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if top_k is not None and (isinstance(top_k, bool) or int(top_k) != top_k or top_k < 0):
        raise ValueError('top_k must be a nonnegative integer or None')
    if top_p is not None and (not math.isfinite(float(top_p)) or not 0 < top_p <= 1):
        raise ValueError('top_p must be in (0, 1] or None')
    logits = logits.float() / float(temperature)
    if top_k and top_k < logits.shape[-1]:
        threshold = logits.topk(int(top_k), -1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    probabilities = logits.softmax(-1)
    if top_p is not None and top_p < 1:
        ordered, indices = probabilities.sort(-1, descending=True)
        remove = ordered.cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, 0.)
        probabilities = torch.zeros_like(probabilities).scatter(-1, indices, ordered)
    return torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1).reshape(probabilities.shape[:-1])


class DCCompoundTransformer(nn.Module):
    """Fixed-position, depth-major LASER adaptation of stacked DC decoders."""

    def __init__(self, *, num_atoms=16384, coeff_vocab_size=2048,
                 block_size=(8, 8, 4), dictionary_dim=256, width=896, heads=14,
                 encoder_ffn_layers=(2, 2, 2, 2),
                 atom_ffn_layers=(2, 2, 2, 4),
                 coefficient_ffn_layers=(2, 2, 2, 2, 2, 7),
                 overlap=8, dropout=.1):
        super().__init__()
        if len(block_size) != 3 or min(block_size) < 1:
            raise ValueError('block_size must contain height, width and depths')
        self.block_size = tuple(block_size)
        self.sites = self.block_size[0] * self.block_size[1]
        self.depths = self.block_size[2]
        self.events = self.sites * self.depths
        if width % heads or not 0 <= overlap < self.sites or num_atoms < self.depths:
            raise ValueError('invalid width, overlap or distinct-atom vocabulary')
        if not all((encoder_ffn_layers, atom_ffn_layers, coefficient_ffn_layers)):
            raise ValueError('encoder and both decoders require blocks')
        self.num_atoms, self.coeff_vocab_size = num_atoms, coeff_vocab_size
        self.dictionary_dim, self.width, self.overlap = dictionary_dim, width, overlap
        self.amp_dtype = torch.bfloat16
        self.atom_embedding = nn.Embedding(num_atoms, width)
        self.coefficient_embedding = nn.Embedding(coeff_vocab_size, width)
        self.physical_projection = nn.Linear(dictionary_dim, width, bias=False)
        self.pair_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.site_embedding = nn.Embedding(self.sites, width)
        self.depth_embedding = nn.Embedding(self.depths, width)
        self.chunk_position = nn.Parameter(torch.empty(1, self.sites + overlap, width))
        self.start = nn.Parameter(torch.empty(1, 1, width))
        self.prefix_projection = nn.Linear(dictionary_dim, width, bias=False)
        self.prefix_site_embedding = nn.Embedding(self.sites, width)
        self.prefix_depth_embedding = nn.Embedding(self.depths, width)
        self.prefix_encoder = nn.ModuleList([
            ReZeroPARBlock(width, heads, count, dropout, self.sites, causal=False)
            for count in encoder_ffn_layers])
        self.prefix_output_norm = nn.LayerNorm(width)
        self.atom_decoder = nn.ModuleList([
            ReZeroPARBlock(width, heads, count, dropout, self.sites + overlap, cross_attention=True)
            for count in atom_ffn_layers])
        self.coefficient_decoder = nn.ModuleList([
            ReZeroPARBlock(width, heads, count, dropout, self.sites + overlap, cross_attention=True)
            for count in coefficient_ffn_layers])
        self.coefficient_atom_projection = nn.Linear(dictionary_dim, width, bias=False)
        # Separate norms prevent a learned history residual from overwhelming
        # current-atom identity/geometry or the local partial reconstruction.
        self.coefficient_field_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.atom_classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_atoms))
        self.coefficient_classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, coeff_vocab_size))
        self.apply(self._initialize)
        nn.init.normal_(self.chunk_position, std=.02)
        nn.init.normal_(self.start, std=.02)
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

    def _context(self, amp, device):
        return torch.autocast('cuda', dtype=self.amp_dtype) if amp and device.type == 'cuda' else nullcontext()

    def _validate_aux(self, aux):
        if (tuple(aux.dictionary.shape) != (self.dictionary_dim, self.num_atoms)
                or tuple(aux.coeff_bins.shape) != (self.coeff_vocab_size,)
                or tuple(aux.coeff_scales.shape) != (self.depths,)):
            raise ValueError('frozen dictionary, bins or physical depth scales mismatch')

    def depth_major(self, tensor):
        """[B,H,W,D,...] -> [B,D*H*W,...], with no value/atom separation."""
        if tuple(tensor.shape[1:4]) != self.block_size:
            raise ValueError('expected complete spatial/depth grid')
        axes = (0, 3, 1, 2, *range(4, tensor.ndim))
        return tensor.permute(axes).reshape(tensor.shape[0], self.events, *tensor.shape[4:])

    def raster_depth(self, tensor):
        """[B,D*H*W,...] -> [B,H,W,D,...]."""
        if tensor.shape[1] != self.events:
            raise ValueError('expected complete depth-major sequence')
        shaped = tensor.reshape(tensor.shape[0], self.depths, *self.block_size[:2], *tensor.shape[2:])
        return shaped.permute(0, 2, 3, 1, *range(4, shaped.ndim)).contiguous()

    def pair_physical(self, atoms, coefficients, depths, aux):
        values = aux.coeff_bins[coefficients.long()] * aux.coeff_scales[depths]
        return aux.dictionary.T[atoms.long()] * values[..., None]

    def pair_embedding(self, atoms, coefficients, depths, aux):
        physical = self.pair_physical(atoms, coefficients, depths, aux)
        fields = (self.atom_embedding(atoms.long()), self.coefficient_embedding(coefficients.long()),
                  self.physical_projection(physical))
        return sum(norm(field) for norm, field in zip(self.pair_norms, fields)) / math.sqrt(3.)

    def prefix_latent(self, atoms, coefficients, aux, chunk):
        """Physical 8x8 reconstruction from completed planes, never this plane."""
        if not 0 <= chunk < self.depths:
            raise ValueError('chunk outside depth planes')
        if chunk == 0:
            return aux.dictionary.new_zeros(atoms.shape[0], self.sites, self.dictionary_dim)
        a = atoms.reshape(atoms.shape[0], self.sites, self.depths)[..., :chunk]
        c = coefficients.reshape_as(atoms).reshape(atoms.shape[0], self.sites, self.depths)[..., :chunk]
        depths = torch.arange(chunk, device=atoms.device)
        return self.pair_physical(a, c, depths, aux).sum(-2)

    def encode_prefix(self, atoms, coefficients, aux, chunk):
        latent = self.prefix_latent(atoms, coefficients, aux, chunk)
        sites = torch.arange(self.sites, device=atoms.device)
        chunk_id = torch.tensor(chunk, device=atoms.device)
        encoded = (self.prefix_projection(latent) + self.prefix_site_embedding(sites)[None]
                   + self.prefix_depth_embedding(chunk_id)[None, None])
        for block in self.prefix_encoder:
            encoded = block(encoded)
        return self.prefix_output_norm(encoded)

    def coefficient_inputs(self, hidden, current_atoms, prefix_at_site, aux):
        fields = (hidden, self.atom_embedding(current_atoms),
                  self.coefficient_atom_projection(aux.dictionary.T[current_atoms]), prefix_at_site)
        history, identity, geometry, local = [norm(x) for norm, x in zip(self.coefficient_field_norms, fields)]
        atom = (identity + geometry) / math.sqrt(2.)
        return (history + atom + local) / math.sqrt(3.)

    def _positions(self, indices, offset=0):
        return (self.site_embedding(indices % self.sites)[None]
                + self.depth_embedding(indices // self.sites)[None]
                + self.chunk_position[:, offset:offset + len(indices)])

    def _previous_inputs(self, flat_atoms, flat_coefficients, indices, aux):
        previous_indices = (indices - 1).clamp_min(0)
        previous = self.pair_embedding(flat_atoms[:, previous_indices], flat_coefficients[:, previous_indices],
                                       previous_indices // self.sites, aux)
        # The placeholder at event zero is completely replaced; it carries
        # neither its atom nor its coefficient into the first prediction.
        return torch.where((indices == 0)[None, :, None], self.start.expand_as(previous), previous)

    def _mask_atoms(self, logits, atoms):
        masked = logits.clone()
        for depth in range(1, self.depths):
            masked[..., depth, :].scatter_(-1, atoms[..., :depth].long(), -torch.inf)
        return masked

    def forward(self, packed, model_aux=None, cond=None, amp=False):
        if tuple(packed.shape[1:]) != self.block_size:
            raise ValueError('all spatial/depth pairs are required')
        if cond is not None and torch.count_nonzero(cond):
            raise ValueError('Church model is unconditional')
        self._validate_aux(model_aux)
        atoms, coefficients = packed.long() // self.coeff_vocab_size, packed.long() % self.coeff_vocab_size
        flat_atoms, flat_coefficients = self.depth_major(atoms), self.depth_major(coefficients)
        atom_outputs, coefficient_outputs = [], []
        with self._context(amp, packed.device):
            for chunk in range(self.depths):
                begin = max(0, chunk * self.sites - self.overlap)
                end = (chunk + 1) * self.sites
                indices = torch.arange(begin, end, device=packed.device)
                prefix = self.encode_prefix(atoms, coefficients, model_aux, chunk)
                hidden = self._previous_inputs(flat_atoms, flat_coefficients, indices, model_aux) + self._positions(indices)
                for block in self.atom_decoder:
                    hidden = block(hidden, prefix)
                values = self.coefficient_inputs(hidden, flat_atoms[:, indices], prefix[:, indices % self.sites], model_aux)
                for block in self.coefficient_decoder:
                    values = block(values, prefix)
                # Overlap only supplies context. Every real pair receives one
                # atom and one coefficient prediction, in its own depth chunk.
                atom_outputs.append(self.atom_classifier(hidden[:, -self.sites:]))
                coefficient_outputs.append(self.coefficient_classifier(values[:, -self.sites:]))
            atom_logits = self.raster_depth(torch.cat(atom_outputs, 1))
            coefficient_logits = self.raster_depth(torch.cat(coefficient_outputs, 1))
        return dict(atom_logits=self._mask_atoms(atom_logits, atoms), coeff_logits=coefficient_logits)

    def init_cache(self):
        for block in [*self.prefix_encoder, *self.atom_decoder, *self.coefficient_decoder]:
            block.reset_cache()
        self._next_atom = self._next_coefficient = 0
        self._cached_prefix = self._cached_hidden = None
        self._chunk_begin = None

    def _cached_support(self, flat_atoms, flat_coefficients, aux, event):
        index = torch.tensor([event], device=flat_atoms.device)
        previous = self._previous_inputs(flat_atoms, flat_coefficients, index, aux)
        hidden = previous + self._positions(index, event - self._chunk_begin)
        for block in self.atom_decoder:
            hidden = block(hidden, self._cached_prefix, cached=True)
        return hidden

    def _cached_value(self, hidden, current_atom, aux, event):
        local = self._cached_prefix[:, event % self.sites:event % self.sites + 1]
        values = self.coefficient_inputs(hidden, current_atom[:, None], local, aux)
        for block in self.coefficient_decoder:
            values = block(values, self._cached_prefix, cached=True)
        return values

    @torch.no_grad()
    def cached_atom_logits(self, atoms, coefficients, aux, event, *, amp=False):
        """Consume one depth-major event after all prior pairs are completed."""
        if self.training or event != self._next_atom or event != self._next_coefficient:
            raise ValueError('atom and coefficient calls must alternate in depth-major order')
        if not 0 <= event < self.events:
            raise ValueError('event outside complete sequence')
        self._validate_aux(aux)
        flat_atoms, flat_coefficients = self.depth_major(atoms), self.depth_major(coefficients)
        chunk, site = divmod(event, self.sites)
        with self._context(amp, atoms.device):
            if site == 0:
                for block in [*self.atom_decoder, *self.coefficient_decoder]:
                    block.reset_cache()
                self._cached_prefix = self.encode_prefix(atoms, coefficients, aux, chunk)
                self._chunk_begin = max(0, event - self.overlap)
                for overlap_event in range(self._chunk_begin, event):
                    hidden = self._cached_support(flat_atoms, flat_coefficients, aux, overlap_event)
                    self._cached_value(hidden, flat_atoms[:, overlap_event], aux, overlap_event)
            hidden = self._cached_support(flat_atoms, flat_coefficients, aux, event)
            logits = self.atom_classifier(hidden[:, 0])
        if chunk:
            prior = atoms.reshape(atoms.shape[0], self.sites, self.depths)[:, site, :chunk]
            logits.scatter_(-1, prior.long(), -torch.inf)
        self._cached_hidden = hidden
        self._next_atom += 1
        return logits

    @torch.no_grad()
    def cached_coefficient_logits(self, atoms, coefficients, current_atom, aux, event, *, amp=False):
        if self.training or event != self._next_coefficient or self._next_atom != event + 1:
            raise ValueError('predict current atom before its coefficient')
        with self._context(amp, atoms.device):
            values = self._cached_value(self._cached_hidden, current_atom.long(), aux, event)
            logits = self.coefficient_classifier(values[:, 0])
        self._cached_hidden = None
        self._next_coefficient += 1
        return logits

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, atom_temperature=1.,
                        atom_top_k=0, atom_top_p=None, coeff_temperature=1.,
                        coeff_top_k=0, coeff_top_p=None, amp=True):
        if self.training:
            raise ValueError('sampling requires eval mode')
        if cond is not None and torch.count_nonzero(cond):
            raise ValueError('Church model is unconditional')
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, *self.block_size, device=device, dtype=torch.long)
        coefficients = torch.zeros_like(atoms)
        self.init_cache()
        try:
            for event in range(self.events):
                atom_logits = self.cached_atom_logits(atoms, coefficients, model_aux, event, amp=amp)
                atom = sample_categorical(atom_logits, temperature=atom_temperature, top_k=atom_top_k, top_p=atom_top_p)
                coefficient_logits = self.cached_coefficient_logits(atoms, coefficients, atom, model_aux, event, amp=amp)
                coefficient = sample_categorical(coefficient_logits, temperature=coeff_temperature,
                                                 top_k=coeff_top_k, top_p=coeff_top_p)
                chunk, site = divmod(event, self.sites)
                atoms.reshape(batch_size, self.sites, self.depths)[:, site, chunk] = atom
                coefficients.reshape(batch_size, self.sites, self.depths)[:, site, chunk] = coefficient
        finally:
            self.init_cache()
        return atoms, coefficients


# Descriptive alias for external callers; both refer to the same architecture.
CompoundDCTransformer = DCCompoundTransformer
