"""Checkpoint-compatible compound tokens with explicit full pair history.

Both predictions receive an unpooled causal history of all earlier complete
(atom, coefficient) events. The trained spatial/local decoder remains intact;
a small, gated attention path introduces this additional conditioning.
"""
import copy
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from src.models.physical_pair_scalar_prior import PhysicalPairScalarRQTransformer
from src.models.rqtransformer.transformers import sample_from_logits


class CompoundHistory(nn.Module):
    def __init__(self, trained_block):
        super().__init__()
        self.blocks = nn.ModuleList([copy.deepcopy(trained_block)])

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value

    def cached_forward(self, value):
        for block in self.blocks:
            value = block.cached_forward(value)
        return value

    def init_cache(self):
        for block in self.blocks:
            block.init_cache()


class PhysicalCompoundRQTransformer(PhysicalPairScalarRQTransformer):
    @classmethod
    def from_scalar(cls, model, *, initial_gate=.01):
        if type(model) is not PhysicalPairScalarRQTransformer or model.block_size_cond != 1:
            raise ValueError('Expected a physical scalar prior with one conditioning label')
        model.__class__ = cls
        model.scalar_block_size = model.block_size
        model.compound_depths = model.block_size[-1] // 2
        model.block_size = torch.Size((*model.block_size[:2], model.compound_depths))
        model.events = math.prod(model.block_size)
        model.decoder_type = 'gated_full_history_physical_compound_v2'
        parameter = next(model.parameters())
        # Clone one trained block, without altering any existing module or RNG.
        model.full_history = CompoundHistory(model.head_transformer.blocks[-1])
        model.history_norm = nn.LayerNorm(model.config.embed_dim).to(
            device=parameter.device, dtype=parameter.dtype)
        model.history_gates = nn.Parameter(parameter.new_full((2, model.config.embed_dim), initial_gate))
        model.full_history.train(model.training)
        model.history_norm.train(model.training)
        model.init_cache()
        return model

    def unpack(self, packed, aux):
        if tuple(packed.shape[1:]) != tuple(self.block_size):
            raise ValueError('Full raster/depth sequence of compound tokens required')
        return (packed.div(aux.coeff_vocab_size, rounding_mode='floor').long(),
                packed.remainder(aux.coeff_vocab_size).long())

    def scalar_tokens(self, atoms, coefficients):
        return torch.stack((atoms, coefficients+self.num_atoms), -1).flatten(-2)

    def pair_embedding(self, atoms, coefficients, aux, *, depth=None):
        vectors = aux.dictionary.T[atoms]
        scale = aux.coeff_scales if depth is None else aux.coeff_scales[depth]
        physical = aux.coeff_bins[coefficients] * scale
        contribution = vectors * physical[..., None]
        identity = self.pair_features(torch.cat((vectors, physical[..., None], contribution), -1))
        projected = self.input_mlp(contribution)
        shape = (identity.shape[-1],)
        return (F.layer_norm(identity, shape) + F.layer_norm(projected, shape)) / math.sqrt(2.)

    def positions(self):
        height, width, depths = self.block_size
        return (self.pos_emb_hw[:, :, None] + self.pos_emb_d[:, None, ::2]).reshape(
            1, height*width*depths, self.config.embed_dim)

    def forward(self, packed, model_aux=None, cond=None, amp=False):
        # Retain the existing on-the-fly encoder interface. Internally each
        # alternating scalar pair becomes one complete compound event.
        if tuple(packed.shape[1:]) == tuple(self.scalar_block_size):
            packed = packed[..., 0::2]*model_aux.coeff_vocab_size + packed[..., 1::2]-self.num_atoms
        atoms, coefficients = self.unpack(packed, model_aux)
        tokens = self.scalar_tokens(atoms, coefficients)
        with torch.amp.autocast('cuda') if amp else nullcontext():
            batch, height, width, depth = tokens.shape
            sequence = tokens.reshape(batch, height*width, depth)
            _, _, _, contributions = self._physical_pairs(sequence, model_aux)
            embedded = self.input_mlp(contributions.sum(-2)) + self.pos_emb_hw[:, :height*width]
            if cond is None:
                cond = torch.zeros(batch, 1, device=packed.device, dtype=torch.long)
            start = self.cond_emb(cond.reshape(batch, 1)) + self.pos_emb_cond
            body = self.body_transformer(self.embed_drop(torch.cat((start, embedded[:, :-1]), 1)))
            inputs = self._pair_inputs(body.reshape(-1, body.shape[-1]),
                sequence.reshape(-1, depth), model_aux)
            native = self.head_transformer(inputs).reshape(batch, height, width, depth, -1)
            pairs = self.pair_embedding(atoms, coefficients, model_aux).reshape(batch, self.events, -1)
            previous = torch.cat((start, pairs[:, :-1]), 1)
            # This cache/sequence never resets between spatial sites or depths.
            history = self.history_norm(self.full_history(previous+self.positions()))
            history = history.reshape(batch, height, width, self.compound_depths, -1)
            residual = (history[..., None, :] * self.history_gates).flatten(-3, -2)
            output = self.classify_head_outputs(native+residual)
            for index in range(1, self.compound_depths):
                output['atom_logits'][..., index, :].scatter_(-1, atoms[..., :index], -torch.inf)
            if self.training and getattr(self, 'compound_geometry_config', None):
                from src.training.physical_compound_geometry import candidate_geometry_outputs
                output.update(candidate_geometry_outputs(self, inputs, history, output, atoms, model_aux))
            return output

    def init_cache(self):
        super().init_cache()
        if hasattr(self, 'full_history'):
            self.full_history.init_cache()
        self._next_atom = self._next_coefficient = 0
        self._compound_history = None

    def _native_cached_hidden(self, atoms, coefficients, aux, cond, event, coefficient):
        batch = atoms.shape[0]
        site, d = divmod(event, self.compound_depths)
        h, w = divmod(site, self.block_size[1])
        tokens = self.scalar_tokens(atoms, coefficients)[:, :h+1]
        scalar_depth = 2*self.compound_depths
        history = tokens.reshape(batch, -1, scalar_depth)[:, :site+1]
        if d == 0 and not coefficient:
            _, _, _, contributions = self._physical_pairs(history, aux)
            embedded = self.input_mlp(contributions.sum(-2)) + self.pos_emb_hw[:, :site+1]
            if cond is None:
                cond = torch.zeros(batch, 1, device=atoms.device, dtype=torch.long)
            start = self.cond_emb(cond.reshape(batch, 1)) + self.pos_emb_cond
            values = self.embed_drop(torch.cat((start, embedded[:, :-1]), 1))
            values = values if self._cache['spatial_ctx_hw'] is None else values[:, -1:]
            self._cache['spatial_ctx_hw'] = self.body_transformer.cached_forward(values)[:, -1:]
            self.head_transformer.init_cache()
        context = self._cache['spatial_ctx_hw']
        inputs = self._pair_inputs(context[:, 0], history[:, site], aux)
        scalar_event = 2*d+int(coefficient)
        return self.head_transformer.cached_forward(inputs[:, scalar_event:scalar_event+1])

    @torch.no_grad()
    def cached_atom_logits(self, atoms, coefficients, aux, cond, event, amp=True):
        if self.training or event != self._next_atom or self._next_atom != self._next_coefficient:
            raise ValueError('Atom decoder must consume completed compound tokens in order in eval')
        if not 0 <= event < self.events:
            raise ValueError('Compound event outside context')
        batch = atoms.shape[0]
        with torch.amp.autocast('cuda', enabled=amp):
            if event:
                a = atoms.reshape(batch, -1)[:, event-1]
                c = coefficients.reshape(batch, -1)[:, event-1]
                previous = self.pair_embedding(a, c, aux, depth=(event-1)%self.compound_depths)[:, None]
            else:
                if cond is None:
                    cond = torch.zeros(batch, 1, device=atoms.device, dtype=torch.long)
                previous = self.cond_emb(cond.reshape(batch, 1)) + self.pos_emb_cond
            history = self.history_norm(self.full_history.cached_forward(
                previous+self.positions()[:, event:event+1]))
            native = self._native_cached_hidden(atoms, coefficients, aux, cond, event, False)
            normalized = self.classifier.layer_norm(native+history*self.history_gates[0])[:, 0]
            linear = self.classifier.linear
            logits = F.linear(normalized, linear.weight[:self.num_atoms],
                None if linear.bias is None else linear.bias[:self.num_atoms])
        site, depth = divmod(event, self.compound_depths)
        if depth:
            logits.scatter_(1, atoms.reshape(batch, -1, self.compound_depths)[:, site, :depth], -torch.inf)
        self._compound_history = history
        self._next_atom += 1
        return logits

    @torch.no_grad()
    def cached_coefficient_logits(self, atoms, coefficients, current_atom, aux, event, amp=True):
        if self.training or event != self._next_coefficient or self._next_atom != event+1:
            raise ValueError('Predict the current compound atom before its coefficient')
        # Selected atom is known; its current coefficient is still excluded.
        current_atoms = atoms.clone()
        current_atoms.reshape(atoms.shape[0], -1)[:, event] = current_atom
        with torch.amp.autocast('cuda', enabled=amp):
            native = self._native_cached_hidden(current_atoms, coefficients, aux, None, event, True)
            normalized = self.classifier.layer_norm(
                native+self._compound_history*self.history_gates[1])[:, 0]
            linear = self.classifier.linear
            logits = F.linear(normalized, linear.weight[self.num_atoms:],
                None if linear.bias is None else linear.bias[self.num_atoms:])
        self._next_coefficient += 1
        self._compound_history = None
        return logits

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, *, atom_temperature=.9,
            atom_top_k=0, atom_top_p=.9, coeff_temperature=1., coeff_top_k=0,
            coeff_top_p=.85, amp=True):
        atoms = torch.zeros(batch_size, *self.block_size,
            device=next(self.parameters()).device, dtype=torch.long)
        coefficients = torch.full_like(atoms, model_aux.coeff_vocab_size//2)
        self.init_cache()
        try:
            for event in range(self.events):
                logits = self.cached_atom_logits(atoms, coefficients, model_aux, cond, event, amp=amp)
                atom = sample_from_logits(logits, temperature=atom_temperature,
                    top_k=atom_top_k or None, top_p=atom_top_p)
                logits = self.cached_coefficient_logits(atoms, coefficients, atom, model_aux, event, amp=amp)
                coefficient = sample_from_logits(logits, temperature=coeff_temperature,
                    top_k=coeff_top_k or None, top_p=coeff_top_p)
                atoms.reshape(batch_size, -1)[:, event] = atom
                coefficients.reshape(batch_size, -1)[:, event] = coefficient
        finally:
            self.init_cache()
        return atoms, coefficients

    @torch.no_grad()
    def sample_sparse(self, batch_size, model_aux, cond=None, **kwargs):
        atoms, coefficients = self.sample_compound(batch_size, model_aux, cond, **kwargs)
        return self.scalar_tokens(atoms, coefficients)
