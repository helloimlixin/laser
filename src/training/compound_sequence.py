"""Stacked causal support/value sequence decoders for full-refit OMP pairs.

DCTransformer-inspired factorization: support sees only completed pair history;
the value stage additionally sees the current selected support and support state.
The existing RQ spatial/depth backbone and local atom/value conditioner remain.
"""
import math

import torch

from src.models.compound_coefficient_decoder import CompoundCoefficientHistoryDecoder
from src.training.compound_history import HistoryConditionedCompoundTransformer
from src.training.rqtransformer import CompoundLaserRQTransformer


class SequenceCompoundTransformer(CompoundLaserRQTransformer):
    coefficient_history_inputs = HistoryConditionedCompoundTransformer.coefficient_history_inputs

    def init_cache(self):
        super().init_cache()
        for name in ('support_sequence_decoder', 'coefficient_sequence_decoder'):
            if hasattr(self, name):
                getattr(self, name).reset_cache()
        self._sequence_context = None

    def classify_head_outputs(self, hidden):
        batch = hidden.shape[0]
        previous, prefix = self.coefficient_history_inputs(
            self._teacher_atoms, self._teacher_coeff_ids, self._model_aux)
        self._sequence_context = (previous, prefix, None)
        try:
            support = hidden + self.support_sequence_decoder(
                hidden.reshape(batch, -1, hidden.shape[-1]), None,
                previous, prefix).reshape_as(hidden)
            return super().classify_head_outputs(support)
        finally:
            self._sequence_context = None

    def cached_head_output(self, packed, model_aux, cond, sample_loc, amp=True):
        hidden = super().cached_head_output(packed, model_aux, cond, sample_loc, amp=amp)
        h, w, depth = sample_loc
        batch, _, width, depths = packed.shape
        event = (h * width + w) * depths + depth
        atoms, coeffs = self.unpack(packed)
        if event:
            previous = self.compound_pair_embeddings(
                model_aux, atoms.reshape(batch, -1)[:, event - 1],
                coeffs.reshape(batch, -1)[:, event - 1],
                depth_index=(event - 1) % depths)
        else:
            previous = model_aux.dictionary.new_zeros(batch, self.config.input_embed_dim)
        if depth:
            vectors = model_aux.dictionary.T[atoms[:, h, w, :depth]]
            values = model_aux.coeff_bins[coeffs[:, h, w, :depth]] * model_aux.coeff_scales[:depth]
            prefix = (vectors * values[..., None]).cumsum(dim=1)[:, -1]
        else:
            prefix = torch.zeros_like(previous)
        self._sequence_context = (previous[:, None], prefix[:, None], event)
        # The backbone's internal autocast scope ends before this method resumes.
        with torch.autocast(device_type=hidden.device.type, enabled=amp and hidden.is_cuda):
            residual = self.support_sequence_decoder(
                hidden[:, None], None, previous[:, None], prefix[:, None], event=event)[:, 0]
        return hidden + residual

    def refine_coefficient_hidden(self, hidden, atom_vectors):
        if self._sequence_context is None:
            raise RuntimeError('support sequence must precede coefficient sequence')
        original = super().refine_coefficient_hidden(hidden, atom_vectors)
        previous, prefix, event = self._sequence_context
        batch = hidden.shape[0]
        residual = self.coefficient_sequence_decoder(
            hidden.reshape(batch, -1, hidden.shape[-1]),
            atom_vectors.reshape(batch, -1, atom_vectors.shape[-1]),
            previous, prefix, event=event).reshape_as(hidden)
        if event is not None:
            self._sequence_context = None
        return original + residual


def attach_pair_sequence_decoders(model, *, width=512, layers=2, heads=8, dropout=.1):
    """Attach two independently parameterized stacks, initialized for a fresh run."""
    if type(model) is not CompoundLaserRQTransformer or not model.pair_autoregressive:
        raise ValueError('sequence decoders require a plain full-pair compound transformer')
    if (model.causal_prefix_state or model.contribution_head is not None
            or getattr(model, 'geometry_top_k', 0)):
        raise ValueError('sequence decoders require the standard joint likelihood objective')
    model.__class__ = SequenceCompoundTransformer
    parameter = next(model.parameters())
    for name, conditioned in [('support_sequence_decoder', False),
                              ('coefficient_sequence_decoder', True)]:
        decoder = CompoundCoefficientHistoryDecoder(
            model.config.embed_dim, model.config.input_embed_dim,
            width=width, layers=layers, heads=heads, dropout=dropout,
            max_events=math.prod(model.block_size), atom_conditioned=conditioned,
            zero_output=False).to(device=parameter.device, dtype=parameter.dtype)
        decoder.train(model.training)
        setattr(model, name, decoder)
    model.init_cache()
    return model
