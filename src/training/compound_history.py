"""Opt-in, checkpoint-preserving compound coefficient history decoder."""
import math

import torch

from src.models.compound_coefficient_decoder import CompoundCoefficientHistoryDecoder
from src.training.rqtransformer import CompoundLaserRQTransformer


class HistoryConditionedCompoundTransformer(CompoundLaserRQTransformer):
    def init_cache(self):
        super().init_cache()
        if hasattr(self, 'coefficient_history_decoder'):
            self.coefficient_history_decoder.reset_cache()
        self._coefficient_sample_context = None

    def cached_head_output(self, packed, model_aux, cond, sample_loc, amp=True):
        hidden = super().cached_head_output(packed, model_aux, cond, sample_loc, amp=amp)
        self._coefficient_sample_context = (packed, model_aux, sample_loc)
        return hidden

    def coefficient_history_inputs(self, atoms, coeff_ids, model_aux):
        """Shift completed pairs globally; shift physical sums within each site."""
        batch = atoms.shape[0]
        pairs = self.compound_pair_embeddings(model_aux, atoms, coeff_ids).reshape(batch, -1, self.config.input_embed_dim)
        previous = torch.cat((torch.zeros_like(pairs[:, :1]), pairs[:, :-1]), dim=1)
        physical = model_aux.compound_embeddings(atoms, coeff_ids)
        # Shift *before* summing: cumsum-minus-current has numerical leakage.
        shifted = torch.cat((torch.zeros_like(physical[..., :1, :]), physical[..., :-1, :]), dim=-2)
        prefix = shifted.cumsum(dim=-2).reshape_as(previous)
        return previous, prefix

    def refine_coefficient_hidden(self, hidden, atom_vectors):
        original = super().refine_coefficient_hidden(hidden, atom_vectors)
        decoder = self.coefficient_history_decoder
        if self._teacher_atoms is not None:
            batch = hidden.shape[0]
            previous, prefix = self.coefficient_history_inputs(
                self._teacher_atoms, self._teacher_coeff_ids, self._model_aux,
            )
            residual = decoder(hidden.reshape(batch, -1, hidden.shape[-1]),
                               atom_vectors.reshape(batch, -1, atom_vectors.shape[-1]),
                               previous, prefix).reshape_as(hidden)
        else:
            if self._coefficient_sample_context is None:
                raise RuntimeError('cached_head_output must precede coefficient refinement')
            packed, aux, (h, w, depth) = self._coefficient_sample_context
            batch, _, width, depths = packed.shape
            event = (h * width + w) * depths + depth
            atoms, coeffs = self.unpack(packed)
            if event:
                prev_atom = atoms.reshape(batch, -1)[:, event - 1]
                prev_coeff = coeffs.reshape(batch, -1)[:, event - 1]
                previous = self.compound_pair_embeddings(
                    aux, prev_atom, prev_coeff, depth_index=(event - 1) % depths,
                )
            else:
                previous = torch.zeros_like(atom_vectors)
            if depth:
                vectors = aux.dictionary.T[atoms[:, h, w, :depth]]
                values = aux.coeff_bins[coeffs[:, h, w, :depth]] * aux.coeff_scales[:depth]
                # Same FP32 cumulative reduction as the teacher-forced path.
                prefix = (vectors * values[..., None]).cumsum(dim=1)[:, -1]
            else:
                prefix = torch.zeros_like(atom_vectors)
            residual = decoder(hidden[:, None], atom_vectors[:, None],
                               previous[:, None], prefix[:, None], event=event)[:, 0]
            self._coefficient_sample_context = None
        return original + residual


def attach_coefficient_history_decoder(model, *, width=512, layers=2, heads=8, dropout=0.1):
    """Append new parameters without reordering or replacing checkpoint weights.

    The model must use full-pair autoregression. This adapter retains the old
    local coefficient conditioner and adds an initially zero global decoder
    residual. Thus both logits and the old optimizer parameter order survive.
    """
    if type(model) is not CompoundLaserRQTransformer or not model.pair_autoregressive:
        raise ValueError('history decoder requires a plain full-pair compound transformer')
    if model.causal_prefix_state or model.contribution_head is not None:
        raise ValueError('history experiment requires the standard compound objective')
    model.__class__ = HistoryConditionedCompoundTransformer
    model.coefficient_history_decoder = CompoundCoefficientHistoryDecoder(
        model.config.embed_dim, model.config.input_embed_dim, width=width,
        layers=layers, heads=heads, dropout=dropout, max_events=math.prod(model.block_size),
    ).to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    model.coefficient_history_decoder.train(model.training)
    model._coefficient_sample_context = None
    return model
