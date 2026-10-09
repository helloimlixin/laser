"""Checkpoint-preserving integration of atom and coefficient memory queries."""
from contextlib import nullcontext

import torch

from src.models.pair_memory_cross_attention import PairMemoryQueries
from src.training.rqtransformer import CompoundLaserRQTransformer


class PairMemoryCompoundTransformer(CompoundLaserRQTransformer):
    def init_cache(self):
        super().init_cache()
        if hasattr(self, 'pair_memory_queries'):
            self.pair_memory_queries.reset_cache()
        self._pair_memory_amp = False

    def memory_fields(self, aux, atoms, coefficients, *, depth_index=None):
        vectors = aux.dictionary.T[atoms.long()]
        embedding = self.coeff_token_embedding(coefficients.long())
        scale = aux.coeff_scales if depth_index is None else aux.coeff_scales[depth_index]
        scalar = (aux.coeff_bins[coefficients.long()] * scale)[..., None]
        return vectors, embedding, vectors * scalar, scalar

    def physical_prefix(self, contributions):
        shifted = torch.cat((torch.zeros_like(contributions[..., :1, :]), contributions[..., :-1, :]), dim=-2)
        return shifted.cumsum(dim=-2)

    def classify_head_outputs(self, head_outputs):
        atoms, coefficients = self._teacher_atoms, self._teacher_coeff_ids
        if atoms is None or coefficients is None:
            raise RuntimeError('teacher pairs required')
        batch, hidden_dim = head_outputs.shape[0], head_outputs.shape[-1]
        fields = self.memory_fields(self._model_aux, atoms, coefficients)
        prefix = self.physical_prefix(fields[2]).reshape(batch, -1, self.config.input_embed_dim)
        flat_fields = [value.reshape(batch, -1, value.shape[-1]) for value in fields]
        refined, residual = self.pair_memory_queries(
            head_outputs.reshape(batch, -1, hidden_dim), flat_fields[0], prefix, *flat_fields)
        refined, residual = refined.reshape_as(head_outputs), residual.reshape_as(head_outputs)
        coefficient_hidden = super().refine_coefficient_hidden(refined, fields[0]) + residual
        return dict(atom_logits=self.mask_seen_atoms(self.classifier(refined), atoms),
                    coeff_logits=self.classify_coefficients(coefficient_hidden))

    @torch.no_grad()
    def cached_head_output(self, packed, model_aux, cond, sample_loc, amp=True):
        hidden = super().cached_head_output(packed, model_aux, cond, sample_loc, amp=amp)
        h, w, depth = sample_loc
        batch, _, width, depths = packed.shape
        event = (h * width + w) * depths + depth
        atoms, coefficients = self.unpack(packed)
        if event:
            fields = self.memory_fields(model_aux, atoms.reshape(batch, -1)[:, event-1],
                                        coefficients.reshape(batch, -1)[:, event-1],
                                        depth_index=(event-1) % depths)
        else:
            vector = model_aux.dictionary.new_zeros(batch, self.config.input_embed_dim)
            fields = (vector, vector, vector, vector[:, :1])
        if depth:
            local = self.memory_fields(model_aux, atoms[:, h, w], coefficients[:, h, w])[2]
            prefix = local[:, :depth].sum(dim=1)
        else:
            prefix = torch.zeros_like(fields[0])
        context = torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp) if hidden.is_cuda else nullcontext()
        with context:
            memory = self.pair_memory_queries.encode_memory(*(value[:, None] for value in fields), event=event)
            refined, _ = self.pair_memory_queries.atom_query(hidden[:, None], prefix[:, None], memory, event=event)
        self._pair_memory_amp = amp
        return refined[:, 0]

    def refine_coefficient_hidden(self, hidden, atom_vectors):
        decoder = self.pair_memory_queries
        if decoder._pending is None:
            raise RuntimeError('cached atom query must precede coefficient prediction')
        query, prefix, memory, event = decoder._pending
        context = torch.autocast('cuda', dtype=torch.bfloat16, enabled=self._pair_memory_amp) if hidden.is_cuda else nullcontext()
        with context:
            original = super().refine_coefficient_hidden(hidden, atom_vectors)
            residual = decoder.coefficient_query(query, atom_vectors[:, None], prefix, memory, event=event)
        return original + residual[:, 0]


def attach_pair_memory_queries(model, *, width=512, heads=8, memory_layers=1,
                               query_layers=2, mode='cross'):
    if type(model) is not CompoundLaserRQTransformer or not model.pair_autoregressive:
        raise ValueError('pair memory requires a plain full-pair autoregressive compound transformer')
    if model.causal_prefix_state or model.contribution_head is not None or model.geometry_top_k:
        raise ValueError('pair memory trial requires the standard compound objective')
    model.__class__ = PairMemoryCompoundTransformer
    model.pair_memory_queries = PairMemoryQueries(
        model.config.embed_dim, model.config.input_embed_dim, model.block_size,
        width=width, heads=heads, memory_layers=memory_layers,
        query_layers=query_layers, mode=mode,
    ).to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    model.pair_memory_queries.train(model.training)
    model._pair_memory_amp = False
    return model
