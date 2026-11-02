"""Offset scalar IDs with the unchanged four-position pair RQ backbone.

The public stream is a0, N+c0, a1, N+c1, ... . Internally each completed
pair advances the RQ depth transformer once; its coefficient is conditioned
on its selected atom through the existing small coefficient decoder.
"""
import torch
from torch.nn import functional as F

from src.training.compound_history import HistoryConditionedCompoundTransformer
from src.models.rqtransformer.transformers import sample_from_logits


class OffsetPairRQTransformer(HistoryConditionedCompoundTransformer):
    @classmethod
    def from_pair_backbone(cls, model):
        if type(model) is not HistoryConditionedCompoundTransformer:
            raise ValueError("offset IDs require the full-pair coefficient-history backbone")
        if not model.pair_autoregressive or model.causal_prefix_state:
            raise ValueError("offset IDs require completed-pair autoregression")
        model.__class__ = cls
        model.token_vocabulary_size = model.num_atoms + model.coeff_vocab_size
        model.token_block_size = (*model.block_size[:2], 2 * model.block_size[-1])
        model._offset_coefficients = None
        model._offset_hidden = None
        model._offset_location = None
        model.decoder_type = "offset_scalar_ids_pair_rq_backbone_v1"
        return model

    def fields(self, tokens):
        if tokens.dtype != torch.long or tokens.shape[-1] != self.token_block_size[-1]:
            raise ValueError("expected alternating atom and offset coefficient IDs")
        atoms, coefficients = tokens[..., 0::2], tokens[..., 1::2] - self.num_atoms
        if ((atoms < 0).any() or (atoms >= self.num_atoms).any() or
                (coefficients < 0).any() or (coefficients >= self.coeff_vocab_size).any()):
            raise ValueError("token outside its atom or coefficient vocabulary")
        return atoms, coefficients

    def unpack(self, atoms):
        # The parent receives the atom field as its shape-only carrier. Its
        # embedding hooks read the coefficient field separately, without
        # ever multiplying an atom ID by the coefficient vocabulary size.
        if self._offset_coefficients is None:
            raise RuntimeError("coefficient field is unavailable outside a forward call")
        coefficients = self._offset_coefficients.reshape(atoms.shape[0], -1, self.block_size[-1])
        sites = atoms.numel() // (atoms.shape[0] * self.block_size[-1])
        return atoms, coefficients[:, :sites].reshape_as(atoms)

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        atoms, self._offset_coefficients = self.fields(tokens)
        try:
            return super().forward(atoms, model_aux=model_aux, cond=cond, amp=amp)
        finally:
            self._offset_coefficients = None

    def init_cache(self):
        super().init_cache()
        self._offset_hidden = self._offset_location = None
        self._offset_coefficients = None

    @torch.no_grad()
    def cached_forward(self, tokens, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        h, w, scalar_event = sample_loc
        depth, coefficient = divmod(scalar_event, 2)
        atoms, coefficients = self.fields(tokens)
        location = (h, w, depth)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if not coefficient:
                if self._offset_hidden is not None:
                    raise ValueError("predict the preceding atom's coefficient first")
                self._offset_coefficients = coefficients
                hidden = self.cached_head_output(atoms, model_aux, cond, location, amp=amp)
                self._offset_hidden, self._offset_location = hidden, location
                logits = self.classifier(hidden)
                if depth and self.mask_seen_atoms_training:
                    logits.scatter_(1, atoms[:, h, w, :depth], -torch.inf)
                return F.pad(logits, (0, self.coeff_vocab_size), value=-torch.inf)
            if self._offset_hidden is None or self._offset_location != location:
                raise ValueError("predict the current atom before its coefficient")
            atom_vectors = model_aux.dictionary.T[atoms[:, h, w, depth]]
            refined = self.refine_coefficient_hidden(self._offset_hidden, atom_vectors)
            logits = self.classify_coefficients(refined, depth_index=depth)
            self._offset_hidden = self._offset_location = self._offset_coefficients = None
            return F.pad(logits, (self.num_atoms, 0), value=-torch.inf)

    @torch.no_grad()
    def sample_sparse(self, batch_size, model_aux, cond=None, *, atom_temperature=1.,
                      atom_top_k=0, atom_top_p=.92, coeff_temperature=1.,
                      coeff_top_k=0, coeff_top_p=.92, amp=True):
        if self.training:
            raise ValueError("sampling requires eval mode")
        tokens = torch.zeros(batch_size, *self.token_block_size,
                             device=next(self.parameters()).device, dtype=torch.long)
        tokens[..., 1::2] = self.num_atoms + self.coeff_vocab_size // 2
        self.init_cache()
        try:
            for h in range(self.block_size[0]):
                for w in range(self.block_size[1]):
                    for event in range(self.token_block_size[-1]):
                        logits = self.cached_forward(tokens, model_aux, cond, amp,
                                                     sample_loc=(h, w, event))
                        if event % 2:
                            logits, offset = logits[:, self.num_atoms:], self.num_atoms
                            temperature, top_k, top_p = coeff_temperature, coeff_top_k, coeff_top_p
                        else:
                            logits, offset = logits[:, :self.num_atoms], 0
                            temperature, top_k, top_p = atom_temperature, atom_top_k, atom_top_p
                        tokens[:, h, w, event] = offset + sample_from_logits(
                            logits, temperature=temperature, top_k=top_k or None, top_p=top_p)
        finally:
            self.init_cache()
        return tokens

    @torch.no_grad()
    def joint_coefficient_scorer(self, model_aux):
        """Read the pending atom event; candidate branches cannot commit history."""
        if self._offset_hidden is None or self._coefficient_sample_context is None:
            raise ValueError('start the current atom event before scoring complete pairs')
        packed, aux, (h, w, depth) = self._coefficient_sample_context
        if aux is not model_aux:
            raise ValueError('candidate scorer uses a different frozen tokenizer')
        hidden = self._offset_hidden
        atoms, coefficients = self.unpack(packed)
        event = (h * self.block_size[1] + w) * self.block_size[-1] + depth
        zeros = hidden.new_zeros((len(hidden), model_aux.dictionary.shape[0]))
        previous = self._cached_previous_pair(aux, atoms, coefficients, event) if event else zeros
        prefix = self._cached_prefix(aux, atoms[:, h, w, :depth],
                                     coefficients[:, h, w, :depth]) if depth else zeros
        def score(rows, candidates):
            count = candidates.shape[1]
            vectors = model_aux.dictionary.T[candidates]
            return self.coefficient_candidate_logits(
                hidden[rows, None].expand(-1, count, -1), vectors,
                previous[rows, None], prefix[rows, None],
                depth_index=depth, event=event, rows=rows)
        return score

    @torch.no_grad()
    def sample_joint_sparse(self, batch_size, model_aux, cond=None, *, top_k=4096,
                            temperature=1., top_p=1., candidate_chunk=16384, amp=True,
                            generator=None, return_stats=False):
        """One draw over complete c*atom contributions, with certified global K."""
        from src.joint_pair_topk import sample_joint_pair_topk
        if any(module.training for module in self.modules()):
            raise ValueError('joint sampling requires evaluation mode')
        device = next(self.parameters()).device
        tokens = torch.zeros(batch_size, *self.token_block_size, device=device, dtype=torch.long)
        tokens[..., 1::2] = self.num_atoms + self.coeff_vocab_size // 2
        evaluated = torch.zeros((), device=device, dtype=torch.long)
        maximum = torch.zeros_like(evaluated)
        mass = torch.zeros((), device=device)
        self.init_cache()
        try:
            for h in range(self.block_size[0]):
                for w in range(self.block_size[1]):
                    for depth in range(self.block_size[-1]):
                        event = 2 * depth
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                            atom_logits = self.cached_forward(tokens, model_aux, cond, amp,
                                sample_loc=(h, w, event))[:, :self.num_atoms]
                            scorer = self.joint_coefficient_scorer(model_aux)
                            atoms, coefficients, stats = sample_joint_pair_topk(
                                atom_logits, scorer, top_k=top_k, temperature=temperature,
                                top_p=top_p, candidate_chunk=candidate_chunk, generator=generator)
                            tokens[:, h, w, event] = atoms
                            # Commit exactly the selected atom's history once.
                            self.cached_forward(tokens, model_aux, cond, amp,
                                                sample_loc=(h, w, event+1))
                            tokens[:, h, w, event+1] = coefficients + self.num_atoms
                        evaluated += stats['evaluated_atoms'].sum()
                        maximum = torch.maximum(maximum, stats['evaluated_atoms'].max())
                        mass += stats['retained_mass'].sum()
        finally:
            self.init_cache()
        if return_stats:
            pairs = batch_size * self.block_size[0] * self.block_size[1] * self.block_size[-1]
            return tokens, dict(certified_global_top_k=True, pairs=pairs,
                mean_evaluated_atoms=float(evaluated) / pairs, maximum_evaluated_atoms=int(maximum),
                mean_retained_untempered_mass=float(mass) / pairs)
        return tokens

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, *,
                        causal_prefix_sampling="predicted", joint_pair_top_k=0,
                        joint_pair_temperature=1., joint_pair_top_p=1.,
                        joint_pair_candidate_chunk=16384, **kwargs):
        # Existing preview/FID callers accept separated fields. Keep their
        # interface while all generated IDs use the explicit scalar stream.
        if joint_pair_top_k:
            amp = kwargs.pop('amp', True)
            if kwargs:
                raise ValueError('independent field filters cannot accompany joint pair sampling')
            return self.fields(self.sample_joint_sparse(batch_size, model_aux, cond,
                top_k=joint_pair_top_k, temperature=joint_pair_temperature,
                top_p=joint_pair_top_p, candidate_chunk=joint_pair_candidate_chunk, amp=amp))
        return self.fields(self.sample_sparse(batch_size, model_aux, cond, **kwargs))
