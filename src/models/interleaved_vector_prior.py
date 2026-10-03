"""Shared, interleaved pair decoding for the existing vector likelihood.

This mixin adapts a *fresh* VectorCompoundModel. Its spatial body, frozen
physical embeddings, classifiers, and finite-vector objective stay intact.
The existing depth and coefficient transformer blocks become one causal
stack. Every completed pair supplies its atom, physical coefficient, and
contribution explicitly, rather than only a compressed coefficient state.

The baseline class is supplied by the caller so experiment runtimes remain
isolated. No live model or training configuration is patched on import.
"""
from contextlib import contextmanager
from copy import deepcopy
from inspect import unwrap

import torch
from torch import nn
from torch.nn import functional as F


class InterleavedVectorPriorMixin:
    def configure_interleaved_decoder(self):
        """Call once, immediately after constructing the baseline instance."""
        if getattr(self, "interleaved_vector_decoder", False):
            raise ValueError("Interleaved decoder already configured")
        if self.micro_transformer_layers <= 0 or not self.depth_specific_coeff_heads:
            raise ValueError("Expected a coefficient micro-transformer and depth-specific heads")
        if self.config.embd_pdrop != 0:
            raise ValueError("Shared candidate decoding requires zero embedding dropout")
        self.config = deepcopy(self.config)
        width = int(self.config.embed_dim)
        dimension = int(self.config.input_embed_dim)
        depth = int(self.block_size[-1])
        # Retain both stacks' existing initialized blocks and parameter budget.
        # The coefficient stack is no longer separately evaluated per depth.
        self.head_transformer.blocks.extend(self.coeff_micro_transformer.blocks)
        # Keep the constructor configuration intact for checkpoint round trips:
        # constructing this class will concatenate the two stacks once again.
        self.shared_pair_layers = len(self.head_transformer.blocks)
        self.vector_decoder_mode = "shared_interleaved_physical_pairs_v1"
        self._pair_dropout_type = type(self.head_transformer.blocks[0].mlp[-1])
        self._pair_attention_type = type(self.head_transformer.blocks[0].attn)
        if not hasattr(self.head_transformer.blocks[0].mlp[-1], "active"):
            raise ValueError("Expected the baseline's shared candidate dropout")
        for layer, block in enumerate(self.head_transformer.blocks):
            if block.attn.attn_drop.p != 0:
                raise ValueError("Attention dropout needs shared attention masks")
            block.attn.resid_drop.name = f"pair.{layer}.attention"
            block.mlp[-1].name = f"pair.{layer}.mlp"
        del self.coeff_micro_transformer
        del self.coeff_micro_pos
        self.micro_transformer_layers = 0
        # coeff_atom_proj remains the physical dictionary-vector projection.
        # No learned coefficient IDs or trainable dictionary are introduced.
        reference = self.head_mlp.weight
        self.pair_features = nn.Linear(2 * dimension + 1, width, bias=False,
                                      device=reference.device, dtype=reference.dtype)
        self.pos_emb_d = nn.Parameter(torch.empty(1, 2 * depth, width,
                                      device=reference.device, dtype=reference.dtype))
        nn.init.normal_(self.pos_emb_d, mean=0., std=.02)
        self.interleaved_vector_decoder = True

    def _pair_event_inputs(self, spatial_rows, atoms, coefficient_ids, model_aux):
        """Inputs: spatial, atom0, pair0, atom1, pair1, ..., atom(D-1)."""
        depth = int(self.block_size[-1])
        if atoms.shape != coefficient_ids.shape or atoms.shape[-1] != depth:
            raise ValueError("Expected matching complete atom/coefficient sequences")
        vectors = model_aux.dictionary.t()[atoms]
        physical = model_aux.coeff_bins[coefficient_ids] * model_aux.coeff_scales
        contributions = vectors * physical[..., None]
        partial = contributions.cumsum(-2)
        before = torch.cat((torch.zeros_like(partial[:, :1]), partial[:, :-1]), -2)
        selected_atom = self.coeff_atom_proj(vectors) + self.head_mlp(before)
        completed_pair = self.pair_features(torch.cat(
            (vectors, physical[..., None], contributions), dim=-1)) + self.head_mlp(partial)
        shifted = torch.stack((selected_atom, completed_pair), -2).flatten(-3, -2)
        inputs = torch.cat((spatial_rows[:, None], shifted[:, :-1]), -2)
        return inputs + self.pos_emb_d

    @contextmanager
    def _candidate_dropout(self, masks, site_ids, site_count, depth):
        # The complete eight-event stack shares a mask for every site, layer,
        # event, and channel across candidate suffixes and replay chunks.
        saved, attention_saved = [], []
        for module in self.head_transformer.modules():
            if isinstance(module, self._pair_dropout_type):
                saved.append((module, module.active))
                module.active = (masks, site_ids, site_count) if masks is not None else None
            elif isinstance(module, self._pair_attention_type):
                attention_saved.append((module, getattr(module, "_candidate_scoring_active", False)))
                module._candidate_scoring_active = True
        try:
            yield
        finally:
            for module, active in saved:
                module.active = active
            for module, active in attention_saved:
                module._candidate_scoring_active = active

    def _pair_hidden(self, spatial_rows, atoms, coefficient_ids, model_aux, *,
                     masks=None, site_ids=None, site_count=None):
        if site_ids is None:
            site_ids = torch.arange(len(atoms), device=atoms.device)
            site_count = len(atoms)
        inputs = self._pair_event_inputs(spatial_rows, atoms, coefficient_ids, model_aux)
        with self._candidate_dropout(masks, site_ids, site_count, atoms.shape[-1]):
            return self.head_transformer(inputs)

    def _candidate_logits(self, spatial_rows, atoms, coefficient_ids, model_aux, *,
                          masks=None, site_ids=None, site_count=None):
        hidden = self._pair_hidden(spatial_rows, atoms, coefficient_ids, model_aux,
            masks=masks, site_ids=site_ids, site_count=site_count)
        atom_logits = self.classifier(hidden[:, ::2])
        for depth in range(1, atoms.shape[-1]):
            atom_logits[:, depth].scatter_(1, atoms[:, :depth], -torch.inf)
        return atom_logits, self.classify_coefficients(hidden[:, 1::2])

    def _score_grouped_prefixes(self, spatial_rows, atoms, coefficient_ids, model_aux, *,
                                masks, site_ids, site_count):
        """Reuse classifier distributions only for identical causal prefixes."""
        rows = torch.arange(len(atoms), device=atoms.device)
        depth = atoms.shape[-1]
        cardinality = self.num_atoms * self.coeff_vocab_size
        if max(len(atoms), int(site_count)) * cardinality >= 2**63:
            raise ValueError("Prefix keys overflow int64")

        def groups(keys):
            unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            representatives = torch.full((len(unique),), len(atoms), dtype=torch.long,
                                         device=atoms.device)
            representatives.scatter_reduce_(0, inverse, rows, reduce="amin", include_self=True)
            return inverse, representatives

        hidden = self._pair_hidden(spatial_rows, atoms, coefficient_ids, model_aux,
            masks=masks, site_ids=site_ids, site_count=site_count)
        _, prefix, counts = torch.unique_consecutive(site_ids, return_inverse=True,
                                                     return_counts=True)
        representatives = counts.cumsum(0) - counts
        packed_pairs = atoms * self.coeff_vocab_size + coefficient_ids
        scores = []
        for d in range(depth):
            atom_logits = self.classifier(hidden[representatives, 2 * d])
            if d:
                atom_logits.scatter_(1, atoms[representatives, :d], -torch.inf)
            atom_score = F.log_softmax(atom_logits, -1)[prefix, atoms[:, d]]
            coefficient_groups, coefficient_rows = groups(prefix * self.num_atoms + atoms[:, d])
            coefficient_logits = self.classify_coefficients(
                hidden[coefficient_rows, 2 * d + 1], depth_index=d)
            coefficient_score = F.log_softmax(coefficient_logits, -1)[
                coefficient_groups, coefficient_ids[:, d]]
            scores.append(atom_score + coefficient_score)
            if d + 1 < depth:
                prefix, representatives = groups(prefix * cardinality + packed_pairs[:, d])
        return torch.stack(scores, -1).sum(-1)

    def coefficient_logits(self, hidden, atom_vectors, depth_index=None):
        raise RuntimeError("Use the shared decoder's coefficient event after entering the selected atom")

    @torch.no_grad()
    def cached_pair_event(self, packed, model_aux, cond, sample_loc, amp=False):
        """One causal atom or coefficient event, called in raster/event order."""
        h, w, event = sample_loc
        batch, height, width, depth = packed.shape
        if not 0 <= event < 2 * depth:
            raise ValueError("Pair event out of range")
        site = h * width + w
        history = packed.reshape(batch, -1, depth)[:, :site + 1]
        with torch.amp.autocast("cuda", enabled=amp):
            if event == 0:
                if cond is None:
                    cond = torch.zeros(batch, 1, device=packed.device, dtype=torch.long)
                cond = cond.reshape(batch, 1)
                vectors = self.embed_with_model_aux(history, model_aux).sum(-2)
                embedded = self.input_mlp(vectors) + self.pos_emb_hw[:, :site + 1]
                start = self.cond_emb(cond) + self.pos_emb_cond[:, :1]
                inputs = self.embed_drop(torch.cat((start, embedded[:, :-1]), 1))
                if self._cache["spatial_ctx_hw"] is None:
                    context = self.body_transformer.cached_forward(inputs)[:, -1:]
                else:
                    context = self.body_transformer.cached_forward(inputs[:, -1:])
                self._cache["spatial_ctx_hw"] = context
                self.head_transformer.init_cache()
            if self._cache["spatial_ctx_hw"] is None:
                raise RuntimeError("Call the first atom event before later pair events")
            atoms, coefficients = self.unpack(history[:, site])
            inputs = self._pair_event_inputs(self._cache["spatial_ctx_hw"][:, 0],
                atoms, coefficients, model_aux)
            return self.head_transformer.cached_forward(inputs[:, event:event + 1]).reshape(batch, -1)

    @torch.no_grad()
    def cached_head_output(self, packed, model_aux, cond, sample_loc, amp=False):
        h, w, depth = sample_loc
        return self.cached_pair_event(packed, model_aux, cond, (h, w, 2 * depth), amp=amp)

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, temperature=1.,
                        atom_top_k=16384, atom_top_p=.92, coeff_top_p=.92,
                        atom_temperature=None, coeff_temperature=None, amp=True):
        # Obtain the exact baseline sampler from the inherited method's module;
        # this preserves its filtering and random draw semantics.
        baseline_method = super().sample_compound
        sample_field = unwrap(baseline_method.__func__).__globals__["sample_from_logits"]
        height, width, depth = self.block_size
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, height, width, depth, device=device, dtype=torch.long)
        coefficients = torch.full_like(atoms, self.coeff_vocab_size // 2)
        packed = atoms * self.coeff_vocab_size + coefficients
        atom_temperature = temperature if atom_temperature is None else atom_temperature
        coeff_temperature = temperature if coeff_temperature is None else coeff_temperature
        self.init_cache()
        try:
            for h in range(height):
                for w in range(width):
                    for d in range(depth):
                        hidden = self.cached_pair_event(packed, model_aux, cond, (h, w, 2 * d), amp)
                        logits = self.classifier(hidden)
                        if d:
                            logits.scatter_(1, atoms[:, h, w, :d], -torch.inf)
                        atom = sample_field(logits, temperature=float(atom_temperature),
                            top_k=min(atom_top_k, self.num_atoms), top_p=atom_top_p)
                        atoms[:, h, w, d] = atom
                        packed[:, h, w, d] = atom * self.coeff_vocab_size + coefficients[:, h, w, d]
                        hidden = self.cached_pair_event(packed, model_aux, cond, (h, w, 2 * d + 1), amp)
                        logits = self.classify_coefficients(hidden, depth_index=d)
                        coefficient = sample_field(logits, temperature=float(coeff_temperature),
                            top_k=self.coeff_vocab_size, top_p=coeff_top_p)
                        coefficients[:, h, w, d] = coefficient
                        packed[:, h, w, d] = atom * self.coeff_vocab_size + coefficient
        finally:
            self.init_cache()
        return atoms, coefficients
