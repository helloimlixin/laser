"""Matched five-event Church priors: coefficient pattern before/after support."""
import math

import torch
from torch import nn

from src.training.rqtransformer import CompoundLaserRQTransformer, support_first_objective
from src.church_ffhq_recipe import recipe_config
from src.church_support_pattern_training import pattern_targets
from src.models.rqtransformer.transformers import RQTransformer


def sample_field(logits, temperature=1., top_k=0, top_p=None):
    if temperature <= 0 or (top_p is not None and not 0 < top_p <= 1):
        raise ValueError('Invalid sampling temperature or probability cutoff')
    logits = logits.float() / temperature
    if top_k:
        values = logits.topk(min(top_k, logits.shape[-1])).values
        logits = logits.masked_fill(logits < values[..., -1:], -float('inf'))
    probabilities = logits.softmax(-1)
    if top_p is not None and top_p < 1:
        ordered, indices = probabilities.sort(descending=True)
        # CUDA cumsum is unsupported under strict deterministic execution.
        remove = ordered.cpu().cumsum(-1).to(probabilities.device) >= top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        probabilities = probabilities.masked_fill(remove.scatter(-1, indices, remove), 0.)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return torch.multinomial(probabilities, 1).squeeze(-1)


class ChurchPatternOrderRQTransformer(RQTransformer):
    """Same body/head/parameters for two causal factorizations of the same code.

    Transport has five fields; serialized complete-site integers still use the
    existing four-atom-plus-pattern codec. No target field is consumed before
    its decision. Pattern-first atom context uses signed, weighted prefixes.
    """
    def __init__(self, config, num_atoms, num_patterns, ordering):
        if ordering not in {'pattern-first', 'support-first'}:
            raise ValueError('Unknown field ordering')
        config = config.copy()
        config.cumsum_depth_ctx = False
        super().__init__(config)
        if self.block_size[-1] != 5:
            raise ValueError('Expected one pattern plus four atom events')
        self.num_atoms, self.coefficient_pattern_vocab_size = num_atoms, num_patterns
        self.ordering = ordering
        width, inputs = int(config.embed_dim), int(config.input_embed_dim)
        self.pattern_embedding = nn.Sequential(nn.Linear(4, inputs), nn.SiLU(), nn.Linear(inputs, inputs))
        self.pattern_classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_patterns))
        self._teacher_atoms = None

    def pack(self, atoms, ids):
        if atoms.shape[:-1] != ids.shape or atoms.shape[-1] != 4:
            raise ValueError('Expected four atoms and one pattern per site')
        fields = (ids[..., None], atoms) if self.ordering == 'pattern-first' else (atoms, ids[..., None])
        return torch.cat(fields, -1).long()

    def unpack(self, packed):
        if packed.shape[-1] != 5:
            raise ValueError('Expected five transport fields')
        if self.ordering == 'pattern-first':
            return packed[..., 1:].long(), packed[..., 0].long()
        return packed[..., :4].long(), packed[..., 4].long()

    def embed_with_model_aux(self, packed, aux):
        atoms, ids = self.unpack(packed)
        z = aux.coefficient_pattern_latents(atoms, ids)
        # The shared linear input projection is followed by a sum across fields
        # in RQTransformer. Both orders therefore use projected z + five biases.
        zero = torch.zeros_like(z)
        return torch.stack((zero, zero, zero, zero, z), -2)

    def embed_depth_with_model_aux(self, packed, aux):
        atoms, ids = self.unpack(packed)
        vectors = aux.dictionary.t()[atoms]
        if self.ordering == 'pattern-first':
            coefficients = aux.coefficient_patterns[ids]
            pattern = self.pattern_embedding(coefficients / aux.coeff_scales)
            prefix = torch.zeros_like(vectors[..., 0, :])
            contexts = [pattern]
            for depth in range(4):
                prefix = prefix + vectors[..., depth, :] * coefficients[..., depth, None]
                contexts.append(pattern + prefix)
        else:
            prefix = torch.zeros_like(vectors[..., 0, :])
            contexts = []
            for depth in range(4):
                prefix = prefix + vectors[..., depth, :]
                contexts.append(prefix)
            contexts.append(prefix)  # Last field is shifted out of this site.
        return torch.stack(contexts, -2)

    def classify_head_outputs(self, hidden):
        if self._teacher_atoms is None:
            raise RuntimeError('Teacher support was not set')
        if self.ordering == 'pattern-first':
            pattern_hidden, atom_hidden = hidden[..., 0, :], hidden[..., 1:, :]
        else:
            pattern_hidden, atom_hidden = hidden[..., 4, :], hidden[..., :4, :]
        atoms = CompoundLaserRQTransformer.mask_seen_atoms(self.classifier(atom_hidden), self._teacher_atoms)
        return {'atom_logits': atoms, 'pattern_logits': self.pattern_classifier(pattern_hidden)}

    def forward(self, packed, model_aux=None, cond=None, amp=False):
        self._teacher_atoms, _ = self.unpack(packed)
        try:
            return super().forward(packed, model_aux=model_aux, cond=cond, amp=amp)
        finally:
            self._teacher_atoms = None

    @torch.no_grad()
    def cached_hidden(self, packed, aux, sample_loc, cond=None, amp=True):
        h, w, event = sample_loc
        batch, _, width, fields = packed.shape
        site = h * width + w
        xs = packed.reshape(batch, -1, fields)[:, :site + 1]
        with torch.amp.autocast('cuda', enabled=amp):
            if cond is None:
                cond = torch.zeros(batch, self.block_size_cond, device=xs.device, dtype=torch.long)
            if event == 0:
                embeddings = self.input_mlp(self.embed_with_model_aux(xs, aux)).sum(-2)
                spatial = embeddings + self.pos_emb_hw[:, :site + 1]
                conditions = self.cond_emb(cond) + self.pos_emb_cond[:, :self.block_size_cond]
                inputs = self.embed_drop(torch.cat((conditions, spatial[:, :-1]), 1))
                if self._cache['spatial_ctx_hw'] is None:
                    context = self.body_transformer.cached_forward(inputs)[:, -1:]
                else:
                    context = self.body_transformer.cached_forward(inputs[:, -1:])
                self._cache['spatial_ctx_hw'] = context
                self.head_transformer.init_cache()
            depth = self.head_mlp(self.embed_depth_with_model_aux(xs, aux))[:, site]
            inputs = torch.cat((self._cache['spatial_ctx_hw'], depth[:, :-1]), 1) + self.pos_emb_d
            return self.head_transformer.cached_forward(inputs[:, event:event + 1]).reshape(batch, -1)

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, atom_top_k=2048, atom_top_p=None,
                        coeff_top_k=0, coeff_top_p=None, atom_temperature=1., coeff_temperature=1.,
                        amp=True):
        hmax, wmax, _ = self.block_size
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, hmax, wmax, 4, dtype=torch.long, device=device)
        ids = torch.zeros(batch_size, hmax, wmax, dtype=torch.long, device=device)
        packed = self.pack(atoms, ids)
        self.init_cache()
        try:
            for h in range(hmax):
                for w in range(wmax):
                    for event in range(5):
                        hidden = self.cached_hidden(packed, model_aux, (h, w, event), amp=amp)
                        pattern_event = event == (0 if self.ordering == 'pattern-first' else 4)
                        if pattern_event:
                            with torch.amp.autocast('cuda', enabled=amp):
                                logits = self.pattern_classifier(hidden)
                            value = sample_field(logits, coeff_temperature, coeff_top_k, coeff_top_p)
                            ids[:, h, w] = value
                        else:
                            depth = event - 1 if self.ordering == 'pattern-first' else event
                            with torch.amp.autocast('cuda', enabled=amp):
                                logits = self.classifier(hidden)
                            if depth:
                                logits = logits.clone().scatter_(1, atoms[:, h, w, :depth], -float('inf'))
                            value = sample_field(logits, atom_temperature, atom_top_k, atom_top_p)
                            atoms[:, h, w, depth] = value
                        packed[:, h, w, event] = value
        finally:
            self.init_cache()
        return atoms, ids


def pattern_order_prior(vocabulary, dropout=.15, ordering='pattern-first'):
    config = recipe_config('balanced')
    config.block_size = [8, 8, 5]
    config.body.block.resid_pdrop = float(dropout)
    config.head.block.resid_pdrop = float(dropout)
    return ChurchPatternOrderRQTransformer(config, 16384, vocabulary, ordering)


def order_objective(model, aux, atoms, physical):
    ids = pattern_targets(aux, atoms, physical)
    with torch.autocast(atoms.device.type, dtype=torch.bfloat16, enabled=atoms.is_cuda):
        output = model(model.pack(atoms, ids), model_aux=aux, amp=atoms.is_cuda)
    loss, values = support_first_objective(output['atom_logits'], output['pattern_logits'], atoms, ids,
                                          atom_weight=1., pattern_weight=1.)
    with torch.no_grad():
        prediction = output['pattern_logits'].argmax(-1)
        coefficients = aux.coefficient_patterns[prediction]
        joint = values['atom_nll'].sum(-1) + values['pattern_nll']
        metrics = {'loss':float(loss.detach()), 'atom_nll':float(values['atom_nll'].mean()),
            'pattern_nll':float(values['pattern_nll'].mean()), 'joint_nll':float(joint.mean()),
            'joint_bits_per_site':float(joint.mean()/math.log(2)), 'selection_score':float(joint.mean()),
            'pattern_accuracy':float((prediction == ids).float().mean()),
            'coefficient_argmax_mae':float((coefficients-physical).abs().mean()),
            'sign_accuracy':float(((coefficients >= 0) == (physical >= 0)).float().mean()),
            'coefficient_out_of_range':float((physical.abs() > aux.coeff_scales*3).float().mean())}
        for depth in range(4):
            metrics[f'atom_nll_d{depth}'] = float(values['atom_nll'][...,depth].mean())
    return loss, metrics
