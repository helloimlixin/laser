"""Fresh full-vocabulary OMP teacher and supports-before-coefficients prior.

Within a site the law is q(A|z) q(C|A,z). OMP draws supports sequentially,
refits on the complete support, then draws coefficients. Atom soft labels
therefore condition on support history only. The prior uses the same order:
a0,a1,a2,a3,c0,c1,c2,c3. Completed spatial sites retain full pair embeddings.
"""
from contextlib import nullcontext, contextmanager

import torch
from torch import nn

from src.stochastic_compound import stochastic_omp
from src.models.rqtransformer.transformers import sample_from_logits
from src.training.rqtransformer import CompoundLaserRQTransformer

LATENT_FORMAT = 'laser_encoder_latents_v1'


@contextmanager
def teacher_precision():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class EncoderLatentDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        self.latents, self.labels, self.meta = (payload[k] for k in ('latents', 'labels', 'meta'))
        if self.meta.get('format') != LATENT_FORMAT or self.latents.dtype != torch.float32:
            raise ValueError('online OMP needs FP32 prequantization encoder latents')
        if self.latents.ndim != 4 or len(self.latents) != len(self.labels):
            raise ValueError('latent cache shape/labels mismatch')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.latents[index], self.labels[index]


@torch.no_grad()
def online_omp_targets(signals, aux, *, temperature, coefficient_temperature,
                       site_chunk_size=256, generator=None):
    """All eligible atoms at every draw; no cached trajectories or top-k.

    Chunking limits teacher scratch memory, and is part of RNG replay metadata.
    The dense FP32 targets remain full-vocabulary. Coefficients use the existing
    normalized-bin soft kernel after the final physical least-squares refit.
    """
    if temperature <= 0 or coefficient_temperature <= 0 or site_chunk_size <= 0:
        raise ValueError('positive temperatures and chunk size required')
    shape = signals.shape[:-1]
    x = signals.reshape(-1, signals.shape[-1]).float()
    dictionary = aux.dictionary.float()
    depth, vocab = len(aux.coeff_scales), dictionary.shape[1]
    atoms = torch.empty(len(x), depth, device=x.device, dtype=torch.long)
    centers = x.new_empty(len(x), depth)
    probabilities = x.new_empty(len(x), depth, vocab)
    with teacher_precision(), torch.autocast(device_type=x.device.type, enabled=False):
        # Frozen dictionary; this is derived scratch, not checkpoint state.
        if not hasattr(aux, '_online_omp_gram'):
            aux._online_omp_gram = dictionary.T @ dictionary
        for start in range(0, len(x), site_chunk_size):
            stop = min(start + site_chunk_size, len(x))
            result = stochastic_omp(x[start:stop], dictionary, depth=depth,
                                    temperature=temperature, gram=aux._online_omp_gram,
                                    generator=generator, return_probabilities=True)
            atoms[start:stop] = result['atoms']
            centers[start:stop] = result['coefficients'] / aux.coeff_scales
            probabilities[start:stop] = result['atom_probabilities']
        # Use the same kernel as LaserAux, with an explicit generator for tests.
        coefficient_probs = (-(centers[..., None] - aux.coeff_bins).square()
                             / coefficient_temperature).softmax(-1)
        coefficient_ids = torch.multinomial(coefficient_probs.flatten(0, 1), 1,
                                            generator=generator).reshape_as(atoms)
    return dict(atoms=atoms.reshape(*shape, depth),
                coefficients=centers.reshape(*shape, depth),
                coefficient_ids=coefficient_ids.reshape(*shape, depth),
                atom_probabilities=probabilities.reshape(*shape, depth, vocab),
                coefficient_probabilities=coefficient_probs.reshape(*shape, depth, -1))


class SupportsFirstOMPTransformer(CompoundLaserRQTransformer):
    """Eight causal head events per four-pair site, same spatial architecture."""

    def event_embeddings(self, packed, aux):
        atoms, coefficients = self.unpack(packed)
        return torch.cat((aux.dictionary.T[atoms.long()],
                          self.compound_pair_embeddings(aux, atoms, coefficients)), dim=-2)

    def forward(self, packed, model_aux=None, cond=None, amp=False,
                causal_prefix_reconstructions=None):
        if causal_prefix_reconstructions is not None:
            raise ValueError('supports-first OMP does not use causal prefix reconstructions')
        b, h, w, d = packed.shape
        seq = h * w
        atoms, coefficients = self.unpack(packed)
        xs = packed.reshape(b, seq, d)
        cond = torch.zeros(b, 1, device=packed.device, dtype=torch.long) if cond is None else cond.reshape(b, 1)
        with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
            spatial = self.input_mlp(self.embed_with_model_aux(xs, model_aux)).sum(-2)
            spatial = spatial + self.pos_emb_hw[:, :seq]
            condition = self.cond_emb(cond) + self.pos_emb_cond
            body = self.body_transformer(self.embed_drop(torch.cat((condition, spatial[:, :-1]), 1)))
            events = self.event_embeddings(xs, model_aux)
            if self.config.cumsum_depth_ctx:
                events = events.cumsum(-2)
            events = self.head_mlp(events)
            head_inputs = torch.cat((body[:, :, None], events[:, :, :-1]), -2)
            hidden = self.head_transformer(head_inputs.reshape(b * seq, 2*d, -1) + self.pos_emb_d)
            hidden = hidden.reshape(b, h, w, 2*d, -1)
            return dict(atom_logits=self.mask_seen_atoms(self.classifier(hidden[..., :d, :]), atoms),
                        coeff_logits=self.coefficient_logits(hidden[..., d:, :], model_aux.dictionary.T[atoms.long()]))

    @torch.no_grad()
    def cached_head_output(self, packed, model_aux, cond, sample_loc, amp=True):
        h, w, event = sample_loc
        b, _, width, d = packed.shape
        site = h * width + w
        xs = packed.reshape(b, -1, d)[:, :site+1]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp and packed.is_cuda):
            if event == 0:
                cond = torch.zeros(b, 1, device=packed.device, dtype=torch.long) if cond is None else cond.reshape(b, 1)
                spatial = self.input_mlp(self.embed_with_model_aux(xs, model_aux)).sum(-2)
                spatial = spatial + self.pos_emb_hw[:, :site+1]
                body = self.embed_drop(torch.cat((self.cond_emb(cond)+self.pos_emb_cond, spatial[:, :-1]), 1))
                if self._cache['spatial_ctx_hw'] is not None:
                    body = body[:, -1:]
                self._cache['spatial_ctx_hw'] = self.body_transformer.cached_forward(body)[:, -1:]
                self.head_transformer.init_cache()
                head_input = self._cache['spatial_ctx_hw']
            else:
                events = self.event_embeddings(xs[:, -1:], model_aux)[:, 0]
                if self.config.cumsum_depth_ctx:
                    events = events.cumsum(-2)
                head_input = self.head_mlp(events[:, event-1:event])
            return self.head_transformer.cached_forward(head_input + self.pos_emb_d[:, event:event+1])[:, 0]

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, temperature=1.,
                        atom_top_k=16384, atom_top_p=.92, coeff_top_p=.92,
                        coeff_top_k=0, atom_temperature=None, coeff_temperature=None,
                        amp=True, causal_prefix_sampling='predicted'):
        h, w, d = self.block_size
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, h, w, d, device=device, dtype=torch.long)
        coeffs = torch.full_like(atoms, self.coeff_vocab_size // 2)
        packed = atoms * self.coeff_vocab_size + coeffs
        self.init_cache()
        try:
            for row in range(h):
                for column in range(w):
                    for event in range(2*d):
                        hidden = self.cached_head_output(packed, model_aux, cond, (row, column, event), amp=amp)
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp and packed.is_cuda):
                            if event < d:
                                logits = self.classifier(hidden)
                                if event:
                                    logits.scatter_(1, atoms[:, row, column, :event], -torch.inf)
                                atoms[:, row, column, event] = sample_from_logits(
                                    logits, temperature=temperature if atom_temperature is None else atom_temperature,
                                    top_k=min(atom_top_k or self.num_atoms, self.num_atoms), top_p=atom_top_p)
                                index = event
                            else:
                                index = event-d
                                vectors = model_aux.dictionary.T[atoms[:, row, column, index]]
                                logits = self.coefficient_logits(hidden, vectors, depth_index=index)
                                coeffs[:, row, column, index] = sample_from_logits(
                                    logits, temperature=temperature if coeff_temperature is None else coeff_temperature,
                                    top_k=coeff_top_k or self.coeff_vocab_size, top_p=coeff_top_p)
                        packed[:, row, column, index] = atoms[:, row, column, index] * self.coeff_vocab_size + coeffs[:, row, column, index]
        finally:
            self.init_cache()
        return atoms, coeffs


def attach_supports_first_omp(model):
    if type(model) is not CompoundLaserRQTransformer or model.causal_prefix_state or model.contribution_head is not None:
        raise ValueError('supports-first OMP requires a plain compound transformer')
    if model.block_size_cond != 1 or not model.config.shared_cls_emb:
        raise ValueError('supports-first currently supports unconditional shared atom classification')
    model.__class__ = SupportsFirstOMPTransformer
    extra = torch.empty_like(model.pos_emb_d)
    nn.init.normal_(extra, std=.02)
    model.pos_emb_d = nn.Parameter(torch.cat((model.pos_emb_d.detach(), extra), 1))
    model.event_order = 'all_supports_then_all_coefficients'
    model.init_cache()
    return model
