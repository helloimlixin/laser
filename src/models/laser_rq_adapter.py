"""LASER pair embeddings/masks over the unmodified upstream RQ backbone.

Attention stacks, blocks, MLPs, norms, positions, initialization and KV cache
implementation are inherited from rqvae. Overrides below handle only LASER's
physical atom/coefficient representation and its two vocabulary partitions.
"""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from src.rq_upstream import RQTransformer
from rqvae.utils.utils import sample_from_logits


class LaserPairRQTransformer(RQTransformer):
    def __init__(self, config, num_atoms):
        super().__init__(config)
        if self.block_size[-1] % 2:
            raise ValueError('LASER requires alternating atom/coefficient events')
        self.num_atoms = int(num_atoms)
        dimension, width = int(config.input_embed_dim), int(config.embed_dim)
        self.selected_atom_projection = nn.Linear(dimension, width, bias=False)
        self.pair_features = nn.Linear(2 * dimension + 1, width, bias=False)
        self.config.cumsum_depth_ctx = False
        self.decoder_type = 'shared_interleaved_physical_pairs_scalar_ce_v1'

    def _physical_pairs(self, tokens, aux):
        atoms = tokens[..., 0::2].long()
        ids = tokens[..., 1::2].long() - self.num_atoms
        vectors = aux.dictionary.t()[atoms]
        physical = aux.coeff_bins[ids] * aux.coeff_scales
        return atoms, vectors, physical, vectors * physical[..., None]

    def _context_contributions(self, tokens, aux, contributions):
        """Completed-pair features; the default retains the physical latent."""
        return contributions

    def _pair_inputs(self, spatial, tokens, aux):
        _, vectors, physical, contributions = self._physical_pairs(tokens, aux)
        prefix = self._context_contributions(tokens, aux, contributions).cumsum(-2)
        before = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), -2)
        selected = self.selected_atom_projection(vectors) + self.head_mlp(before)
        completed = self.pair_features(torch.cat(
            (vectors, physical[..., None], contributions), -1)) + self.head_mlp(prefix)
        events = torch.stack((selected, completed), -2).flatten(-3, -2)
        return torch.cat((spatial[:, None], events[:, :-1]), -2) + self.pos_emb_d

    def classify_head_outputs(self, hidden):
        normalized = self.classifier.layer_norm(hidden)
        linear = self.classifier.linear
        return dict(atom_logits=F.linear(normalized[..., 0::2, :],
            linear.weight[:self.num_atoms], None if linear.bias is None else linear.bias[:self.num_atoms]),
            coeff_logits=F.linear(normalized[..., 1::2, :],
            linear.weight[self.num_atoms:], None if linear.bias is None else linear.bias[self.num_atoms:]))

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        with torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext():
            batch, height, width, depth = tokens.shape
            sequence = tokens.reshape(batch, height * width, depth)
            atoms, _, _, contributions = self._physical_pairs(sequence, model_aux)
            contributions = self._context_contributions(sequence, model_aux, contributions)
            # A completed site's pair context enters the upstream body once.
            embedded = self.input_mlp(contributions.sum(-2)) + self.pos_emb_hw[:, :height * width]
            if cond is None:
                cond = torch.zeros(batch, self.block_size_cond, dtype=torch.long, device=tokens.device)
            start = self.cond_emb(cond.reshape(batch, self.block_size_cond)) + self.pos_emb_cond
            body = self.body_transformer(self.embed_drop(torch.cat((start, embedded[:, :-1]), 1)))
            spatial = body[:, self.block_size_cond - 1:]
            inputs = self._pair_inputs(spatial.reshape(-1, spatial.shape[-1]),
                                       sequence.reshape(-1, depth), model_aux)
            hidden = self.head_transformer(inputs).reshape(batch, height, width, depth, -1)
            output = self.classify_head_outputs(hidden)
            atoms = atoms.reshape(batch, height, width, depth // 2)
            for index in range(1, depth // 2):
                output['atom_logits'][..., index, :].scatter_(-1, atoms[..., :index], -torch.inf)
            if self.block_size_cond > 1:
                return output, self.cond_classifier(body[:, :self.block_size_cond - 1])
            return output

    @torch.no_grad()
    def cached_forward(self, tokens, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        h, w, event = sample_loc
        batch, _, width, depth = tokens.shape
        site = h * width + w
        history = tokens.reshape(batch, -1, depth)[:, :site + 1]
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            if event == 0:
                _, _, _, contributions = self._physical_pairs(history, model_aux)
                contributions = self._context_contributions(history, model_aux, contributions)
                embedded = self.input_mlp(contributions.sum(-2)) + self.pos_emb_hw[:, :site + 1]
                if cond is None:
                    cond = torch.zeros(batch, self.block_size_cond, dtype=torch.long, device=tokens.device)
                start = self.cond_emb(cond.reshape(batch, self.block_size_cond)) + self.pos_emb_cond
                inputs = self.embed_drop(torch.cat((start, embedded[:, :-1]), 1))
                inputs = inputs if self._cache['spatial_ctx_hw'] is None else inputs[:, -1:]
                self._cache['spatial_ctx_hw'] = self.body_transformer.cached_forward(inputs)[:, -1:]
                self.head_transformer.init_cache()
            context = self._cache['spatial_ctx_hw']
            if context is None:
                raise RuntimeError('Generate the first atom before later pair events')
            inputs = self._pair_inputs(context[:, 0], history[:, site], model_aux)
            hidden = self.head_transformer.cached_forward(inputs[:, event:event + 1])
            normalized = self.classifier.layer_norm(hidden).reshape(batch, -1)
            linear = self.classifier.linear
            if event % 2 == 0:
                logits = F.linear(normalized, linear.weight[:self.num_atoms],
                                  None if linear.bias is None else linear.bias[:self.num_atoms])
                if event:
                    logits.scatter_(1, history[:, site, :event:2], -torch.inf)
                return F.pad(logits, (0, linear.weight.shape[0] - self.num_atoms), value=-torch.inf)
            logits = F.linear(normalized, linear.weight[self.num_atoms:],
                              None if linear.bias is None else linear.bias[self.num_atoms:])
            return F.pad(logits, (self.num_atoms, 0), value=-torch.inf)

    @torch.no_grad()
    def sample_sparse(self, batch_size, model_aux, cond=None, *,
                      atom_temperature=1., atom_top_k=0, atom_top_p=.95,
                      coeff_temperature=1., coeff_top_k=0, coeff_top_p=.95, amp=True):
        height, width, depth = self.block_size
        if depth != 2 * model_aux.sparsity_level:
            raise ValueError('LASER pair depth differs from the tokenizer')
        device = next(self.parameters()).device
        tokens = torch.zeros(batch_size, height, width, depth, dtype=torch.long, device=device)
        tokens[..., 1::2] = self.num_atoms + model_aux.coeff_vocab_size // 2
        self.init_cache()
        try:
            for h in range(height):
                for w in range(width):
                    for event in range(depth):
                        logits = self.cached_forward(tokens[:, :h + 1], model_aux, cond,
                            amp=amp, sample_loc=(h, w, event))
                        atom = event % 2 == 0
                        offset = 0 if atom else self.num_atoms
                        logits = logits[:, :self.num_atoms] if atom else logits[:, self.num_atoms:]
                        selected = sample_from_logits(logits,
                            temperature=atom_temperature if atom else coeff_temperature,
                            top_k=int(atom_top_k) if atom and 0 < int(atom_top_k) < self.num_atoms
                                else int(coeff_top_k) if not atom and 0 < int(coeff_top_k) < model_aux.coeff_vocab_size
                                else None,
                            top_p=atom_top_p if atom else coeff_top_p)
                        tokens[:, h, w, event] = selected + offset
        finally:
            self.init_cache()
        return tokens
