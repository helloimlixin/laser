"""Physical pair context for the ImageNet scalar-token cross-entropy prior.

Port the Church shared decoder's context and uniqueness rules while retaining
the ImageNet vocabulary, six depth blocks, and existing training objective.
"""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from src.training.rqtransformer import LaserRQTransformer


class PhysicalPairScalarRQTransformer(LaserRQTransformer):
    def __init__(self, config, num_atoms):
        super().__init__(config, num_atoms)
        if self.block_size_cond != 1 or self.block_size[-1] % 2:
            raise ValueError("Physical pairs require one class token and alternating scalar fields")
        dimension, width = int(config.input_embed_dim), int(config.embed_dim)
        self.selected_atom_projection = nn.Linear(dimension, width, bias=False)
        self.pair_features = nn.Linear(2 * dimension + 1, width, bias=False)
        self.config.cumsum_depth_ctx = False
        self.decoder_type = "shared_interleaved_physical_pairs_scalar_ce_v1"

    def _physical_pairs(self, tokens, aux):
        atoms = tokens[..., 0::2].long()
        ids = tokens[..., 1::2].long() - self.num_atoms
        vectors = aux.dictionary.t()[atoms]
        physical = aux.coeff_bins[ids] * aux.coeff_scales
        return atoms, vectors, physical, vectors * physical[..., None]

    def _pair_inputs(self, spatial, tokens, aux):
        _, vectors, physical, contributions = self._physical_pairs(tokens, aux)
        prefix = contributions.cumsum(-2)
        before = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), -2)
        selected = self.selected_atom_projection(vectors) + self.head_mlp(before)
        completed = self.pair_features(torch.cat(
            (vectors, physical[..., None], contributions), -1)) + self.head_mlp(prefix)
        events = torch.stack((selected, completed), -2).flatten(-3, -2)
        return torch.cat((spatial[:, None], events[:, :-1]), -2) + self.pos_emb_d

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        with torch.amp.autocast("cuda") if amp else nullcontext():
            batch, height, width, depth = tokens.shape
            sequence = tokens.reshape(batch, height * width, depth)
            atoms, _, _, contributions = self._physical_pairs(sequence, model_aux)
            # Project the completed physical vector once; scalar events must
            # not multiply the projection bias or enter spatial context alone.
            embedded = self.input_mlp(contributions.sum(-2)) + self.pos_emb_hw[:, :height * width]
            if cond is None:
                cond = torch.zeros(batch, 1, dtype=torch.long, device=tokens.device)
            start = self.cond_emb(cond.reshape(batch, 1)) + self.pos_emb_cond
            spatial = self.body_transformer(self.embed_drop(torch.cat((start, embedded[:, :-1]), 1)))
            inputs = self._pair_inputs(spatial.reshape(-1, spatial.shape[-1]),
                                       sequence.reshape(-1, depth), model_aux)
            hidden = self.head_transformer(inputs).reshape(batch, height, width, depth, -1)
            output = self.classify_head_outputs(hidden)
            atoms = atoms.reshape(batch, height, width, depth // 2)
            for index in range(1, depth // 2):
                output["atom_logits"][..., index, :].scatter_(-1, atoms[..., :index], -torch.inf)
            return output

    @torch.no_grad()
    def cached_forward(self, tokens, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        h, w, event = sample_loc
        batch, _, width, depth = tokens.shape
        site = h * width + w
        history = tokens.reshape(batch, -1, depth)[:, :site + 1]
        with torch.amp.autocast("cuda", enabled=amp):
            if event == 0:
                _, _, _, contributions = self._physical_pairs(history, model_aux)
                embedded = self.input_mlp(contributions.sum(-2)) + self.pos_emb_hw[:, :site + 1]
                if cond is None:
                    cond = torch.zeros(batch, 1, dtype=torch.long, device=tokens.device)
                start = self.cond_emb(cond.reshape(batch, 1)) + self.pos_emb_cond
                inputs = self.embed_drop(torch.cat((start, embedded[:, :-1]), 1))
                inputs = inputs if self._cache["spatial_ctx_hw"] is None else inputs[:, -1:]
                self._cache["spatial_ctx_hw"] = self.body_transformer.cached_forward(inputs)[:, -1:]
                self.head_transformer.init_cache()
            context = self._cache["spatial_ctx_hw"]
            if context is None:
                raise RuntimeError("Generate the first atom before later pair events")
            inputs = self._pair_inputs(context[:, 0], history[:, site], model_aux)
            hidden = self.head_transformer.cached_forward(inputs[:, event:event + 1])
            normalized = self.classifier.layer_norm(hidden).reshape(batch, -1)
            linear = self.classifier.linear
            if event % 2 == 0:
                logits = F.linear(normalized, linear.weight[:self.num_atoms],
                                  None if linear.bias is None else linear.bias[:self.num_atoms])
                if event:
                    logits.scatter_(1, history[:, site, :event:2], -torch.inf)
                logits = F.pad(logits, (0, linear.weight.shape[0] - self.num_atoms), value=-torch.inf)
            else:
                logits = F.linear(normalized, linear.weight[self.num_atoms:],
                                  None if linear.bias is None else linear.bias[self.num_atoms:])
                logits = F.pad(logits, (self.num_atoms, 0), value=-torch.inf)
            return logits
