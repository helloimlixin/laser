"""Checkpoint-compatible historical LASER pairs over upstream RQ forward/cache.

Only pair embeddings and coefficient prediction are adapted. Spatial and depth
execution call the actual upstream implementation, including projection bias.
"""
import math

import torch
from torch.nn import functional as F

from src.models.compound_coefficient_decoder import CompoundCoefficientHistoryDecoder
from src.models.laser_compound_rq_adapter import LaserCompoundRQTransformer
from src.rq_upstream import RQTransformer, RQTransformerConfig


class LaserHistoricalRQTransformer(LaserCompoundRQTransformer):
    def __init__(self, config, num_atoms, coefficient_bins, depth=4,
                 history_width=512, history_layers=2, history_heads=8, history_dropout=.1):
        super().__init__(config, num_atoms, coefficient_bins, depth)
        self.config.cumsum_depth_ctx = True
        self.coefficient_history_decoder = CompoundCoefficientHistoryDecoder(
            int(config.embed_dim), int(config.input_embed_dim), width=history_width,
            layers=history_layers, heads=history_heads, dropout=history_dropout,
            max_events=math.prod(self.block_size[:2]) * depth)
        self._native_hidden = None
        self.classifier.register_forward_pre_hook(self._capture_native_hidden)
        self.decoder_type = 'historical_full_pair_history_upstream_forward_v1'

    def _capture_native_hidden(self, module, inputs):
        self._native_hidden = inputs[0]

    def pack(self, tokens):
        return tokens[..., 0::2] * self.coeff_vocab_size + tokens[..., 1::2] - self.num_atoms

    def scalar_tokens(self, packed):
        tokens = torch.empty(*packed.shape[:-1], 2 * self.pair_depth,
                             dtype=torch.long, device=packed.device)
        tokens[..., 0::2] = packed.div(self.coeff_vocab_size, rounding_mode='floor')
        tokens[..., 1::2] = packed.remainder(self.coeff_vocab_size) + self.num_atoms
        return tokens

    def embed_with_model_aux(self, packed, model_aux):
        return self.pair_embeddings(self.scalar_tokens(packed), model_aux)

    def init_cache(self):
        RQTransformer.init_cache(self)
        self.coefficient_history_decoder.reset_cache()
        self._pair_hidden = self._native_hidden = None

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            try:
                atom_logits = RQTransformer.forward(self, self.pack(tokens), model_aux, cond, amp)
                hidden = self._native_hidden
                atoms, vectors, _, physical = self._physical_pairs(tokens, model_aux)
                for d in range(1, self.pair_depth):
                    atom_logits[..., d, :].scatter_(-1, atoms[..., :d], -torch.inf)
                batch = len(tokens)
                refined = self.coefficient_hidden(hidden, vectors)
                pairs = self.pair_embeddings(tokens, model_aux).reshape(batch, -1, vectors.shape[-1])
                previous = torch.cat((torch.zeros_like(pairs[:, :1]), pairs[:, :-1]), 1)
                shifted = torch.cat((torch.zeros_like(physical[..., :1, :]), physical[..., :-1, :]), -2)
                prefix = shifted.cumsum(-2).reshape_as(previous)
                residual = self.coefficient_history_decoder(
                    hidden.reshape(batch, -1, hidden.shape[-1]), vectors.reshape_as(pairs),
                    previous, prefix).reshape_as(hidden)
                refined = refined + residual
                coefficients = torch.stack([head(refined[..., d, :])
                    for d, head in enumerate(self.coeff_classifier)], -2)
                return dict(atom_logits=atom_logits, coeff_logits=coefficients)
            finally:
                self._native_hidden = None

    @torch.no_grad()
    def cached_forward(self, tokens, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        h, w, event = sample_loc
        d = event // 2
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            if event % 2 == 0:
                logits = RQTransformer.cached_forward(self, self.pack(tokens), model_aux, cond,
                    amp, sample_loc=(h, w, d))
                self._pair_hidden, self._native_hidden = self._native_hidden, None
                if d:
                    logits.scatter_(1, tokens[:, h, w, :event:2], -torch.inf)
                return F.pad(logits, (0, self.coeff_vocab_size), value=-torch.inf)
            site = h * tokens.shape[2] + w
            history = tokens.reshape(len(tokens), -1, 2 * self.pair_depth)[:, :site + 1]
            _, vectors, _, physical = self._physical_pairs(history[:, site], model_aux)
            vector = vectors[:, d:d + 1]
            index = site * self.pair_depth + d
            if index:
                pairs = self.pair_embeddings(history, model_aux).reshape(len(tokens), -1, vector.shape[-1])
                previous = pairs[:, index - 1:index]
            else:
                previous = torch.zeros_like(vector)
            prefix = physical[:, :d].cumsum(-2)[:, -1:] if d else torch.zeros_like(vector)
            refined = self.coefficient_hidden(self._pair_hidden, vector)
            residual = self.coefficient_history_decoder(
                self._pair_hidden, vector, previous, prefix, event=index)
            refined = refined + residual
            logits = self.coeff_classifier[d](refined)[:, 0]
            return F.pad(logits, (self.num_atoms, 0), value=-torch.inf)


def build_historical_pair_model(arch, *, num_atoms, coefficient_bins, depth=4, **history_options):
    from omegaconf import OmegaConf
    arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    arch.block_size = [*arch.block_size[:2], depth]
    arch.vocab_size = num_atoms
    return LaserHistoricalRQTransformer(RQTransformerConfig.create(arch), num_atoms,
                                       coefficient_bins, depth, **history_options)
