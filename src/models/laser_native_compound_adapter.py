"""Archived atom-then-coefficient head on the actual upstream RQ execution."""
import torch
from torch.nn import functional as F

from src.models.laser_compound_rq_adapter import LaserCompoundRQTransformer
from src.rq_upstream import RQTransformer, RQTransformerConfig


class LaserNativeCompoundRQTransformer(LaserCompoundRQTransformer):
    """Keep all 837 historical parameter names and use native forward/cache.

    In particular, the spatial path projects each pair before summing, so a
    trained input projection bias is included four times, as in upstream RQ.
    """
    def __init__(self, config, num_atoms, coefficient_bins, depth=4):
        super().__init__(config, num_atoms, coefficient_bins, depth)
        self.config.cumsum_depth_ctx = True
        self._native_hidden = None
        self.classifier.register_forward_pre_hook(self._capture_hidden)
        self.decoder_type = 'archived_compound_head_native_rq_execution_v1'

    def _capture_hidden(self, module, inputs):
        self._native_hidden = inputs[0]

    def pack(self, tokens):
        return tokens[..., 0::2] * self.coeff_vocab_size + tokens[..., 1::2] - self.num_atoms

    def scalar_tokens(self, packed):
        result = torch.empty(*packed.shape[:-1], 2*self.pair_depth,
                             device=packed.device, dtype=torch.long)
        result[..., 0::2] = packed.div(self.coeff_vocab_size, rounding_mode='floor')
        result[..., 1::2] = packed.remainder(self.coeff_vocab_size) + self.num_atoms
        return result

    def embed_with_model_aux(self, packed, model_aux):
        return self.pair_embeddings(self.scalar_tokens(packed), model_aux)

    def init_cache(self):
        RQTransformer.init_cache(self)
        self._native_hidden = self._pair_hidden = None

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            try:
                atoms = RQTransformer.forward(self, self.pack(tokens), model_aux, cond, amp)
                hidden = self._native_hidden
                selected, vectors, _, _ = self._physical_pairs(tokens, model_aux)
                for depth in range(1, self.pair_depth):
                    atoms[..., depth, :].scatter_(-1, selected[..., :depth], -torch.inf)
                refined = self.coefficient_hidden(hidden, vectors)
                coefficients = torch.stack([head(refined[..., d, :])
                    for d, head in enumerate(self.coeff_classifier)], -2)
                return dict(atom_logits=atoms, coeff_logits=coefficients)
            finally:
                self._native_hidden = None

    @torch.no_grad()
    def cached_forward(self, tokens, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        h, w, event = sample_loc
        depth = event//2
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            if event % 2 == 0:
                logits = RQTransformer.cached_forward(self, self.pack(tokens), model_aux, cond,
                    amp, sample_loc=(h, w, depth))
                self._pair_hidden, self._native_hidden = self._native_hidden, None
                if depth:
                    logits.scatter_(-1, tokens[:, h, w, :event:2], -torch.inf)
                return F.pad(logits, (0, self.coeff_vocab_size), value=-torch.inf)
            _, vectors, _, _ = self._physical_pairs(tokens[:, h, w], model_aux)
            refined = self.coefficient_hidden(self._pair_hidden, vectors[:, depth:depth+1])
            logits = self.coeff_classifier[depth](refined)[:, 0]
            return F.pad(logits, (self.num_atoms, 0), value=-torch.inf)


def build_native_compound_model(arch, *, num_atoms, coefficient_bins, depth=4):
    from omegaconf import OmegaConf
    arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    arch.block_size = [*arch.block_size[:2], depth]
    arch.vocab_size = num_atoms
    return LaserNativeCompoundRQTransformer(RQTransformerConfig.create(arch), num_atoms,
                                           coefficient_bins, depth)
