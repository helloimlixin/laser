"""One-based offset transport for the existing four-pair RQ prior.

ID zero is unused. Atoms occupy 1..N and coefficients N+1..N+C.
Embedding/classifier indices remain zero-based inside the model.
"""
import torch
from torch.nn import functional as F

from src.models.offset_pair_prior import OffsetPairRQTransformer
from src.models.rqtransformer.transformers import sample_from_logits


class OneBasedOffsetPairRQTransformer(OffsetPairRQTransformer):
    @classmethod
    def from_offset_model(cls, model):
        if type(model) is not OffsetPairRQTransformer:
            raise TypeError("requires the existing offset pair RQ backbone")
        model.__class__ = cls
        model.token_vocabulary_size = model.num_atoms + model.coeff_vocab_size + 1
        model.decoder_type = "one_based_offset_scalar_ids_pair_rq_backbone_v1"
        return model

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        return super().forward(tokens - 1, model_aux, cond, amp)

    @torch.no_grad()
    def cached_forward(self, tokens, model_aux=None, cond=None, amp=False,
                       sample_loc=(0, 0, 0)):
        logits = super().cached_forward(tokens - 1, model_aux, cond, amp, sample_loc)
        return F.pad(logits, (1, 0), value=-torch.inf)

    @torch.no_grad()
    def sample_sparse(self, batch_size, model_aux, cond=None, *, atom_temperature=1.,
                      atom_top_k=0, atom_top_p=.92, coeff_temperature=1.,
                      coeff_top_k=0, coeff_top_p=.92, amp=True):
        if self.training:
            raise ValueError("sampling requires eval mode")
        tokens = torch.ones(batch_size, *self.token_block_size,
                            device=next(self.parameters()).device, dtype=torch.long)
        tokens[..., 1::2] = self.num_atoms + 1 + self.coeff_vocab_size // 2
        self.init_cache()
        try:
            for h in range(self.block_size[0]):
                for w in range(self.block_size[1]):
                    for event in range(self.token_block_size[-1]):
                        logits = self.cached_forward(tokens, model_aux, cond, amp,
                                                     sample_loc=(h, w, event))
                        if event % 2:
                            logits, offset = logits[:, self.num_atoms + 1:], self.num_atoms + 1
                            temperature, top_k, top_p = coeff_temperature, coeff_top_k, coeff_top_p
                        else:
                            logits, offset = logits[:, 1:self.num_atoms + 1], 1
                            temperature, top_k, top_p = atom_temperature, atom_top_k, atom_top_p
                        tokens[:, h, w, event] = offset + sample_from_logits(
                            logits, temperature=temperature, top_k=top_k or None, top_p=top_p)
        finally:
            self.init_cache()
        return tokens

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, *,
                        causal_prefix_sampling="predicted", **kwargs):
        return self.fields(self.sample_sparse(batch_size, model_aux, cond, **kwargs) - 1)
