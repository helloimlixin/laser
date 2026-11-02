"""Two exact autoregressive orders over the same frozen LASER codes.

The native RQ spatial/depth stacks handle all eight events. Support-first
predicts all atom IDs before any coefficients; the interleaved control uses
the same parameters, embeddings and computation with a different event order.
Neither changes the external interleaved codec or frozen reconstruction.
"""
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from src.rq_upstream import RQTransformer, RQTransformerConfig
from rqvae.utils.utils import sample_from_logits


class _TypedClassifier(nn.Module):
    """Partition the native shared classifier without allocating padded logits."""
    def __init__(self, native, num_atoms, depth, order):
        super().__init__()
        self.layer_norm, self.linear, self.logit_mask = native.layer_norm, native.linear, native.logit_mask
        self.num_atoms, self.depth, self.order = num_atoms, depth, order
        self.cached_atom = None

    def _linear(self, hidden, atom):
        rows = slice(None, self.num_atoms) if atom else slice(self.num_atoms, None)
        bias = None if self.linear.bias is None else self.linear.bias[rows]
        return F.linear(hidden, self.linear.weight[rows], bias)

    def forward(self, hidden):
        hidden = self.layer_norm(hidden)
        if hidden.ndim == 5:
            if self.order == 'support_first':
                atoms, coefficients = hidden[..., :self.depth, :], hidden[..., self.depth:, :]
            else:
                atoms, coefficients = hidden[..., 0::2, :], hidden[..., 1::2, :]
            return dict(atom_logits=self._linear(atoms, True),
                        coeff_logits=self._linear(coefficients, False))
        if hidden.ndim != 3 or self.cached_atom is None:
            raise RuntimeError('Cached classification requires an explicit event type')
        logits = self._linear(hidden, self.cached_atom)
        padding = (0, self.linear.out_features-self.num_atoms) if self.cached_atom else (self.num_atoms, 0)
        return F.pad(logits, padding, value=-torch.inf)


class LaserNativeSupportOrderRQTransformer(RQTransformer):
    def __init__(self, config, num_atoms, coefficient_bins, *, order, depth=4):
        if order not in ('interleaved', 'support_first'):
            raise ValueError('Unknown support/coefficient order')
        if (list(config.block_size)[-1] != 2*depth or not config.shared_cls_emb
                or not config.input_emb_vqvae or not config.head_emb_vqvae
                or not config.cumsum_depth_ctx or config.block_size_cond > 1):
            raise ValueError('Require eight-event native RQ with shared classifier and cumulative frozen embeddings')
        super().__init__(config)
        self.num_atoms, self.coeff_vocab_size, self.pair_depth = int(num_atoms), int(coefficient_bins), int(depth)
        self.order = order
        self.event_positions = tuple(range(2*depth)) if order == 'interleaved' else tuple(range(0, 2*depth, 2))+tuple(range(1, 2*depth, 2))
        self.classifier = _TypedClassifier(self.classifier, num_atoms, depth, order)
        self.decoder_type = 'native_rq_eight_event_'+order

    def event_tokens(self, tokens):
        if self.order == 'interleaved':return tokens
        return torch.cat((tokens[..., 0::2], tokens[..., 1::2]), -1)

    def codec_tokens(self, events):
        if self.order == 'interleaved':return events
        return torch.stack((events[..., :self.pair_depth], events[..., self.pair_depth:]), -1).flatten(-2)

    def embed_with_model_aux(self, events, model_aux):
        if self.order == 'support_first':
            atoms, ids = events[..., :self.pair_depth], events[..., self.pair_depth:]-self.num_atoms
        else:
            atoms, ids = events[..., 0::2], events[..., 1::2]-self.num_atoms
        vectors = model_aux.dictionary.T[atoms]
        physical = model_aux.coeff_bins[ids]*model_aux.coeff_scales
        # Atom events reveal identity only. A later coefficient corrects that
        # vector to its physical contribution. Summing a completed site gives
        # exactly sum_i D[atom_i] * coefficient_i (up to roundoff), in either order.
        corrections = vectors*(physical-1)[..., None]
        if self.order == 'support_first':return torch.cat((vectors, corrections), -2)
        return torch.stack((vectors, corrections), -2).flatten(-3, -2)

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        output = RQTransformer.forward(self, self.event_tokens(tokens), model_aux, cond, amp)
        atoms = tokens[..., 0::2]
        for depth in range(1, self.pair_depth):
            output['atom_logits'][..., depth, :].scatter_(-1, atoms[..., :depth], -torch.inf)
        return output

    @torch.no_grad()
    def cached_forward(self, tokens, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        h, w, event = sample_loc
        external = self.event_positions[event]
        atom = external % 2 == 0
        self.classifier.cached_atom = atom
        try:
            output = RQTransformer.cached_forward(self, self.event_tokens(tokens), model_aux,
                                                  cond, amp, sample_loc=sample_loc)
        finally:self.classifier.cached_atom = None
        if atom and external:
            output.scatter_(-1, tokens[:, h, w, :external:2], -torch.inf)
        return output

    @torch.no_grad()
    def sample_sparse(self, batch_size, model_aux, cond=None, *, atom_temperature=1., atom_top_k=0,
                      atom_top_p=.95, coeff_temperature=1., coeff_top_k=0, coeff_top_p=.95, amp=True):
        height, width, events = self.block_size
        device = next(self.parameters()).device
        tokens = torch.zeros(batch_size, height, width, events, dtype=torch.long, device=device)
        tokens[..., 1::2] = self.num_atoms+self.coeff_vocab_size//2
        self.init_cache()
        try:
            for h in range(height):
                for w in range(width):
                    for event, external in enumerate(self.event_positions):
                        atom = external%2 == 0
                        logits = self.cached_forward(tokens[:, :h+1], model_aux, cond, amp,
                                                     sample_loc=(h, w, event))
                        logits = logits[:, :self.num_atoms] if atom else logits[:, self.num_atoms:]
                        size = self.num_atoms if atom else self.coeff_vocab_size
                        requested_k = int(atom_top_k if atom else coeff_top_k)
                        selected = sample_from_logits(logits,
                            temperature=atom_temperature if atom else coeff_temperature,
                            top_k=requested_k if 0 < requested_k < size else None,
                            top_p=atom_top_p if atom else coeff_top_p).reshape(batch_size)
                        tokens[:, h, w, external] = selected+(0 if atom else self.num_atoms)
        finally:self.init_cache()
        return tokens


def build_native_support_order_model(arch, *, num_atoms, coefficient_bins, order, depth=4):
    arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    arch.block_size = [*arch.block_size[:2], 2*depth]
    arch.vocab_size = num_atoms+coefficient_bins
    arch.shared_cls_emb = True
    arch.input_emb_vqvae = arch.head_emb_vqvae = arch.cumsum_depth_ctx = True
    model = LaserNativeSupportOrderRQTransformer(RQTransformerConfig.create(arch), num_atoms,
        coefficient_bins, order=order, depth=depth)
    model.apply(model._init_weights)
    return model
