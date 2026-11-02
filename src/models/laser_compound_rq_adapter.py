"""Complete LASER pairs over the same upstream spatial/depth transformers.

The depth stack predicts four supports. Each coefficient additionally sees its
selected atom through a two-token conditional decoder. Scalar tokens remain
the external codec interface; packing changes no probability factorization.
"""
from contextlib import nullcontext
import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.rq_upstream import RQTransformer, RQTransformerConfig
from src.models.laser_rq_adapter import LaserPairRQTransformer
from rqvae.models.rqtransformer.attentions import AttentionStack


class LaserCompoundRQTransformer(RQTransformer):
    def __init__(self, config, num_atoms, coefficient_bins, depth=4):
        super().__init__(config)
        self.num_atoms = int(num_atoms)
        self.coeff_vocab_size = int(coefficient_bins)
        self.pair_depth = int(depth)
        assert self.block_size[-1] == depth
        # Keep the existing eight-integer codec and sampler interface.
        self.block_size = torch.Size((*self.block_size[:2], 2 * depth))
        dimension, width = int(config.input_embed_dim), int(config.embed_dim)
        self.coeff_token_embedding = nn.Embedding(coefficient_bins, dimension)
        self.pair_embedding_adapter = nn.Sequential(
            nn.LayerNorm(3 * dimension), nn.Linear(3 * dimension, dimension),
            nn.SiLU(), nn.Linear(dimension, dimension))
        self.coeff_atom_proj = nn.Linear(dimension, width, bias=False)
        micro = copy.deepcopy(config.head)
        micro.n_layer = 2
        self.coeff_micro_transformer = AttentionStack(micro)
        self.coeff_micro_pos = nn.Parameter(torch.empty(1, 2, width))
        self.coeff_micro_pos.data.normal_(std=.02)
        self.coeff_classifier = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, coefficient_bins))
            for _ in range(depth)])
        self.decoder_type = 'upstream_four_pairs_atom_conditioned_coefficients_v1'
        self._pair_hidden = None

    _physical_pairs = LaserPairRQTransformer._physical_pairs
    sample_sparse = LaserPairRQTransformer.sample_sparse

    def pair_embeddings(self, tokens, aux):
        atoms, _, _, contributions = self._physical_pairs(tokens, aux)
        # Atom identity stays fixed across support prefixes. In orthogonal
        # coordinates Q describes the contribution, while D describes the ID.
        vectors = aux.dictionary.T[atoms]
        coefficient_ids = tokens[..., 1::2] - self.num_atoms
        identity = self.coeff_token_embedding(coefficient_ids)
        return contributions + self.pair_embedding_adapter(
            torch.cat((vectors, identity, contributions), -1))

    def coefficient_hidden(self, hidden, vectors):
        selected = self.coeff_atom_proj(vectors)
        pair = torch.stack((hidden, selected), -2) + self.coeff_micro_pos
        decoded = self.coeff_micro_transformer(pair.reshape(-1, 2, pair.shape[-1]))
        return hidden + decoded.reshape_as(pair)[..., -1, :]

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        with torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext():
            batch, height, width, _ = tokens.shape
            sequence = tokens.reshape(batch, height * width, 2 * self.pair_depth)
            pairs = self.pair_embeddings(sequence, model_aux)
            embedded = self.input_mlp(pairs.sum(-2)) + self.pos_emb_hw[:, :height * width]
            if cond is None:
                cond = torch.zeros(batch, self.block_size_cond, device=tokens.device, dtype=torch.long)
            start = self.cond_emb(cond.reshape(batch, self.block_size_cond)) + self.pos_emb_cond
            body = self.body_transformer(self.embed_drop(torch.cat((start, embedded[:, :-1]), 1)))
            spatial = body[:, self.block_size_cond - 1:]
            local = self.head_mlp(pairs.cumsum(-2))
            inputs = torch.cat((spatial[:, :, None], local[:, :, :-1]), -2)
            inputs = inputs.reshape(-1, self.pair_depth, spatial.shape[-1]) + self.pos_emb_d
            hidden = self.head_transformer(inputs).reshape(batch, height, width, self.pair_depth, -1)
            atom_logits = self.classifier(hidden)
            atoms, vectors, _, _ = self._physical_pairs(tokens, model_aux)
            for d in range(1, self.pair_depth):
                atom_logits[..., d, :].scatter_(-1, atoms[..., :d], -torch.inf)
            refined = self.coefficient_hidden(hidden, vectors)
            coefficients = torch.stack([head(refined[..., d, :])
                                         for d, head in enumerate(self.coeff_classifier)], -2)
            return dict(atom_logits=atom_logits, coeff_logits=coefficients)

    @torch.no_grad()
    def cached_forward(self, tokens, model_aux=None, cond=None, amp=False, sample_loc=(0, 0, 0)):
        h, w, event = sample_loc
        batch, _, width, _ = tokens.shape
        site, d = h * width + w, event // 2
        history = tokens.reshape(batch, -1, 2 * self.pair_depth)[:, :site + 1]
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            if event == 0:
                pairs = self.pair_embeddings(history, model_aux)
                embedded = self.input_mlp(pairs.sum(-2)) + self.pos_emb_hw[:, :site + 1]
                if cond is None:
                    cond = torch.zeros(batch, self.block_size_cond, device=tokens.device, dtype=torch.long)
                start = self.cond_emb(cond.reshape(batch, self.block_size_cond)) + self.pos_emb_cond
                inputs = self.embed_drop(torch.cat((start, embedded[:, :-1]), 1))
                if self._cache['spatial_ctx_hw'] is not None:
                    inputs = inputs[:, -1:]
                self._cache['spatial_ctx_hw'] = self.body_transformer.cached_forward(inputs)[:, -1:]
                self.head_transformer.init_cache()
            if event % 2 == 0:
                if d == 0:
                    inputs = self._cache['spatial_ctx_hw']
                else:
                    pairs = self.pair_embeddings(history[:, site], model_aux)
                    inputs = self.head_mlp(pairs[:, :d].sum(-2))[:, None]
                self._pair_hidden = self.head_transformer.cached_forward(inputs + self.pos_emb_d[:, d:d + 1])
                logits = self.classifier(self._pair_hidden)[:, 0]
                if d:
                    logits.scatter_(1, history[:, site, :event:2], -torch.inf)
                return F.pad(logits, (0, self.coeff_vocab_size), value=-torch.inf)
            _, vectors, _, _ = self._physical_pairs(history[:, site], model_aux)
            refined = self.coefficient_hidden(self._pair_hidden, vectors[:, d:d + 1])
            logits = self.coeff_classifier[d](refined)[:, 0]
            return F.pad(logits, (self.num_atoms, 0), value=-torch.inf)


def build_compound_model(arch, *, num_atoms, coefficient_bins, depth=4):
    from omegaconf import OmegaConf
    arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    arch.block_size = [*arch.block_size[:2], depth]
    arch.vocab_size = num_atoms
    model = LaserCompoundRQTransformer(RQTransformerConfig.create(arch), num_atoms, coefficient_bins, depth)
    model.apply(model._init_weights)
    return model
