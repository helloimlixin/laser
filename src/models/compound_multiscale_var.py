"""Original next-scale VAR with a causal, shared-vocabulary sparse-token head.

Spatial attention uses only preceding scales. At each spatial site a small
causal transformer predicts [atom_1, atoms+bin_1, atom_2, atoms+bin_2, ...].
The frozen tokenizer and sampled reconstruction use the same scalar bins.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .multiscale_laser_var import VAR, sample_with_top_k_top_p_


class CompoundLaserVAR(VAR):
    def __init__(self, tokenizer, depth=16, width=None, heads=None,
                 num_classes=1000, token_width=256, token_layers=2, token_heads=4):
        q = tokenizer.quantize
        self.num_atoms = int(q.vocab_size)
        self.coefficient_bins = int(q.coefficient_bins)
        if self.coefficient_bins != 512:
            raise ValueError('Compound LASER VAR requires exactly 512 coefficient bins')
        proxy = SimpleNamespace(Cvae=q.Cvae,
                                vocab_size=self.num_atoms + self.coefficient_bins,
                                quantize=q)
        super().__init__(proxy, num_classes=num_classes, depth=depth,
                         embed_dim=width or depth * 64, num_heads=heads or depth,
                         patch_nums=q.v_patch_nums, attn_l2_norm=True,
                         drop_path_rate=.1 * depth / 24, norm_eps=1e-6,
                         flash_if_available=False, fused_if_available=False)
        self.sparsity = int(q.sparsity)
        self.tokens_per_site = 2 * self.sparsity
        self.spatial_token_length = self.L
        self.categorical_decisions_per_image = self.L * self.tokens_per_site
        self._q = (q,)
        self.token_embedding = nn.Embedding(self.V, token_width)
        self.token_position = nn.Embedding(self.tokens_per_site, token_width)
        self.token_start = nn.Parameter(torch.empty(1, 1, token_width))
        self.spatial_condition = nn.Linear(self.C, token_width)
        layer = nn.TransformerEncoderLayer(
            token_width, token_heads, dim_feedforward=4 * token_width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True,
            layer_norm_eps=1e-6)
        self.token_transformer = nn.TransformerEncoder(
            layer, token_layers, enable_nested_tensor=False)
        self.token_output = nn.Sequential(nn.LayerNorm(token_width, eps=1e-6),
                                          nn.Linear(token_width, self.C))
        causal = torch.ones(self.tokens_per_site, self.tokens_per_site,
                            dtype=torch.bool).triu(1)
        self.register_buffer('token_causal_mask', causal)
        self.init_weights(init_adaln=.5, init_adaln_gamma=1e-3,
                          init_head=.02, init_std=-1.)
        nn.init.trunc_normal_(self.token_start, std=token_width ** -.5)
        for block in self.token_transformer.layers:
            nn.init.trunc_normal_(block.self_attn.in_proj_weight,
                                  std=token_width ** -.5)

    def get_logits(self, hidden, condition):
        # VAR.forward supplies scale-causal features; defer the categorical head
        # until after the independently causal within-site token decoder.
        return self.head_nm(hidden.float(), condition).float()

    def pack_tokens(self, atoms, coefficients):
        if atoms.shape != coefficients.shape or atoms.shape[-1] != self.sparsity:
            raise ValueError('Sparse support shape does not match the tokenizer')
        if ((atoms < 0) | (atoms >= self.num_atoms)).any():
            raise ValueError('Atom IDs must be in [0, num_atoms)')
        if ((coefficients < 0) | (coefficients >= self.coefficient_bins)).any():
            raise ValueError('Coefficient bins must be in [0, 512)')
        if self.sparsity > 1:
            ordered = atoms.sort(dim=-1).values
            if (ordered[..., 1:] == ordered[..., :-1]).any():
                raise ValueError('OMP supports must contain distinct atoms')
        return torch.stack((atoms, self.num_atoms + coefficients), -1).flatten(-2)

    def prefix_logits(self, features, prefix):
        """Return the next token logits from a prefix of length 0..2K-1."""
        batch, sites, _ = features.shape
        length = prefix.shape[-1] + 1
        if length > self.tokens_per_site:
            raise ValueError('Compound prefix exceeds sparse support depth')
        previous = self.token_embedding(prefix).reshape(
            batch * sites, length - 1, self.token_embedding.embedding_dim)
        start = self.token_start.expand(batch * sites, -1, -1)
        x = torch.cat((start, previous), 1)
        positions = self.token_position.weight[:length]
        condition = self.spatial_condition(features).reshape(batch * sites, 1, -1)
        x = x + positions + condition
        x = self.token_transformer(x, mask=self.token_causal_mask[:length, :length])
        return self.head(self.token_output(x[:, -1])).float().reshape(batch, sites, self.V)

    def token_logits(self, features, tokens):
        """Teacher forcing with a strict shift, type mask, and OMP support mask."""
        batch, sites, length = tokens.shape
        if length != self.tokens_per_site:
            raise ValueError('Expected 2*sparsity compound tokens per site')
        previous = self.token_embedding(tokens[..., :-1]).reshape(batch * sites, length - 1, -1)
        x = torch.cat((self.token_start.expand(batch * sites, -1, -1), previous), 1)
        x = x + self.token_position.weight + self.spatial_condition(features).reshape(batch * sites, 1, -1)
        x = self.token_transformer(x, mask=self.token_causal_mask)
        logits = self.head(self.token_output(x)).float().reshape(batch, sites, length, self.V)
        valid = (torch.arange(self.V, device=tokens.device) < self.num_atoms)
        atom_slot = torch.arange(length, device=tokens.device).remainder(2) == 0
        logits = logits.masked_fill(atom_slot[:, None] != valid[None, :], -torch.inf)
        for slot in range(2, length, 2):
            logits[:, :, slot] = logits[:, :, slot].scatter(
                -1, tokens[:, :, :slot:2], -torch.inf)
        return logits

    def forward(self, labels, inputs, atoms, coefficients):
        tokens = self.pack_tokens(atoms, coefficients)
        features = super().forward(labels, inputs)
        logits = self.token_logits(features, tokens)
        atom_nll = F.cross_entropy(logits[:, :, ::2, :self.num_atoms].reshape(-1, self.num_atoms),
                                   atoms.reshape(-1))
        coefficient_nll = F.cross_entropy(logits[:, :, 1::2, self.num_atoms:].reshape(-1, self.coefficient_bins),
                                          coefficients.reshape(-1))
        return atom_nll + coefficient_nll, torch.stack((atom_nll.detach(), coefficient_nll.detach()))

    @torch.no_grad()
    def sample(self, labels, *, cfg=1.5, top_k=900, top_p=.96, seed=0):
        q, batch = self._q[0], len(labels)
        rng = torch.Generator(device=labels.device).manual_seed(seed)
        condition = self.class_emb(torch.cat((labels, torch.full_like(labels, self.num_classes))))
        positions = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        x = condition[:, None] + self.pos_start + positions[:, :self.first_l]
        accumulated = condition.new_zeros(batch, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        offset = 0
        for block in self.blocks:
            block.attn.kv_caching(True)
        try:
            for scale, pn in enumerate(self.patch_nums):
                for block in self.blocks:
                    x = block(x=x, cond_BD=self.shared_ada_lin(condition), attn_bias=None)
                features = self.get_logits(x, condition)
                tokens = torch.empty(batch, pn * pn, 0, dtype=torch.long, device=labels.device)
                guidance = cfg * scale / self.num_stages_minus_1
                for slot in range(self.tokens_per_site):
                    scores = self.prefix_logits(features, tokens.repeat(2, 1, 1))
                    # Apply CFG only to finite, type-valid logits. Mixing masked
                    # -inf entries would create NaNs, even at zero guidance.
                    scores = scores[..., :self.num_atoms] if slot % 2 == 0 else scores[..., self.num_atoms:]
                    scores = (1 + guidance) * scores[:batch] - guidance * scores[batch:]
                    if slot % 2 == 0 and slot:
                        scores.scatter_(-1, tokens[..., ::2], -torch.inf)
                    selected = sample_with_top_k_top_p_(
                        scores.clone(), rng=rng,
                        top_k=min(top_k, self.num_atoms) if slot % 2 == 0 else 0,
                        top_p=top_p)[:, :, 0]
                    if slot % 2:
                        selected = selected + self.num_atoms
                    tokens = torch.cat((tokens, selected[..., None]), -1)
                atoms, coefficients = tokens[..., ::2], tokens[..., 1::2] - self.num_atoms
                with torch.autocast(device_type=labels.device.type, enabled=False):
                    vectors = q.embed(atoms, coefficients, scale)
                    accumulated = accumulated + q.contribution(vectors.float(), scale)
                offset += pn * pn
                if scale < self.num_stages_minus_1:
                    x = (self.word_embed(q.next_input(accumulated, scale + 1)) +
                         positions[:, offset:offset + self.patch_nums[scale + 1] ** 2]).repeat(2, 1, 1)
            return accumulated
        finally:
            for block in self.blocks:
                block.attn.kv_caching(False)
