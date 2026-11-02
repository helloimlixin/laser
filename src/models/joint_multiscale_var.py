"""Finite atom/coefficient products for the frozen multiscale VAR tokenizer.

Each contribution is ONE categorical ID. Tables contain signed coefficients,
indexed only by scale and ID; decoding never needs image-specific side data.
The spatial VAR sequence is unchanged; sparse depth remains autoregressive.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .multiscale_laser_var import VAR, sample_with_top_k_top_p_


class JointMultiScaleQuantizer(nn.Module):
    def __init__(self, source, dictionary, coefficients):
        super().__init__()
        dictionary = torch.as_tensor(dictionary).detach().float().clone()
        coefficients = torch.as_tensor(coefficients).detach().float().clone()
        if dictionary.ndim != 2 or dictionary.shape[0] != source.Cvae:
            raise ValueError('Expected dictionary [channels, directions]')
        if (coefficients.ndim != 3 or coefficients.shape[:2] !=
                (len(source.v_patch_nums), dictionary.shape[1]) or coefficients.shape[-1] < 2):
            raise ValueError('Expected coefficient tables [scales, directions, levels]')
        if (not torch.isfinite(dictionary).all() or not torch.isfinite(coefficients).all()
                or (dictionary.norm(dim=0) < 1.e-8).any()):
            raise ValueError('Finite nonzero directions and finite coefficients required')
        if dictionary.shape[1] < source.sparsity:
            raise ValueError('Need distinct directions for the sparse support')
        self.source = source.requires_grad_(False).eval()
        self.Cvae, self.sparsity = source.Cvae, source.sparsity
        self.v_patch_nums = source.v_patch_nums
        self.num_directions, self.levels = coefficients.shape[1:]
        self.vocab_size = self.num_directions * self.levels
        self.register_buffer('dictionary', dictionary)
        self.register_buffer('coefficients', coefficients)

    def embed_joint(self, codes, scale):
        atoms, bins = codes.div(self.levels, rounding_mode='floor'), codes.remainder(self.levels)
        values = self.coefficients[scale, atoms, bins]
        return F.embedding(atoms, self.dictionary.T) * values[..., None]

    def contribution(self, vectors, scale):
        return self.source.contribution(vectors, scale)

    def next_input(self, accumulated, scale):
        return self.source.next_input(accumulated, scale)

    def mask_previous(self, logits, prefix):
        if prefix.shape[-1]:
            # Every coefficient level of a used direction is excluded.
            blocked = prefix.div(self.levels, rounding_mode='floor')[..., None] * self.levels
            blocked = (blocked + torch.arange(self.levels, device=prefix.device)).flatten(-2)
            logits = logits.scatter(-1, blocked, -torch.inf)
        return logits

    @torch.no_grad()
    def quantize_scale(self, signals, scale, chunk_size=256):
        shape = signals.shape[:-1]
        x = signals.reshape(-1, self.Cvae).float()
        out = []
        table = self.coefficients[scale]
        norms = self.dictionary.square().sum(0)
        for chunk in x.split(chunk_size):
            residual = chunk.clone()
            prefix = torch.empty(len(chunk), 0, device=x.device, dtype=torch.long)
            for _ in range(self.sparsity):
                correlation = residual @ self.dictionary
                scores = (2 * correlation[..., None] * table -
                          norms[:, None] * table.square()).flatten(-2)
                scores = self.mask_previous(scores, prefix)
                chosen = scores.argmax(-1)
                prefix = torch.cat((prefix, chosen[:, None]), -1)
                residual.sub_(self.embed_joint(chosen, scale))
            out.append(prefix)
        codes = torch.cat(out).reshape(*shape, self.sparsity)
        return codes, self.embed_joint(codes, scale).sum(-2)

    @torch.no_grad()
    def decompose(self, latent):
        z = latent.float()
        expected = (self.Cvae, self.v_patch_nums[-1], self.v_patch_nums[-1])
        if z.shape[1:] != expected or not torch.isfinite(z).all():
            raise ValueError('Invalid encoder latent')
        accumulated = torch.zeros_like(z)
        codes, inputs = [], []
        for scale, pn in enumerate(self.v_patch_nums):
            if scale:
                inputs.append(self.next_input(accumulated, scale))
            residual = F.interpolate(z - accumulated, (pn, pn), mode='area')
            signals = residual.flatten(2).transpose(1, 2)
            ids, vectors = self.quantize_scale(signals, scale)
            codes.append(ids)
            accumulated = accumulated + self.contribution(vectors, scale)
        return dict(codes=torch.cat(codes, 1), inputs=torch.cat(inputs, 1), latent=accumulated)

    def from_codes(self, codes):
        if (codes.dtype != torch.long or codes.ndim != 3 or
                codes.shape[1:] != (sum(p*p for p in self.v_patch_nums), self.sparsity)
                or (codes < 0).any() or (codes >= self.vocab_size).any()):
            raise ValueError('Joint IDs do not match the multiscale layout')
        accumulated = self.dictionary.new_zeros(len(codes), self.Cvae,
                                               self.v_patch_nums[-1], self.v_patch_nums[-1])
        inputs, offset = [], 0
        for scale, pn in enumerate(self.v_patch_nums):
            if scale:
                inputs.append(self.next_input(accumulated, scale))
            vectors = self.embed_joint(codes[:, offset:offset+pn*pn], scale).sum(-2)
            accumulated = accumulated + self.contribution(vectors, scale)
            offset += pn*pn
        return accumulated, torch.cat(inputs, 1)


class JointLaserVAR(VAR):
    """Original spatial VAR with one shared categorical head per contribution."""
    def __init__(self, quantizer, depth=16, width=None, heads=None, num_classes=1000):
        q = quantizer
        proxy = SimpleNamespace(Cvae=q.Cvae, vocab_size=q.vocab_size, quantize=q)
        super().__init__(proxy, num_classes=num_classes, depth=depth,
                         embed_dim=width or depth*64, num_heads=heads or depth,
                         patch_nums=q.v_patch_nums, attn_l2_norm=True,
                         drop_path_rate=.1*depth/24, norm_eps=1e-6,
                         flash_if_available=False, fused_if_available=False)
        self._q = (q,)
        self.sparsity = q.sparsity
        self.depth_embedding = nn.Embedding(q.sparsity, self.C)
        self.depth_context = nn.Sequential(nn.Linear(q.Cvae, self.C), nn.SiLU(),
                                           nn.Linear(self.C, self.C))
        self.init_weights(init_adaln=.5, init_adaln_gamma=1.e-3, init_head=.02, init_std=-1.)

    def get_logits(self, hidden, condition):
        return self.head_nm(hidden.float(), condition).float()

    def prefix_logits(self, features, prefix, depth):
        return self.head(features + self.depth_context(prefix) + self.depth_embedding.weight[depth]).float()

    def token_logits(self, features, codes):
        q = self._q[0]
        vectors = torch.cat([q.embed_joint(codes[:, lo:hi], s)
                             for s, (lo, hi) in enumerate(self.begin_ends)], 1).detach()
        prefix = features.new_zeros(*features.shape[:2], self.Cvae)
        logits = []
        for depth in range(self.sparsity):
            scores = self.prefix_logits(features, prefix, depth)
            logits.append(q.mask_previous(scores, codes[..., :depth]))
            prefix = prefix + vectors[..., depth, :]
        return torch.stack(logits, 2)

    def forward(self, labels, inputs, codes):
        features = super().forward(labels, inputs)
        logits = self.token_logits(features, codes)
        return F.cross_entropy(logits.flatten(0, 2), codes.flatten())

    @torch.no_grad()
    def sample(self, labels, cfg=1.5, top_k=900, top_p=.96, seed=0):
        q, batch = self._q[0], len(labels)
        rng = torch.Generator(device=labels.device).manual_seed(seed)
        condition = self.class_emb(torch.cat((labels, torch.full_like(labels, self.num_classes))))
        positions = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        x = condition[:, None] + self.pos_start + positions[:, :self.first_l]
        accumulated = q.dictionary.new_zeros(batch, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        offset = 0
        for block in self.blocks:
            block.attn.kv_caching(True)
        try:
            for scale, pn in enumerate(self.patch_nums):
                for block in self.blocks:
                    x = block(x=x, cond_BD=self.shared_ada_lin(condition), attn_bias=None)
                features = self.get_logits(x, condition)
                prefix = accumulated.new_zeros(batch, pn*pn, self.Cvae)
                codes = torch.empty(batch, pn*pn, 0, dtype=torch.long, device=labels.device)
                guidance = cfg*scale/self.num_stages_minus_1
                for depth in range(self.sparsity):
                    scores = self.prefix_logits(features, prefix.repeat(2, 1, 1), depth)
                    # Mix finite logits before imposing the support constraint.
                    scores = (1+guidance)*scores[:batch] - guidance*scores[batch:]
                    scores = q.mask_previous(scores, codes)
                    ids = sample_with_top_k_top_p_(scores.clone(), rng=rng,
                        top_k=min(top_k, self.V), top_p=top_p)[:, :, 0]
                    codes = torch.cat((codes, ids[..., None]), -1)
                    prefix = prefix + q.embed_joint(ids, scale)
                with torch.autocast(device_type=labels.device.type, enabled=False):
                    accumulated = accumulated + q.contribution(prefix.float(), scale)
                offset += pn*pn
                if scale < self.num_stages_minus_1:
                    x = (self.word_embed(q.next_input(accumulated, scale+1)) +
                         positions[:, offset:offset+self.patch_nums[scale+1]**2]).repeat(2, 1, 1)
            return accumulated
        finally:
            for block in self.blocks:
                block.attn.kv_caching(False)
