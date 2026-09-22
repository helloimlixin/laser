"""Compound sparse events on the original next-scale VAR spatial sequence.

The depth chain is p(a_d | previous pairs, previous scales) times
p(c_d | a_d, previous pairs, previous scales). Completed pairs retain atom
identity, coefficient identity, and their physical contribution, as in the
Church compound model. The tokenizer's learned coefficient grid is unchanged.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from src.stochastic_compound import stochastic_omp
from .multiscale_laser_var import LaserVAR, VAR, sample_with_top_k_top_p_


def short_causal_attention(q, k, v):
    """Exact one/two-token causal attention without a general tiled attention kernel."""
    if q.shape[-2] == 1:
        return v
    if q.shape[-2] == 2:
        # The first query has only one legal key. The second has two, whose
        # softmax reduces to a sigmoid of their logit difference.
        score = (q[..., 1, :] * (k[..., 1, :] - k[..., 0, :])).sum(-1, keepdim=True)
        weight = torch.sigmoid(score * q.shape[-1] ** -.5)
        second = v[..., 0, :] + weight * (v[..., 1, :] - v[..., 0, :])
        return torch.stack((v[..., 0, :], second), dim=-2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class ShortCausalBlock(nn.Module):
    """FP32 attention for the short local pair and coefficient sequences."""
    def __init__(self, width, heads, dropout=0.):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                 nn.Linear(width * 4, width), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.fast_short_attention = True

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        with torch.autocast(device_type=x.device.type, enabled=False):
            attended = (short_causal_attention(q.float(), k.float(), v.float()) if self.fast_short_attention else
                        F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True))
        attended = attended.transpose(1, 2).reshape(batch, length, width).to(x.dtype)
        x = x + self.dropout(self.proj(attended))
        return x + self.mlp(self.norm2(x))


@torch.no_grad()
def compound_decompose(q, latent, *, atom_temperatures=None, coefficient_temperatures=None,
                       stochastic=False, generator=None):
    """Use the same hard-token trajectory codec as tokenized stage-1 training."""
    from .sparse_token_codec import decompose_sparse_tokens
    return decompose_sparse_tokens(q, latent, atom_temperatures=atom_temperatures,
        coefficient_temperatures=coefficient_temperatures, stochastic=stochastic, generator=generator)


class CompoundLaserVAR(LaserVAR):
    def __init__(self, tokenizer, *, depth=16, width=None, heads=None, num_classes=2,
                 local_width=256, local_heads=4, local_layers=2, micro_layers=2,
                 dropout=.1, atom_loss_weight=1.5, coefficient_top_p=1., scale_loss_weights=None):
        super().__init__(tokenizer, depth=depth, width=width, heads=heads, num_classes=num_classes)
        q = self._q[0]
        for name in ('coefficient_head', 'atom_context', 'depth_context', 'depth_embedding'):
            delattr(self, name)
        self.atom_loss_weight = float(atom_loss_weight)
        self.coefficient_top_p = float(coefficient_top_p)
        position_weights = None
        if scale_loss_weights is not None:
            if len(scale_loss_weights) != len(q.v_patch_nums) or any(
                    not math.isfinite(float(w)) or float(w) <= 0 for w in scale_loss_weights):
                raise ValueError('Scale loss weights must be positive and match the tokenizer scales')
            position_weights = torch.cat([torch.full((pn * pn,), float(weight))
                                          for pn, weight in zip(q.v_patch_nums, scale_loss_weights)])
            position_weights /= position_weights.mean()
        self.register_buffer('position_loss_weights', position_weights, persistent=False)
        self.coefficient_embedding = nn.Embedding(q.coefficient_bins, q.Cvae)
        self.pair_adapter = nn.Sequential(nn.LayerNorm(q.Cvae * 3), nn.Linear(q.Cvae * 3, q.Cvae),
                                          nn.SiLU(), nn.Linear(q.Cvae, q.Cvae))
        self.pair_projection = nn.Linear(q.Cvae, local_width)
        self.spatial_projection = nn.Linear(self.C, local_width)
        self.depth_positions = nn.Parameter(torch.zeros(1, q.sparsity, local_width))
        self.depth_blocks = nn.Sequential(*[ShortCausalBlock(local_width, local_heads, dropout)
                                           for _ in range(local_layers)])
        self.atom_projection = nn.Linear(q.Cvae, local_width, bias=False)
        self.micro_positions = nn.Parameter(torch.zeros(1, 2, local_width))
        self.micro_blocks = nn.Sequential(*[ShortCausalBlock(local_width, local_heads, dropout)
                                           for _ in range(micro_layers)])
        self.coefficient_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(local_width), nn.Linear(local_width, q.coefficient_bins))
            for _ in range(q.sparsity)])
        # Reuse the VAR atom classifier after a local residual correction.
        self.atom_output = nn.Linear(local_width, self.C)
        self.init_weights(init_adaln=.5, init_adaln_gamma=1e-3, init_head=.02, init_std=-1.)
        nn.init.normal_(self.depth_positions, std=.02)
        nn.init.normal_(self.micro_positions, std=.02)

    def pair_embeddings(self, atoms, coefficients, scale_ids):
        q = self._q[0]
        vectors = F.embedding(atoms, q.normalized_dictionary().T).detach()
        grid = torch.stack([q.coefficient_values(torch.arange(q.coefficient_bins, device=atoms.device), i)
                            for i in range(len(q.v_patch_nums))])
        values = grid[scale_ids[None, :, None], coefficients]
        contribution = vectors * values[..., None]
        features = torch.cat((vectors, self.coefficient_embedding(coefficients), contribution), -1)
        return contribution + self.pair_adapter(features)

    def local_hidden(self, features, past_pairs):
        batch, sites, _ = features.shape
        length = past_pairs.shape[2] + 1
        previous = F.pad(past_pairs, (0, 0, 1, 0))
        x = self.spatial_projection(features)[..., None, :] + self.pair_projection(previous)
        x = x + self.depth_positions[:, :length]
        return self.depth_blocks(x.reshape(batch * sites, length, -1)).reshape(batch, sites, length, -1)

    def coefficient_logits(self, hidden, atom_vectors, depth):
        pair = torch.stack((hidden, self.atom_projection(atom_vectors)), -2) + self.micro_positions
        refined = self.micro_blocks(pair.reshape(-1, 2, pair.shape[-1])).reshape_as(pair)
        return self.coefficient_heads[depth](hidden + refined[..., -1, :])

    def token_logits(self, features, atoms, coefficients, scale_ids=None):
        if scale_ids is None:
            scale_ids = self.lvl_1L[0]
        pairs = self.pair_embeddings(atoms, coefficients, scale_ids)
        hidden = self.local_hidden(features, pairs[:, :, :-1])
        vectors = F.embedding(atoms, self._q[0].normalized_dictionary().T).detach()
        atom_logits, coefficient_logits = [], []
        for depth in range(self.sparsity):
            logits = self.head(features + self.atom_output(hidden[:, :, depth]))
            if depth:
                logits = logits.scatter(-1, atoms[:, :, :depth], -torch.inf)
            atom_logits.append(logits)
            coefficient_logits.append(self.coefficient_logits(hidden[:, :, depth], vectors[:, :, depth], depth))
        return torch.stack(atom_logits, 2), torch.stack(coefficient_logits, 2)

    def forward(self, labels, inputs, atoms, coefficients, coefficient_probabilities=None):
        features = VAR.forward(self, labels, inputs)
        a, c = self.token_logits(features, atoms, coefficients)
        if self.position_loss_weights is None:
            atom_ce = F.cross_entropy(a.float().flatten(0, 2), atoms.flatten())
            coefficient_ce = (F.cross_entropy(c.float().flatten(0, 2), coefficients.flatten())
                              if coefficient_probabilities is None else
                              -(coefficient_probabilities * c.float().log_softmax(-1)).sum(-1).mean())
        else:
            weights = self.position_loss_weights[None, :, None]
            atom_ce = (F.cross_entropy(a.float().flatten(0, 2), atoms.flatten(), reduction='none')
                       .reshape_as(atoms) * weights).mean()
            coefficients_ce = (F.cross_entropy(c.float().flatten(0, 2), coefficients.flatten(), reduction='none')
                               .reshape_as(coefficients) if coefficient_probabilities is None else
                               -(coefficient_probabilities * c.float().log_softmax(-1)).sum(-1))
            coefficient_ce = (coefficients_ce * weights).mean()
        loss = (self.atom_loss_weight * atom_ce + coefficient_ce) / (self.atom_loss_weight + 1)
        return loss, torch.stack((atom_ce.detach(), coefficient_ce.detach()))

    @torch.no_grad()
    def sample(self, labels, *, cfg=1.5, top_k=250, top_p=1., seed=0, teacher_codes=None,
               return_details=False, teacher_scales=None, atom_temperature=1.,
               coefficient_temperature=1., coefficient_top_p=None, return_codes=False):
        q, batch = self._q[0], len(labels)
        def temperatures(value):
            values = [float(value)] * len(self.patch_nums) if isinstance(value, (int, float)) else list(value)
            if len(values) != len(self.patch_nums) or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError('Sampling needs a positive finite temperature for each scale')
            return values
        atom_temperatures, coefficient_temperatures = temperatures(atom_temperature), temperatures(coefficient_temperature)
        forced_scales = (len(self.patch_nums) if teacher_codes is not None else 0) if teacher_scales is None else teacher_scales
        if not isinstance(forced_scales, int) or not 0 <= forced_scales <= len(self.patch_nums):
            raise ValueError('Teacher scales must be a valid prefix length')
        if forced_scales and teacher_codes is None:
            raise ValueError('Teacher codes are required for a forced prefix')
        coefficient_top_p = self.coefficient_top_p if coefficient_top_p is None else float(coefficient_top_p)
        if not 0 < coefficient_top_p <= 1:
            raise ValueError('Coefficient top-p must be in (0,1]')
        rng = torch.Generator(device=labels.device).manual_seed(seed)
        cond = self.class_emb(torch.cat((labels, torch.full_like(labels, self.num_classes))))
        positions = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        x = cond[:, None] + self.pos_start + positions[:, :self.first_l]
        accumulated = cond.new_zeros(batch, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        offset, all_atoms, all_coefficients, all_logits = 0, [], [], []
        for block in self.blocks:
            block.attn.kv_caching(True)
        try:
            for scale, pn in enumerate(self.patch_nums):
                for block in self.blocks:
                    x = block(x=x, cond_BD=self.shared_ada_lin(cond), attn_bias=None)
                features = self.get_logits(x, cond)
                atoms = torch.empty(batch, pn * pn, 0, device=labels.device, dtype=torch.long)
                coefficients = atoms.clone()
                scale_ids = torch.full((pn * pn,), scale, device=labels.device, dtype=torch.long)
                scale_a, scale_c = [], []
                guidance = cfg * scale / self.num_stages_minus_1
                for depth in range(self.sparsity):
                    pairs = self.pair_embeddings(atoms, coefficients, scale_ids)
                    h = self.local_hidden(features, pairs.repeat(2, 1, 1, 1))[:, :, -1]
                    a = self.head(features + self.atom_output(h))
                    a = (1 + guidance) * a[:batch] - guidance * a[batch:]
                    if depth:
                        a.scatter_(-1, atoms, -torch.inf)
                    atom = (teacher_codes['atoms'][:, offset:offset + pn * pn, depth]
                            if scale < forced_scales else
                            sample_with_top_k_top_p_(a.clone().div_(atom_temperatures[scale]), rng=rng,
                                                    top_k=min(top_k, self.V), top_p=top_p)[:, :, 0])
                    vectors = F.embedding(atom, q.normalized_dictionary().T)
                    c = self.coefficient_logits(h, vectors.repeat(2, 1, 1), depth)
                    c = (1 + guidance) * c[:batch] - guidance * c[batch:]
                    coefficient = (teacher_codes['coefficients'][:, offset:offset + pn * pn, depth]
                                   if scale < forced_scales else
                                   sample_with_top_k_top_p_(c.clone().div_(coefficient_temperatures[scale]), rng=rng,
                                                           top_k=0, top_p=coefficient_top_p)[:, :, 0])
                    atoms = torch.cat((atoms, atom[..., None]), -1)
                    coefficients = torch.cat((coefficients, coefficient[..., None]), -1)
                    scale_a.append(a)
                    scale_c.append(c)
                with torch.autocast(device_type=labels.device.type, enabled=False):
                    accumulated += q.contribution(q.embed(atoms, coefficients, scale), scale)
                all_atoms.append(atoms)
                all_coefficients.append(coefficients)
                if return_details:
                    all_logits.append((torch.stack(scale_a, 2), torch.stack(scale_c, 2)))
                offset += pn * pn
                if scale < self.num_stages_minus_1:
                    x = (self.word_embed(q.next_input(accumulated, scale + 1)) +
                         positions[:, offset:offset + self.patch_nums[scale + 1] ** 2]).repeat(2, 1, 1)
            if return_details or return_codes:
                result = dict(latent=accumulated, atoms=torch.cat(all_atoms, 1),
                              coefficients=torch.cat(all_coefficients, 1))
                if return_details:
                    result.update(atom_logits=torch.cat([x[0] for x in all_logits], 1),
                                  coefficient_logits=torch.cat([x[1] for x in all_logits], 1))
                return result
            return accumulated
        finally:
            for block in self.blocks:
                block.attn.kv_caching(False)
