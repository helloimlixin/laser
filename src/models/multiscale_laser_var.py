"""VAR next-scale prediction with LASER OMP dictionaries and scalar coefficients.

The spatial transformer is the pinned FoundationVision implementation. Sparse
depth is autoregressive within each site, so a target atom/coeff never enters
the spatial transformer at its own scale. No expanded vector codebook is used.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .dictionary_learner import DictionaryLearning

UPSTREAM = Path(__file__).resolve().parents[2] / "third_party/FoundationVision_VAR"
UPSTREAM_REVISION = "78b95394fc5896192e3a003e4b295f8ea743c48f"
if not (UPSTREAM / "models/var.py").is_file():
    raise ImportError("Install FoundationVision/VAR as described in docs/imagenet-laser-var.md")
sys.path.insert(0, str(UPSTREAM))
from models.var import VAR
from models.vqvae import VQVAE
from models.helpers import sample_with_top_k_top_p_

PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


class MultiScaleLaser(nn.Module):
    """Shared normalized LASER dictionary, independently sparse at each scale.

    OMP refits all selected coefficients within a scale. Earlier scales remain
    fixed. The decoder and prior both consume the same discretized coefficients.
    """
    def __init__(self, channels=32, atoms=4096, sparsity=2, patch_nums=PATCH_NUMS,
                 coefficient_bins=257, coefficient_max=8., quant_resi=None,
                 beta=.25, omp_chunk_size=2048):
        super().__init__()
        self.Cvae, self.vocab_size = channels, atoms
        self.sparsity = int(sparsity)
        self.v_patch_nums = tuple(patch_nums)
        if (len(patch_nums) < 2 or patch_nums[0] != 1 or
                any(a >= b for a, b in zip(patch_nums, patch_nums[1:]))):
            raise ValueError("Scales must increase strictly, starting at 1")
        if coefficient_bins < 3 or coefficient_bins % 2 != 1 or coefficient_max <= 0:
            raise ValueError("Use an odd coefficient vocabulary and positive range")
        self.coefficient_bins = int(coefficient_bins)
        self.beta, self.omp_chunk_size = beta, int(omp_chunk_size)
        self.coefficient_range_decay = None
        self.dictionary = DictionaryLearning(
            num_embeddings=atoms, embedding_dim=channels, sparsity_level=sparsity,
            omp_ridge=1e-5, omp_max_support_coherence=.999,
            dictionary_collective_backend="default")
        self.quant_resi = quant_resi
        self.register_buffer("coefficient_max", torch.full((len(patch_nums),), float(coefficient_max)))
        self.register_buffer("companding", torch.tensor(10.))

    def normalized_dictionary(self):
        return F.normalize(self.dictionary.dictionary.float(), dim=0)

    def coefficient_values(self, ids, scale):
        unit = ids.float() * (2. / (self.coefficient_bins - 1)) - 1
        return torch.sinh(unit * torch.asinh(self.companding)) / self.companding * self.coefficient_max[scale]

    def coefficient_ids(self, values, scale):
        unit = torch.asinh(values / self.coefficient_max[scale] * self.companding) / torch.asinh(self.companding)
        return ((unit.clamp(-1, 1) + 1) * ((self.coefficient_bins - 1) / 2)).round().long()

    def embed(self, atoms, coefficients, scale):
        vectors = F.embedding(atoms, self.normalized_dictionary().T)
        return (vectors * self.coefficient_values(coefficients, scale)[..., None]).sum(-2)

    def contribution(self, vectors, scale):
        pn, final = self.v_patch_nums[scale], self.v_patch_nums[-1]
        x = vectors.transpose(1, 2).reshape(vectors.shape[0], self.Cvae, pn, pn)
        if pn != final:
            x = F.interpolate(x, (final, final), mode="bicubic", align_corners=False)
        return self.quant_resi[scale / (len(self.v_patch_nums) - 1)](x) if self.quant_resi is not None else x

    def next_input(self, accumulated, scale):
        pn = self.v_patch_nums[scale]
        return F.interpolate(accumulated, (pn, pn), mode="area").flatten(2).transpose(1, 2)

    def decompose(self, latent, *, calibrate=False):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            z = latent.float()
            if z.shape[1:] != (self.Cvae, self.v_patch_nums[-1], self.v_patch_nums[-1]):
                raise ValueError(f"Unexpected latent shape {tuple(z.shape)}")
            if not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite encoder output")
            accumulated = torch.zeros_like(z)
            dictionary = self.normalized_dictionary()
            atom_maps, coeff_maps, inputs, losses, clipping = [], [], [], [], []
            for scale, pn in enumerate(self.v_patch_nums):
                if scale:
                    inputs.append(self.next_input(accumulated.detach(), scale))
                residual = F.interpolate(z.detach() - accumulated.detach(), (pn, pn), mode="area")
                signals = residual.permute(0, 2, 3, 1).reshape(-1, self.Cvae)
                with torch.no_grad():
                    pairs = [self.dictionary.batch_omp_with_support(chunk.T, dictionary.detach())
                             for chunk in signals.split(self.omp_chunk_size)]
                    atoms = torch.cat([p[0] for p in pairs])
                    values = torch.cat([p[1] for p in pairs])
                    if calibrate:
                        bound = torch.quantile(values.abs().flatten(), .9995).clamp_min(.1)
                        self.coefficient_max[scale].copy_(torch.maximum(self.coefficient_max[scale], bound))
                    elif self.training and self.coefficient_range_decay is not None:
                        # A scratch encoder's latent scale changes while learning.
                        # Track training-only ranges, synchronized across ranks;
                        # evaluation and the frozen prior never update these bins.
                        bound = torch.quantile(values.abs().flatten(), .9995).clamp_min(.1) * 1.2
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(bound, op=torch.distributed.ReduceOp.MAX)
                        decay = self.coefficient_range_decay
                        self.coefficient_max[scale].mul_(decay).add_(bound * (1-decay))
                    clipping.append((values.abs() > self.coefficient_max[scale]).float().mean())
                    coefficients = self.coefficient_ids(values, scale)
                atoms = atoms.reshape(z.shape[0], pn * pn, self.sparsity)
                coefficients = coefficients.reshape_as(atoms)
                # Gradients update the shared dictionary and residual convolutions;
                # support selection and least-squares coefficients are fixed-code.
                accumulated = accumulated + self.contribution(self.embed(atoms, coefficients, scale), scale)
                losses.append(F.mse_loss(accumulated, z.detach()) + self.beta * F.mse_loss(accumulated.detach(), z))
                atom_maps.append(atoms)
                coeff_maps.append(coefficients)
            return dict(latent=accumulated, atoms=torch.cat(atom_maps, 1),
                        coefficients=torch.cat(coeff_maps, 1), inputs=torch.cat(inputs, 1),
                        loss=torch.stack(losses).mean(), clip_fraction=torch.stack(clipping).mean())

    def forward(self, latent, ret_usages=False):
        result = self.decompose(latent)
        self.last_atom_ids = result['atoms'].detach()
        self.last_clip_fraction = result['clip_fraction'].detach()
        straight_through = latent.float() + (result["latent"] - latent.float()).detach()
        return straight_through, None, result["loss"]

    def from_codes(self, atoms, coefficients):
        final = self.v_patch_nums[-1]
        accumulated = self.coefficient_max.new_zeros(atoms.shape[0], self.Cvae, final, final)
        inputs, offset = [], 0
        with torch.autocast(device_type=atoms.device.type, enabled=False):
            for scale, pn in enumerate(self.v_patch_nums):
                if scale:
                    inputs.append(self.next_input(accumulated, scale))
                end = offset + pn * pn
                accumulated = accumulated + self.contribution(self.embed(atoms[:, offset:end], coefficients[:, offset:end], scale), scale)
                offset = end
        if offset != atoms.shape[1] or atoms.shape != coefficients.shape:
            raise ValueError("Sparse codes do not match scale layout")
        return accumulated, torch.cat(inputs, 1)


class LaserVQVAE(VQVAE):
    def __init__(self, *, pretrained=None, sparsity=2, coefficient_bins=257, ch=160,
                 channels=32, atoms=4096, patch_nums=PATCH_NUMS):
        super().__init__(vocab_size=atoms, z_channels=channels, ch=ch,
                         v_patch_nums=patch_nums, test_mode=False)
        if pretrained:
            # Strictly load the released VQ backbone before replacing its bottleneck.
            super().load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
        old = self.quantize
        self.quantize = MultiScaleLaser(channels, atoms, sparsity, patch_nums,
                                       coefficient_bins, quant_resi=old.quant_resi)
        with torch.no_grad():
            self.quantize.dictionary.dictionary.copy_(F.normalize(old.embedding.weight.T.float(), dim=0))

    @torch.no_grad()
    def tokenize(self, images, calibrate=False):
        return self.quantize.decompose(self.quant_conv(self.encoder(images)), calibrate=calibrate)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # VQVAE's override assumes EMA VQ buffers, which LASER intentionally lacks.
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)


class LaserVAR(VAR):
    """Official VAR body, with a small causal sparse-depth output boundary."""
    def __init__(self, tokenizer, depth=16, width=None, heads=None, num_classes=1000):
        q = tokenizer.quantize
        proxy = SimpleNamespace(Cvae=q.Cvae, vocab_size=q.vocab_size, quantize=q)
        super().__init__(proxy, num_classes=num_classes, depth=depth,
                         embed_dim=width or depth * 64, num_heads=heads or depth,
                         patch_nums=q.v_patch_nums, attn_l2_norm=True,
                         drop_path_rate=.1 * depth / 24, norm_eps=1e-6,
                         flash_if_available=False, fused_if_available=False)
        expected_length = sum(pn * pn for pn in q.v_patch_nums)
        if self.L != expected_length:
            raise RuntimeError(
                f"LASER changed the VAR spatial sequence: {self.L} != {expected_length}"
            )
        # Sparse depth is deliberately local to each spatial position. It must
        # never be flattened into the transformer's sequence dimension.
        self.spatial_token_length = self.L
        self.sparsity, self.coefficient_bins = q.sparsity, q.coefficient_bins
        self.sparse_pairs_per_image = self.L * self.sparsity
        self.categorical_decisions_per_image = self.sparse_pairs_per_image * 2
        self.coefficient_head = nn.Linear(self.C, q.coefficient_bins)
        self.atom_context = nn.Linear(q.Cvae, self.C, bias=False)
        self.depth_context = nn.Sequential(nn.Linear(q.Cvae, self.C), nn.SiLU(), nn.Linear(self.C, self.C))
        self.depth_embedding = nn.Embedding(q.sparsity, self.C)
        self.init_weights(init_adaln=.5, init_adaln_gamma=1e-3, init_head=.02, init_std=-1.)
        self.coefficient_head.weight.data.mul_(.02)
        self._q = (q,)

    def get_logits(self, hidden, condition):
        return self.head_nm(hidden.float(), condition).float()

    def depth_features(self, features, prefix, depth):
        return features + self.depth_context(prefix) + self.depth_embedding.weight[depth]

    def token_logits(self, features, atoms, coefficients):
        q = self._q[0]
        atom_vectors = F.embedding(atoms, q.normalized_dictionary().T).detach()
        values = torch.cat([q.coefficient_values(coefficients[:, lo:hi], si)
                            for si, (lo, hi) in enumerate(self.begin_ends)], 1)
        prefix = features.new_zeros(*features.shape[:2], self.Cvae)
        atom_logits, coefficient_logits = [], []
        for depth in range(self.sparsity):
            h = self.depth_features(features, prefix, depth)
            logits = self.head(h)
            if depth:
                logits = logits.scatter(-1, atoms[:, :, :depth], -torch.inf)
            atom_logits.append(logits)
            coefficient_logits.append(self.coefficient_head(h + self.atom_context(atom_vectors[:, :, depth])))
            prefix = prefix + atom_vectors[:, :, depth] * values[:, :, depth, None]
        return torch.stack(atom_logits, 2), torch.stack(coefficient_logits, 2)

    def forward(self, labels, inputs, atoms, coefficients):
        features = super().forward(labels, inputs)
        a, c = self.token_logits(features, atoms, coefficients)
        atom_nll = F.cross_entropy(a.float().flatten(0, 2), atoms.flatten(), reduction="mean")
        coeff_nll = F.cross_entropy(c.float().flatten(0, 2), coefficients.flatten(), reduction="mean")
        # Sum of the conditional NLLs is the joint NLL per sparse pair.
        return atom_nll + coeff_nll, torch.stack((atom_nll.detach(), coeff_nll.detach()))

    @torch.no_grad()
    def sample(self, labels, *, cfg=1.5, top_k=900, top_p=.96, seed=0):
        q, B = self._q[0], len(labels)
        rng = torch.Generator(device=labels.device).manual_seed(seed)
        cond = self.class_emb(torch.cat((labels, torch.full_like(labels, self.num_classes))))
        positions = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        x = cond[:, None] + self.pos_start + positions[:, :self.first_l]
        accumulated = cond.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        offset = 0
        for block in self.blocks:
            block.attn.kv_caching(True)
        try:
            for scale, pn in enumerate(self.patch_nums):
                for block in self.blocks:
                    x = block(x=x, cond_BD=self.shared_ada_lin(cond), attn_bias=None)
                features = self.get_logits(x, cond)
                prefix = accumulated.new_zeros(B, pn * pn, self.Cvae)
                guidance = cfg * scale / self.num_stages_minus_1
                for depth in range(self.sparsity):
                    h = self.depth_features(features, prefix.repeat(2, 1, 1), depth)
                    logits = self.head(h)
                    logits = (1 + guidance) * logits[:B] - guidance * logits[B:]
                    if depth:
                        # Match OMP's no-duplicate support constraint at inference.
                        logits.scatter_(-1, torch.stack(selected, -1), -torch.inf)
                    atoms = sample_with_top_k_top_p_(logits.clone(), rng=rng, top_k=min(top_k, self.V), top_p=top_p)[:, :, 0]
                    if depth == 0:
                        selected = []
                    selected.append(atoms)
                    vectors = F.embedding(atoms, q.normalized_dictionary().T)
                    coeff_logits = self.coefficient_head(h + self.atom_context(vectors.repeat(2, 1, 1)))
                    coeff_logits = (1 + guidance) * coeff_logits[:B] - guidance * coeff_logits[B:]
                    ids = sample_with_top_k_top_p_(coeff_logits.clone(), rng=rng, top_k=0, top_p=top_p)[:, :, 0]
                    prefix = prefix + vectors * q.coefficient_values(ids, scale)[..., None]
                with torch.autocast(device_type=labels.device.type, enabled=False):
                    accumulated = accumulated + q.contribution(prefix.float(), scale)
                offset += pn * pn
                if scale < self.num_stages_minus_1:
                    x = (self.word_embed(q.next_input(accumulated, scale + 1)) +
                         positions[:, offset:offset + self.patch_nums[scale + 1] ** 2]).repeat(2, 1, 1)
            return accumulated
        finally:
            for block in self.blocks:
                block.attn.kv_caching(False)
