"""Frozen multiscale residual vectors and a class-conditional VAR prior.

Each ID selects an entire vector in the scale/depth table. No atom mask,
continuous coefficients, or coefficient side channel participates in decoding.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .joint_multiscale_var import JointLaserVAR
from .multiscale_laser_var import VAR, sample_with_top_k_top_p_


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, geometry, vectors, temperature=.5, chunk_size=8192):
        super().__init__()
        vectors = torch.as_tensor(vectors).detach().float().clone()
        assert vectors.ndim == 4 and vectors.shape[0] == len(geometry.v_patch_nums)
        assert vectors.shape[1] == geometry.sparsity and vectors.shape[-1] == geometry.Cvae
        assert torch.isfinite(vectors).all() and (vectors[:, :, 0] == 0).all()
        self.source = geometry.requires_grad_(False).eval()
        self.Cvae, self.sparsity = geometry.Cvae, geometry.sparsity
        self.v_patch_nums = tuple(geometry.v_patch_nums)
        self.vocab_size, self.temperature, self.chunk_size = vectors.shape[2], temperature, chunk_size
        self.register_buffer('vectors', vectors)

    def contribution(self, vectors, scale):
        return self.source.contribution(vectors, scale)

    def next_input(self, accumulated, scale):
        return self.source.next_input(accumulated, scale)

    def embed(self, ids, scale, depth):
        return F.embedding(ids, self.vectors[scale, depth])

    def embed_joint(self, codes, scale):
        return torch.stack([self.embed(codes[..., d], scale, d) for d in range(self.sparsity)], -2)

    def distances(self, residual, scale, depth):
        book = self.vectors[scale, depth]
        return torch.addmm(residual.square().sum(-1, keepdim=True) + book.square().sum(-1),
                           residual, book.T, alpha=-2.)

    @torch.no_grad()
    def decompose(self, latent, stochastic=False):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            z = latent.float()
            assert z.shape[1:] == (self.Cvae, self.v_patch_nums[-1], self.v_patch_nums[-1])
            accumulated = torch.zeros_like(z)
            codes, inputs, residuals = [], [], []
            for scale, pn in enumerate(self.v_patch_nums):
                if scale:
                    inputs.append(self.next_input(accumulated, scale))
                r = F.interpolate(z - accumulated, (pn, pn), mode='area').flatten(2).transpose(1, 2).contiguous()
                ids, teacher_r = [], []
                total = torch.zeros_like(r)
                for depth in range(self.sparsity):
                    teacher_r.append(r.clone())
                    parts = []
                    for part in r.reshape(-1, self.Cvae).split(self.chunk_size):
                        distance = self.distances(part, scale, depth)
                        chosen = (torch.multinomial((-distance / self.temperature).softmax(-1), 1).squeeze(-1)
                                  if stochastic else distance.argmin(-1))
                        parts.append(chosen)
                    chosen = torch.cat(parts).reshape(len(z), pn * pn)
                    value = self.embed(chosen, scale, depth)
                    ids.append(chosen); total.add_(value); r = r - value
                codes.append(torch.stack(ids, -1)); residuals.append(torch.stack(teacher_r, -2))
                accumulated = accumulated + self.contribution(total, scale)
            return dict(codes=torch.cat(codes, 1), inputs=torch.cat(inputs, 1),
                        latent=accumulated, residuals=torch.cat(residuals, 1))

    def from_codes(self, codes):
        assert codes.dtype == torch.long and codes.shape[1:] == (sum(p*p for p in self.v_patch_nums), self.sparsity)
        assert (codes >= 0).all() and (codes < self.vocab_size).all()
        accumulated = self.vectors.new_zeros(len(codes), self.Cvae, self.v_patch_nums[-1], self.v_patch_nums[-1])
        inputs, offset = [], 0
        for scale, pn in enumerate(self.v_patch_nums):
            if scale: inputs.append(self.next_input(accumulated, scale))
            # Add in the same order as decompose for a bit-exact decoder interface.
            total = torch.zeros(len(codes), pn*pn, self.Cvae, device=codes.device)
            for depth in range(self.sparsity): total.add_(self.embed(codes[:, offset:offset+pn*pn, depth], scale, depth))
            accumulated = accumulated + self.contribution(total, scale)
            offset += pn*pn
        return accumulated, torch.cat(inputs, 1)


class ResidualVectorVAR(JointLaserVAR):
    def forward(self, labels, inputs, codes, residuals=None):
        features = VAR.forward(self, labels, inputs)
        q = self._q[0]
        prefix = features.new_zeros(*features.shape[:2], self.Cvae)
        loss = features.new_zeros((), dtype=torch.float32)
        for depth in range(self.sparsity):
            scores = self.prefix_logits(features, prefix, depth)
            if residuals is None:
                loss = loss + F.cross_entropy(scores.flatten(0, 1), codes[..., depth].flatten()) / self.sparsity
            else:
                for scale, (lo, hi) in enumerate(self.begin_ends):
                    with torch.no_grad(), torch.autocast(device_type=labels.device.type, enabled=False):
                        r = residuals[:, lo:hi, depth].reshape(-1, self.Cvae).float()
                        target = (-q.distances(r, scale, depth) / q.temperature).softmax(-1)
                    logits = scores[:, lo:hi].reshape(-1, self.V)
                    loss = loss + F.cross_entropy(logits.float(), target) * ((hi-lo) / (self.L*self.sparsity))
            prefix = prefix + torch.cat([q.embed(codes[:, lo:hi, depth], scale, depth)
                                         for scale, (lo, hi) in enumerate(self.begin_ends)], 1)
        return loss

    @torch.no_grad()
    def sample(self, labels, cfg=1.5, top_k=900, top_p=.96, seed=0):
        q, batch = self._q[0], len(labels)
        rng = torch.Generator(device=labels.device).manual_seed(seed)
        condition = self.class_emb(torch.cat((labels, torch.full_like(labels, self.num_classes))))
        positions = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        x = condition[:, None] + self.pos_start + positions[:, :self.first_l]
        accumulated = q.vectors.new_zeros(batch, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        offset = 0
        for block in self.blocks: block.attn.kv_caching(True)
        try:
            for scale, pn in enumerate(self.patch_nums):
                for block in self.blocks: x = block(x=x, cond_BD=self.shared_ada_lin(condition), attn_bias=None)
                features = self.get_logits(x, condition)
                prefix = accumulated.new_zeros(batch, pn*pn, self.Cvae)
                guidance = cfg*scale/self.num_stages_minus_1
                for depth in range(self.sparsity):
                    scores = self.prefix_logits(features, prefix.repeat(2, 1, 1), depth)
                    scores = (1+guidance)*scores[:batch] - guidance*scores[batch:]
                    ids = sample_with_top_k_top_p_(scores, rng=rng, top_k=min(top_k, self.V), top_p=top_p)[:, :, 0]
                    prefix.add_(q.embed(ids, scale, depth))
                with torch.autocast(device_type=labels.device.type, enabled=False):
                    accumulated = accumulated + q.contribution(prefix.float(), scale)
                offset += pn*pn
                if scale < self.num_stages_minus_1:
                    x = (self.word_embed(q.next_input(accumulated, scale+1)) + positions[:, offset:offset+self.patch_nums[scale+1]**2]).repeat(2, 1, 1)
            return accumulated
        finally:
            for block in self.blocks: block.attn.kv_caching(False)


class AnchoredResidualVectorQuantizer(ResidualVectorQuantizer):
    """Vector-ID catalog centered on the frozen stage-1 contributions.

    Hard IDs reproduce stage 1 exactly. The full categorical teacher perturbs
    complete contributions, compensates the first sampled vector at depth two,
    and re-encodes each later spatial scale against the sampled latent prefix.
    """
    def __init__(self, anchor_quantizer, vectors, **kwargs):
        super().__init__(anchor_quantizer.source, vectors, **kwargs)
        self._anchor = (anchor_quantizer.requires_grad_(False).eval(),)

    @torch.no_grad()
    def decompose(self, latent, stochastic=False):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            z=latent.float();anchor=self._anchor[0];accumulated=torch.zeros_like(z)
            codes,inputs,residuals=[],[],[]
            for scale,pn in enumerate(self.v_patch_nums):
                if scale:inputs.append(self.next_input(accumulated,scale))
                signal=F.interpolate(z-accumulated,(pn,pn),mode='area').flatten(2).transpose(1,2).contiguous()
                old_ids,_=anchor.quantize_scale(signal,scale,chunk_size=2048)
                old_vectors=anchor.embed_joint(old_ids,scale)
                total=torch.zeros_like(signal);correction=torch.zeros_like(signal)
                new_ids=[];teacher_r=[]
                for depth in range(self.sparsity):
                    center=old_vectors[...,depth,:]+correction
                    teacher_r.append(center)
                    if stochastic:
                        pieces=[]
                        for part in center.reshape(-1,self.Cvae).split(self.chunk_size):
                            distance=self.distances(part,scale,depth)
                            pieces.append(torch.multinomial((-distance/self.temperature).softmax(-1),1).squeeze(-1))
                        chosen=torch.cat(pieces).reshape(len(z),pn*pn)
                    else:chosen=old_ids[...,depth]+1
                    value=self.embed(chosen,scale,depth)
                    new_ids.append(chosen);total.add_(value)
                    correction=center-value
                codes.append(torch.stack(new_ids,-1));residuals.append(torch.stack(teacher_r,-2))
                accumulated=accumulated+self.contribution(total,scale)
            return dict(codes=torch.cat(codes,1),inputs=torch.cat(inputs,1),latent=accumulated,residuals=torch.cat(residuals,1))


class SharedResidualVectorQuantizer(ResidualVectorQuantizer):
    """One physical vector table, identical at every scale and sparse depth."""
    def __init__(self, geometry, vectors, temperature=.5, chunk_size=8192):
        nn.Module.__init__(self)
        vectors = torch.as_tensor(vectors).detach().float().clone()
        if (vectors.ndim != 2 or vectors.shape[1] != geometry.Cvae or
                len(vectors) < 2 or not torch.isfinite(vectors).all() or
                (vectors[0] != 0).any()):
            raise ValueError('Expected one finite [vocabulary, channels] table with zero ID 0')
        self.source = geometry.requires_grad_(False).eval()
        self.Cvae, self.sparsity = geometry.Cvae, geometry.sparsity
        self.v_patch_nums = tuple(geometry.v_patch_nums)
        self.vocab_size, self.temperature, self.chunk_size = len(vectors), temperature, chunk_size
        self.register_buffer('vectors', vectors)

    def embed(self, ids, scale, depth):
        return F.embedding(ids, self.vectors)

    def distances(self, residual, scale, depth):
        return torch.addmm(residual.square().sum(-1, keepdim=True) + self.vectors.square().sum(-1),
                           residual, self.vectors.T, alpha=-2.)

    def from_codes(self, codes):
        with torch.autocast(device_type=codes.device.type, enabled=False):
            return super().from_codes(codes)


class SharedAnchoredResidualVectorQuantizer(SharedResidualVectorQuantizer):
    """Re-encode source contributions using a strictly shared complete-vector book.

    The source supplies encoder targets only. Every emitted ID is decoded by the
    same global vector table, without scale/depth gains or an auxiliary channel.
    """
    def __init__(self, anchor_quantizer, vectors, **kwargs):
        super().__init__(anchor_quantizer.source, vectors, **kwargs)
        self._anchor = (anchor_quantizer.requires_grad_(False).eval(),)

    @torch.no_grad()
    def decompose(self, latent, stochastic=False):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            z = latent.float()
            anchor = self._anchor[0]
            accumulated = torch.zeros_like(z)
            codes, inputs, residuals = [], [], []
            for scale, pn in enumerate(self.v_patch_nums):
                if scale:
                    inputs.append(self.next_input(accumulated, scale))
                signal = F.interpolate(z-accumulated, (pn,pn), mode='area').flatten(2).transpose(1,2).contiguous()
                old_ids, _ = anchor.quantize_scale(signal, scale, chunk_size=2048)
                desired = anchor.embed_joint(old_ids, scale)
                total, correction = torch.zeros_like(signal), torch.zeros_like(signal)
                selected, centers = [], []
                for depth in range(self.sparsity):
                    center = desired[...,depth,:] + correction
                    centers.append(center)
                    pieces = []
                    for part in center.reshape(-1,self.Cvae).split(self.chunk_size):
                        distance = self.distances(part, scale, depth)
                        chosen = (torch.multinomial((-distance/self.temperature).softmax(-1),1).squeeze(-1)
                                  if stochastic else distance.argmin(-1))
                        pieces.append(chosen)
                    chosen = torch.cat(pieces).reshape(len(z),pn*pn)
                    value = self.embed(chosen,scale,depth)
                    selected.append(chosen)
                    total.add_(value)
                    correction = center-value
                codes.append(torch.stack(selected,-1))
                residuals.append(torch.stack(centers,-2))
                accumulated = accumulated+self.contribution(total,scale)
            return dict(codes=torch.cat(codes,1),inputs=torch.cat(inputs,1),latent=accumulated,
                        residuals=torch.cat(residuals,1))
