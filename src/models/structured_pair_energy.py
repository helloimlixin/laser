"""A normalized joint pair classifier with a factorized physical interaction.

For a pair (a,b), s(a,b|h) = u_a(h) + v_b(h) + c_b <D_a,q(h)>.
The physical term has separation rank one in atom/coefficient indices. Only
A+B output rows and a latent query are stored, never an A*B classifier table.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _partition_forward(coupling, bias, bins):
    scores = coupling[:, :, None]*bins + bias[:, None, :]
    partition = scores.logsumexp(-1)
    mean = (torch.exp(scores-partition[:, :, None])*bins).sum(-1)
    return partition, mean


def _partition_bias_backward(coupling, bias, bins, partition, gradient):
    probability = torch.exp(coupling[:, :, None]*bins + bias[:, None, :]-partition[:, :, None])
    return (probability*gradient[:, :, None]).sum(1)


# Fusing these reductions avoids retaining the Cartesian score grid. Eager CPU
# execution is also available for tiny numerical oracles and gradient checks.
_compiled_forward = torch.compile(_partition_forward, fullgraph=True)
_compiled_backward = torch.compile(_partition_bias_backward, fullgraph=True)


class _CoefficientPartition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coupling, bias, bins, row_chunk):
        forward = _compiled_forward if coupling.is_cuda else _partition_forward
        partition, means = torch.empty_like(coupling), torch.empty_like(coupling)
        for first in range(0, len(coupling), row_chunk):
            end = first+row_chunk
            partition[first:end], means[first:end] = forward(coupling[first:end], bias[first:end], bins)
        ctx.save_for_backward(coupling, bias, bins, partition, means)
        ctx.row_chunk = row_chunk
        return partition

    @staticmethod
    def backward(ctx, gradient):
        coupling, bias, bins, partition, means = ctx.saved_tensors
        backward = _compiled_backward if coupling.is_cuda else _partition_bias_backward
        bias_gradient = torch.empty_like(bias)
        for first in range(0, len(coupling), ctx.row_chunk):
            end = first+ctx.row_chunk
            bias_gradient[first:end] = backward(coupling[first:end], bias[first:end], bins,
                                                 partition[first:end], gradient[first:end])
        return gradient*means, bias_gradient, None, None


def coefficient_partition(coupling, bias, bins, *, row_chunk=32):
    """Exact logsumexp over all coefficient bins for every atom, bounded memory.

    Supplies first derivatives for model training. Bins are frozen codebook
    values. No full Cartesian target grid is stored in the autograd graph.
    """
    if (coupling.ndim != 2 or bias.ndim != 2 or bins.ndim != 1 or
            len(coupling) != len(bias) or bias.shape[-1] != len(bins) or row_chunk<1):
        raise ValueError('expected coupling [N,A], bias [N,B], bins [B]')
    if bins.requires_grad:
        raise ValueError('this partition uses frozen coefficient bins')
    return _CoefficientPartition.apply(coupling, bias, bins, row_chunk)


class StructuredPairEnergyRQ(nn.Module):
    """Reuse the causal pair RQ backbone; classify complete pairs directly."""
    def __init__(self, backbone, dictionary, bins, *, physical_interaction=True, row_chunk=32):
        super().__init__()
        if backbone.mask_seen_atoms_training:
            raise ValueError('fixed-pair residual quantization must permit repeated atoms')
        self.backbone = backbone
        backbone.capture_shared_hidden = True
        self.register_buffer('dictionary', dictionary.detach().float().clone())
        self.register_buffer('bins', bins.detach().float().clone())
        self.physical_interaction = bool(physical_interaction)
        self.row_chunk = int(row_chunk)
        self.query = nn.Linear(backbone.config.embed_dim, dictionary.shape[0], bias=False)
        nn.init.zeros_(self.query.weight)
        if not self.physical_interaction:
            self.query.requires_grad_(False)

    def classify(self, hidden, current_atoms, *, depth_index=None):
        atom_bias = self.backbone.classifier(hidden).float()
        coefficient_bias = self.backbone.classify_coefficients(hidden, depth_index=depth_index).float()
        if self.physical_interaction:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                features = self.backbone.classifier.layer_norm(hidden.float())
                query = self.query(features)
                coupling = F.linear(query, self.dictionary.T)
                partition = coefficient_partition(coupling.reshape(-1, coupling.shape[-1]),
                    coefficient_bias.reshape(-1, coefficient_bias.shape[-1]), self.bins,
                    row_chunk=self.row_chunk).reshape_as(coupling)
        else:
            coupling = torch.zeros_like(atom_bias)
            partition = coefficient_bias.logsumexp(-1, keepdim=True).expand_as(coupling)
        selected = coupling.gather(-1, current_atoms[..., None])
        return dict(atom_logits=atom_bias+partition,
                    coeff_logits=coefficient_bias+selected*self.bins,
                    atom_bias=atom_bias, coefficient_bias=coefficient_bias,
                    coupling=coupling, coefficient_partition=partition)

    def forward(self, tokens, model_aux=None, cond=None, amp=False):
        hidden = self.backbone(tokens, model_aux=model_aux, cond=cond, amp=amp)['shared_hidden']
        atoms, _ = self.backbone.fields(tokens)
        return self.classify(hidden, atoms)

    def init_cache(self):
        self.backbone.init_cache()

    @torch.no_grad()
    def cached_pair(self, tokens, model_aux, cond=None, *, sample_loc=(0,0,0), amp=False):
        """Exactly one native RQ step; hypothetical pairs cannot alter history."""
        h,w,depth = sample_loc
        atoms, coefficients = self.backbone.fields(tokens)
        self.backbone._offset_coefficients = coefficients
        try:
            hidden = self.backbone.cached_head_output(atoms, model_aux, cond, sample_loc, amp=amp)
            return self.classify(hidden, atoms[:, h,w,depth], depth_index=depth)
        finally:
            self.backbone._offset_coefficients = None
            self.backbone._coefficient_sample_context = None

    @torch.no_grad()
    def sample_joint_sparse(self, batch_size, model_aux, cond=None, *, top_k=4096,
                            temperature=1., top_p=1., amp=True, generator=None):
        from src.joint_pair_topk import sample_joint_pair_topk
        if self.training:
            raise ValueError('sampling requires evaluation mode')
        if top_k==0 and (temperature!=1. or top_p!=1.):
            raise ValueError('unfiltered joint draws use temperature=1 and top_p=1')
        tokens = model_aux.empty_tokens(batch_size, *self.backbone.block_size[:2])
        self.init_cache()
        try:
            for h in range(self.backbone.block_size[0]):
                for w in range(self.backbone.block_size[1]):
                    for depth in range(self.backbone.block_size[-1]):
                        with torch.autocast(device_type=tokens.device.type, dtype=torch.bfloat16, enabled=amp):
                            output = self.cached_pair(tokens, model_aux, cond, sample_loc=(h,w,depth), amp=amp)
                        def coefficient_scorer(rows, candidates):
                            interaction = output['coupling'][rows[:,None], candidates]
                            return output['coefficient_bias'][rows,None] + interaction[...,None]*self.bins
                        if top_k==0:
                            a=torch.multinomial(output['atom_logits'].float().softmax(-1),1,generator=generator).squeeze(-1)
                            c=coefficient_scorer(torch.arange(batch_size,device=a.device),a[:,None])[:,0]
                            b=torch.multinomial(c.float().softmax(-1),1,generator=generator).squeeze(-1)
                        else:
                            a,b,_ = sample_joint_pair_topk(output['atom_logits'], coefficient_scorer,
                                top_k=top_k, temperature=temperature, top_p=top_p, generator=generator)
                        tokens[:,h,w,2*depth] = a
                        tokens[:,h,w,2*depth+1] = b+self.backbone.num_atoms
        finally:
            self.init_cache()
        return tokens


def structured_pair_cross_entropy(output, targets, bins):
    """Full joint soft CE over all teacher candidates, without sampling a field.

    The model normalizer covers A*B outcomes exactly; only the teacher is
    truncated to its certified top K. Both field gradients see every candidate.
    """
    a,b,w = targets.candidate_atoms, targets.candidate_bins, targets.pair_weights
    u = output['atom_bias'].gather(-1,a)
    v = output['coefficient_bias'].gather(-1,b)
    coupling = output['coupling'].gather(-1,a)
    g = output['coefficient_partition'].gather(-1,a)
    logz = output['atom_logits'].logsumexp(-1)
    atom_ce = logz-((u+g)*w).sum(-1)
    coeff_ce = ((g-v-coupling*bins[b])*w).sum(-1)
    axes = tuple(range(atom_ce.ndim-1))
    return (atom_ce+coeff_ce).mean(), atom_ce.mean(axes), coeff_ce.mean(axes)
