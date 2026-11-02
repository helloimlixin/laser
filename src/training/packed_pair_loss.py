"""Exact single-softmax CE for sparse labels on full Cartesian pair IDs."""
import torch


class _SparsePairCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, atoms, coefficient_probabilities, row_chunk_size):
        bins = coefficient_probabilities.shape[-1]
        if (logits.shape[:-1] != atoms.shape or coefficient_probabilities.shape[:-1] != atoms.shape
                or logits.shape[-1] % bins or atoms.dtype != torch.long
                or coefficient_probabilities.requires_grad or row_chunk_size < 1):
            raise ValueError('Incompatible joint logits, sparse labels, or row chunk size')
        ctx.save_for_backward(logits, atoms, coefficient_probabilities)
        ctx.row_chunk_size = int(row_chunk_size)
        x = logits.reshape(-1, logits.shape[-1])
        a, p = atoms.reshape(-1), coefficient_probabilities.reshape(-1, bins)
        offsets = torch.arange(bins, device=logits.device)
        loss = logits.new_zeros((), dtype=torch.float32)
        for first in range(0, len(x), ctx.row_chunk_size):
            rows = x[first:first + ctx.row_chunk_size].float()
            probabilities = p[first:first + ctx.row_chunk_size].float()
            ids = a[first:first + len(rows), None] * bins + offsets
            selected = rows.gather(-1, ids)
            loss += (rows.logsumexp(-1) * probabilities.sum(-1)
                     - (selected * probabilities).sum(-1)).sum()
        return loss / len(x)

    @staticmethod
    def backward(ctx, grad_output):
        logits, atoms, coefficient_probabilities = ctx.saved_tensors
        bins = coefficient_probabilities.shape[-1]
        x = logits.reshape(-1, logits.shape[-1])
        a, p = atoms.reshape(-1), coefficient_probabilities.reshape(-1, bins)
        offsets = torch.arange(bins, device=logits.device)
        gradient = torch.empty_like(x)
        for first in range(0, len(x), ctx.row_chunk_size):
            rows = x[first:first + ctx.row_chunk_size].float()
            probabilities = p[first:first + ctx.row_chunk_size].float()
            ids = a[first:first + len(rows), None] * bins + offsets
            g = rows.softmax(-1) * probabilities.sum(-1, keepdim=True)
            g.scatter_add_(-1, ids, -probabilities)
            gradient[first:first + len(rows)] = g * (grad_output / len(x))
        return gradient.reshape_as(logits), None, None, None


def sparse_pair_cross_entropy(logits, atoms, coefficient_probabilities, row_chunk_size=1):
    """Normalize over *all* pairs; sparse labels only save target memory.

    This is exactly native soft-target CE with target[atom*C+bin]=p(bin).
    It is not a sum/mean of independently normalized atom and coefficient CE.
    The dense classifier logits still exist, and must be budgeted separately.
    """
    return _SparsePairCrossEntropy.apply(logits, atoms, coefficient_probabilities, row_chunk_size)
