"""Exact chunked global linear/softmax CE for a large integer vocabulary.

Only execution is tiled. Every class contributes to the normalizer and every
classifier row receives its dense softmax gradient. Targets may be sparse.
"""
import torch
from torch.nn import functional as F


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight, bias, ids, probabilities, row_chunk, class_chunk):
        if (features.ndim != 2 or weight.ndim != 2 or features.shape[1] != weight.shape[1]
                or ids.ndim != 2 or ids.shape != probabilities.shape or len(ids) != len(features)
                or ids.dtype != torch.long or probabilities.requires_grad
                or row_chunk < 1 or class_chunk < 1):
            raise ValueError('Invalid global pair-ID classifier or sparse targets')
        dtype = features.dtype
        log_z = features.new_full((len(features),), -torch.inf, dtype=torch.float32)
        target_logits = features.new_zeros((len(features),), dtype=torch.float32)
        for first in range(0, len(features), row_chunk):
            row = features[first:first+row_chunk]
            normalizer = log_z[first:first+len(row)]
            for start in range(0, len(weight), class_chunk):
                b = None if bias is None else bias[start:start+class_chunk].to(dtype)
                logits = F.linear(row, weight[start:start+class_chunk].to(dtype), b).float()
                normalizer.copy_(torch.logaddexp(normalizer, logits.logsumexp(-1)))
            selected = ids[first:first+len(row)]
            w = weight[selected].to(dtype).float()
            # Match the classifier's dot product and output dtype, including
            # AMP rounding, without expanding dense labels or full logits.
            selected_logits = (w * row.float()[:,None,:]).sum(-1)
            if bias is not None:
                selected_logits += bias[selected].to(dtype).float()
            selected_logits = selected_logits.to(dtype).float()
            target_logits[first:first+len(row)] = (
                selected_logits * probabilities[first:first+len(row)].float()).sum(-1)
        saved_bias = bias if bias is not None else features.new_empty(0)
        ctx.save_for_backward(features, weight, saved_bias, ids, probabilities, log_z)
        ctx.has_bias, ctx.row_chunk, ctx.class_chunk = bias is not None, row_chunk, class_chunk
        return (log_z * probabilities.float().sum(-1) - target_logits).mean()

    @staticmethod
    def backward(ctx, grad_output):
        features, weight, bias, ids, probabilities, log_z = ctx.saved_tensors
        dx = torch.zeros_like(features, dtype=torch.float32)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(bias) if ctx.has_bias else None
        dtype = features.dtype
        scale = grad_output.float() / len(features)
        for start in range(0, len(weight), ctx.class_chunk):
            end = min(start+ctx.class_chunk, len(weight))
            w = weight[start:end].to(dtype)
            b = bias[start:end].to(dtype) if ctx.has_bias else None
            for first in range(0, len(features), ctx.row_chunk):
                row = features[first:first+ctx.row_chunk]
                selected = ids[first:first+len(row)]
                p = probabilities[first:first+len(row)].float()
                logits = F.linear(row, w, b).float()
                gradient = (logits - log_z[first:first+len(row), None]).exp() * p.sum(-1, keepdim=True)
                local = (selected-start).clamp(0, end-start-1)
                valid = (selected >= start) & (selected < end)
                gradient.scatter_add_(-1, local, -p * valid)
                gradient.mul_(scale)
                g = gradient.to(dtype)
                dx[first:first+len(row)].add_((g @ w).float())
                dw[start:end].add_((g.T @ row).to(dw.dtype))
                if db is not None:
                    db[start:end].add_(gradient.sum(0).to(db.dtype))
        return dx.to(dtype), dw, db, None, None, None, None


def chunked_pair_linear_cross_entropy(features, weight, bias, ids, probabilities,
                                      row_chunk=128, class_chunk=65536):
    shape = features.shape[:-1]
    if ids.shape[:-1] != shape or probabilities.shape != ids.shape:
        raise ValueError('Sparse pair targets must match the feature grid')
    return _ChunkedLinearCrossEntropy.apply(features.reshape(-1,features.shape[-1]),weight,bias,
        ids.reshape(-1,ids.shape[-1]), probabilities.reshape(-1,probabilities.shape[-1]),
        int(row_chunk), int(class_chunk))
