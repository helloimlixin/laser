"""Experimental depth-weighted soft CE with bounded FP32 intermediates."""
import math

import torch


def normalize_depth_weights(values, depth):
    values = [float(value) for value in values]
    if len(values) != depth or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('One positive finite loss weight is required per depth')
    maximum = max(values)
    scaled = [value / maximum for value in values]
    total = math.fsum(scaled)
    normalized = [depth * value / total for value in scaled]
    if any(value < torch.finfo(torch.float32).tiny for value in normalized):
        raise ValueError('Normalized loss weights must remain positive in FP32')
    return normalized


class _DepthWeightedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, weights, chunk_size):
        ctx.save_for_backward(logits, targets, weights)
        ctx.chunk_size = chunk_size
        depth = len(weights)
        xs = logits.reshape(-1, logits.shape[-1])
        ys = targets.reshape_as(xs)
        totals = logits.new_zeros(2, depth, dtype=torch.float32)
        for start in range(0, len(xs), chunk_size):
            x, y = xs[start:start + chunk_size].float(), ys[start:start + chunk_size]
            positions = torch.arange(start, start + len(x), device=logits.device) % depth
            ce = -(y * x.log_softmax(-1)).sum(-1)
            entropy = -(y * y.clamp_min(1.e-30).log()).sum(-1)
            totals[0].scatter_add_(0, positions, ce)
            totals[1].scatter_add_(0, positions, entropy)
        stats = totals / (len(xs) // depth)
        loss = (stats[0] * weights).mean()
        ctx.mark_non_differentiable(stats)
        return loss, stats

    @staticmethod
    def backward(ctx, grad_loss, grad_stats):
        logits, targets, weights = ctx.saved_tensors
        xs = logits.reshape(-1, logits.shape[-1])
        ys = targets.reshape_as(xs)
        gradient = torch.empty_like(xs)
        for start in range(0, len(xs), ctx.chunk_size):
            x, y = xs[start:start + ctx.chunk_size], ys[start:start + ctx.chunk_size]
            positions = torch.arange(start, start + len(x), device=logits.device) % len(weights)
            delta = x.float().softmax(-1) * y.sum(-1, keepdim=True) - y
            gradient[start:start + len(x)] = delta * weights[positions, None] * (grad_loss / len(xs))
        return gradient.reshape_as(logits), None, None, None


def depth_weighted_soft_cross_entropy(logits, targets, depth_weights, chunk_size=128):
    """Return weighted CE and detached, unweighted per-depth diagnostics.

    The penultimate axis is residual depth. Weights are normalized to mean one;
    global-batch/DDP scaling remains the caller's responsibility. Targets are
    fixed FP32 distributions and retain their actual row sums in backward.
    """
    if logits.ndim < 2 or logits.shape != targets.shape or logits.numel() == 0:
        raise ValueError('Expected matching nonempty [..., depth, vocabulary] tensors')
    if targets.requires_grad or targets.dtype != torch.float32 or targets.device != logits.device:
        raise ValueError('Expected fixed FP32 targets on the logits device')
    if chunk_size < 1 or not isinstance(chunk_size, int):
        raise ValueError('Positive integer chunk size required')
    weights = logits.new_tensor(normalize_depth_weights(depth_weights, logits.shape[-2]), dtype=torch.float32)
    loss, stats = _DepthWeightedCE.apply(logits, targets, weights, chunk_size)
    metrics = dict(unweighted_ce=stats[0].mean(), target_entropy=stats[1].mean(),
                   target_kl=(stats[0] - stats[1]).mean())
    for d in range(len(weights)):
        metrics[f'ce_depth{d + 1}'] = stats[0, d]
        metrics[f'kl_depth{d + 1}'] = stats[0, d] - stats[1, d]
    return loss, metrics
