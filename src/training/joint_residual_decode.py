"""Decode fixed joint codes with explicit physical residual weights."""
import math

import torch


@torch.no_grad()
def decode_residuals(tokenizer, codes, scales):
    depth = int(tokenizer.quantizer.depth)
    if (len(scales) != depth or codes.shape[-1] != depth or
            any(not math.isfinite(s) or s < 0 for s in scales)):
        raise ValueError('One finite nonnegative scale is required per residual depth')
    if all(s == 1 for s in scales):
        return tokenizer.decode_code(codes)
    embeddings = tokenizer.quantizer.embed(codes)
    weights = embeddings.new_tensor(scales).reshape(*([1]*(embeddings.ndim-2)),depth,1)
    return tokenizer.decode((embeddings*weights).sum(-2))
