"""Shared causal-transformer utilities for maintained model code."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = list[Tuple[torch.Tensor, torch.Tensor]]

_CUDA_GRID_MAX = 65535


def private_long_tensor(
    x: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a private int64 copy for embedding/scatter paths."""
    return x.to(device=device, dtype=torch.long).clone()


def init_transformer_module_weights(module: nn.Module) -> None:
    """GPT-style weight init shared by the maintained transformer models."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        qkv_bias: bool = False,
        out_proj_bias: bool = False,
        chunk_causal_batches: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.attn_drop_p = float(dropout)
        self.chunk_causal_batches = bool(chunk_causal_batches)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.resid_dropout = nn.Dropout(dropout)

    def _scaled_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        drop_p: float,
        use_cache: bool,
    ) -> torch.Tensor:
        if use_cache:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        if self.chunk_causal_batches and q.size(0) * self.n_heads > _CUDA_GRID_MAX:
            chunk_b = max(1, _CUDA_GRID_MAX // self.n_heads)
            return torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q[start : start + chunk_b],
                        k[start : start + chunk_b],
                        v[start : start + chunk_b],
                        is_causal=True,
                        dropout_p=drop_p,
                    )
                    for start in range(0, q.size(0), chunk_b)
                ],
                dim=0,
            )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=drop_p,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        drop_p = self.attn_drop_p if self.training else 0.0
        out = self._scaled_attention(q, k, v, drop_p=drop_p, use_cache=(kv_cache is not None))
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out, (k, v)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional KV cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        *,
        attn_bias: bool = False,
        out_proj_bias: bool = False,
        chunk_causal_batches: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            dropout,
            qkv_bias=attn_bias,
            out_proj_bias=out_proj_bias,
            chunk_causal_batches=chunk_causal_batches,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


def run_transformer_blocks(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    final_norm: Optional[nn.Module],
    *,
    kv_cache: Optional[KVCache] = None,
) -> Tuple[torch.Tensor, KVCache]:
    """Run a stack of cached transformer blocks with an optional final norm."""
    new_cache: KVCache = []
    for block_idx, block in enumerate(blocks):
        block_cache = None if kv_cache is None else kv_cache[block_idx]
        x, block_kv = block(x, kv_cache=block_cache)
        new_cache.append(block_kv)
    if final_norm is not None:
        x = final_norm(x)
    return x, new_cache
