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
        window_size: Optional[int] = None,
        n_global_prefix_tokens: int = 0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.attn_drop_p = float(dropout)
        self.chunk_causal_batches = bool(chunk_causal_batches)
        self.window_size = None if window_size is None else int(window_size)
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError(f"window_size must be positive when set, got {self.window_size}")
        self.n_global_prefix_tokens = int(n_global_prefix_tokens)
        if self.n_global_prefix_tokens < 0:
            raise ValueError(
                f"n_global_prefix_tokens must be >= 0, got {self.n_global_prefix_tokens}"
            )
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.resid_dropout = nn.Dropout(dropout)

    def _trim_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.window_size is None:
            return k, v
        prefix = min(int(self.n_global_prefix_tokens), int(k.size(2)))
        max_total = prefix + int(self.window_size)
        if k.size(2) <= max_total:
            return k, v
        if prefix > 0:
            return (
                torch.cat([k[:, :, :prefix, :], k[:, :, -int(self.window_size) :, :]], dim=2),
                torch.cat([v[:, :, :prefix, :], v[:, :, -int(self.window_size) :, :]], dim=2),
            )
        return (
            k[:, :, -int(self.window_size) :, :],
            v[:, :, -int(self.window_size) :, :],
        )

    def _windowed_attention_mask(
        self,
        *,
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype,
        n_global_prefix_tokens: int = 0,
        k_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_pos = torch.arange(q_start, q_end, device=device)
        if k_pos is None:
            k_pos = torch.arange(k_start, k_end, device=device)
        else:
            k_pos = torch.as_tensor(k_pos, device=device)
        causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        local = k_pos.unsqueeze(0) > (q_pos.unsqueeze(1) - int(window_size))
        if int(n_global_prefix_tokens) > 0:
            global_prefix = k_pos.unsqueeze(0) < int(n_global_prefix_tokens)
            allowed = causal & (global_prefix | local)
        else:
            allowed = causal & local
        mask = torch.full((1, 1, q_pos.numel(), k_pos.numel()), float("-inf"), device=device, dtype=dtype)
        mask.masked_fill_(allowed.unsqueeze(0).unsqueeze(0), 0.0)
        return mask

    def _windowed_causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        drop_p: float,
        window_size: int,
        n_global_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        seq_len = q.size(2)
        full_window = int(window_size) + max(0, int(n_global_prefix_tokens))
        if seq_len <= full_window:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=drop_p,
            )

        outs = []
        chunk_size = int(window_size)
        for q_start in range(0, seq_len, chunk_size):
            q_end = min(seq_len, q_start + chunk_size)
            q_chunk = q[:, :, q_start:q_end, :]
            if int(n_global_prefix_tokens) > 0:
                prefix = min(int(n_global_prefix_tokens), q_end)
                local_start = max(prefix, q_start - int(window_size) + 1)
                prefix_k = k[:, :, :prefix, :]
                prefix_v = v[:, :, :prefix, :]
                tail_k = k[:, :, local_start:q_end, :]
                tail_v = v[:, :, local_start:q_end, :]
                if tail_k.size(2) > 0:
                    k_chunk = torch.cat([prefix_k, tail_k], dim=2)
                    v_chunk = torch.cat([prefix_v, tail_v], dim=2)
                    k_pos = torch.cat(
                        [
                            torch.arange(0, prefix, device=q.device),
                            torch.arange(local_start, q_end, device=q.device),
                        ],
                        dim=0,
                    )
                else:
                    k_chunk = prefix_k
                    v_chunk = prefix_v
                    k_pos = torch.arange(0, prefix, device=q.device)
                attn_mask = self._windowed_attention_mask(
                    q_start=q_start,
                    q_end=q_end,
                    k_start=0,
                    k_end=0,
                    window_size=int(window_size),
                    device=q.device,
                    dtype=q.dtype,
                    n_global_prefix_tokens=int(n_global_prefix_tokens),
                    k_pos=k_pos,
                )
            else:
                k_start = max(0, q_start - int(window_size) + 1)
                k_chunk = k[:, :, k_start:q_end, :]
                v_chunk = v[:, :, k_start:q_end, :]
                attn_mask = self._windowed_attention_mask(
                    q_start=q_start,
                    q_end=q_end,
                    k_start=k_start,
                    k_end=q_end,
                    window_size=int(window_size),
                    device=q.device,
                    dtype=q.dtype,
                )
            outs.append(
                F.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attn_mask=attn_mask,
                    dropout_p=drop_p,
                )
            )
        return torch.cat(outs, dim=2)

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
        if self.window_size is not None and q.size(2) > self.window_size:
            return self._windowed_causal_attention(
                q,
                k,
                v,
                drop_p=drop_p,
                window_size=int(self.window_size),
                n_global_prefix_tokens=int(self.n_global_prefix_tokens),
            )
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
            if seq_len == 1:
                # Cached decoding feeds one position at a time; trim the stored
                # window here so generation memory stays O(prefix + window).
                k, v = self._trim_kv_cache(k, v)

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
        window_size: Optional[int] = None,
        n_global_prefix_tokens: int = 0,
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
            window_size=window_size,
            n_global_prefix_tokens=n_global_prefix_tokens,
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
