"""
Simple minGPT-style prior for quantized LASER sparse codes.

The model consumes the shared quantized token stream directly:
  atom ids:   [0, num_atoms)
  coeff bins: [num_atoms, num_atoms + num_coeff_bins)

Inputs/outputs use the same [B, H*W, D] token-grid layout as SpatialDepthPrior
so stage-2 training, sampling, and decoding can stay generic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .transformer_core import (
    TransformerBlock,
    init_transformer_module_weights,
    private_long_tensor,
    run_transformer_blocks,
)

_private_long_tensor = private_long_tensor


@dataclass
class MinGPTQuantizedPriorConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    atom_vocab_size: int
    coeff_vocab_size: int
    window_sites: int = 0
    n_global_spatial_tokens: int = 0
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 1024
    dropout: float = 0.1


def build_mingpt_quantized_prior_config(
    bottleneck,
    *,
    H: int,
    W: int,
    D: int,
    window_sites: int,
    n_global_spatial_tokens: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
) -> MinGPTQuantizedPriorConfig:
    atom_vocab_size = int(bottleneck.num_embeddings)
    coeff_vocab_size = int(bottleneck.n_bins)
    vocab_size = atom_vocab_size + coeff_vocab_size
    return MinGPTQuantizedPriorConfig(
        vocab_size=vocab_size,
        H=int(H),
        W=int(W),
        D=int(D),
        atom_vocab_size=atom_vocab_size,
        coeff_vocab_size=coeff_vocab_size,
        window_sites=int(window_sites),
        n_global_spatial_tokens=int(n_global_spatial_tokens),
        d_model=int(d_model),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        d_ff=int(d_ff),
        dropout=float(dropout),
    )


class MinGPTQuantizedPrior(nn.Module):
    def __init__(self, cfg: MinGPTQuantizedPriorConfig):
        super().__init__()
        if int(cfg.D) % 2 != 0:
            raise ValueError(f"Quantized MinGPT prior expects even token depth, got D={cfg.D}")
        self.cfg = cfg
        self.real_valued_coeffs = False
        self.gaussian_coeffs = False
        self.autoregressive_coeffs = True
        self.atom_vocab_size = int(cfg.atom_vocab_size)
        self.coeff_vocab_size = int(cfg.coeff_vocab_size)
        self.content_vocab_size = int(cfg.vocab_size)
        self.sequence_length = int(cfg.H) * int(cfg.W) * int(cfg.D)
        self.n_global_spatial_tokens = int(cfg.n_global_spatial_tokens)
        if self.n_global_spatial_tokens < 0:
            raise ValueError(
                f"n_global_spatial_tokens must be >= 0, got {self.n_global_spatial_tokens}"
            )
        self.total_sequence_length = self.sequence_length + self.n_global_spatial_tokens
        self.window_sites = int(cfg.window_sites)
        if self.window_sites < 0:
            raise ValueError(f"window_sites must be >= 0, got {self.window_sites}")
        self.n_sites = int(cfg.H) * int(cfg.W)
        self.window_tokens = self._window_tokens()
        self.bos_token_id = int(cfg.vocab_size)

        self.token_emb = nn.Embedding(self.content_vocab_size + 1, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.total_sequence_length, cfg.d_model))
        self.global_spatial_tokens = None
        if self.n_global_spatial_tokens > 0:
            self.global_spatial_tokens = nn.Parameter(
                torch.zeros(1, self.n_global_spatial_tokens, cfg.d_model)
            )
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    cfg.dropout,
                    window_size=self.window_tokens,
                    n_global_prefix_tokens=self.n_global_spatial_tokens,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, self.content_vocab_size, bias=False)

        self.register_buffer(
            "_atom_vocab_size_tensor",
            torch.tensor(self.atom_vocab_size, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "_coeff_vocab_size_tensor",
            torch.tensor(self.coeff_vocab_size, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "_bos_token_id_tensor",
            torch.tensor(self.bos_token_id, dtype=torch.int64),
            persistent=True,
        )

        self.apply(init_transformer_module_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        if self.global_spatial_tokens is not None:
            nn.init.normal_(self.global_spatial_tokens, mean=0.0, std=0.02)

    def _window_tokens(self) -> Optional[int]:
        if self.window_sites <= 0 or self.window_sites >= self.n_sites:
            return None
        return self.window_sites * int(self.cfg.D)

    def _is_atom_step(self, pos: torch.Tensor) -> torch.Tensor:
        return pos.remainder(self.cfg.D).remainder(2).eq(0)

    def _site_start(self, step: int) -> int:
        return step - (step % self.cfg.D)

    def _ban_step_vocab(self, logits: torch.Tensor, *, start_pos: int) -> torch.Tensor:
        pos = torch.arange(start_pos, start_pos + logits.size(1), device=logits.device)
        atom_steps = self._is_atom_step(pos)
        if atom_steps.any():
            logits[:, atom_steps, self.atom_vocab_size :] = float("-inf")
        coeff_steps = ~atom_steps
        if coeff_steps.any():
            logits[:, coeff_steps, : self.atom_vocab_size] = float("-inf")
        return logits

    def _ban_used_atoms(self, logits: torch.Tensor, seq: torch.Tensor, *, step: int) -> torch.Tensor:
        if step <= 0 or (step % 2) != 0:
            return logits
        site_start = self._site_start(step)
        used_atoms = seq[:, site_start:step:2]
        if used_atoms.numel() == 0:
            return logits
        logits = logits.clone()
        logits.scatter_(1, used_atoms.clamp(0, self.atom_vocab_size - 1), float("-inf"))
        return logits

    def _flatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B, H*W, D], got {tuple(tokens.shape)}")
        B, T, D = tokens.shape
        expected = (self.cfg.H * self.cfg.W, self.cfg.D)
        if (T, D) != expected:
            raise ValueError(f"Expected token grid {(B, *expected)}, got {tuple(tokens.shape)}")
        return _private_long_tensor(tokens.reshape(B, self.sequence_length), device=tokens.device)

    def _embed(
        self,
        idx: torch.Tensor,
        *,
        past_len: int = 0,
        type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del type_ids
        x = self.token_emb(idx)
        pos_start = self.n_global_spatial_tokens + int(past_len)
        x = x + self.pos_emb[:, pos_start:pos_start + idx.size(1), :]
        return self.dropout(x)

    def _prefix(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.global_spatial_tokens is None:
            return None
        prefix = self.global_spatial_tokens.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        prefix = prefix + self.pos_emb[:, : self.n_global_spatial_tokens, :].to(device=device, dtype=dtype)
        return self.dropout(prefix)

    def _masked_logits(self, logits: torch.Tensor, *, start_pos: int) -> torch.Tensor:
        return self._ban_step_vocab(logits, start_pos=start_pos)

    def _forward_sequence(
        self,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        x = self._embed(idx, past_len=0)
        prefix = self._prefix(idx.size(0), device=idx.device, dtype=x.dtype)
        if prefix is not None:
            x = torch.cat([prefix, x], dim=1)
        x, _ = run_transformer_blocks(x, self.blocks, self.ln_f)
        if self.n_global_spatial_tokens > 0:
            x = x[:, self.n_global_spatial_tokens :, :]
        logits = self.token_head(x)
        return self._masked_logits(logits, start_pos=0)

    def forward(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
        return_features: bool = False,
        mask_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del coeffs, return_features
        source_flat = self._flatten_tokens(tokens)
        target_flat = source_flat if mask_tokens is None else self._flatten_tokens(mask_tokens)
        bos = torch.full(
            (source_flat.size(0), 1),
            self.bos_token_id,
            device=source_flat.device,
            dtype=torch.long,
        )
        inp = torch.cat([bos, source_flat[:, :-1]], dim=1)
        logits = self._forward_sequence(inp)
        if target_flat.shape != source_flat.shape:
            raise ValueError(
                f"Expected target_flat shape {tuple(source_flat.shape)}, got {tuple(target_flat.shape)}"
            )
        return logits.view(source_flat.size(0), self.cfg.H * self.cfg.W, self.cfg.D, self.content_vocab_size)

    def _forward_step(
        self,
        idx: torch.Tensor,
        *,
        kv_cache: Optional[list[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int,
    ) -> Tuple[torch.Tensor, list[Tuple[torch.Tensor, torch.Tensor]]]:
        x = self._embed(idx, past_len=start_pos)
        x, new_cache = run_transformer_blocks(
            x,
            self.blocks,
            self.ln_f,
            kv_cache=kv_cache,
        )
        logits = self.token_head(x)
        logits = self._masked_logits(logits, start_pos=start_pos)
        return logits, new_cache

    def _prime_prefix_cache(self, batch_size: int, *, device: torch.device) -> Optional[list[Tuple[torch.Tensor, torch.Tensor]]]:
        prefix = self._prefix(batch_size, device=device, dtype=self.token_emb.weight.dtype)
        if prefix is None:
            return None
        _, kv_cache = run_transformer_blocks(prefix, self.blocks, self.ln_f)
        return kv_cache

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        logits = logits / max(float(temperature), 1e-8)
        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), int(logits.size(-1)))
            values, indices = torch.topk(logits, k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, indices, values)
            logits = masked
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        coeff_temperature: Optional[float] = None,
        coeff_sample_mode: str = "gaussian",
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        prompt_coeffs: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del coeff_temperature, coeff_sample_mode, prompt_coeffs
        device = next(self.parameters()).device
        if prompt_tokens is not None:
            prompt_flat = self._flatten_tokens(prompt_tokens)
            expected_shape = (batch_size, self.cfg.H * self.cfg.W, self.cfg.D)
            if tuple(prompt_tokens.shape) != expected_shape:
                raise ValueError(f"Expected prompt_tokens shape {expected_shape}, got {tuple(prompt_tokens.shape)}")
        else:
            prompt_flat = None
        if prompt_mask is not None:
            prompt_mask = torch.as_tensor(prompt_mask, device=device, dtype=torch.bool)
            if tuple(prompt_mask.shape) == (batch_size, self.cfg.H * self.cfg.W):
                prompt_mask = prompt_mask.unsqueeze(-1).expand(batch_size, self.cfg.H * self.cfg.W, self.cfg.D)
            elif tuple(prompt_mask.shape) == (batch_size, self.cfg.H * self.cfg.W, 1):
                prompt_mask = prompt_mask.expand(batch_size, self.cfg.H * self.cfg.W, self.cfg.D)
            elif tuple(prompt_mask.shape) != (batch_size, self.cfg.H * self.cfg.W, self.cfg.D):
                raise ValueError(
                    "Expected prompt_mask shape "
                    f"{(batch_size, self.cfg.H * self.cfg.W)}, "
                    f"{(batch_size, self.cfg.H * self.cfg.W, 1)}, or "
                    f"{(batch_size, self.cfg.H * self.cfg.W, self.cfg.D)}; "
                    f"got {tuple(prompt_mask.shape)}"
                )
            prompt_mask = prompt_mask.reshape(batch_size, self.sequence_length)
        elif prompt_flat is not None:
            prompt_mask = torch.ones(batch_size, self.sequence_length, device=device, dtype=torch.bool)

        seq = torch.empty(batch_size, self.sequence_length, dtype=torch.long, device=device)
        bos = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=torch.long)
        kv_cache = self._prime_prefix_cache(batch_size, device=device)
        logits, kv_cache = self._forward_step(bos, kv_cache=kv_cache, start_pos=0)

        steps = tqdm(
            range(self.sequence_length),
            desc=(progress_desc or "[Stage2] sampling tokens"),
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for step in steps:
            step_logits = self._ban_used_atoms(logits[:, -1, :], seq, step=step)
            sampled = self._sample_from_logits(
                step_logits,
                temperature=float(temperature),
                top_k=top_k,
            )
            if prompt_flat is not None and prompt_mask is not None:
                clamp_now = prompt_mask[:, step]
                if clamp_now.any():
                    sampled = torch.where(clamp_now, prompt_flat[:, step], sampled)
            seq[:, step] = sampled
            if step + 1 < self.sequence_length:
                logits, kv_cache = self._forward_step(
                    sampled.unsqueeze(1),
                    kv_cache=kv_cache,
                    start_pos=step + 1,
                )

        return seq.view(batch_size, self.cfg.H * self.cfg.W, self.cfg.D)
