"""
Standalone two-stage spatial+depth transformer prior for the LASER sparse
dictionary autoencoder.

Spatial stage:  Causal transformer over H*W positions in raster order.
                Each position receives a fused summary of the previous
                position's D atom embeddings, optionally conditioned on a
                learned bank of global prefix tokens.
Depth stage:    Small causal transformer that, conditioned on the spatial
                hidden state, autoregressively generates D atom IDs and
                their scalar coefficients per spatial position.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def soft_clamp(x: torch.Tensor, max_val: float) -> torch.Tensor:
    """Tanh-based soft clamp: approximately linear near zero, smoothly
    saturates towards ±max_val instead of the hard discontinuity of clamp."""
    return max_val * torch.tanh(x / max_val)


@dataclass
class SpatialDepthPriorConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    atom_vocab_size: Optional[int] = None
    coeff_vocab_size: Optional[int] = None
    coeff_bin_values: Optional[torch.Tensor] = None
    real_valued_coeffs: bool = True
    d_model: int = 256
    n_heads: int = 8
    n_spatial_layers: int = 6
    n_depth_layers: int = 4
    n_global_spatial_tokens: int = 0
    d_ff: int = 1024
    dropout: float = 0.1
    coeff_max: float = 24.0


def build_spatial_depth_prior_config(
    bottleneck,
    *,
    H: int,
    W: int,
    D: int,
    d_model: int,
    n_heads: int,
    n_spatial_layers: int,
    n_depth_layers: int,
    d_ff: int,
    dropout: float,
    n_global_spatial_tokens: int,
    real_valued_coeffs: bool,
    coeff_max_fallback: float,
) -> SpatialDepthPriorConfig:
    coeff_vocab_size = None
    coeff_bin_values = None
    if not real_valued_coeffs:
        coeff_vocab_size = int(bottleneck.n_bins)
        coeff_bin_values = bottleneck._dequantize_coeff(
            torch.arange(coeff_vocab_size, dtype=torch.long)
        ).detach().cpu()

    return SpatialDepthPriorConfig(
        vocab_size=int(bottleneck.content_vocab_size),
        H=int(H),
        W=int(W),
        D=int(D),
        atom_vocab_size=int(bottleneck.num_embeddings),
        coeff_vocab_size=coeff_vocab_size,
        coeff_bin_values=coeff_bin_values,
        real_valued_coeffs=bool(real_valued_coeffs),
        d_model=int(d_model),
        n_heads=int(n_heads),
        n_spatial_layers=int(n_spatial_layers),
        n_depth_layers=int(n_depth_layers),
        n_global_spatial_tokens=int(n_global_spatial_tokens),
        d_ff=int(d_ff),
        dropout=float(dropout),
        coeff_max=float(getattr(bottleneck, "coef_max", coeff_max_fallback)),
    )


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_kv = (k, v)
        drop_p = self.attn_drop_p if self.training else 0.0

        _CUDA_GRID_MAX = 65535
        if kv_cache is not None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        elif B * self.n_heads > _CUDA_GRID_MAX:
            chunk_b = _CUDA_GRID_MAX // self.n_heads
            out = torch.cat([
                F.scaled_dot_product_attention(
                    q[i:i+chunk_b], k[i:i+chunk_b], v[i:i+chunk_b],
                    is_causal=True, dropout_p=drop_p,
                )
                for i in range(0, B, chunk_b)
            ], dim=0)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=drop_p,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out, new_kv


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional KV cache."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
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
        h = self.ln1(x)
        attn_out, new_kv = self.attn(h, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


class SpatialDepthPrior(nn.Module):

    def __init__(self, cfg: SpatialDepthPriorConfig):
        super().__init__()
        self.cfg = cfg
        self.real_valued_coeffs = bool(cfg.real_valued_coeffs)
        self.n_global_spatial_tokens = int(cfg.n_global_spatial_tokens)
        if self.n_global_spatial_tokens < 0:
            raise ValueError(
                f"n_global_spatial_tokens must be >= 0, got {self.n_global_spatial_tokens}"
            )
        self.atom_vocab_size = int(cfg.atom_vocab_size if cfg.atom_vocab_size is not None else cfg.vocab_size)
        if self.atom_vocab_size <= 0 or self.atom_vocab_size > int(cfg.vocab_size):
            raise ValueError(
                f"atom_vocab_size must be in [1, vocab_size], got {self.atom_vocab_size} for vocab_size={cfg.vocab_size}"
            )
        self.coeff_vocab_size = None if cfg.coeff_vocab_size is None else int(cfg.coeff_vocab_size)
        if self.real_valued_coeffs:
            self.content_vocab_size = self.atom_vocab_size
            if self.coeff_vocab_size not in (None, 0):
                raise ValueError("coeff_vocab_size must be unset when real_valued_coeffs=True")
            self.coeff_vocab_size = None
        else:
            if self.coeff_vocab_size is None or self.coeff_vocab_size <= 0:
                raise ValueError("coeff_vocab_size must be set when real_valued_coeffs=False")
            self.content_vocab_size = self.atom_vocab_size + self.coeff_vocab_size
            if self.content_vocab_size > int(cfg.vocab_size):
                raise ValueError(
                    "atom_vocab_size + coeff_vocab_size must be <= vocab_size, "
                    f"got {self.content_vocab_size} for vocab_size={cfg.vocab_size}"
                )
            atom_steps = (int(cfg.D) + 1) // 2
            if self.atom_vocab_size < atom_steps:
                raise ValueError(
                    "quantized sparse coeff path requires at least one unique atom id per atom step, "
                    f"got atom_vocab_size={self.atom_vocab_size} for D={cfg.D} "
                    f"({atom_steps} atom steps)"
                )

        # Shared token embedding for atom IDs and output classes.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.row_emb = nn.Embedding(cfg.H, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)
        self.start_emb = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.start_emb, std=0.02)
        self.global_spatial_tokens = None
        if self.n_global_spatial_tokens > 0:
            self.global_spatial_tokens = nn.Parameter(
                torch.zeros(1, self.n_global_spatial_tokens, cfg.d_model)
            )
            nn.init.normal_(self.global_spatial_tokens, std=0.02)
        self.coeff_proj = nn.Linear(1, cfg.d_model)
        self.coeff_token_emb = None
        self.token_type_emb = None
        if not self.real_valued_coeffs:
            coeff_bin_values = cfg.coeff_bin_values
            if coeff_bin_values is None:
                coeff_bin_values = torch.linspace(
                    -cfg.coeff_max,
                    cfg.coeff_max,
                    steps=self.coeff_vocab_size,
                    dtype=torch.float32,
                )
            else:
                coeff_bin_values = torch.as_tensor(coeff_bin_values, dtype=torch.float32).reshape(-1)
                if coeff_bin_values.numel() != self.coeff_vocab_size:
                    raise ValueError(
                        "coeff_bin_values length must match coeff_vocab_size, "
                        f"got {coeff_bin_values.numel()} for coeff_vocab_size={self.coeff_vocab_size}"
                    )
            self.register_buffer("_coeff_bin_values", coeff_bin_values, persistent=False)
            self.coeff_token_emb = nn.Embedding(self.coeff_vocab_size, cfg.d_model)
            self.token_type_emb = nn.Embedding(2, cfg.d_model)
        else:
            self._coeff_bin_values = None

        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_spatial_layers)
        ])
        self.depth_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_depth_layers)
        ])
        self.spatial_ln = nn.LayerNorm(cfg.d_model)
        self.depth_ln = nn.LayerNorm(cfg.d_model)

        self.spatial_fuse = nn.Linear(cfg.D * cfg.d_model, cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        if self.real_valued_coeffs:
            self.atom_coeff_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.coeff_head = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1),
            )
        else:
            self.atom_coeff_emb = None
            self.coeff_head = None

        spatial_pos = torch.arange(cfg.H * cfg.W, dtype=torch.long)
        self.register_buffer(
            "_rows", spatial_pos // cfg.W, persistent=False,
        )
        self.register_buffer(
            "_cols", spatial_pos % cfg.W, persistent=False,
        )
        depth_token_mask = torch.zeros(cfg.D, cfg.vocab_size, dtype=torch.bool)
        if self.real_valued_coeffs:
            depth_token_mask[:, :self.atom_vocab_size] = True
        else:
            depth_token_mask[0::2, :self.atom_vocab_size] = True
            depth_token_mask[1::2, self.atom_vocab_size:self.content_vocab_size] = True
        self.register_buffer("_depth_token_mask", depth_token_mask, persistent=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        global_key = prefix + "global_spatial_tokens"
        if self.n_global_spatial_tokens <= 0:
            state_dict.pop(global_key, None)
        else:
            expected = self.global_spatial_tokens.detach().clone()
            loaded = state_dict.get(global_key, None)
            if loaded is None:
                state_dict[global_key] = expected
            else:
                loaded = loaded.to(device=expected.device, dtype=expected.dtype)
                if tuple(loaded.shape) != tuple(expected.shape):
                    patched = expected
                    if loaded.ndim == expected.ndim == 3 and loaded.size(0) == expected.size(0) and loaded.size(2) == expected.size(2):
                        keep = min(int(loaded.size(1)), int(expected.size(1)))
                        patched[:, :keep].copy_(loaded[:, :keep])
                    state_dict[global_key] = patched
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _spatial_pos(self, T: int, device: torch.device) -> torch.Tensor:
        rows = self._rows[:T].to(device)
        cols = self._cols[:T].to(device)
        return self.row_emb(rows) + self.col_emb(cols)

    def _prepend_global_spatial_tokens(self, spatial_in: torch.Tensor) -> torch.Tensor:
        if self.global_spatial_tokens is None:
            return spatial_in
        global_tokens = self.global_spatial_tokens.expand(spatial_in.size(0), -1, -1)
        return torch.cat([global_tokens, spatial_in], dim=1)

    def _prime_spatial_kv_with_global_tokens(
        self,
        batch_size: int,
        spatial_kv: list[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.global_spatial_tokens is None:
            return spatial_kv

        global_h = self.global_spatial_tokens.expand(batch_size, -1, -1)
        for i, blk in enumerate(self.spatial_blocks):
            global_h, spatial_kv[i] = blk(global_h, kv_cache=spatial_kv[i])
        return spatial_kv

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.real_valued_coeffs:
            return self.token_emb(tokens)
        embed_shape = (*tokens.shape, self.cfg.d_model)
        embeds = torch.zeros(
            embed_shape,
            device=tokens.device,
            dtype=self.start_emb.dtype,
        )
        atom_mask = tokens < self.atom_vocab_size
        if atom_mask.any():
            atom_ids = tokens.masked_select(atom_mask)
            atom_emb = self.token_emb(atom_ids)
            atom_emb = atom_emb + self.token_type_emb.weight[0].unsqueeze(0)
            embeds[atom_mask] = atom_emb
        coeff_mask = ~atom_mask
        if coeff_mask.any():
            coeff_idx = tokens.masked_select(coeff_mask) - self.atom_vocab_size
            coeff_idx = coeff_idx.clamp(0, self.coeff_vocab_size - 1)
            coeff_vals = self._coeff_bin_values.to(
                device=tokens.device,
                dtype=self.start_emb.dtype,
            ).index_select(0, coeff_idx)
            coeff_emb = self.coeff_token_emb(coeff_idx)
            coeff_emb = coeff_emb + self.coeff_proj(coeff_vals.unsqueeze(-1))
            coeff_emb = coeff_emb + self.token_type_emb.weight[1].unsqueeze(0)
            embeds[coeff_mask] = coeff_emb
        return embeds

    def _masked_training_logits(
        self,
        logits: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected [B, T, D, V] logits, got {tuple(logits.shape)}")
        masked = logits.masked_fill(
            (~self._depth_token_mask.to(device=logits.device)).view(1, 1, self.cfg.D, self.cfg.vocab_size),
            float("-inf"),
        )
        if tokens is not None:
            if tokens.shape != logits.shape[:3]:
                raise ValueError(
                    f"Expected tokens with shape {tuple(logits.shape[:3])}, got {tuple(tokens.shape)}"
                )
            # Match sampling-time support constraints by masking previously used
            # atoms at later atom slots during teacher-forced training.
            if self.real_valued_coeffs:
                depth_range = range(1, self.cfg.D)
                prev_atom_slice = lambda depth_idx: tokens[:, :, :depth_idx]
            else:
                depth_range = range(2, self.cfg.D, 2)
                prev_atom_slice = lambda depth_idx: tokens[:, :, :depth_idx:2]
            for depth_idx in depth_range:
                prev_atoms = prev_atom_slice(depth_idx).to(torch.long)
                masked[:, :, depth_idx, :].scatter_(2, prev_atoms, float("-inf"))
        return masked

    def _forward_depth_hidden(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = tokens.shape
        d_model = self.cfg.d_model
        device = tokens.device

        if self.real_valued_coeffs:
            if coeffs is None:
                raise ValueError("coeffs must be provided when real_valued_coeffs=True")
            coeffs = coeffs.to(device=device, dtype=self.start_emb.dtype)
        else:
            if coeffs is not None:
                raise ValueError("coeffs must be omitted when real_valued_coeffs=False")

        spatial_in = torch.zeros(B, T, d_model, device=device)
        spatial_in[:, 0] = self.start_emb.squeeze(1)
        if T > 1:
            prev_token_emb = self._embed_tokens(tokens[:, :-1])
            if self.real_valued_coeffs:
                prev_token_emb = prev_token_emb + self.coeff_proj(coeffs[:, :-1].unsqueeze(-1))
            prev_flat = prev_token_emb.reshape(B, T - 1, D * d_model)
            spatial_in[:, 1:] = self.spatial_fuse(prev_flat)

        spatial_in = spatial_in + self._spatial_pos(T, device).unsqueeze(0)
        spatial_h = self._prepend_global_spatial_tokens(spatial_in)
        for blk in self.spatial_blocks:
            spatial_h, _ = blk(spatial_h)
        spatial_h = self.spatial_ln(spatial_h)
        if self.n_global_spatial_tokens > 0:
            spatial_h = spatial_h[:, self.n_global_spatial_tokens:]

        bt = B * T
        spatial_h_flat = spatial_h.reshape(bt, d_model)
        depth_in = torch.zeros(bt, D, d_model, device=device)
        depth_in[:, 0] = spatial_h_flat
        if D > 1:
            prev_depth_tokens = tokens[:, :, :-1].reshape(bt, D - 1)
            depth_in[:, 1:] = self._embed_tokens(prev_depth_tokens)
            if self.real_valued_coeffs:
                prev_depth_coeffs = coeffs[:, :, :-1].reshape(bt, D - 1, 1)
                depth_in[:, 1:] = depth_in[:, 1:] + self.coeff_proj(prev_depth_coeffs)

        depth_pos = self.depth_emb(torch.arange(D, device=device))
        depth_in = depth_in + depth_pos.unsqueeze(0)

        depth_h = depth_in
        for blk in self.depth_blocks:
            depth_h, _ = blk(depth_h)
        depth_h = self.depth_ln(depth_h)
        return depth_h.reshape(B, T, D, d_model)

    def predict_coeffs_for_atoms(
        self,
        depth_h: torch.Tensor,
        atom_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.real_valued_coeffs or self.atom_coeff_emb is None or self.coeff_head is None:
            raise RuntimeError("predict_coeffs_for_atoms is only valid when real_valued_coeffs=True")
        if depth_h.ndim != 4:
            raise ValueError(f"Expected depth_h with shape [B, T, D, d_model], got {tuple(depth_h.shape)}")
        if tuple(atom_ids.shape) != tuple(depth_h.shape[:3]):
            raise ValueError(
                f"Expected atom_ids with shape {tuple(depth_h.shape[:3])}, got {tuple(atom_ids.shape)}"
            )

        B, T, D, d_model = depth_h.shape
        depth_h_flat = depth_h.reshape(B * T, D, d_model)
        atom_ids_flat = atom_ids.reshape(B * T, D).to(torch.long)
        coeff_emb = self.atom_coeff_emb(atom_ids_flat)
        coeff_in = torch.cat([depth_h_flat, coeff_emb], dim=-1)
        coeff_pred = soft_clamp(
            self.coeff_head(coeff_in).squeeze(-1),
            self.cfg.coeff_max,
        )
        return coeff_pred.reshape(B, T, D)

    def forward_features(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_h = self._forward_depth_hidden(tokens, coeffs)
        token_logits = self.token_head(depth_h)
        token_logits = self._masked_training_logits(token_logits, tokens=tokens)
        return token_logits, depth_h

    def _mask_generation_logits(
        self,
        logits: torch.Tensor,
        depth_idx: int,
        prev_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = logits.masked_fill(
            ~self._depth_token_mask[depth_idx].to(device=logits.device).unsqueeze(0),
            float("-inf"),
        )
        prev_atoms = None
        if prev_tokens is not None and depth_idx > 0:
            if self.real_valued_coeffs:
                prev_atoms = prev_tokens[:, :depth_idx]
            elif (depth_idx % 2) == 0:
                prev_atoms = prev_tokens[:, :depth_idx:2]
        if prev_atoms is not None:
            if prev_atoms.numel() > 0:
                logits.scatter_(1, prev_atoms, float("-inf"))
        return logits

    def forward(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        """Training forward with teacher forcing.

        Args:
            tokens: [B, H*W, D] ground-truth atom ids in real-valued mode or
                interleaved atom/bin tokens in quantized mode.
            coeffs: [B, H*W, D] ground-truth coefficients for real-valued mode.

        Returns:
            real-valued mode: (atom_logits [B, H*W, D, vocab], coeff_pred [B, H*W, D])
            quantized mode:   token_logits [B, H*W, D, vocab]
        """
        token_logits, depth_h = self.forward_features(tokens, coeffs)
        if not self.real_valued_coeffs:
            if return_features:
                return token_logits, depth_h
            return token_logits

        coeff_pred = self.predict_coeffs_for_atoms(depth_h, tokens)
        if return_features:
            return token_logits, coeff_pred, depth_h
        return token_logits, coeff_pred

    def _prepare_prompt_inputs(
        self,
        batch_size: int,
        T: int,
        D: int,
        device: torch.device,
        prompt_tokens: Optional[torch.Tensor],
        prompt_coeffs: Optional[torch.Tensor],
        prompt_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if prompt_tokens is None and prompt_mask is None and prompt_coeffs is None:
            return None, None, None
        if prompt_tokens is None:
            raise ValueError("prompt_tokens must be provided when using prompt_mask or prompt_coeffs")

        expected_shape = (batch_size, T, D)
        if tuple(prompt_tokens.shape) != expected_shape:
            raise ValueError(f"Expected prompt_tokens shape {expected_shape}, got {tuple(prompt_tokens.shape)}")
        prompt_tokens = prompt_tokens.to(device=device, dtype=torch.long)

        if prompt_mask is None:
            prompt_mask = torch.ones(expected_shape, device=device, dtype=torch.bool)
        else:
            prompt_mask = torch.as_tensor(prompt_mask, device=device, dtype=torch.bool)
            if tuple(prompt_mask.shape) == (batch_size, T):
                prompt_mask = prompt_mask.unsqueeze(-1).expand(batch_size, T, D)
            elif tuple(prompt_mask.shape) == (batch_size, T, 1):
                prompt_mask = prompt_mask.expand(batch_size, T, D)
            elif tuple(prompt_mask.shape) != expected_shape:
                raise ValueError(
                    f"Expected prompt_mask shape {(batch_size, T)}, {(batch_size, T, 1)}, "
                    f"or {expected_shape}; got {tuple(prompt_mask.shape)}"
                )

        if prompt_coeffs is None:
            if self.real_valued_coeffs and bool(prompt_mask.any().item()):
                raise ValueError("prompt_coeffs must be provided when clamping real-valued coefficients")
            return prompt_tokens, None, prompt_mask

        if not self.real_valued_coeffs:
            raise ValueError("prompt_coeffs is only valid when real_valued_coeffs=True")
        if tuple(prompt_coeffs.shape) != expected_shape:
            raise ValueError(f"Expected prompt_coeffs shape {expected_shape}, got {tuple(prompt_coeffs.shape)}")
        prompt_coeffs = prompt_coeffs.to(device=device, dtype=self.start_emb.dtype)
        return prompt_tokens, prompt_coeffs, prompt_mask

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        prompt_coeffs: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unconditional generation with KV-cached spatial transformer.

        Returns:
            real-valued mode: (atom_ids [B, H*W, D], coeffs [B, H*W, D])
            quantized mode:   tokens [B, H*W, D]
        """
        device = next(self.parameters()).device
        cfg = self.cfg
        T = cfg.H * cfg.W
        D = cfg.D
        prompt_tokens, prompt_coeffs, prompt_mask = self._prepare_prompt_inputs(
            batch_size=batch_size,
            T=T,
            D=D,
            device=device,
            prompt_tokens=prompt_tokens,
            prompt_coeffs=prompt_coeffs,
            prompt_mask=prompt_mask,
        )

        tokens = torch.zeros(batch_size, T, D, dtype=torch.long, device=device)
        coeffs = torch.zeros(batch_size, T, D, device=device) if self.real_valued_coeffs else None

        spatial_kv: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * len(self.spatial_blocks)
        spatial_kv = self._prime_spatial_kv_with_global_tokens(batch_size, spatial_kv)

        steps = tqdm(
            range(T),
            desc=(progress_desc or "[Stage2] sampling tokens"),
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for t in steps:
            # -- spatial input for position t --
            if t == 0:
                x_new = self.start_emb.expand(batch_size, -1, -1)
            else:
                prev_emb = self._embed_tokens(tokens[:, t - 1])    # [B, D, d]
                if self.real_valued_coeffs:
                    prev_emb = prev_emb + self.coeff_proj(coeffs[:, t - 1].unsqueeze(-1))
                x_new = self.spatial_fuse(
                    prev_emb.reshape(batch_size, -1),
                ).unsqueeze(1)                                   # [B, 1, d]

            x_new = x_new + self.row_emb(self._rows[t]) + self.col_emb(self._cols[t])

            spatial_h = x_new
            for i, blk in enumerate(self.spatial_blocks):
                spatial_h, spatial_kv[i] = blk(
                    spatial_h, kv_cache=spatial_kv[i],
                )
            h_t = self.spatial_ln(spatial_h).squeeze(1)          # [B, d]

            # -- depth stage: autoregressive, no KV cache (D is small) --
            depth_seq: list[torch.Tensor] = []
            for d in range(D):
                if d == 0:
                    step_in = h_t.unsqueeze(1)
                else:
                    step_in = self._embed_tokens(tokens[:, t, d - 1]).unsqueeze(1)
                    if self.real_valued_coeffs:
                        step_in = step_in + self.coeff_proj(coeffs[:, t, d - 1].view(batch_size, 1, 1))
                step_in = step_in + self.depth_emb.weight[d]
                depth_seq.append(step_in)

                depth_h = torch.cat(depth_seq, dim=1)            # [B, d+1, dm]
                for blk in self.depth_blocks:
                    depth_h, _ = blk(depth_h)
                depth_h = self.depth_ln(depth_h)
                last_h = depth_h[:, -1]                          # [B, dm]

                logits = self.token_head(last_h) / max(float(temperature), 1e-8)
                logits = self._mask_generation_logits(logits, d, prev_tokens=tokens[:, t, :d])
                if top_k is not None and int(top_k) > 0:
                    k = min(int(top_k), int(logits.size(-1)))
                    v, ix = torch.topk(logits, k, dim=-1)
                    mask = torch.full_like(logits, float("-inf"))
                    mask.scatter_(1, ix, v)
                    logits = mask
                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                if prompt_mask is not None:
                    clamp_now = prompt_mask[:, t, d]
                    if clamp_now.any():
                        sampled = torch.where(clamp_now, prompt_tokens[:, t, d], sampled)
                tokens[:, t, d] = sampled

                if self.real_valued_coeffs:
                    c_emb = self.atom_coeff_emb(sampled)
                    c_in = torch.cat([last_h, c_emb], dim=-1)
                    c_val = soft_clamp(self.coeff_head(c_in).squeeze(-1), cfg.coeff_max)
                    if prompt_mask is not None:
                        clamp_now = prompt_mask[:, t, d]
                        if clamp_now.any():
                            c_val = torch.where(clamp_now, prompt_coeffs[:, t, d], c_val)
                    coeffs[:, t, d] = c_val

        if self.real_valued_coeffs:
            return tokens, coeffs
        return tokens

if __name__ == "__main__":
    cfg = SpatialDepthPriorConfig(
        vocab_size=64,
        H=4, W=4, D=4,
        d_model=64,
        n_heads=4,
        n_spatial_layers=2,
        n_depth_layers=2,
        d_ff=128,
    )
    model = SpatialDepthPrior(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    B = 2
    ids = torch.randint(0, cfg.vocab_size, (B, cfg.H * cfg.W, cfg.D))
    cs = torch.randn(B, cfg.H * cfg.W, cfg.D)

    atom_logits, coeff_pred = model(ids, cs)
    print(f"forward  -> atom_logits {tuple(atom_logits.shape)}, "
          f"coeff_pred {tuple(coeff_pred.shape)}")
    assert atom_logits.shape == (B, cfg.H * cfg.W, cfg.D, cfg.vocab_size)
    assert coeff_pred.shape == (B, cfg.H * cfg.W, cfg.D)

    model.eval()
    gen_atoms, gen_coeffs = model.generate(batch_size=2, show_progress=True)
    print(f"generate -> atoms {tuple(gen_atoms.shape)}, "
          f"coeffs {tuple(gen_coeffs.shape)}")
    assert gen_atoms.shape == (B, cfg.H * cfg.W, cfg.D)
    assert gen_coeffs.shape == (B, cfg.H * cfg.W, cfg.D)
    print("Smoke test passed.")
