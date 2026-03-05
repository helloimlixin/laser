"""
Standalone two-stage spatial+depth transformer prior for the LASER sparse
dictionary autoencoder.

Spatial stage:  Causal transformer over H*W positions in raster order.
                Each position receives a fused summary of the previous
                position's D atom embeddings.
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
    d_model: int = 256
    n_heads: int = 8
    n_spatial_layers: int = 6
    n_depth_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    coeff_max: float = 24.0


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

        # Shared atom embedding
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.row_emb = nn.Embedding(cfg.H, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)
        self.start_emb = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.start_emb, std=0.02)

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
        self.atom_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.atom_coeff_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.coeff_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

        spatial_pos = torch.arange(cfg.H * cfg.W, dtype=torch.long)
        self.register_buffer(
            "_rows", spatial_pos // cfg.W, persistent=False,
        )
        self.register_buffer(
            "_cols", spatial_pos % cfg.W, persistent=False,
        )

    def _spatial_pos(self, T: int, device: torch.device) -> torch.Tensor:
        rows = self._rows[:T].to(device)
        cols = self._cols[:T].to(device)
        return self.row_emb(rows) + self.col_emb(cols)

    def forward(
        self,
        atom_ids: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward with teacher forcing.

        Args:
            atom_ids: [B, H*W, D] ground-truth atom indices.
            coeffs:   [B, H*W, D] ground-truth coefficients (used only
                      implicitly — atom_ids drive teacher forcing).

        Returns:
            atom_logits: [B, H*W, D, vocab_size]
            coeff_pred:  [B, H*W, D]
        """
        B, T, D = atom_ids.shape
        d_model = self.cfg.d_model
        device = atom_ids.device

        # ---- Spatial stage ----
        spatial_in = torch.zeros(B, T, d_model, device=device)
        spatial_in[:, 0] = self.start_emb.squeeze(1)
        if T > 1:
            prev_emb = self.token_emb(atom_ids[:, :-1])        # [B, T-1, D, d]
            prev_flat = prev_emb.reshape(B, T - 1, D * d_model)
            spatial_in[:, 1:] = self.spatial_fuse(prev_flat)

        spatial_in = spatial_in + self._spatial_pos(T, device).unsqueeze(0)

        spatial_h = spatial_in
        for blk in self.spatial_blocks:
            spatial_h, _ = blk(spatial_h)
        spatial_h = self.spatial_ln(spatial_h)                  # [B, T, d]

        # ---- Depth stage (all spatial positions in parallel) ----
        bt = B * T
        spatial_h_flat = spatial_h.reshape(bt, d_model)

        depth_in = torch.zeros(bt, D, d_model, device=device)
        depth_in[:, 0] = spatial_h_flat
        if D > 1:
            prev_depth_atoms = atom_ids[:, :, :-1].reshape(bt, D - 1)
            depth_in[:, 1:] = self.token_emb(prev_depth_atoms)

        depth_pos = self.depth_emb(torch.arange(D, device=device))
        depth_in = depth_in + depth_pos.unsqueeze(0)

        depth_h = depth_in
        for blk in self.depth_blocks:
            depth_h, _ = blk(depth_h)
        depth_h = self.depth_ln(depth_h)                       # [bt, D, d]

        atom_logits = self.atom_head(depth_h).reshape(B, T, D, self.cfg.vocab_size)

        atom_ids_flat = atom_ids.reshape(bt, D)
        coeff_emb = self.atom_coeff_emb(atom_ids_flat)          # [bt, D, d]
        coeff_in = torch.cat([depth_h, coeff_emb], dim=-1)      # [bt, D, 2d]
        coeff_pred = self.coeff_head(coeff_in).squeeze(-1).reshape(B, T, D)

        return atom_logits, coeff_pred

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unconditional generation with KV-cached spatial transformer.

        Returns:
            atom_ids: [B, H*W, D]
            coeffs:   [B, H*W, D]
        """
        device = next(self.parameters()).device
        cfg = self.cfg
        T = cfg.H * cfg.W
        D = cfg.D

        atom_ids = torch.zeros(batch_size, T, D, dtype=torch.long, device=device)
        coeffs = torch.zeros(batch_size, T, D, device=device)

        spatial_kv: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * len(self.spatial_blocks)

        steps = tqdm(
            range(T), desc="generating", leave=False, disable=not show_progress,
        )
        for t in steps:
            # -- spatial input for position t --
            if t == 0:
                x_new = self.start_emb.expand(batch_size, -1, -1)
            else:
                prev_emb = self.token_emb(atom_ids[:, t - 1])   # [B, D, d]
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
                    step_in = self.token_emb(atom_ids[:, t, d - 1]).unsqueeze(1)
                step_in = step_in + self.depth_emb.weight[d]
                depth_seq.append(step_in)

                depth_h = torch.cat(depth_seq, dim=1)            # [B, d+1, dm]
                for blk in self.depth_blocks:
                    depth_h, _ = blk(depth_h)
                depth_h = self.depth_ln(depth_h)
                last_h = depth_h[:, -1]                          # [B, dm]

                logits = self.atom_head(last_h)
                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                atom_ids[:, t, d] = sampled

                c_emb = self.atom_coeff_emb(sampled)
                c_in = torch.cat([last_h, c_emb], dim=-1)
                c_val = self.coeff_head(c_in).squeeze(-1)
                coeffs[:, t, d] = soft_clamp(c_val, cfg.coeff_max)

        return atom_ids, coeffs


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
