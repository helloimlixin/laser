"""
Standalone decoder-only dual-head transformer prior.

Single-stream causal decoder over flattened LASER sequence [H*W*D] with:
- decomposed positional encoding (row + col + depth)
- atom-id classification head
- coefficient regression head (real-valued sparse coefficients)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=(kv_cache is None and seq_len > 1),
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.out_proj(out)), (k, v)


class TransformerBlock(nn.Module):
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
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


@dataclass
class DualHeadDecoderConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    num_classes: int = 0
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    coeff_pred_atom_mix: float = 0.5
    coeff_atom_coherence_weight: float = 0.1
    # Kept for CLI/checkpoint compatibility; unused by regressor version.
    n_coeff_bins: int = 256
    coeff_mu: float = 255.0
    coeff_max_val: float = 24.0


class DualHeadDecoderPrior(nn.Module):
    """
    Decoder-only prior with two heads:
      - atom logits over dictionary ids
      - coefficient regression for real-valued sparse coeffs

    Input/Output compatibility:
      forward_tokens(tokens_flat, coeffs_flat, class_ids=None)
        -> atom_logits [B, H*W, D, vocab_size], coeff_pred [B, H*W, D]
      generate(batch_size, ...)
        -> tokens [B, H*W*D], coeffs [B, H*W*D]
    """

    def __init__(self, cfg: DualHeadDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.positions_per_image = cfg.H * cfg.W
        self.seq_len = self.positions_per_image * cfg.D
        self.bos_token = cfg.vocab_size
        self.total_vocab = cfg.vocab_size + 1

        self.token_emb = nn.Embedding(self.total_vocab, cfg.d_model)
        self.coeff_value_proj = nn.Linear(1, cfg.d_model)

        self.row_emb = nn.Embedding(cfg.H, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)
        seq_pos = torch.arange(self.seq_len, dtype=torch.long)
        # Location-major order: all depths for spatial position 0, then position 1, ...
        spatial_pos = torch.div(seq_pos, cfg.D, rounding_mode="floor")
        depth_pos = torch.remainder(seq_pos, cfg.D)
        self.register_buffer(
            "_pos_rows",
            torch.div(spatial_pos, cfg.W, rounding_mode="floor"),
            persistent=False,
        )
        self.register_buffer(
            "_pos_cols",
            torch.remainder(spatial_pos, cfg.W),
            persistent=False,
        )
        self.register_buffer(
            "_pos_depths",
            depth_pos,
            persistent=False,
        )

        self.class_emb = (
            nn.Embedding(cfg.num_classes, cfg.d_model)
            if cfg.num_classes > 0 else None
        )
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(cfg.d_model)

        self.atom_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.coeff_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _class_bias(
        self,
        class_ids: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.class_emb is None:
            return None
        if class_ids is None:
            raise ValueError("class_ids required when num_classes > 0")
        if class_ids.numel() != batch_size:
            raise ValueError(
                f"class_ids shape mismatch: expected {batch_size} elements, "
                f"got {tuple(class_ids.shape)}"
            )
        return self.class_emb(class_ids.long().to(device)).unsqueeze(1)

    def _pos_encoding(self, device: torch.device) -> torch.Tensor:
        return (
            self.row_emb(self._pos_rows.to(device))
            + self.col_emb(self._pos_cols.to(device))
            + self.depth_emb(self._pos_depths.to(device))
        )

    def _pos_encoding_at(self, idx: int, device: torch.device) -> torch.Tensor:
        r = self._pos_rows[idx].to(device)
        c = self._pos_cols[idx].to(device)
        d = self._pos_depths[idx].to(device)
        return self.row_emb(r) + self.col_emb(c) + self.depth_emb(d)

    def _run_no_cache_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.training:
                x = checkpoint(
                    lambda inp, m=blk: m(inp)[0],
                    x,
                    use_reentrant=False,
                )
            else:
                x, _ = blk(x)
        return x

    def _shifted_inputs(
        self, tokens: torch.Tensor, coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = tokens.size(0)
        in_tokens = torch.full(
            (bsz, self.seq_len),
            self.bos_token,
            dtype=torch.long,
            device=tokens.device,
        )
        in_coeffs = torch.zeros(
            (bsz, self.seq_len), dtype=torch.float32, device=tokens.device,
        )
        if self.seq_len > 1:
            in_tokens[:, 1:] = tokens[:, :-1]
            in_coeffs[:, 1:] = coeffs[:, :-1]
        return in_tokens, in_coeffs

    def _atom_emb_from_logits(self, atom_logits_flat: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(atom_logits_flat, dim=-1)
        atom_table = self.token_emb.weight[: self.cfg.vocab_size]
        return probs @ atom_table

    def _forward_core(
        self,
        tokens_flat: torch.Tensor,
        coeffs_flat: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = tokens_flat.device
        bsz = tokens_flat.size(0)
        tokens = tokens_flat.long().view(bsz, self.seq_len)
        coeffs = coeffs_flat.float().view(bsz, self.seq_len)
        in_tokens, in_coeffs = self._shifted_inputs(tokens, coeffs)

        x = self.token_emb(in_tokens) + self.coeff_value_proj(
            in_coeffs.unsqueeze(-1),
        )
        x = x + self._pos_encoding(device).unsqueeze(0)
        class_bias = self._class_bias(class_ids, bsz, device)
        if class_bias is not None:
            x = x + class_bias
        x = self.drop(x)

        h = self._run_no_cache_blocks(x)
        h = self.final_ln(h)

        atom_logits_flat = self.atom_head(h)
        atom_emb_gt = self.token_emb(tokens)
        atom_emb_pred = self._atom_emb_from_logits(atom_logits_flat)
        mix = min(max(float(self.cfg.coeff_pred_atom_mix), 0.0), 1.0)
        atom_emb_ctx = (1.0 - mix) * atom_emb_gt + mix * atom_emb_pred

        coeff_pred_flat = self.coeff_head(
            torch.cat([h, atom_emb_ctx], dim=-1),
        ).squeeze(-1)
        return atom_logits_flat, coeff_pred_flat, atom_emb_gt, atom_emb_pred

    def forward_tokens(
        self,
        tokens_flat: torch.Tensor,
        coeffs_flat: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced forward pass with real-valued coefficients.

        Args:
            tokens_flat: [B, H*W*D] atom indices.
            coeffs_flat: [B, H*W*D] real-valued sparse coefficients.
            class_ids:   [B] optional class labels.

        Returns:
            atom_logits: [B, H*W, D, vocab_size]
            coeff_pred:  [B, H*W, D]
        """
        bsz = tokens_flat.size(0)
        atom_logits_flat, coeff_pred_flat, _, _ = self._forward_core(
            tokens_flat, coeffs_flat, class_ids=class_ids,
        )

        atom_logits = atom_logits_flat.view(
            bsz, self.positions_per_image, self.cfg.D, self.cfg.vocab_size,
        )
        coeff_pred = coeff_pred_flat.view(bsz, self.positions_per_image, self.cfg.D)
        return atom_logits, coeff_pred

    def forward_loss(
        self,
        tokens_flat: torch.Tensor,
        coeffs_flat: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        coeff_loss_weight: float = 0.25,
    ) -> torch.Tensor:
        bsz = tokens_flat.size(0)
        atom_logits_flat, coeff_pred_flat, atom_emb_gt, atom_emb_pred = self._forward_core(
            tokens_flat, coeffs_flat, class_ids=class_ids,
        )
        atom_target = tokens_flat.long().view(bsz, self.seq_len)
        coeff_target = coeffs_flat.float().view(bsz, self.seq_len)

        atom_loss = F.cross_entropy(
            atom_logits_flat.reshape(-1, self.cfg.vocab_size),
            atom_target.reshape(-1),
        )
        coeff_loss = F.smooth_l1_loss(
            coeff_pred_flat, coeff_target, beta=4.0,
        )
        total = atom_loss + max(float(coeff_loss_weight), 0.0) * coeff_loss

        coh_w = max(float(self.cfg.coeff_atom_coherence_weight), 0.0)
        if coh_w > 0.0:
            coherence = F.smooth_l1_loss(
                atom_emb_pred, atom_emb_gt.detach(), beta=0.5,
            )
            total = total + coh_w * coherence
        return total

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        class_ids: Optional[torch.Tensor] = None,
        show_progress: bool = False,
        coeff_temperature: Optional[float] = None,
        coeff_top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive generation with atom sampling + coeff regression."""
        del temperature, top_k, coeff_temperature, coeff_top_k
        device = next(self.parameters()).device
        class_bias = self._class_bias(class_ids, batch_size, device)

        tokens = torch.zeros(
            batch_size, self.seq_len, dtype=torch.long, device=device,
        )
        coeffs = torch.zeros(batch_size, self.seq_len, dtype=torch.float32, device=device)

        kv_cache: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = (
            [None] * len(self.blocks)
        )
        steps = tqdm(
            range(self.seq_len),
            desc="sampling",
            leave=False,
            disable=(not show_progress),
        )

        for idx in steps:
            if idx == 0:
                prev_tok = torch.full(
                    (batch_size,), self.bos_token, dtype=torch.long, device=device,
                )
                prev_coeff = torch.zeros(batch_size, device=device)
            else:
                prev_tok = tokens[:, idx - 1]
                prev_coeff = coeffs[:, idx - 1]

            x_new = (
                self.token_emb(prev_tok) + self.coeff_value_proj(prev_coeff.unsqueeze(-1))
            ).unsqueeze(1)
            x_new = x_new + self._pos_encoding_at(idx, device).view(1, 1, -1)
            if class_bias is not None:
                x_new = x_new + class_bias
            x_new = self.drop(x_new)

            for i, blk in enumerate(self.blocks):
                x_new, kv_cache[i] = blk(x_new, kv_cache=kv_cache[i])
            h_t = self.final_ln(x_new).squeeze(1)

            logits = self.atom_head(h_t)
            tok_next = torch.multinomial(
                F.softmax(logits, dim=-1), 1,
            ).squeeze(-1)
            tokens[:, idx] = tok_next

            atom_emb = self.token_emb(tok_next)
            pred_coeff = self.coeff_head(
                torch.cat([h_t, atom_emb], dim=-1),
            ).squeeze(-1)
            coeffs[:, idx] = pred_coeff

        return tokens, coeffs
