"""Plain minGPT for autoregressive modeling of sparse codes.

Sparse codes are serialized as interleaved integer tokens:
  [atom_id_0, coeff_bin_0, atom_id_1, coeff_bin_1, ...]

Token ranges (for vocab_size=V, n_coeff_bins=C):
  atom ids:   0 .. V-1
  coeff bins: V .. V+C-1
  BOS:        V+C
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

KVCache = list[Tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=(kv_cache is None and T > 1),
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out)), (k, v)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# MinGPT
# ---------------------------------------------------------------------------

class MinGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        num_classes: Optional[int] = None,
        n_layer: int = 12,
        n_head: int = 8,
        n_embd: int = 256,
        dropout: float = 0.1,
        n_token_types: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.type_emb = (
            nn.Embedding(n_token_types, n_embd) if n_token_types > 0 else None
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.class_emb = (
            nn.Embedding(num_classes, n_embd) if num_classes else None
        )

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        print(f"MinGPT: block_size={block_size}, vocab_size={vocab_size}")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _embed(
        self, idx: torch.Tensor, class_idx, past_len: int,
        type_ids: Optional[torch.Tensor] = None,
    ):
        T = idx.size(1)
        x = self.tok_emb(idx) + self.pos_emb[:, past_len:past_len + T, :]
        if self.type_emb is not None and type_ids is not None:
            x = x + self.type_emb(type_ids)
        if self.class_emb is not None and class_idx is not None:
            x = x + self.class_emb(class_idx).unsqueeze(1).expand_as(x)
        return self.drop(x)

    def forward(self, idx, targets=None, class_idx=None, type_ids=None):
        x = self._embed(idx, class_idx, past_len=0, type_ids=type_ids)
        for block in self.blocks:
            x, _ = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1),
            )
        return logits, loss

    def forward_step(
        self, idx: torch.Tensor,
        class_idx=None,
        kv_cache: Optional[KVCache] = None,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        past_len = kv_cache[0][0].size(2) if kv_cache else 0
        x = self._embed(idx, class_idx, past_len, type_ids=type_ids)
        new_cache: KVCache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, kv_cache=kv_cache[i] if kv_cache else None)
            new_cache.append(kv)
        return self.head(self.ln_f(x)), new_cache

    @staticmethod
    def _top_k(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
        if not k or k <= 0:
            return logits
        v, ix = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(-1, ix, v)
        return out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, class_idx=None,
                 temperature=1.0, top_k=None):
        self.eval()
        B = idx.size(0)
        _ = top_k  # Top-k truncation is intentionally disabled for sampling.
        prompt = idx[:, -self.block_size:]
        P = prompt.size(1)
        out = torch.empty(B, P + max_new_tokens, dtype=idx.dtype, device=idx.device)
        out[:, :P] = prompt
        logits, cache = self.forward_step(prompt, class_idx=class_idx)
        for step in range(max_new_tokens):
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            tok = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            out[:, P + step] = tok
            if step + 1 < max_new_tokens:
                logits, cache = self.forward_step(
                    tok.unsqueeze(1), class_idx=class_idx, kv_cache=cache,
                )
        return out


# ---------------------------------------------------------------------------
# Coefficient quantizer
# ---------------------------------------------------------------------------

class CoefficientQuantizer:
    """Sparse coefficient quantizer with optional mu-law companding."""

    def __init__(
        self, n_bins: int = 1024, max_val: float = 24.0, mu: float = 255.0,
    ):
        self.n_bins = max(int(n_bins), 2)
        self.max_val = max(float(max_val), 1e-6)
        self.mu = max(float(mu), 0.0)
        self.use_mu_law = self.mu > 0.0
        self._log_mu1 = math.log1p(self.mu) if self.use_mu_law else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x.float().clamp(-self.max_val, self.max_val) / self.max_val
        if self.use_mu_law:
            x_norm = (
                torch.sign(x_norm)
                * torch.log1p(self.mu * x_norm.abs())
                / self._log_mu1
            )
        return (
            ((x_norm + 1.0) / 2.0 * (self.n_bins - 1))
            .round()
            .long()
            .clamp(0, self.n_bins - 1)
        )

    def decode(self, bins: torch.Tensor) -> torch.Tensor:
        x_norm = bins.float().clamp(0, self.n_bins - 1) / (self.n_bins - 1) * 2.0 - 1.0
        if self.use_mu_law:
            x_norm = (
                torch.sign(x_norm)
                * (torch.pow(1.0 + self.mu, x_norm.abs()) - 1.0)
                / self.mu
            )
        return x_norm * self.max_val


# ---------------------------------------------------------------------------
# MinGPT sparse-code wrapper
# ---------------------------------------------------------------------------

@dataclass
class MinGPTSparseConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    num_classes: int = 0
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    dropout: float = 0.1
    n_coeff_bins: int = 1024
    coeff_mu: float = 255.0
    coeff_max_val: float = 24.0
    coeff_loss_weight: float = 1.0


class MinGPTSparse(nn.Module):
    """minGPT over interleaved [atom_id, coeff_bin] token stream."""

    def __init__(self, cfg: MinGPTSparseConfig):
        super().__init__()
        self.cfg = cfg
        self.sparse_len = cfg.H * cfg.W * cfg.D
        self.seq_len = 2 * self.sparse_len
        self.coeff_offset = cfg.vocab_size
        self.bos_token = cfg.vocab_size + cfg.n_coeff_bins
        self.total_vocab = self.bos_token + 1

        self.quantizer = CoefficientQuantizer(
            cfg.n_coeff_bins, cfg.coeff_max_val, cfg.coeff_mu,
        )
        self.gpt = MinGPT(
            vocab_size=self.total_vocab,
            block_size=self.seq_len,
            num_classes=cfg.num_classes if cfg.num_classes > 0 else None,
            n_layer=cfg.n_layers,
            n_head=cfg.n_heads,
            n_embd=cfg.d_model,
            dropout=cfg.dropout,
            n_token_types=2,
        )
        type_ids = torch.zeros(self.seq_len, dtype=torch.long)
        type_ids[1::2] = 1
        self.register_buffer("_type_ids", type_ids, persistent=False)

    def _interleave(self, atoms: torch.Tensor, coeff_bins: torch.Tensor):
        B = atoms.size(0)
        seq = torch.empty(B, self.seq_len, dtype=torch.long, device=atoms.device)
        seq[:, 0::2] = atoms.view(B, self.sparse_len)
        seq[:, 1::2] = coeff_bins.view(B, self.sparse_len) + self.coeff_offset
        return seq

    def _mask_logits_by_slot(
        self, logits: torch.Tensor, step: int,
    ) -> torch.Tensor:
        """Mask logits so each step samples only valid token type."""
        masked = torch.full_like(logits, float("-inf"))
        if step % 2 == 0:
            masked[:, :self.cfg.vocab_size] = logits[:, :self.cfg.vocab_size]
        else:
            lo = self.coeff_offset
            hi = self.coeff_offset + self.cfg.n_coeff_bins
            masked[:, lo:hi] = logits[:, lo:hi]
        return masked

    def forward_loss(
        self, tokens_flat: torch.Tensor, coeff_bins_flat: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weighted per-type CE loss over the interleaved sequence."""
        target = self._interleave(tokens_flat, coeff_bins_flat)
        B = target.size(0)
        bos = torch.full((B, 1), self.bos_token,
                         dtype=torch.long, device=target.device)
        inp = torch.cat([bos, target[:, :-1]], dim=1)
        # Input stream is [BOS, target[:-1]], so type ids must be shifted too.
        type_ids = torch.zeros(B, self.seq_len, dtype=torch.long, device=target.device)
        if self.seq_len > 1:
            type_ids[:, 1:] = self._type_ids[:-1].unsqueeze(0).expand(B, -1)
        cls = class_ids.long().to(target.device) if class_ids is not None else None
        logits, _ = self.gpt(inp, class_idx=cls, type_ids=type_ids)

        atom_mask = self._type_ids == 0
        coeff_mask = ~atom_mask
        atom_logits = logits[:, atom_mask, :].reshape(-1, self.gpt.vocab_size)
        coeff_logits = logits[:, coeff_mask, :].reshape(-1, self.gpt.vocab_size)
        atom_targets = target[:, atom_mask].reshape(-1)
        coeff_targets = target[:, coeff_mask].reshape(-1)

        atom_loss = F.cross_entropy(atom_logits, atom_targets)
        coeff_loss = F.cross_entropy(coeff_logits, coeff_targets)
        w = float(self.cfg.coeff_loss_weight)
        return (atom_loss + w * coeff_loss) / (1.0 + w)

    @torch.no_grad()
    def generate(
        self, batch_size: int, temperature: float = 1.0,
        top_k: Optional[int] = None, class_ids: Optional[torch.Tensor] = None,
        show_progress: bool = False, **_kw,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Type-aware autoregressive sampling with temperature."""
        device = next(self.parameters()).device
        _ = top_k  # Top-k truncation is intentionally disabled for sampling.
        cls = class_ids.long().to(device) if class_ids is not None else None
        bos = torch.full((batch_size, 1), self.bos_token,
                         dtype=torch.long, device=device)
        seq = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=device)
        bos_type = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        logits, cache = self.gpt.forward_step(bos, class_idx=cls, type_ids=bos_type)

        steps = tqdm(
            range(self.seq_len),
            desc="sampling",
            leave=False,
            disable=(not show_progress),
        )
        for step in steps:
            step_logits = self._mask_logits_by_slot(
                logits[:, -1, :] / max(float(temperature), 1e-8),
                step,
            )
            tok = torch.multinomial(F.softmax(step_logits, dim=-1), 1).squeeze(-1)
            seq[:, step] = tok
            if step + 1 < self.seq_len:
                # The token we feed now is seq[:, step], so use that slot's type.
                step_type = self._type_ids[step : step + 1].unsqueeze(0).expand(
                    batch_size, -1,
                )
                logits, cache = self.gpt.forward_step(
                    tok.unsqueeze(1), class_idx=cls, kv_cache=cache,
                    type_ids=step_type,
                )

        atoms = seq[:, 0::2]
        cbins = seq[:, 1::2] - self.coeff_offset
        coeffs = self.quantizer.decode(cbins).contiguous().view(batch_size, -1)
        return atoms, coeffs


# Backward-compat aliases.
MinGPTPriorConfig = MinGPTSparseConfig
MinGPTPrior = MinGPTSparse
