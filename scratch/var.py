"""
LASER stage-2 prior inspired by VAR-style next-scale prediction.

The original VAR paper models images as a coarse-to-fine pyramid and predicts
the next spatial scale instead of the next raster token. For LASER, the sparse
latent already has a natural coarse-to-fine axis inside each latent site: the
progressive sparse support itself. This module reinterprets the k-th sparse slot
as the k-th "scale" and predicts the next atom/coeff pair for every latent site
in parallel.

Concretely, stage k predicts:
  - the k-th atom id for every H x W latent site
  - the associated quantized coefficient bin or real-valued coefficient

conditioned on all previous sparse stages 0..k-1. This gives a next-sparsity
prediction analogue of VAR that is aligned with LASER's dictionary manifold.

This formulation is most faithful when the stage-1 bottleneck keeps the greedy
OMP selection order, so `force_greedy_omp_slot_order(ae)` is provided as a
convenience helper.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def soft_clamp(x: torch.Tensor, max_val: float) -> torch.Tensor:
    """Smoothly bound values to [-max_val, max_val] without a hard clip."""
    if max_val <= 0:
        raise ValueError(f"max_val must be > 0, got {max_val}")
    return max_val * torch.tanh(x / max_val)


@dataclass
class SparsityVARConfig:
    H: int
    W: int
    num_stages: int
    atom_vocab_size: int
    coeff_vocab_size: Optional[int] = None
    coeff_bin_values: Optional[torch.Tensor] = None
    real_valued_coeffs: bool = False
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    n_global_tokens: int = 0
    cond_channels: int = 0
    coeff_max: float = 24.0


def build_sparsity_var_config(
    bottleneck,
    *,
    H: int,
    W: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    n_global_tokens: int = 0,
    use_reconstruction_conditioning: bool = True,
    real_valued_coeffs: Optional[bool] = None,
    coeff_max_fallback: float = 24.0,
) -> SparsityVARConfig:
    """Build a config from a LASER bottleneck."""
    if real_valued_coeffs is None:
        real_valued_coeffs = not bool(getattr(bottleneck, "quantize_sparse_coeffs", False))

    coeff_vocab_size = None
    coeff_bin_values = None
    if not real_valued_coeffs:
        coeff_vocab_size = int(bottleneck.n_bins)
        coeff_bin_values = bottleneck._dequantize_coeff(
            torch.arange(coeff_vocab_size, dtype=torch.long)
        ).detach().cpu()

    cond_channels = int(getattr(bottleneck, "embedding_dim", 0)) if use_reconstruction_conditioning else 0

    return SparsityVARConfig(
        H=int(H),
        W=int(W),
        num_stages=int(getattr(bottleneck, "sparsity_level")),
        atom_vocab_size=int(getattr(bottleneck, "num_embeddings")),
        coeff_vocab_size=coeff_vocab_size,
        coeff_bin_values=coeff_bin_values,
        real_valued_coeffs=bool(real_valued_coeffs),
        d_model=int(d_model),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        d_ff=int(d_ff),
        dropout=float(dropout),
        n_global_tokens=int(n_global_tokens),
        cond_channels=int(cond_channels),
        coeff_max=float(getattr(bottleneck, "coef_max", coeff_max_fallback)),
    )


def build_var_config(**kwargs) -> SparsityVARConfig:
    return build_sparsity_var_config(**kwargs)


def force_greedy_omp_slot_order(ae) -> None:
    """Make stage-1 emit slots in greedy OMP order instead of canonicalized order."""
    bottleneck = getattr(ae, "bottleneck", None)
    if bottleneck is None:
        raise AttributeError("ae must expose a .bottleneck attribute")
    if hasattr(bottleneck, "canonicalize_sparse_slots"):
        bottleneck.canonicalize_sparse_slots = False


class SpatialStageBlock(nn.Module):
    """Full-attention block over latent sites within one sparsity stage."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x


class SparsityLevelVAR(nn.Module):
    """Predict LASER sparse codes stage-by-stage instead of token-by-token."""

    def __init__(self, cfg: SparsityVARConfig):
        super().__init__()
        self.cfg = cfg
        self.real_valued_coeffs = bool(cfg.real_valued_coeffs)
        self.atom_vocab_size = int(cfg.atom_vocab_size)
        self.num_stages = int(cfg.num_stages)
        self.n_global_tokens = int(cfg.n_global_tokens)
        self.coeff_vocab_size = None if cfg.coeff_vocab_size is None else int(cfg.coeff_vocab_size)

        if cfg.H <= 0 or cfg.W <= 0:
            raise ValueError(f"H and W must be positive, got {(cfg.H, cfg.W)}")
        if self.num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {self.num_stages}")
        if self.atom_vocab_size <= 0:
            raise ValueError(f"atom_vocab_size must be positive, got {self.atom_vocab_size}")
        if self.num_stages > self.atom_vocab_size:
            raise ValueError(
                "num_stages cannot exceed atom_vocab_size when duplicate atoms are disallowed, "
                f"got num_stages={self.num_stages} atom_vocab_size={self.atom_vocab_size}"
            )
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})")
        if self.real_valued_coeffs:
            if self.coeff_vocab_size not in (None, 0):
                raise ValueError("coeff_vocab_size must be unset when real_valued_coeffs=True")
            self.coeff_vocab_size = None
            self.token_depth = self.num_stages
        else:
            if self.coeff_vocab_size is None or self.coeff_vocab_size <= 0:
                raise ValueError("coeff_vocab_size must be set when real_valued_coeffs=False")
            self.token_depth = 2 * self.num_stages

        self.total_sites = int(cfg.H) * int(cfg.W)

        self.atom_emb = nn.Embedding(self.atom_vocab_size, cfg.d_model)
        self.row_emb = nn.Embedding(cfg.H, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.W, cfg.d_model)
        self.stage_emb = nn.Embedding(self.num_stages, cfg.d_model)
        self.start_site = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.start_site, std=0.02)

        self.global_tokens = None
        if self.n_global_tokens > 0:
            self.global_tokens = nn.Parameter(torch.zeros(1, self.n_global_tokens, cfg.d_model))
            nn.init.normal_(self.global_tokens, std=0.02)

        self.coeff_proj = nn.Linear(1, cfg.d_model)
        self.coeff_emb = None
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
                        f"got {coeff_bin_values.numel()} and {self.coeff_vocab_size}"
                    )
            self.register_buffer("_coeff_bin_values", coeff_bin_values, persistent=False)
            self.coeff_emb = nn.Embedding(self.coeff_vocab_size, cfg.d_model)
        else:
            self._coeff_bin_values = None

        self.state_proj = nn.Sequential(
            nn.Linear(self.num_stages * cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

        self.cond_proj = None
        if int(cfg.cond_channels) > 0:
            self.cond_proj = nn.Conv2d(int(cfg.cond_channels), cfg.d_model, kernel_size=1)

        self.blocks = nn.ModuleList(
            [SpatialStageBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.atom_head = nn.Linear(cfg.d_model, self.atom_vocab_size)
        coeff_out_dim = 1 if self.real_valued_coeffs else self.coeff_vocab_size
        self.coeff_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, coeff_out_dim),
        )

        spatial_pos = torch.arange(self.total_sites, dtype=torch.long)
        self.register_buffer("_rows", spatial_pos // cfg.W, persistent=False)
        self.register_buffer("_cols", spatial_pos % cfg.W, persistent=False)

    def _flatten_sites(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.ndim == 3:
            if x.size(1) != self.total_sites:
                raise ValueError(f"Expected {self.total_sites} sites, got {x.size(1)}")
            return x
        if x.ndim == 4:
            if tuple(x.shape[1:3]) != (self.cfg.H, self.cfg.W):
                raise ValueError(
                    f"Expected spatial shape {(self.cfg.H, self.cfg.W)}, got {tuple(x.shape[1:3])}"
                )
            return x.view(x.size(0), self.total_sites, x.size(-1))
        raise ValueError(f"Expected rank-3 or rank-4 tensor, got {tuple(x.shape)}")

    def _unflatten_sites(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(f"Expected at least rank-3 tensor, got {tuple(x.shape)}")
        return x.view(x.size(0), self.cfg.H, self.cfg.W, *x.shape[2:])

    def _spatial_pos(self, device: torch.device) -> torch.Tensor:
        rows = self._rows.to(device)
        cols = self._cols.to(device)
        return self.row_emb(rows) + self.col_emb(cols)

    def _prepend_global_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_tokens is None:
            return x
        global_tokens = self.global_tokens.expand(x.size(0), -1, -1)
        return torch.cat([global_tokens, x], dim=1)

    def _drop_global_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_global_tokens <= 0:
            return x
        return x[:, self.n_global_tokens :]

    def _coeff_input_embed(
        self,
        coeff_inputs: torch.Tensor,
    ) -> torch.Tensor:
        if self.real_valued_coeffs:
            coeff_values = coeff_inputs.to(dtype=self.start_site.dtype).unsqueeze(-1)
            return self.coeff_proj(coeff_values)

        coeff_idx = coeff_inputs.to(torch.long).clamp(0, self.coeff_vocab_size - 1)
        coeff_values = self._coeff_bin_values.to(device=coeff_idx.device, dtype=self.start_site.dtype)[coeff_idx]
        coeff_embed = self.coeff_emb(coeff_idx)
        return coeff_embed + self.coeff_proj(coeff_values.unsqueeze(-1))

    def _build_prefix_state(
        self,
        atom_prefix: torch.Tensor,
        coeff_prefix: torch.Tensor,
    ) -> torch.Tensor:
        B, T, prefix_len = atom_prefix.shape
        if prefix_len == 0:
            flat = torch.zeros(
                B,
                T,
                self.num_stages * self.cfg.d_model,
                device=atom_prefix.device,
                dtype=self.start_site.dtype,
            )
            return self.state_proj(flat)

        stage_ids = torch.arange(prefix_len, device=atom_prefix.device)
        stage_embed = self.stage_emb(stage_ids).view(1, 1, prefix_len, self.cfg.d_model)
        atom_embed = self.atom_emb(atom_prefix.to(torch.long))
        coeff_embed = self._coeff_input_embed(coeff_prefix)
        slot_embed = atom_embed + coeff_embed + stage_embed

        full_slots = torch.zeros(
            B,
            T,
            self.num_stages,
            self.cfg.d_model,
            device=slot_embed.device,
            dtype=slot_embed.dtype,
        )
        full_slots[:, :, :prefix_len] = slot_embed
        flat = full_slots.reshape(B, T, self.num_stages * self.cfg.d_model)
        return self.state_proj(flat)

    def _prepare_cond(self, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if cond is None:
            return None
        if self.cond_proj is None:
            raise ValueError("cond was provided but cond_channels=0 in the config")
        if cond.ndim != 4:
            raise ValueError(f"Expected cond with shape [B,C,H,W], got {tuple(cond.shape)}")
        if tuple(cond.shape[-2:]) != (self.cfg.H, self.cfg.W):
            cond = F.interpolate(cond, size=(self.cfg.H, self.cfg.W), mode="bilinear", align_corners=False)
        cond = self.cond_proj(cond)
        return cond.permute(0, 2, 3, 1).reshape(cond.size(0), self.total_sites, self.cfg.d_model)

    def _stage_hidden(
        self,
        atom_prefix: torch.Tensor,
        coeff_prefix: torch.Tensor,
        stage_idx: int,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        site_state = self._build_prefix_state(atom_prefix, coeff_prefix)
        x = self.start_site.expand(site_state.size(0), self.total_sites, -1) + site_state
        x = x + self.stage_emb.weight[stage_idx].view(1, 1, -1)
        x = x + self._spatial_pos(x.device).unsqueeze(0)
        cond_flat = self._prepare_cond(cond)
        if cond_flat is not None:
            x = x + cond_flat
        x = self._prepend_global_tokens(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self._drop_global_tokens(x)
        return x

    def _mask_duplicate_atom_logits(
        self,
        logits: torch.Tensor,
        atom_prefix: torch.Tensor,
    ) -> torch.Tensor:
        if atom_prefix.numel() == 0:
            return logits
        masked = logits.clone()
        masked.scatter_(2, atom_prefix.to(torch.long), float("-inf"))
        return masked

    def _predict_coeff_output(
        self,
        stage_hidden: torch.Tensor,
        atom_ids: torch.Tensor,
    ) -> torch.Tensor:
        atom_embed = self.atom_emb(atom_ids.to(torch.long))
        coeff_in = torch.cat([stage_hidden, atom_embed], dim=-1)
        coeff_out = self.coeff_head(coeff_in)
        if self.real_valued_coeffs:
            coeff_out = soft_clamp(coeff_out.squeeze(-1), self.cfg.coeff_max)
        return coeff_out

    def _split_inputs(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._flatten_sites(tokens)
        if self.real_valued_coeffs:
            if coeffs is None:
                raise ValueError("coeffs must be provided when real_valued_coeffs=True")
            coeffs = self._flatten_sites(coeffs)
            if tokens.shape != coeffs.shape:
                raise ValueError(f"tokens and coeffs shape mismatch: {tokens.shape} vs {coeffs.shape}")
            if tokens.size(-1) != self.num_stages:
                raise ValueError(f"Expected token depth {self.num_stages}, got {tokens.size(-1)}")
            return tokens.to(torch.long), coeffs.to(self.start_site.dtype)

        if coeffs is not None:
            raise ValueError("coeffs must be omitted when real_valued_coeffs=False")
        if tokens.size(-1) != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {tokens.size(-1)}")
        atom_ids = tokens[..., 0::2].to(torch.long)
        coeff_bins = tokens[..., 1::2].to(torch.long) - self.atom_vocab_size
        coeff_bins = coeff_bins.clamp(0, self.coeff_vocab_size - 1)
        return atom_ids, coeff_bins

    def pack_quantized_tokens(
        self,
        atom_ids: torch.Tensor,
        coeff_bins: torch.Tensor,
        *,
        spatial: bool = True,
    ) -> torch.Tensor:
        atom_ids = self._flatten_sites(atom_ids)
        coeff_bins = self._flatten_sites(coeff_bins)
        if atom_ids.shape != coeff_bins.shape:
            raise ValueError(f"atom_ids and coeff_bins shape mismatch: {atom_ids.shape} vs {coeff_bins.shape}")
        if atom_ids.size(-1) != self.num_stages:
            raise ValueError(f"Expected num_stages={self.num_stages}, got {atom_ids.size(-1)}")
        tokens = torch.empty(
            atom_ids.size(0),
            atom_ids.size(1),
            self.token_depth,
            device=atom_ids.device,
            dtype=torch.long,
        )
        tokens[..., 0::2] = atom_ids.to(torch.long)
        tokens[..., 1::2] = coeff_bins.to(torch.long) + self.atom_vocab_size
        return self._unflatten_sites(tokens) if spatial else tokens

    def unpack_quantized_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_ids, coeff_bins = self._split_inputs(tokens)
        return self._unflatten_sites(atom_ids), self._unflatten_sites(coeff_bins)

    def _prefix_latent(
        self,
        bottleneck,
        atom_prefix: torch.Tensor,
        coeff_prefix: torch.Tensor,
    ) -> torch.Tensor:
        if atom_prefix.size(-1) == 0:
            return torch.zeros(
                atom_prefix.size(0),
                int(getattr(bottleneck, "embedding_dim")),
                self.cfg.H,
                self.cfg.W,
                device=atom_prefix.device,
                dtype=self.start_site.dtype,
            )

        support = torch.zeros(
            atom_prefix.size(0),
            self.cfg.H,
            self.cfg.W,
            self.num_stages,
            device=atom_prefix.device,
            dtype=torch.long,
        )
        coeffs = torch.zeros(
            atom_prefix.size(0),
            self.cfg.H,
            self.cfg.W,
            self.num_stages,
            device=atom_prefix.device,
            dtype=self.start_site.dtype,
        )
        prefix_len = atom_prefix.size(-1)
        support[..., :prefix_len] = self._unflatten_sites(atom_prefix)
        coeff_values = coeff_prefix
        if not self.real_valued_coeffs:
            coeff_values = bottleneck._dequantize_coeff(coeff_prefix.to(torch.long))
        coeffs[..., :prefix_len] = self._unflatten_sites(coeff_values.to(self.start_site.dtype))
        return bottleneck._reconstruct_sparse(support, coeffs)

    def forward(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
        *,
        bottleneck=None,
        return_features: bool = False,
    ):
        atom_ids, coeff_inputs = self._split_inputs(tokens, coeffs)
        _, _, K = atom_ids.shape
        atom_logits = []
        coeff_outputs = []
        hidden_states = []

        for stage_idx in range(K):
            prefix_atoms = atom_ids[:, :, :stage_idx]
            prefix_coeffs = coeff_inputs[:, :, :stage_idx]
            cond = None
            if bottleneck is not None and self.cond_proj is not None:
                cond = self._prefix_latent(bottleneck, prefix_atoms, prefix_coeffs)

            stage_hidden = self._stage_hidden(prefix_atoms, prefix_coeffs, stage_idx, cond=cond)
            stage_atom_logits = self._mask_duplicate_atom_logits(
                self.atom_head(stage_hidden),
                prefix_atoms,
            )
            stage_coeff_out = self._predict_coeff_output(stage_hidden, atom_ids[:, :, stage_idx])

            atom_logits.append(stage_atom_logits)
            coeff_outputs.append(stage_coeff_out)
            if return_features:
                hidden_states.append(stage_hidden)

        atom_logits = torch.stack(atom_logits, dim=2)
        coeff_outputs = torch.stack(coeff_outputs, dim=2)

        if return_features:
            features = torch.stack(hidden_states, dim=2)
            return atom_logits, coeff_outputs, features
        return atom_logits, coeff_outputs

    def teacher_forced_argmax(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
        *,
        bottleneck=None,
        spatial: bool = True,
    ):
        atom_logits, coeff_outputs = self(tokens, coeffs, bottleneck=bottleneck, return_features=False)
        pred_atoms = atom_logits.argmax(dim=-1)
        if self.real_valued_coeffs:
            pred_coeffs = coeff_outputs
            if spatial:
                return self._unflatten_sites(pred_atoms), self._unflatten_sites(pred_coeffs)
            return pred_atoms, pred_coeffs

        pred_coeff_bins = coeff_outputs.argmax(dim=-1)
        packed = self.pack_quantized_tokens(pred_atoms, pred_coeff_bins, spatial=spatial)
        return packed

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        if top_k is not None:
            top_k = int(top_k)
        if top_k is not None and 0 < top_k < logits.size(-1):
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            cutoff = top_vals[..., -1:].expand_as(logits)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        if top_p is not None:
            top_p = float(top_p)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove, dtype=torch.bool)
            remove.scatter_(dim=-1, index=sorted_idx, src=sorted_remove)
            logits = logits.masked_fill(remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all():
            raise RuntimeError("Sampling probabilities contain non-finite values")
        next_token = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1)
        return next_token.view(*probs.shape[:-1])

    def _stage_temperature(
        self,
        stage_idx: int,
        *,
        temperature: float,
        temperature_end: Optional[float] = None,
    ) -> float:
        if temperature_end is None or self.num_stages <= 1:
            return float(temperature)
        ratio = float(stage_idx) / float(max(1, self.num_stages - 1))
        return float(temperature) + (float(temperature_end) - float(temperature)) * ratio

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        *,
        temperature: float = 1.0,
        temperature_end: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        bottleneck=None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        spatial: bool = True,
    ):
        device = next(self.parameters()).device
        atom_ids = torch.zeros(batch_size, self.total_sites, self.num_stages, device=device, dtype=torch.long)
        if self.real_valued_coeffs:
            coeff_data = torch.zeros(
                batch_size,
                self.total_sites,
                self.num_stages,
                device=device,
                dtype=self.start_site.dtype,
            )
        else:
            coeff_data = torch.zeros(batch_size, self.total_sites, self.num_stages, device=device, dtype=torch.long)

        iterator = range(self.num_stages)
        if show_progress:
            iterator = tqdm(iterator, desc=progress_desc or "Generating sparsity stages")

        for stage_idx in iterator:
            prefix_atoms = atom_ids[:, :, :stage_idx]
            prefix_coeffs = coeff_data[:, :, :stage_idx]
            stage_temperature = self._stage_temperature(
                stage_idx,
                temperature=temperature,
                temperature_end=temperature_end,
            )
            cond = None
            if bottleneck is not None and self.cond_proj is not None:
                cond = self._prefix_latent(bottleneck, prefix_atoms, prefix_coeffs)

            stage_hidden = self._stage_hidden(prefix_atoms, prefix_coeffs, stage_idx, cond=cond)
            stage_atom_logits = self._mask_duplicate_atom_logits(
                self.atom_head(stage_hidden),
                prefix_atoms,
            )
            sampled_atoms = self._sample_from_logits(
                stage_atom_logits,
                temperature=stage_temperature,
                top_k=top_k,
                top_p=top_p,
            )
            atom_ids[:, :, stage_idx] = sampled_atoms

            stage_coeff_out = self._predict_coeff_output(stage_hidden, sampled_atoms)
            if self.real_valued_coeffs:
                coeff_data[:, :, stage_idx] = stage_coeff_out
            else:
                sampled_bins = self._sample_from_logits(
                    stage_coeff_out,
                    temperature=stage_temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                coeff_data[:, :, stage_idx] = sampled_bins

        if self.real_valued_coeffs:
            if spatial:
                return self._unflatten_sites(atom_ids), self._unflatten_sites(coeff_data)
            return atom_ids, coeff_data

        return self.pack_quantized_tokens(atom_ids, coeff_data, spatial=spatial)

    @torch.no_grad()
    def decode_with_bottleneck(
        self,
        bottleneck,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.real_valued_coeffs:
            if coeffs is None:
                raise ValueError("coeffs must be provided when decoding real-valued outputs")
            tokens = self._unflatten_sites(self._flatten_sites(tokens))
            coeffs = self._unflatten_sites(self._flatten_sites(coeffs))
            return bottleneck.tokens_to_latent(tokens, coeffs=coeffs)

        tokens = self._unflatten_sites(self._flatten_sites(tokens))
        return bottleneck.tokens_to_latent(tokens)


SparsityVAR = SparsityLevelVAR
VAR = SparsityLevelVAR
VARConfig = SparsityVARConfig

__all__ = [
    "SparsityVARConfig",
    "SparsityLevelVAR",
    "SparsityVAR",
    "VARConfig",
    "VAR",
    "build_sparsity_var_config",
    "build_var_config",
    "force_greedy_omp_slot_order",
]
