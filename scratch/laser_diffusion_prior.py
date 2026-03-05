"""
laser_diffusion_prior.py

Standalone diffusion-based prior for the LASER sparse dictionary autoencoder.

Instead of autoregressive generation, a denoising diffusion model generates the
full token sequence (atom IDs + coefficients) in parallel.  The discrete atom
IDs are embedded into a continuous space, concatenated with the scalar
coefficients, and processed by a bidirectional transformer with AdaLN
conditioning on the diffusion timestep.  At inference the continuous atom
embeddings are decoded back to discrete IDs via nearest-neighbour lookup.

Supports both DDPM and (optionally faster) DDIM sampling.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiffusionPriorConfig:
    vocab_size: int
    H: int
    W: int
    D: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    num_timesteps: int = 1000
    coeff_max: float = 24.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_timestep_embedding(t: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar timesteps.

    Args:
        t: [B] integer or float timesteps.
        d: embedding dimension (must be even).

    Returns:
        [B, d] embedding tensor.
    """
    assert d % 2 == 0
    half = d // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Return pre-computed diffusion schedule tensors."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas.float(), alphas.float(), alpha_bar.float()


# ---------------------------------------------------------------------------
# Adaptive Layer Norm (AdaLN)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Bidirectional Self-Attention (no causal mask)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


# ---------------------------------------------------------------------------
# Diffusion Transformer Block (AdaLN + bidirectional attention)
# ---------------------------------------------------------------------------

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.adaln1 = AdaLN(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.adaln2 = AdaLN(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.adaln1(x, t_emb))
        x = x + self.ffn(self.adaln2(x, t_emb))
        return x


# ---------------------------------------------------------------------------
# Diffusion Prior
# ---------------------------------------------------------------------------

class DiffusionPrior(nn.Module):
    def __init__(self, cfg: DiffusionPriorConfig):
        super().__init__()
        self.cfg = cfg
        S = cfg.H * cfg.W
        feat_dim = cfg.D * (cfg.d_model + 1)

        # Embeddings
        self.atom_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.row_emb = nn.Embedding(cfg.H, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(cfg.D, cfg.d_model)

        # Timestep embedding: sinusoidal → MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Projections
        self.input_proj = nn.Linear(feat_dim, cfg.d_model)
        self.output_proj = nn.Linear(cfg.d_model, feat_dim)

        # Transformer
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Noise schedule (registered as buffers so they move with .to(device))
        betas, alphas, alpha_bar = linear_beta_schedule(cfg.num_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        # Precompute spatial index grids
        rows = torch.arange(cfg.H).unsqueeze(1).expand(cfg.H, cfg.W).reshape(S)
        cols = torch.arange(cfg.W).unsqueeze(0).expand(cfg.H, cfg.W).reshape(S)
        self.register_buffer("row_idx", rows)
        self.register_buffer("col_idx", cols)

    # ---- internal helpers ------------------------------------------------

    def _encode_input(self, atom_ids: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Embed atom IDs and concat with coefficients → flat feature per spatial position.

        Args:
            atom_ids: [B, S, D] integer atom indices.
            coeffs:   [B, S, D] float coefficients.

        Returns:
            x0_flat: [B, S, D*(d_model+1)]  the clean signal.
        """
        x_atoms = self.atom_emb(atom_ids)                        # [B, S, D, d_model]
        x = torch.cat([x_atoms, coeffs.unsqueeze(-1)], dim=-1)   # [B, S, D, d_model+1]
        B, S = x.shape[:2]
        return x.reshape(B, S, -1)                               # [B, S, D*(d_model+1)]

    def _pos_embed(self, B: int) -> torch.Tensor:
        """Spatial positional embedding broadcast to batch. [B, S, d_model]"""
        return (self.row_emb(self.row_idx) + self.col_emb(self.col_idx)).unsqueeze(0).expand(B, -1, -1)

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Timestep conditioning vector. [B, 1, d_model] (broadcastable over S)."""
        t_emb = sinusoidal_timestep_embedding(t, self.cfg.d_model)
        return self.time_mlp(t_emb).unsqueeze(1)

    def _run_transformer(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h, t_emb)
        return self.ln_f(h)

    def _predict_x0(self, xt_proj: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Given projected noisy input and timestep, predict clean x0.

        Args:
            xt_proj: [B, S, d_model]  (already through input_proj + pos_emb).
            t:       [B] timestep indices.

        Returns:
            x0_pred: [B, S, feat_dim]
        """
        B = xt_proj.shape[0]
        t_emb = self._time_embed(t)
        h = self._run_transformer(xt_proj, t_emb)
        return self.output_proj(h)

    # ---- training --------------------------------------------------------

    def forward(self, atom_ids: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute diffusion training loss.

        Args:
            atom_ids: [B, H*W, D] long tensor of atom indices.
            coeffs:   [B, H*W, D] float tensor of coefficients.

        Returns:
            Scalar MSE loss.
        """
        B, S, D = atom_ids.shape
        device = atom_ids.device

        x0_flat = self._encode_input(atom_ids, coeffs)           # [B, S, feat_dim]

        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=device)
        noise = torch.randn_like(x0_flat)
        ab_t = self.alpha_bar[t].view(B, 1, 1)
        xt = ab_t.sqrt() * x0_flat + (1.0 - ab_t).sqrt() * noise

        xt_proj = self.input_proj(xt) + self._pos_embed(B)
        x0_pred = self._predict_x0(xt_proj, t)

        return F.mse_loss(x0_pred, x0_flat)

    # ---- DDPM sampling ---------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDPM ancestral sampling.

        Returns:
            atom_ids: [B, S, D] decoded atom indices.
            coeffs:   [B, S, D] predicted coefficients.
        """
        cfg = self.cfg
        S = cfg.H * cfg.W
        feat_dim = cfg.D * (cfg.d_model + 1)
        device = self.betas.device

        xt = torch.randn(batch_size, S, feat_dim, device=device)

        steps = reversed(range(cfg.num_timesteps))
        if show_progress:
            steps = tqdm(steps, total=cfg.num_timesteps, desc="DDPM sampling")

        for t_val in steps:
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            xt_proj = self.input_proj(xt) + self._pos_embed(batch_size)
            x0_pred = self._predict_x0(xt_proj, t)

            # DDPM posterior
            alpha_t = self.alphas[t_val]
            alpha_bar_t = self.alpha_bar[t_val]
            alpha_bar_prev = self.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=device)
            beta_t = self.betas[t_val]

            # Posterior mean
            coeff1 = beta_t * alpha_bar_prev.sqrt() / (1.0 - alpha_bar_t)
            coeff2 = (1.0 - alpha_bar_prev) * alpha_t.sqrt() / (1.0 - alpha_bar_t)
            mu = coeff1 * x0_pred + coeff2 * xt

            if t_val > 0:
                sigma = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t).sqrt()
                xt = mu + sigma * torch.randn_like(xt)
            else:
                xt = mu

        return self._decode_output(x0_pred)

    # ---- DDIM sampling ---------------------------------------------------

    @torch.no_grad()
    def generate_ddim(
        self,
        batch_size: int,
        num_steps: int = 50,
        eta: float = 0.0,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDIM sampling with a reduced number of steps.

        Args:
            batch_size: number of samples.
            num_steps:  number of denoising steps (≤ num_timesteps).
            eta:        0 = fully deterministic (DDIM), 1 = DDPM-equivalent.
            show_progress: show tqdm bar.

        Returns:
            atom_ids, coeffs (same shapes as generate).
        """
        cfg = self.cfg
        S = cfg.H * cfg.W
        feat_dim = cfg.D * (cfg.d_model + 1)
        device = self.betas.device

        step_size = cfg.num_timesteps // num_steps
        timesteps = list(range(cfg.num_timesteps - 1, -1, -step_size))

        xt = torch.randn(batch_size, S, feat_dim, device=device)

        pairs = list(zip(timesteps, timesteps[1:] + [-1]))
        if show_progress:
            pairs = tqdm(pairs, desc="DDIM sampling")

        x0_pred = None
        for t_val, t_prev_val in pairs:
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            xt_proj = self.input_proj(xt) + self._pos_embed(batch_size)
            x0_pred = self._predict_x0(xt_proj, t)

            if t_prev_val < 0:
                break

            alpha_bar_t = self.alpha_bar[t_val]
            alpha_bar_prev = self.alpha_bar[t_prev_val]

            eps_pred = (xt - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt()

            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
            dir_xt = (1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * eps_pred
            xt = alpha_bar_prev.sqrt() * x0_pred + dir_xt
            if sigma > 0:
                xt = xt + sigma * torch.randn_like(xt)

        return self._decode_output(x0_pred)

    # ---- output decoding -------------------------------------------------

    def _decode_output(self, x0_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split predicted flat features into atom IDs (via NN-lookup) and coefficients.

        Args:
            x0_pred: [B, S, D*(d_model+1)]

        Returns:
            atom_ids: [B, S, D] long
            coeffs:   [B, S, D] float (clamped to [-coeff_max, coeff_max])
        """
        cfg = self.cfg
        B, S, _ = x0_pred.shape
        x0_pred = x0_pred.reshape(B, S, cfg.D, cfg.d_model + 1)
        atom_emb_pred = x0_pred[..., :cfg.d_model]    # [B, S, D, d_model]
        coeff_pred = x0_pred[..., cfg.d_model]          # [B, S, D]

        # Nearest-neighbour lookup via cosine similarity
        weight = self.atom_emb.weight                           # [V, d_model]
        atom_emb_flat = atom_emb_pred.reshape(-1, cfg.d_model)  # [B*S*D, d_model]
        atom_emb_flat = F.normalize(atom_emb_flat, dim=-1)
        weight_norm = F.normalize(weight, dim=-1)
        sims = atom_emb_flat @ weight_norm.t()                  # [B*S*D, V]
        atom_ids = sims.argmax(dim=-1).reshape(B, S, cfg.D)

        coeffs = coeff_pred.clamp(-cfg.coeff_max, cfg.coeff_max)
        return atom_ids, coeffs


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = DiffusionPriorConfig(
        vocab_size=512,
        H=4,
        W=4,
        D=4,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        num_timesteps=20,
    )

    model = DiffusionPrior(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiffusionPrior  params: {n_params:,}")

    B, S, D = 2, cfg.H * cfg.W, cfg.D
    atom_ids = torch.randint(0, cfg.vocab_size, (B, S, D), device=device)
    coeffs = torch.randn(B, S, D, device=device)

    # Training forward pass
    loss = model(atom_ids, coeffs)
    print(f"Training loss:  {loss.item():.4f}")
    loss.backward()
    print("Backward pass OK")

    # DDPM generation
    gen_ids, gen_coeffs = model.generate(batch_size=2, show_progress=True)
    print(f"DDPM  generate: ids {gen_ids.shape}, coeffs {gen_coeffs.shape}")

    # DDIM generation
    gen_ids2, gen_coeffs2 = model.generate_ddim(batch_size=2, num_steps=5, show_progress=True)
    print(f"DDIM  generate: ids {gen_ids2.shape}, coeffs {gen_coeffs2.shape}")

    print("\nSmoke test passed.")
