import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _nan_to_num_tensor(
    x: torch.Tensor,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def _gaussian_kl_to_fixed_mean(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: float,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians where p uses a fixed std and mean."""
    target_std = float(max(target_std, 1e-6))
    target_var = target_std * target_std
    var = logvar.exp().clamp_min(1e-8)
    sq_mean = (mu - target_mean).square()
    kl = 0.5 * ((var + sq_mean) / target_var - 1.0 + math.log(target_var) - logvar)
    return kl.mean()


def _canonical_patch_reconstruction(
    patch_reconstruction: Optional[str],
    *,
    patch_size: Optional[int],
    patch_stride: Optional[int],
) -> str:
    mode = "center_crop" if patch_reconstruction is None else str(patch_reconstruction).strip().lower()
    if mode not in ("center_crop", "hann", "tile"):
        raise ValueError("patch_reconstruction must be 'center_crop', 'hann', or 'tile'")
    if patch_size is not None and patch_stride is not None:
        try:
            patch_size_i = int(patch_size)
            patch_stride_i = int(patch_stride)
        except (TypeError, ValueError):
            patch_size_i = None
            patch_stride_i = None
        if patch_size_i is not None and patch_stride_i is not None and patch_stride_i == patch_size_i:
            # Non-overlapping patches can be stitched directly without overlap-aware blending.
            return "tile"
    return mode


def _batched_omp_with_support(
    X: torch.Tensor,
    D: torch.Tensor,
    sparsity_level: int,
    diag_eps: float = 1e-4,
    cholesky_eps: float = 1e-6,
    return_history: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Numerically damped batched OMP that returns support indices and ordered coefficients."""
    if X.ndim != 2 or D.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got X={tuple(X.shape)} D={tuple(D.shape)}")
    if sparsity_level > int(D.size(1)):
        raise ValueError(
            f"sparsity_level ({int(sparsity_level)}) must be <= num_atoms ({int(D.size(1))})"
        )

    X = _nan_to_num_tensor(X)
    D = _nan_to_num_tensor(D)

    _, batch_size = X.size()
    device = D.device
    dtype = D.dtype
    batch_idx = torch.arange(batch_size, device=device)

    Dt = D.t()
    G = Dt.mm(D)
    if diag_eps > 0.0:
        G = G + float(diag_eps) * torch.eye(G.size(0), device=device, dtype=dtype)
    h_bar = Dt.mm(X).t()
    h = h_bar.clone()
    x = torch.zeros_like(h_bar)
    L = torch.empty(batch_size, 0, 0, device=device, dtype=dtype)
    I = torch.empty(batch_size, 0, device=device, dtype=torch.long)
    I_logic = torch.zeros_like(h_bar, dtype=torch.bool)
    support_history = [] if return_history else None
    coeff_history = [] if return_history else None

    def _update_logical(logical: torch.Tensor, to_add: torch.Tensor) -> None:
        logical[batch_idx, to_add] = True

    while I.size(1) < int(sparsity_level):
        scores = h.abs().masked_fill(I_logic, -1.0)
        index = scores.argmax(dim=1)
        _update_logical(I_logic, index)

        selected = int(I.size(1))
        diag_g = G[index, index].view(batch_size, 1, 1)
        if selected == 0:
            L = torch.sqrt(torch.clamp(diag_g, min=cholesky_eps))
        else:
            expanded_batch_idx = batch_idx.unsqueeze(0).expand(selected, batch_size).t()
            G_stack = G[I[batch_idx, :], index[expanded_batch_idx]].view(batch_size, selected, 1)
            w = torch.linalg.solve_triangular(L, G_stack, upper=False)
            w_t = w.transpose(1, 2)
            w_corner = torch.sqrt(
                torch.clamp(diag_g - (w_t ** 2).sum(dim=2, keepdim=True), min=cholesky_eps)
            )
            k_zeros = torch.zeros(batch_size, selected, 1, device=device, dtype=dtype)
            L = torch.cat(
                (
                    torch.cat((L, k_zeros), dim=2),
                    torch.cat((w_t, w_corner), dim=2),
                ),
                dim=1,
            )

        I = torch.cat([I, index.unsqueeze(1)], dim=1)
        support_size = int(I.size(1))
        expanded_batch_idx = batch_idx.unsqueeze(0).expand(support_size, batch_size).t()
        h_stack = h_bar[expanded_batch_idx, I].view(batch_size, support_size, 1)
        try:
            x_stack = torch.cholesky_solve(h_stack, L)
        except RuntimeError:
            gram_support = torch.bmm(L, L.transpose(1, 2))
            reg_eye = torch.eye(support_size, device=device, dtype=dtype).expand(batch_size, -1, -1)
            x_stack = torch.linalg.solve(gram_support + cholesky_eps * reg_eye, h_stack)
        x_stack = torch.nan_to_num(x_stack, nan=0.0, posinf=0.0, neginf=0.0)
        x[batch_idx.unsqueeze(1), I] = x_stack.squeeze(-1)
        coeffs_ordered = x[batch_idx.unsqueeze(1), I]
        coeffs_ordered = torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)
        if return_history:
            padded_support = torch.zeros(batch_size, int(sparsity_level), device=device, dtype=torch.long)
            padded_coeffs = torch.zeros(batch_size, int(sparsity_level), device=device, dtype=dtype)
            padded_support[:, :support_size] = I
            padded_coeffs[:, :support_size] = coeffs_ordered
            support_history.append(padded_support)
            coeff_history.append(padded_coeffs)

        beta = (
            coeffs_ordered
            .unsqueeze(1)
            .bmm(G[I[batch_idx], :])
            .squeeze(1)
        )
        h = torch.nan_to_num(h_bar - beta, nan=0.0, posinf=0.0, neginf=0.0)

    coeffs_ordered = x[batch_idx.unsqueeze(1), I]
    coeffs_ordered = torch.nan_to_num(coeffs_ordered, nan=0.0, posinf=0.0, neginf=0.0)
    if not return_history:
        return I, coeffs_ordered
    return (
        I,
        coeffs_ordered,
        torch.stack(support_history, dim=1),
        torch.stack(coeff_history, dim=1),
    )


def nonlinearity(x):
    return F.silu(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** -0.5)
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    """RQ-VAE encoder with ResNet blocks, optional attention, and progressive downsampling."""
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=False, use_mid_attention=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_mid_attention = bool(use_mid_attention)

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in) if self.use_mid_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """RQ-VAE decoder with ResNet blocks, optional attention, and progressive upsampling."""
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, use_mid_attention=True, extra_res_blocks=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.use_mid_attention = bool(use_mid_attention)
        self.extra_res_blocks = max(0, int(extra_res_blocks))

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in) if self.use_mid_attention else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.blocks_per_level = max(1, self.num_res_blocks + self.extra_res_blocks)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.blocks_per_level):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.blocks_per_level):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# -----------------------------
# Dictionary learning bottleneck (batch OMP) + Option-A tokenization
# -----------------------------

class DictionaryLearningTokenized(nn.Module):
    """
    Dictionary-learning bottleneck with batched Orthogonal Matching Pursuit (OMP) sparse coding.
    Tokenization modes:
    - Quantized-mode: alternating token pairs [atom_id, coeff_bin + num_atoms].
    - Regressor-mode: token = atom_id only, coefficients are modeled with a separate head.

    Outputs, per latent pixel, a token stack of length:
    - 2 * sparsity_level in quantized mode
    - sparsity_level in regressor mode

    Important simplifications (good for a quick test):
      - OMP runs under torch.no_grad() like in LASER: we do NOT backprop through sparse coding.
      - We reconstruct the latent using quantized coefficients, then apply STE so the encoder
        still receives gradients (VQ-VAE style).
    """
    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 16,
        sparsity_level: int = 8,
        n_bins: int = 16,
        coef_max: float = 3.0,
        quantize_sparse_coeffs: bool = False,
        coef_quantization: str = "uniform",
        coef_mu: float = 0.0,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
        canonicalize_sparse_slots: bool = True,
        variational_coeffs: bool = False,
        variational_coeff_kl_weight: float = 0.0,
        variational_coeff_prior_std: float = 0.25,
        variational_coeff_min_std: float = 0.01,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sparsity_level = int(sparsity_level)
        self.n_bins = int(n_bins)
        self.coef_max = float(coef_max)
        self.quantize_sparse_coeffs = bool(quantize_sparse_coeffs)
        self.coef_quantization = str(coef_quantization)
        self.coef_mu = float(coef_mu)
        if self.coef_quantization not in ("uniform", "mu_law"):
            raise ValueError(
                "coef_quantization must be one of {'uniform', 'mu_law'}"
            )
        if self.coef_quantization == "mu_law" and self.coef_mu <= 0.0:
            raise ValueError(f"coef_mu must be > 0, got {self.coef_mu}")
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        self.canonicalize_sparse_slots = bool(canonicalize_sparse_slots)
        self.variational_coeffs = bool(variational_coeffs)
        self.variational_coeff_kl_weight = float(variational_coeff_kl_weight)
        self.variational_coeff_prior_std = float(variational_coeff_prior_std)
        self.variational_coeff_min_std = float(variational_coeff_min_std)
        if self.variational_coeff_kl_weight < 0.0:
            raise ValueError(
                f"variational_coeff_kl_weight must be >= 0, got {self.variational_coeff_kl_weight}"
            )
        if self.variational_coeff_prior_std <= 0.0:
            raise ValueError(
                f"variational_coeff_prior_std must be > 0, got {self.variational_coeff_prior_std}"
            )
        if self.variational_coeff_min_std <= 0.0:
            raise ValueError(
                f"variational_coeff_min_std must be > 0, got {self.variational_coeff_min_std}"
            )
        if self.variational_coeff_min_std > self.variational_coeff_prior_std:
            raise ValueError(
                "variational_coeff_min_std cannot exceed variational_coeff_prior_std: "
                f"{self.variational_coeff_min_std} > {self.variational_coeff_prior_std}"
            )
        if self.variational_coeffs and self.quantize_sparse_coeffs:
            raise ValueError("variational_coeffs currently requires quantize_sparse_coeffs=False")

        # Dictionary shape [C, K] (matches LASER)
        self.dictionary = nn.Parameter(torch.randn(self.embedding_dim, self.num_embeddings) * 0.02)
        self._last_coeff_kl_loss = torch.zeros(())
        self._last_weighted_coeff_kl_loss = torch.zeros(())
        self._last_extra_bottleneck_loss = torch.zeros(())
        self._last_coeff_posterior_std = torch.zeros(())
        self._last_coeff_prior_std = torch.tensor(self.variational_coeff_prior_std)
        if self.variational_coeffs:
            hidden_dim = max(32, min(128, self.embedding_dim * 4))
            self.coeff_variational_atom_emb = nn.Embedding(self.num_embeddings, hidden_dim)
            self.coeff_variational_posterior = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.coeff_variational_atom_emb = None
            self.coeff_variational_posterior = None

        # Coefficient bin centers (uniform)
        centers = torch.linspace(-self.coef_max, self.coef_max, steps=self.n_bins)
        self.register_buffer("coef_bin_centers", centers)
        mu_invlog1p = 1.0
        if self.coef_quantization == "mu_law":
            mu_invlog1p = 1.0 / math.log1p(self.coef_mu)
        self.register_buffer(
            "coef_mu_invlog1p",
            torch.tensor(mu_invlog1p),
        )

        # Special tokens (for the transformer)
        if self.quantize_sparse_coeffs:
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = 2 * self.sparsity_level
            self.content_vocab_size = self.num_embeddings + self.n_bins
            self.pad_token_id = self.content_vocab_size
            self.bos_token_id = self.pad_token_id + 1
            self.vocab_size = self.content_vocab_size + 2
        else:
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = self.sparsity_level
            self.content_vocab_size = self.num_embeddings
            self.pad_token_id = self.num_embeddings
            self.bos_token_id = self.num_embeddings + 1
            self.vocab_size = self.num_embeddings + 2

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
        variational_prefixes = (
            prefix + "coeff_variational_atom_emb.",
            prefix + "coeff_variational_posterior.",
        )
        if self.variational_coeffs and self.coeff_variational_atom_emb is not None and self.coeff_variational_posterior is not None:
            expected_state = {
                prefix + "coeff_variational_atom_emb.weight": self.coeff_variational_atom_emb.weight.detach().clone(),
            }
            for subkey, value in self.coeff_variational_posterior.state_dict().items():
                expected_state[prefix + "coeff_variational_posterior." + subkey] = value.detach().clone()
            for key, expected in expected_state.items():
                loaded = state_dict.get(key, None)
                if loaded is None or tuple(loaded.shape) != tuple(expected.shape):
                    state_dict[key] = expected
        else:
            for key in list(state_dict.keys()):
                if key.startswith(variational_prefixes):
                    state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _normalize_dict(self) -> torch.Tensor:
        return F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

    def _quantize_coeff(self, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize coefficients to bins; return (bin_idx, bin_center_value)."""
        if self.coef_quantization == "uniform":
            c = coeff.clamp(-self.coef_max, self.coef_max)
            scaled = (c + self.coef_max) / (2 * self.coef_max)  # [0,1]
            bin_f = scaled * (self.n_bins - 1)
            bin_idx = torch.round(bin_f).to(torch.long).clamp(0, self.n_bins - 1)
            coeff_q = self.coef_bin_centers[bin_idx]
            return bin_idx, coeff_q

        # μ-law companding: finer resolution near zero for sparse code magnitudes.
        c = coeff.clamp(-self.coef_max, self.coef_max) / self.coef_max
        c_abs = c.abs()
        encoded = torch.sign(c) * torch.log1p(c_abs * self.coef_mu) * self.coef_mu_invlog1p
        scaled = (encoded + 1.0) * ((self.n_bins - 1) / 2.0)
        bin_idx = torch.round(scaled).to(torch.long).clamp(0, self.n_bins - 1)
        decoded = self._dequantize_coeff(bin_idx)
        return bin_idx, decoded

    def _dequantize_coeff(self, bin_idx: torch.Tensor) -> torch.Tensor:
        """Decode bin indices back to quantized coefficients."""
        if self.coef_quantization == "uniform":
            return self.coef_bin_centers[bin_idx]

        # Inverse μ-law companding.
        z = bin_idx.float() * (2.0 / (self.n_bins - 1)) - 1.0
        z_abs = z.abs()
        decoded_norm = torch.sign(z) * (torch.expm1(z_abs / self.coef_mu_invlog1p) / self.coef_mu)
        return decoded_norm * self.coef_max

    def _pack_quantized_tokens(self, support: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        """Interleave atom tokens and coefficient-bin tokens along the token depth axis."""
        if support.shape != bin_idx.shape:
            raise ValueError(f"support and bin_idx shape mismatch: {support.shape} vs {bin_idx.shape}")
        if support.size(-1) != self.sparsity_level:
            raise ValueError(f"Expected sparse depth {self.sparsity_level}, got {support.size(-1)}")

        tokens = torch.empty(
            *support.shape[:-1],
            self.token_depth,
            device=support.device,
            dtype=torch.long,
        )
        tokens[..., 0::2] = support.to(torch.long)
        tokens[..., 1::2] = bin_idx.to(torch.long) + self.coeff_token_offset
        return tokens

    def _unpack_quantized_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode alternating [atom, coeff_bin] tokens back to atom ids and coefficients."""
        if tokens.size(-1) != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {tokens.size(-1)}")

        atom_tokens = tokens[..., 0::2].to(torch.long)
        coeff_tokens = tokens[..., 1::2].to(torch.long)

        atom_invalid = (atom_tokens < 0) | (atom_tokens >= self.num_embeddings)
        coeff_bin = coeff_tokens - self.coeff_token_offset
        coeff_invalid = (coeff_bin < 0) | (coeff_bin >= self.n_bins)
        invalid = atom_invalid | coeff_invalid

        atom_ids = atom_tokens.clamp(0, self.num_embeddings - 1)
        coeff_bin = coeff_bin.clamp(0, self.n_bins - 1)
        coeffs = self._dequantize_coeff(coeff_bin)

        atom_ids = atom_ids.masked_fill(invalid, 0)
        coeffs = coeffs.masked_fill(invalid, 0.0)
        return atom_ids, coeffs

    def _encode_sparse_codes(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run OMP and return support atom ids and continuous coefficients."""
        B, C, H, W = z_e.shape
        n_signals = B * H * W
        dictionary = self._normalize_dict()
        signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t().to(dictionary.dtype)
        with torch.no_grad():
            support, coeffs = self.batch_omp_with_support(signals, dictionary)
        if support.ndim != 2 or coeffs.ndim != 2:
            raise RuntimeError(
                f"OMP returned invalid rank: support={tuple(support.shape)} coeffs={tuple(coeffs.shape)}"
            )
        if support.size(0) != n_signals or coeffs.size(0) != n_signals:
            raise RuntimeError(
                f"OMP returned invalid batch size: expected {n_signals}, "
                f"got support={support.size(0)} coeffs={coeffs.size(0)}"
            )
        # Defensive shape guard: keep a fixed D stack even if OMP exits short due to numerical edge cases.
        if support.size(1) != self.sparsity_level or coeffs.size(1) != self.sparsity_level:
            cur_d = min(support.size(1), coeffs.size(1))
            if cur_d > 0:
                support = support[:, :cur_d]
                coeffs = coeffs[:, :cur_d]
            else:
                support = torch.zeros((n_signals, 0), device=support.device, dtype=support.dtype)
                coeffs = torch.zeros((n_signals, 0), device=coeffs.device, dtype=coeffs.dtype)
            if cur_d < self.sparsity_level:
                pad = self.sparsity_level - cur_d
                support_pad = torch.zeros((n_signals, pad), device=support.device, dtype=support.dtype)
                coeffs_pad = torch.zeros((n_signals, pad), device=coeffs.device, dtype=coeffs.dtype)
                support = torch.cat([support, support_pad], dim=1)
                coeffs = torch.cat([coeffs, coeffs_pad], dim=1)
        if self.canonicalize_sparse_slots:
            # Canonicalize sparse slots so stage-2 does not need to model arbitrary OMP selection order.
            order = coeffs.abs().argsort(dim=1, descending=True)
            support = support.gather(1, order)
            coeffs = coeffs.gather(1, order)
        return (
            support.view(B, H, W, self.sparsity_level),
            coeffs.view(B, H, W, self.sparsity_level),
        )

    def clamp_sparse_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Project coefficients onto the stage-1 decoder manifold."""
        coeffs = _nan_to_num_tensor(coeffs)
        if self.quantize_sparse_coeffs:
            return coeffs.clamp(-self.coef_max, self.coef_max)
        return coeffs

    def _coeff_posterior_stats(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.variational_coeffs:
            raise RuntimeError("_coeff_posterior_stats requires variational_coeffs=True")
        if self.coeff_variational_atom_emb is None or self.coeff_variational_posterior is None:
            raise RuntimeError("variational coefficient modules were not initialized")
        if support.shape != coeffs.shape:
            raise ValueError(f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}")

        support_clamped = support.to(torch.long).clamp(0, self.num_embeddings - 1)
        coeffs_base = self.clamp_sparse_coeffs(coeffs.to(torch.float32))
        atom_emb = self.coeff_variational_atom_emb(support_clamped)
        posterior_in = torch.cat([atom_emb, coeffs_base.unsqueeze(-1)], dim=-1)
        posterior_raw = self.coeff_variational_posterior(posterior_in)

        mean_offset = self.variational_coeff_prior_std * torch.tanh(posterior_raw[..., 0])
        posterior_mu = self.clamp_sparse_coeffs(coeffs_base + mean_offset)

        std_range = max(self.variational_coeff_prior_std - self.variational_coeff_min_std, 0.0)
        posterior_std = self.variational_coeff_min_std + std_range * torch.sigmoid(posterior_raw[..., 1])
        posterior_logvar = 2.0 * torch.log(posterior_std.clamp_min(1e-6))
        return posterior_mu, posterior_logvar

    def project_sparse_coeffs(self, support: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs_clamped = self.clamp_sparse_coeffs(coeffs)
        if not self.variational_coeffs:
            return coeffs_clamped
        coeff_mu, _ = self._coeff_posterior_stats(support, coeffs_clamped)
        return coeff_mu

    def _reconstruct_sparse(
        self, support: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct latent map from atom ids + coefficients."""
        if support.shape != coeffs.shape:
            raise ValueError(
                f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}"
            )

        B, H, W, D = support.shape
        if D != self.sparsity_level:
            raise ValueError(f"Expected D={self.sparsity_level}, got {D}")

        dictionary = self._normalize_dict().t()  # [num_embeddings, C]
        support = support.to(torch.long).clamp(0, self.num_embeddings - 1)
        coeffs = self.clamp_sparse_coeffs(coeffs.to(dictionary.dtype))
        support_flat = support.reshape(-1, D)
        coeffs_flat = coeffs.reshape(-1, D)
        atoms = dictionary[support_flat]  # [B*H*W, D, C]
        recon_flat = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)  # [N, C]
        return recon_flat.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

    def batch_omp_with_support(self, X: torch.Tensor, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched OMP adapted from LASER's DictionaryLearning.batch_omp.
        Runs exactly sparsity_level steps (no early-stop) so stack depth is fixed.

        Args:
            X: [M, B] signals
            D: [M, N] normalized dictionary
        Returns:
            support: [B, K] indices in selection order (K = sparsity_level)
            coeffs:  [B, K] coefficients aligned with support (same order)
        """
        return _batched_omp_with_support(
            X=X,
            D=D,
            sparsity_level=self.sparsity_level,
        )

    def batch_omp_with_trajectory(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched OMP with exact per-iteration refit history.

        Returns:
            support: [B, K] final support indices in greedy selection order.
            coeffs: [B, K] final coefficients aligned with support.
            support_history: [B, K, K] padded support prefixes after each OMP step.
            coeff_history: [B, K, K] padded refit coefficients after each OMP step.
        """
        return _batched_omp_with_support(
            X=X,
            D=D,
            sparsity_level=self.sparsity_level,
            return_history=True,
        )

    def _encode_sparse_codes_with_trajectory(
        self,
        z_e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run OMP and return the exact greedy refit trajectory.

        The returned history tensors are padded to the full sparsity level so the
        k-th OMP state can be decoded directly with `_reconstruct_sparse`.
        """
        if self.canonicalize_sparse_slots:
            raise RuntimeError(
                "Exact OMP trajectories require canonicalize_sparse_slots=False because canonicalization destroys "
                "the greedy step order."
            )
        B, C, H, W = z_e.shape
        n_signals = B * H * W
        signals = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C).t()
        dictionary = self._normalize_dict()
        with torch.no_grad():
            support, coeffs, support_hist, coeff_hist = self.batch_omp_with_trajectory(signals, dictionary)
        if support.ndim != 2 or coeffs.ndim != 2 or support_hist.ndim != 3 or coeff_hist.ndim != 3:
            raise RuntimeError(
                "OMP trajectory returned invalid ranks: "
                f"support={tuple(support.shape)} coeffs={tuple(coeffs.shape)} "
                f"support_hist={tuple(support_hist.shape)} coeff_hist={tuple(coeff_hist.shape)}"
            )
        if support.size(0) != n_signals or coeffs.size(0) != n_signals:
            raise RuntimeError(
                f"OMP trajectory returned invalid batch size: expected {n_signals}, "
                f"got support={support.size(0)} coeffs={coeffs.size(0)}"
            )
        expected_shape = (n_signals, self.sparsity_level, self.sparsity_level)
        if tuple(support_hist.shape) != expected_shape or tuple(coeff_hist.shape) != expected_shape:
            raise RuntimeError(
                f"OMP trajectory returned invalid history shapes: expected {expected_shape}, "
                f"got support_hist={tuple(support_hist.shape)} coeff_hist={tuple(coeff_hist.shape)}"
            )
        return (
            support.view(B, H, W, self.sparsity_level),
            coeffs.view(B, H, W, self.sparsity_level),
            support_hist.view(B, H, W, self.sparsity_level, self.sparsity_level),
            coeff_hist.view(B, H, W, self.sparsity_level, self.sparsity_level),
        )

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, C, H, W]
        Returns:
            z_q_ste: [B, C, H, W]
            loss: scalar bottleneck loss
            tokens: [B, H, W, token_depth]
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(z_e.shape)}")
        B, C, H, W = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(f"Expected channel dim {self.embedding_dim}, got {C}")

        z_e = _nan_to_num_tensor(z_e)
        support, coeffs = self._encode_sparse_codes(z_e)
        support_flat = support.view(-1, self.sparsity_level)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.view(-1, self.sparsity_level))

        if self.quantize_sparse_coeffs:
            # Quantize coefficients and interleave atom + coefficient-bin tokens.
            bin_idx, coeff_q = self._quantize_coeff(coeffs_flat)  # both [Nsig, D]
            tokens = self._pack_quantized_tokens(
                support_flat.view(B, H, W, self.sparsity_level),
                bin_idx.view(B, H, W, self.sparsity_level),
            )
            coeffs_for_recon = coeff_q
        else:
            tokens = support.view(B, H, W, self.sparsity_level).long()
            coeffs_for_recon = self.clamp_sparse_coeffs(coeffs_flat)

        coeff_kl_loss = z_e.new_zeros(())
        weighted_coeff_kl_loss = z_e.new_zeros(())
        if (not self.quantize_sparse_coeffs) and self.variational_coeffs:
            coeffs_base = coeffs_for_recon.reshape(B, H, W, self.sparsity_level)
            coeff_mu, coeff_logvar = self._coeff_posterior_stats(support, coeffs_base)
            if self.training:
                coeff_eps = torch.randn_like(coeff_mu)
                coeff_std = (0.5 * coeff_logvar).exp()
                coeffs_for_recon = self.clamp_sparse_coeffs(coeff_mu + coeff_std * coeff_eps)
            else:
                coeffs_for_recon = coeff_mu
            coeff_kl_loss = _gaussian_kl_to_fixed_mean(
                coeff_mu,
                coeff_logvar,
                coeffs_base,
                target_std=self.variational_coeff_prior_std,
            )
            weighted_coeff_kl_loss = float(self.variational_coeff_kl_weight) * coeff_kl_loss
            self._last_coeff_posterior_std = (0.5 * coeff_logvar).exp().mean().detach()
            self._last_coeff_prior_std = torch.as_tensor(
                self.variational_coeff_prior_std,
                device=z_e.device,
                dtype=z_e.dtype,
            )
        else:
            coeffs_for_recon = coeffs_for_recon.reshape(B, H, W, self.sparsity_level)
            self._last_coeff_posterior_std = z_e.new_zeros(())
            self._last_coeff_prior_std = torch.as_tensor(
                self.variational_coeff_prior_std,
                device=z_e.device,
                dtype=z_e.dtype,
            )

        z_q = _nan_to_num_tensor(self._reconstruct_sparse(support, coeffs_for_recon))

        # Bottleneck loss (LASER-style)
        dl_latent_loss = F.mse_loss(z_q, z_e.detach())
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss + weighted_coeff_kl_loss
        self._last_dl_latent_loss = dl_latent_loss
        self._last_e_latent_loss = e_latent_loss
        self._last_coeff_kl_loss = coeff_kl_loss
        self._last_weighted_coeff_kl_loss = weighted_coeff_kl_loss
        self._last_extra_bottleneck_loss = weighted_coeff_kl_loss
        self._last_bottleneck_loss = loss

        # Straight-through estimator to encoder
        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, loss, tokens

    @torch.no_grad()
    def tokens_to_latent(self, tokens: torch.Tensor, coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode tokens back to a latent map.
        Args:
            tokens: [B, H, W, token_depth]
            coeffs: [B, H, W, D] (only used in non-quantized mode)
        Returns:
            z_q: [B, C, H, W]
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,H,W,D], got {tuple(tokens.shape)}")
        B, H, W, D = tokens.shape
        if D != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {D}")

        if self.quantize_sparse_coeffs:
            atom_ids, coeff_q = self._unpack_quantized_tokens(tokens)
            return self._reconstruct_sparse(atom_ids, coeff_q)

        if coeffs is None:
            raise ValueError("coeffs must be provided when quantize_sparse_coeffs=False")

        coeffs_clamped = self.clamp_sparse_coeffs(coeffs.to(self._normalize_dict().dtype))
        return self._reconstruct_sparse(tokens.to(torch.long), coeffs_clamped)


SparseBottleneck = DictionaryLearningTokenized


# -----------------------------
# Patch-based Dictionary Learning bottleneck
# -----------------------------

class PatchDictionaryLearningTokenized(nn.Module):
    """
    Patch-based dictionary learning bottleneck.

    Extracts overlapping patches from the latent feature map using F.unfold,
    runs batched OMP on each patch vector, then reconstructs the latent via
    one of three stitching strategies:

      "center_crop" (default) — take only the center patch_stride×patch_stride
          region of each reconstructed patch and tile non-overlappingly.
          No averaging; each output pixel comes from exactly one patch center.

      "hann" — weighted overlap-add using a 2D Hann window. The window
          up-weights the patch center and fades to zero at edges. With 50%%
          overlap (patch_stride = patch_size // 2) this satisfies the COLA
          condition so the weight map is constant and there is no blurring
          on the exact signal; for OMP-approximated patches it blends
          smoothly rather than averaging equally.

      "tile" — direct patch tiling with no crop or overlap weighting. This is
          the natural reconstruction for non-overlapping patches
          (patch_stride == patch_size), and overlapping requests are
          automatically normalized to this mode when patches do not overlap.

    All modes pad by (patch_size - patch_stride) // 2 before unfolding so
    that the output covers the full H × W spatial extent.

    The dictionary has shape [patch_dim, num_embeddings] where
        patch_dim = patch_size * patch_size * embedding_dim.

    Token output shape: [B, nph, npw, token_depth]  (nph = H // patch_stride).
    """

    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 16,
        patch_size: int = 8,
        patch_stride: int = 4,
        sparsity_level: int = 8,
        n_bins: int = 16,
        coef_max: float = 3.0,
        quantize_sparse_coeffs: bool = False,
        coef_quantization: str = "uniform",
        coef_mu: float = 0.0,
        commitment_cost: float = 0.25,
        epsilon: float = 1e-10,
        patch_reconstruction: str = "center_crop",
        canonicalize_sparse_slots: bool = True,
        variational_coeffs: bool = False,
        variational_coeff_kl_weight: float = 0.0,
        variational_coeff_prior_std: float = 0.25,
        variational_coeff_min_std: float = 0.01,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.patch_dim = self.patch_size * self.patch_size * self.embedding_dim
        self.sparsity_level = int(sparsity_level)
        self.n_bins = int(n_bins)
        self.coef_max = float(coef_max)
        self.quantize_sparse_coeffs = bool(quantize_sparse_coeffs)
        self.coef_quantization = str(coef_quantization)
        self.coef_mu = float(coef_mu)
        if self.coef_quantization not in ("uniform", "mu_law"):
            raise ValueError("coef_quantization must be one of {'uniform', 'mu_law'}")
        if self.coef_quantization == "mu_law" and self.coef_mu <= 0.0:
            raise ValueError(f"coef_mu must be > 0, got {self.coef_mu}")
        self.commitment_cost = float(commitment_cost)
        self.epsilon = float(epsilon)
        self.canonicalize_sparse_slots = bool(canonicalize_sparse_slots)
        self.variational_coeffs = bool(variational_coeffs)
        self.variational_coeff_kl_weight = float(variational_coeff_kl_weight)
        self.variational_coeff_prior_std = float(variational_coeff_prior_std)
        self.variational_coeff_min_std = float(variational_coeff_min_std)
        if self.variational_coeff_kl_weight < 0.0:
            raise ValueError(
                f"variational_coeff_kl_weight must be >= 0, got {self.variational_coeff_kl_weight}"
            )
        if self.variational_coeff_prior_std <= 0.0:
            raise ValueError(
                f"variational_coeff_prior_std must be > 0, got {self.variational_coeff_prior_std}"
            )
        if self.variational_coeff_min_std <= 0.0:
            raise ValueError(
                f"variational_coeff_min_std must be > 0, got {self.variational_coeff_min_std}"
            )
        if self.variational_coeff_min_std > self.variational_coeff_prior_std:
            raise ValueError(
                "variational_coeff_min_std cannot exceed variational_coeff_prior_std: "
                f"{self.variational_coeff_min_std} > {self.variational_coeff_prior_std}"
            )
        self.patch_reconstruction = _canonical_patch_reconstruction(
            patch_reconstruction,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
        if self.variational_coeffs and self.quantize_sparse_coeffs:
            raise ValueError("variational_coeffs currently requires quantize_sparse_coeffs=False")

        # Dictionary shape: [patch_dim, num_embeddings]
        self.dictionary = nn.Parameter(
            torch.randn(self.patch_dim, self.num_embeddings) * 0.02
        )
        self._last_coeff_kl_loss = torch.zeros(())
        self._last_weighted_coeff_kl_loss = torch.zeros(())
        self._last_extra_bottleneck_loss = torch.zeros(())
        self._last_coeff_posterior_std = torch.zeros(())
        self._last_coeff_prior_std = torch.tensor(self.variational_coeff_prior_std)
        if self.variational_coeffs:
            hidden_dim = max(64, min(256, self.embedding_dim * self.patch_size))
            self.coeff_variational_atom_emb = nn.Embedding(self.num_embeddings, hidden_dim)
            self.coeff_variational_posterior = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.coeff_variational_atom_emb = None
            self.coeff_variational_posterior = None

        centers = torch.linspace(-self.coef_max, self.coef_max, steps=self.n_bins)
        self.register_buffer("coef_bin_centers", centers)
        mu_invlog1p = 1.0
        if self.coef_quantization == "mu_law":
            mu_invlog1p = 1.0 / math.log1p(self.coef_mu)
        self.register_buffer("coef_mu_invlog1p", torch.tensor(mu_invlog1p))

        # Pre-compute the 2D Hann window (channel-tiled) as a buffer.
        # Shape: [patch_dim] = [C * patch_size * patch_size]
        hann_1d = torch.hann_window(self.patch_size, periodic=False)
        window_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)   # [ps, ps]
        window_flat = window_2d.flatten().unsqueeze(0).expand(
            self.embedding_dim, -1
        ).reshape(-1)                                               # [patch_dim]
        self.register_buffer("_hann_win", window_flat.clone())

        if self.quantize_sparse_coeffs:
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = 2 * self.sparsity_level
            self.content_vocab_size = self.num_embeddings + self.n_bins
            self.pad_token_id = self.content_vocab_size
            self.bos_token_id = self.pad_token_id + 1
            self.vocab_size = self.content_vocab_size + 2
        else:
            self.coeff_token_offset = self.num_embeddings
            self.token_depth = self.sparsity_level
            self.content_vocab_size = self.num_embeddings
            self.pad_token_id = self.num_embeddings
            self.bos_token_id = self.num_embeddings + 1
            self.vocab_size = self.num_embeddings + 2

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
        variational_prefixes = (
            prefix + "coeff_variational_atom_emb.",
            prefix + "coeff_variational_posterior.",
        )
        if self.variational_coeffs and self.coeff_variational_atom_emb is not None and self.coeff_variational_posterior is not None:
            expected_state = {
                prefix + "coeff_variational_atom_emb.weight": self.coeff_variational_atom_emb.weight.detach().clone(),
            }
            for subkey, value in self.coeff_variational_posterior.state_dict().items():
                expected_state[prefix + "coeff_variational_posterior." + subkey] = value.detach().clone()
            for key, expected in expected_state.items():
                loaded = state_dict.get(key, None)
                if loaded is None or tuple(loaded.shape) != tuple(expected.shape):
                    state_dict[key] = expected
        else:
            for key in list(state_dict.keys()):
                if key.startswith(variational_prefixes):
                    state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _normalize_dict(self) -> torch.Tensor:
        """Return column-normalised dictionary [patch_dim, num_embeddings]."""
        return F.normalize(self.dictionary, p=2, dim=0, eps=self.epsilon)

    # ---- coefficient quantisation (identical to DictionaryLearningTokenized) ----

    def _quantize_coeff(self, coeff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.coef_quantization == "uniform":
            c = coeff.clamp(-self.coef_max, self.coef_max)
            scaled = (c + self.coef_max) / (2 * self.coef_max)
            bin_f = scaled * (self.n_bins - 1)
            bin_idx = torch.round(bin_f).to(torch.long).clamp(0, self.n_bins - 1)
            coeff_q = self.coef_bin_centers[bin_idx]
            return bin_idx, coeff_q
        c = coeff.clamp(-self.coef_max, self.coef_max) / self.coef_max
        c_abs = c.abs()
        encoded = torch.sign(c) * torch.log1p(c_abs * self.coef_mu) * self.coef_mu_invlog1p
        scaled = (encoded + 1.0) * ((self.n_bins - 1) / 2.0)
        bin_idx = torch.round(scaled).to(torch.long).clamp(0, self.n_bins - 1)
        decoded = self._dequantize_coeff(bin_idx)
        return bin_idx, decoded

    def _dequantize_coeff(self, bin_idx: torch.Tensor) -> torch.Tensor:
        if self.coef_quantization == "uniform":
            return self.coef_bin_centers[bin_idx]
        z = bin_idx.float() * (2.0 / (self.n_bins - 1)) - 1.0
        z_abs = z.abs()
        decoded_norm = torch.sign(z) * (torch.expm1(z_abs / self.coef_mu_invlog1p) / self.coef_mu)
        return decoded_norm * self.coef_max

    def _pack_quantized_tokens(self, support: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        tokens = torch.empty(
            *support.shape[:-1],
            self.token_depth,
            device=support.device,
            dtype=torch.long,
        )
        tokens[..., 0::2] = support.to(torch.long)
        tokens[..., 1::2] = bin_idx.to(torch.long) + self.coeff_token_offset
        return tokens

    def _unpack_quantized_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_tokens = tokens[..., 0::2].to(torch.long)
        coeff_tokens = tokens[..., 1::2].to(torch.long)
        atom_invalid = (atom_tokens < 0) | (atom_tokens >= self.num_embeddings)
        coeff_bin = coeff_tokens - self.coeff_token_offset
        coeff_invalid = (coeff_bin < 0) | (coeff_bin >= self.n_bins)
        invalid = atom_invalid | coeff_invalid
        atom_ids = atom_tokens.clamp(0, self.num_embeddings - 1)
        coeff_bin = coeff_bin.clamp(0, self.n_bins - 1)
        coeffs = self._dequantize_coeff(coeff_bin)
        atom_ids = atom_ids.masked_fill(invalid, 0)
        coeffs = coeffs.masked_fill(invalid, 0.0)
        return atom_ids, coeffs

    # ---- patch extraction / reconstruction ----

    def _extract_patches(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Pad then unfold z_e into overlapping patches.

        Two padding passes are applied:
          1. Symmetric padding of cs = (patch_size - patch_stride) // 2 so that
             the center of each reconstructed patch aligns with a non-overlapping
             patch_stride × patch_stride tile of the original latent.
          2. Asymmetric right/bottom padding to cover any remainder when H (or W)
             is not divisible by patch_stride — making this work for any stride.

        Returns:
            patches      : [B, patch_dim, L]  (L = nph * npw)
            nph, npw     : patch grid height / width
            H_orig, W_orig : original spatial dims before any padding
        """
        _, _, H, W = z_e.shape
        cs = (self.patch_size - self.patch_stride) // 2

        # Minimum patch count to cover the original extent after centering.
        nph = math.ceil(H / self.patch_stride)
        npw = math.ceil(W / self.patch_stride)

        # Total padded size required by unfold for nph / npw patches.
        H_pad_need = (nph - 1) * self.patch_stride + self.patch_size
        W_pad_need = (npw - 1) * self.patch_stride + self.patch_size

        pad_top = cs
        pad_left = cs
        pad_bottom = H_pad_need - H - cs   # may be > cs when H % stride != 0
        pad_right  = W_pad_need - W - cs

        z_e = F.pad(z_e, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        patches = F.unfold(z_e, kernel_size=self.patch_size, stride=self.patch_stride)
        return patches, nph, npw, H, W

    # ---- OMP (same algorithm as DictionaryLearningTokenized) ----

    def batch_omp_with_support(
        self, X: torch.Tensor, D: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched OMP.
        Args:
            X: [M, B] signals  (M = patch_dim)
            D: [M, N] normalised dictionary
        Returns:
            support: [B, K]
            coeffs : [B, K]
        """
        return _batched_omp_with_support(
            X=X,
            D=D,
            sparsity_level=self.sparsity_level,
        )

    def _encode_sparse_codes(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run OMP on every patch and return atom ids + coefficients.

        Returns:
            support : [B, nph, npw, K]
            coeffs  : [B, nph, npw, K]
        """
        patches, nph, npw, H, W = self._extract_patches(z_e)
        B = z_e.shape[0]
        L = patches.shape[2]  # nph * npw
        dictionary = self._normalize_dict()
        signals = patches.permute(0, 2, 1).contiguous().view(-1, self.patch_dim).t().to(dictionary.dtype)
        n_signals = B * L
        with torch.no_grad():
            support, coeffs = self.batch_omp_with_support(signals, dictionary)
        cur_d = min(support.size(1), coeffs.size(1))
        if cur_d < self.sparsity_level:
            pad = self.sparsity_level - cur_d
            support = torch.cat([support, torch.zeros(n_signals, pad, device=support.device, dtype=support.dtype)], dim=1)
            coeffs = torch.cat([coeffs, torch.zeros(n_signals, pad, device=coeffs.device, dtype=coeffs.dtype)], dim=1)
        else:
            support = support[:, :self.sparsity_level]
            coeffs = coeffs[:, :self.sparsity_level]
        if self.canonicalize_sparse_slots:
            # Canonicalize sparse slots so stage-2 sees a stable per-patch ordering.
            order = coeffs.abs().argsort(dim=1, descending=True)
            support = support.gather(1, order)
            coeffs = coeffs.gather(1, order)
        return (
            support.view(B, nph, npw, self.sparsity_level),
            coeffs.view(B, nph, npw, self.sparsity_level),
            H, W,
        )

    def clamp_sparse_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Project coefficients onto the stage-1 decoder manifold."""
        coeffs = _nan_to_num_tensor(coeffs)
        if self.quantize_sparse_coeffs:
            return coeffs.clamp(-self.coef_max, self.coef_max)
        return coeffs

    def _coeff_posterior_stats(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.variational_coeffs:
            raise RuntimeError("_coeff_posterior_stats requires variational_coeffs=True")
        if self.coeff_variational_atom_emb is None or self.coeff_variational_posterior is None:
            raise RuntimeError("variational coefficient modules were not initialized")
        if support.shape != coeffs.shape:
            raise ValueError(f"support and coeffs shape mismatch: {support.shape} vs {coeffs.shape}")

        support_clamped = support.to(torch.long).clamp(0, self.num_embeddings - 1)
        coeffs_base = self.clamp_sparse_coeffs(coeffs.to(torch.float32))
        atom_emb = self.coeff_variational_atom_emb(support_clamped)
        posterior_in = torch.cat([atom_emb, coeffs_base.unsqueeze(-1)], dim=-1)
        posterior_raw = self.coeff_variational_posterior(posterior_in)

        mean_offset = self.variational_coeff_prior_std * torch.tanh(posterior_raw[..., 0])
        posterior_mu = self.clamp_sparse_coeffs(coeffs_base + mean_offset)

        std_range = max(self.variational_coeff_prior_std - self.variational_coeff_min_std, 0.0)
        posterior_std = self.variational_coeff_min_std + std_range * torch.sigmoid(posterior_raw[..., 1])
        posterior_logvar = 2.0 * torch.log(posterior_std.clamp_min(1e-6))
        return posterior_mu, posterior_logvar

    def project_sparse_coeffs(self, support: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        coeffs_clamped = self.clamp_sparse_coeffs(coeffs)
        if not self.variational_coeffs:
            return coeffs_clamped
        coeff_mu, _ = self._coeff_posterior_stats(support, coeffs_clamped)
        return coeff_mu

    def _reconstruct_sparse(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """Dispatch to the requested patch stitching strategy."""
        if self.patch_reconstruction == "tile":
            return self._reconstruct_tile(support, coeffs, H, W)
        if self.patch_reconstruction == "hann":
            return self._reconstruct_hann(support, coeffs, H, W)
        return self._reconstruct_center_crop(support, coeffs, H, W)

    def _reconstruct_tile(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Direct patch tiling for non-overlapping patches.

        H, W: original latent spatial dims. When provided, output is cropped
        to [B, C, H, W].
        """
        if self.patch_stride != self.patch_size:
            raise ValueError(
                "tile reconstruction requires non-overlapping patches: "
                f"patch_stride={self.patch_stride}, patch_size={self.patch_size}"
            )
        B, nph, npw, D = support.shape
        C = self.embedding_dim

        dictionary = self._normalize_dict().t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, D)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.to(dictionary.dtype)).reshape(-1, D)
        atoms = dictionary[support_flat]
        recon = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)

        recon = recon.view(B, nph, npw, C, self.patch_size, self.patch_size)
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(B, C, nph * self.patch_size, npw * self.patch_size)
        if H is not None and W is not None:
            recon = recon[:, :, :H, :W]
        return recon

    def _reconstruct_center_crop(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Center-crop stitching: each patch contributes only its center
        patch_stride × patch_stride region, forming a non-overlapping tiling.
        No averaging; every output pixel comes from exactly one patch.

        H, W: original latent spatial dims. When provided, output is cropped
        to [B, C, H, W], supporting any patch_stride regardless of divisibility.
        """
        B, nph, npw, D = support.shape
        s = self.patch_stride
        cs = (self.patch_size - self.patch_stride) // 2
        C = self.embedding_dim

        dictionary = self._normalize_dict().t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, D)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.to(dictionary.dtype)).reshape(-1, D)
        atoms = dictionary[support_flat]                          # [N, D, patch_dim]
        recon = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)   # [N, patch_dim]

        recon = recon.view(B * nph * npw, C, self.patch_size, self.patch_size)
        recon = recon[:, :, cs:cs + s, cs:cs + s]                # [N, C, s, s]

        recon = recon.view(B, nph, npw, C, s, s)
        recon = recon.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(B, C, nph * s, npw * s)
        if H is not None and W is not None:
            recon = recon[:, :, :H, :W]
        return recon

    def _reconstruct_hann(
        self,
        support: torch.Tensor,
        coeffs: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Weighted overlap-add with a 2D Hann window.

        H, W: original latent spatial dims. When provided, output is cropped
        to [B, C, H, W] after stripping the centering pad.
        """
        B, nph, npw, D = support.shape
        s = self.patch_stride
        cs = (self.patch_size - self.patch_stride) // 2
        C = self.embedding_dim
        # Padded fold dimensions matching what _extract_patches produced.
        H_pad = (nph - 1) * s + self.patch_size
        W_pad = (npw - 1) * s + self.patch_size

        dictionary = self._normalize_dict().t()
        support_flat = support.to(torch.long).clamp(0, self.num_embeddings - 1).reshape(-1, D)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.to(dictionary.dtype)).reshape(-1, D)
        atoms = dictionary[support_flat]                          # [N, D, patch_dim]
        recon = (atoms * coeffs_flat.unsqueeze(-1)).sum(dim=1)   # [N, patch_dim]

        win = self._hann_win.to(recon.dtype)
        recon = recon * win.unsqueeze(0)
        recon = recon.view(B, nph * npw, self.patch_dim).permute(0, 2, 1)

        weighted = F.fold(recon, output_size=(H_pad, W_pad),
                          kernel_size=self.patch_size, stride=s)
        win_map = F.fold(
            win.view(1, -1, 1).expand(B, -1, nph * npw),
            output_size=(H_pad, W_pad),
            kernel_size=self.patch_size, stride=s,
        )
        out = weighted / win_map.clamp_min(1e-8)

        # strip centering pad then crop to original H × W
        out = out[:, :, cs:cs + nph * s, cs:cs + npw * s]
        if H is not None and W is not None:
            out = out[:, :, :H, :W]
        return out

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e : [B, C, H, W]
        Returns:
            z_q_ste : [B, C, H, W]
            loss    : scalar bottleneck loss
            tokens  : [B, nph, npw, token_depth]
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(z_e.shape)}")
        B, C, _, _ = z_e.shape
        if C != self.embedding_dim:
            raise ValueError(f"Expected channel dim {self.embedding_dim}, got {C}")

        z_e = _nan_to_num_tensor(z_e)
        support, coeffs, H, W = self._encode_sparse_codes(z_e)
        _, nph, npw, _ = support.shape

        support_flat = support.view(-1, self.sparsity_level)
        coeffs_flat = self.clamp_sparse_coeffs(coeffs.view(-1, self.sparsity_level))

        if self.quantize_sparse_coeffs:
            bin_idx, coeff_q = self._quantize_coeff(coeffs_flat)
            tokens = self._pack_quantized_tokens(
                support_flat.view(B, nph, npw, self.sparsity_level),
                bin_idx.view(B, nph, npw, self.sparsity_level),
            )
            coeffs_for_recon = coeff_q.reshape(B, nph, npw, self.sparsity_level)
        else:
            tokens = support.view(B, nph, npw, self.sparsity_level).long()
            coeffs_for_recon = self.clamp_sparse_coeffs(coeffs_flat).reshape(B, nph, npw, self.sparsity_level)

        coeff_kl_loss = z_e.new_zeros(())
        weighted_coeff_kl_loss = z_e.new_zeros(())
        if (not self.quantize_sparse_coeffs) and self.variational_coeffs:
            coeffs_base = coeffs_for_recon
            coeff_mu, coeff_logvar = self._coeff_posterior_stats(support, coeffs_base)
            if self.training:
                coeff_eps = torch.randn_like(coeff_mu)
                coeff_std = (0.5 * coeff_logvar).exp()
                coeffs_for_recon = self.clamp_sparse_coeffs(coeff_mu + coeff_std * coeff_eps)
            else:
                coeffs_for_recon = coeff_mu
            coeff_kl_loss = _gaussian_kl_to_fixed_mean(
                coeff_mu,
                coeff_logvar,
                coeffs_base,
                target_std=self.variational_coeff_prior_std,
            )
            weighted_coeff_kl_loss = float(self.variational_coeff_kl_weight) * coeff_kl_loss
            self._last_coeff_posterior_std = (0.5 * coeff_logvar).exp().mean().detach()
            self._last_coeff_prior_std = torch.as_tensor(
                self.variational_coeff_prior_std,
                device=z_e.device,
                dtype=z_e.dtype,
            )
        else:
            self._last_coeff_posterior_std = z_e.new_zeros(())
            self._last_coeff_prior_std = torch.as_tensor(
                self.variational_coeff_prior_std,
                device=z_e.device,
                dtype=z_e.dtype,
            )

        z_q = _nan_to_num_tensor(self._reconstruct_sparse(support, coeffs_for_recon, H, W))

        dl_latent_loss = F.mse_loss(z_q, z_e.detach())
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        loss = dl_latent_loss + self.commitment_cost * e_latent_loss + weighted_coeff_kl_loss
        self._last_dl_latent_loss = dl_latent_loss
        self._last_e_latent_loss = e_latent_loss
        self._last_coeff_kl_loss = coeff_kl_loss
        self._last_weighted_coeff_kl_loss = weighted_coeff_kl_loss
        self._last_extra_bottleneck_loss = weighted_coeff_kl_loss
        self._last_bottleneck_loss = loss

        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, loss, tokens

    @torch.no_grad()
    def tokens_to_latent(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Decode tokens back to a latent map.

        Args:
            tokens : [B, nph, npw, token_depth]
            coeffs : [B, nph, npw, K]  (only used in non-quantized mode)
            latent_hw : optional original latent spatial size (H, W)
        Returns:
            z_q : [B, C, H, W]
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B,nph,npw,D], got {tuple(tokens.shape)}")
        _, _, _, D = tokens.shape
        if D != self.token_depth:
            raise ValueError(f"Expected token depth {self.token_depth}, got {D}")

        if self.quantize_sparse_coeffs:
            atom_ids, coeff_q = self._unpack_quantized_tokens(tokens)
            if latent_hw is None:
                return self._reconstruct_sparse(atom_ids, coeff_q)
            return self._reconstruct_sparse(atom_ids, coeff_q, int(latent_hw[0]), int(latent_hw[1]))

        if coeffs is None:
            raise ValueError("coeffs must be provided when quantize_sparse_coeffs=False")
        coeffs_clamped = self.clamp_sparse_coeffs(coeffs.to(self._normalize_dict().dtype))
        if latent_hw is None:
            return self._reconstruct_sparse(
                tokens.to(torch.long),
                coeffs_clamped,
            )
        return self._reconstruct_sparse(
            tokens.to(torch.long),
            coeffs_clamped,
            int(latent_hw[0]),
            int(latent_hw[1]),
        )


# -----------------------------
# Stage-1 model: Encoder + Dictionary bottleneck + Decoder
# -----------------------------

class LASER(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_hiddens: int = 128,
        num_downsamples: int = 2,
        num_residual_layers: int = 2,
        resolution: int = 128,
        attn_resolutions: tuple = (),
        dropout: float = 0.0,
        max_ch_mult: int = 2,
        decoder_extra_residual_layers: int = 1,
        use_mid_attention: bool = True,
        embedding_dim: int = 16,
        num_embeddings: int = 1024,
        sparsity_level: int = 8,
        commitment_cost: float = 0.25,
        n_bins: int = 16,
        coef_max: float = 3.0,
        coef_quantization: str = "uniform",
        coef_mu: float = 50.0,
        out_tanh: bool = True,
        quantize_sparse_coeffs: bool = False,
        patch_based: bool = False,
        patch_size: int = 8,
        patch_stride: int = 4,
        patch_reconstruction: str = "center_crop",
        variational_coeffs: bool = False,
        variational_coeff_kl_weight: float = 0.0,
        variational_coeff_prior_std: float = 0.25,
        variational_coeff_min_std: float = 0.01,
    ):
        super().__init__()
        self.out_tanh = bool(out_tanh)
        self.max_ch_mult = int(max_ch_mult)
        self.decoder_extra_residual_layers = int(decoder_extra_residual_layers)
        self.use_mid_attention = bool(use_mid_attention)

        if self.max_ch_mult <= 0:
            raise ValueError(f"max_ch_mult must be positive, got {self.max_ch_mult}")
        if self.decoder_extra_residual_layers < 0:
            raise ValueError(
                f"decoder_extra_residual_layers must be non-negative, got {self.decoder_extra_residual_layers}"
            )

        # ch_mult controls the channel multiplier at each resolution level;
        # len(ch_mult) - 1 equals the number of spatial downsampling steps.
        # Cap multipliers to keep the max width bounded without changing the
        # number of encoder or decoder stages.
        ch_mult = tuple(min(2 ** i, self.max_ch_mult) for i in range(num_downsamples + 1))

        enc_dec_kwargs = dict(
            ch=num_hiddens,
            out_ch=in_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_residual_layers,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            use_mid_attention=self.use_mid_attention,
            resamp_with_conv=True,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=embedding_dim,
        )
        dec_kwargs = dict(enc_dec_kwargs, extra_res_blocks=self.decoder_extra_residual_layers)
        self.encoder = Encoder(**enc_dec_kwargs)
        bottleneck_kwargs = dict(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparsity_level=sparsity_level,
            n_bins=n_bins,
            coef_max=coef_max,
            quantize_sparse_coeffs=quantize_sparse_coeffs,
            coef_quantization=coef_quantization,
            coef_mu=coef_mu,
            commitment_cost=commitment_cost,
            variational_coeffs=variational_coeffs,
            variational_coeff_kl_weight=variational_coeff_kl_weight,
            variational_coeff_prior_std=variational_coeff_prior_std,
            variational_coeff_min_std=variational_coeff_min_std,
        )
        if patch_based:
            self.bottleneck = PatchDictionaryLearningTokenized(
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_reconstruction=patch_reconstruction,
                **bottleneck_kwargs,
            )
        else:
            self.bottleneck = DictionaryLearningTokenized(**bottleneck_kwargs)
        self.decoder = Decoder(**dec_kwargs)
        self._last_latent_hw: Optional[Tuple[int, int]] = None

    def _remember_latent_hw(self, z: torch.Tensor) -> None:
        self._last_latent_hw = (int(z.shape[-2]), int(z.shape[-1]))

    def _resolve_patch_latent_hw(self, latent_hw: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        if not isinstance(self.bottleneck, PatchDictionaryLearningTokenized):
            return None
        if latent_hw is not None:
            return (int(latent_hw[0]), int(latent_hw[1]))
        decoder_z_shape = getattr(self.decoder, "z_shape", None)
        if isinstance(decoder_z_shape, tuple) and len(decoder_z_shape) >= 4:
            return (int(decoder_z_shape[-2]), int(decoder_z_shape[-1]))
        if self._last_latent_hw is None:
            raise RuntimeError(
                "Patch-based decoding requires the original latent spatial size. "
                "Run an encode/forward pass first or pass latent_hw explicitly."
            )
        return self._last_latent_hw

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        self._remember_latent_hw(z)
        bottleneck_ctx = (
            torch.autocast(device_type=z.device.type, enabled=False)
            if z.is_cuda and torch.is_autocast_enabled()
            else nullcontext()
        )
        with bottleneck_ctx:
            z_q, b_loss, tokens = self.bottleneck(z.float())
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon, b_loss, tokens

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        z = self.encoder(x)
        self._remember_latent_hw(z)
        _, _, tokens = self.bottleneck(z)
        return tokens, tokens.shape[1], tokens.shape[2]

    @torch.no_grad()
    def encode_to_atoms_and_coeffs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        z = self.encoder(x)
        self._remember_latent_hw(z)
        encoded = self.bottleneck._encode_sparse_codes(z)
        if isinstance(self.bottleneck, PatchDictionaryLearningTokenized):
            atoms, coeffs, _, _ = encoded
        else:
            atoms, coeffs = encoded
        coeffs = self.bottleneck.project_sparse_coeffs(atoms, coeffs)
        return atoms, coeffs, atoms.shape[1], atoms.shape[2]

    def clamp_sparse_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self.bottleneck.clamp_sparse_coeffs(coeffs)

    @torch.no_grad()
    def decode_from_tokens(
        self,
        tokens: torch.Tensor,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        patch_latent_hw = self._resolve_patch_latent_hw(latent_hw)
        if isinstance(self.bottleneck, PatchDictionaryLearningTokenized):
            z_q = self.bottleneck.tokens_to_latent(tokens, latent_hw=patch_latent_hw)
        else:
            z_q = self.bottleneck.tokens_to_latent(tokens)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon

    @torch.no_grad()
    def decode_from_atoms_and_coeffs(
        self,
        atoms: torch.Tensor,
        coeffs: torch.Tensor,
        latent_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        coeffs = self.clamp_sparse_coeffs(coeffs)
        patch_latent_hw = self._resolve_patch_latent_hw(latent_hw)
        if isinstance(self.bottleneck, PatchDictionaryLearningTokenized):
            z_q = self.bottleneck._reconstruct_sparse(
                atoms,
                coeffs,
                int(patch_latent_hw[0]),
                int(patch_latent_hw[1]),
            )
        else:
            z_q = self.bottleneck._reconstruct_sparse(atoms, coeffs)
        recon = self.decoder(z_q)
        if self.out_tanh:
            recon = torch.tanh(recon)
        return recon


# Backward-compatible alias for older scratch experiments.
SparseDictAE = LASER

RQAE = LASER
