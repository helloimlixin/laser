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

import math

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
try:
    from transformer_core import (
        TransformerBlock,
        private_long_tensor,
        run_transformer_blocks,
    )
except ModuleNotFoundError:
    from scratch.transformer_core import (
        TransformerBlock,
        private_long_tensor,
        run_transformer_blocks,
    )


def soft_clamp(x: torch.Tensor, max_val: float) -> torch.Tensor:
    """Tanh-based soft clamp: approximately linear near zero, smoothly
    saturates towards ±max_val instead of the hard discontinuity of clamp."""
    return max_val * torch.tanh(x / max_val)


_private_long_tensor = private_long_tensor


def _autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        if device_type == "cuda":
            return bool(torch.is_autocast_enabled())
        if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            return bool(torch.is_autocast_cpu_enabled())
    return False


def _autocast_dtype(device_type: str) -> Optional[torch.dtype]:
    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):
        if device_type == "cuda" and hasattr(torch, "get_autocast_gpu_dtype"):
            return torch.get_autocast_gpu_dtype()
        if device_type == "cpu" and hasattr(torch, "get_autocast_cpu_dtype"):
            return torch.get_autocast_cpu_dtype()
    return None


def _activation_dtype(device: torch.device, fallback: torch.dtype) -> torch.dtype:
    if _autocast_enabled(device.type):
        autocast_dtype = _autocast_dtype(device.type)
        if autocast_dtype is not None:
            return autocast_dtype
    return fallback


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
    gaussian_coeffs: bool = False
    coeff_prior_std: float = 0.25
    coeff_min_std: float = 0.01
    autoregressive_coeffs: bool = True


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
    autoregressive_coeffs: bool = True,
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
        gaussian_coeffs=bool(getattr(bottleneck, "variational_coeffs", False)),
        coeff_prior_std=float(getattr(bottleneck, "variational_coeff_prior_std", 0.25)),
        coeff_min_std=float(getattr(bottleneck, "variational_coeff_min_std", 0.01)),
        autoregressive_coeffs=bool(autoregressive_coeffs),
    )


class SpatialDepthPrior(nn.Module):

    def __init__(self, cfg: SpatialDepthPriorConfig):
        super().__init__()
        self.cfg = cfg
        self.real_valued_coeffs = bool(cfg.real_valued_coeffs)
        self.gaussian_coeffs = bool(self.real_valued_coeffs and cfg.gaussian_coeffs)
        self.autoregressive_coeffs = bool(cfg.autoregressive_coeffs)
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
            if not self.autoregressive_coeffs:
                raise ValueError(
                    "Quantized stage-2 priors require autoregressive_coeffs=True so the "
                    "shared atom/coeff token stream stays interleaved."
                )

        self.output_depth = int(cfg.D)
        if self.real_valued_coeffs or self.autoregressive_coeffs:
            self.rollout_depth = self.output_depth
        else:
            self.rollout_depth = self.output_depth // 2

        # Shared token embedding for atom IDs and output classes.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.row_emb = nn.Embedding(cfg.H, cfg.d_model)
        self.col_emb = nn.Embedding(cfg.W, cfg.d_model)
        self.depth_emb = nn.Embedding(self.rollout_depth, cfg.d_model)
        self.start_emb = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.start_emb, std=0.02)
        self.global_spatial_tokens = None
        if self.n_global_spatial_tokens > 0:
            self.global_spatial_tokens = nn.Parameter(
                torch.zeros(1, self.n_global_spatial_tokens, cfg.d_model)
            )
            nn.init.normal_(self.global_spatial_tokens, std=0.02)
        self.coeff_proj = (
            nn.Linear(1, cfg.d_model)
            if (self.real_valued_coeffs and self.autoregressive_coeffs)
            else None
        )
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
        else:
            self._coeff_bin_values = None

        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_ff,
                cfg.dropout,
                chunk_causal_batches=True,
            )
            for _ in range(cfg.n_spatial_layers)
        ])
        self.depth_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_depth_layers)
        ])
        self.spatial_ln = nn.LayerNorm(cfg.d_model)
        self.depth_ln = nn.LayerNorm(cfg.d_model)

        self.spatial_fuse = nn.Linear(self.rollout_depth * cfg.d_model, cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        if self.real_valued_coeffs:
            self.atom_coeff_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.coeff_head = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1),
            )
            if self.gaussian_coeffs:
                self.coeff_logvar_head = nn.Sequential(
                    nn.Linear(cfg.d_model * 2, cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_model, 1),
                )
                init_target = max(float(cfg.coeff_prior_std) - float(cfg.coeff_min_std), 1e-6)
                nn.init.zeros_(self.coeff_logvar_head[-1].weight)
                nn.init.constant_(self.coeff_logvar_head[-1].bias, math.log(math.expm1(init_target)))
            else:
                self.coeff_logvar_head = None
            self.coeff_token_head = None
        elif self.autoregressive_coeffs:
            self.atom_coeff_emb = None
            self.coeff_head = None
            self.coeff_logvar_head = None
            self.coeff_token_head = None
        else:
            self.atom_coeff_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.coeff_head = None
            self.coeff_logvar_head = None
            self.coeff_token_head = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, self.coeff_vocab_size),
            )

        spatial_pos = torch.arange(cfg.H * cfg.W, dtype=torch.long)
        self.register_buffer(
            "_rows", spatial_pos // cfg.W, persistent=False,
        )
        self.register_buffer(
            "_cols", spatial_pos % cfg.W, persistent=False,
        )
        depth_token_mask = torch.zeros(self.rollout_depth, cfg.vocab_size, dtype=torch.bool)
        if self.real_valued_coeffs:
            depth_token_mask[:, :self.atom_vocab_size] = True
        elif not self.autoregressive_coeffs:
            depth_token_mask[:, :self.atom_vocab_size] = True
        else:
            depth_token_mask[0::2, :self.atom_vocab_size] = True
            depth_token_mask[1::2, self.atom_vocab_size:self.content_vocab_size] = True
        self.register_buffer("_depth_token_mask", depth_token_mask, persistent=False)
        self.register_buffer(
            "_autoregressive_coeffs_flag",
            torch.tensor(1 if self.autoregressive_coeffs else 0, dtype=torch.int64),
            persistent=True,
        )

    def _clamp_conditioning_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        coeff_max = float(self.cfg.coeff_max)
        if not math.isfinite(coeff_max) or coeff_max <= 0.0:
            return coeffs
        return coeffs.clamp(-coeff_max, coeff_max)

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
        def _sync_tensor(
            tensor_name: str,
            expected: torch.Tensor,
            *,
            remap=None,
        ) -> None:
            full_key = prefix + tensor_name
            loaded = state_dict.get(full_key, None)
            expected = expected.detach().clone()
            if loaded is None:
                state_dict[full_key] = expected
                return
            loaded = loaded.to(device=expected.device, dtype=expected.dtype)
            if tuple(loaded.shape) == tuple(expected.shape):
                return
            if remap is not None:
                patched = remap(loaded, expected)
                if patched is not None and tuple(patched.shape) == tuple(expected.shape):
                    state_dict[full_key] = patched
                    return
            state_dict[full_key] = expected

        def _sync_optional_module(module_name: str, module: Optional[nn.Module]) -> None:
            module_prefix = prefix + module_name + "."
            if module is None:
                for key in list(state_dict.keys()):
                    if key.startswith(module_prefix):
                        state_dict.pop(key, None)
                return
            for subkey, expected in module.state_dict().items():
                full_key = module_prefix + subkey
                expected = expected.detach().clone()
                loaded = state_dict.get(full_key, None)
                if loaded is None or tuple(loaded.shape) != tuple(expected.shape):
                    state_dict[full_key] = expected

        def _remap_depth_embedding(
            loaded: torch.Tensor,
            expected: torch.Tensor,
        ) -> Optional[torch.Tensor]:
            if loaded.ndim != 2 or expected.ndim != 2 or loaded.size(1) != expected.size(1):
                return None
            patched = expected.clone()
            if not self.real_valued_coeffs:
                if (not self.autoregressive_coeffs) and loaded.size(0) == expected.size(0) * 2:
                    patched.copy_(loaded[0::2][:expected.size(0)])
                    return patched
                if self.autoregressive_coeffs and expected.size(0) == loaded.size(0) * 2:
                    patched[0::2][:loaded.size(0)].copy_(loaded)
                    return patched
            keep = min(int(loaded.size(0)), int(expected.size(0)))
            patched[:keep].copy_(loaded[:keep])
            return patched

        def _remap_spatial_fuse_weight(
            loaded: torch.Tensor,
            expected: torch.Tensor,
        ) -> Optional[torch.Tensor]:
            if loaded.ndim != 2 or expected.ndim != 2 or loaded.size(0) != expected.size(0):
                return None
            d_model = int(self.cfg.d_model)
            if loaded.size(1) % d_model != 0 or expected.size(1) % d_model != 0:
                return None
            loaded_blocks = loaded.size(1) // d_model
            expected_blocks = expected.size(1) // d_model
            patched = expected.clone()
            patched_blocks = patched.view(expected.size(0), expected_blocks, d_model)
            loaded_blocks_view = loaded.view(loaded.size(0), loaded_blocks, d_model)
            if not self.real_valued_coeffs:
                if (not self.autoregressive_coeffs) and loaded_blocks == expected_blocks * 2:
                    patched_blocks.copy_(loaded_blocks_view[:, 0::2][:, :expected_blocks, :])
                    return patched
                if self.autoregressive_coeffs and expected_blocks == loaded_blocks * 2:
                    patched_blocks[:, 0::2][:, :loaded_blocks, :].copy_(loaded_blocks_view)
                    return patched
            keep = min(int(loaded_blocks), int(expected_blocks))
            patched_blocks[:, :keep, :].copy_(loaded_blocks_view[:, :keep, :])
            return patched

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
        coeff_logvar_prefix = prefix + "coeff_logvar_head."
        if self.coeff_logvar_head is None:
            for key in list(state_dict.keys()):
                if key.startswith(coeff_logvar_prefix):
                    state_dict.pop(key, None)
        else:
            for subkey, expected in self.coeff_logvar_head.state_dict().items():
                full_key = coeff_logvar_prefix + subkey
                loaded = state_dict.get(full_key, None)
                expected = expected.detach().clone()
                if loaded is None or tuple(loaded.shape) != tuple(expected.shape):
                    state_dict[full_key] = expected
        coeff_flag_key = prefix + "_autoregressive_coeffs_flag"
        flag_expected = self._autoregressive_coeffs_flag.detach().clone()
        flag_loaded = state_dict.get(coeff_flag_key, None)
        if flag_loaded is None or tuple(flag_loaded.shape) != tuple(flag_expected.shape):
            state_dict[coeff_flag_key] = flag_expected
        _sync_tensor("depth_emb.weight", self.depth_emb.weight, remap=_remap_depth_embedding)
        _sync_tensor("spatial_fuse.weight", self.spatial_fuse.weight, remap=_remap_spatial_fuse_weight)
        _sync_tensor("spatial_fuse.bias", self.spatial_fuse.bias)
        _sync_optional_module("coeff_proj", self.coeff_proj)
        _sync_optional_module("coeff_token_emb", self.coeff_token_emb)
        _sync_optional_module("token_type_emb", self.token_type_emb)
        _sync_optional_module("atom_coeff_emb", self.atom_coeff_emb)
        _sync_optional_module("coeff_head", self.coeff_head)
        _sync_optional_module("coeff_token_head", self.coeff_token_head)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _activation_dtype(self, device: torch.device) -> torch.dtype:
        return _activation_dtype(device, self.start_emb.dtype)

    def _spatial_pos(
        self,
        T: int,
        device: torch.device,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        rows = self._rows[:T].to(device)
        cols = self._cols[:T].to(device)
        pos = self.row_emb(rows) + self.col_emb(cols)
        if dtype is not None and pos.dtype != dtype:
            pos = pos.to(dtype=dtype)
        return pos

    def _prepend_global_spatial_tokens(self, spatial_in: torch.Tensor) -> torch.Tensor:
        if self.global_spatial_tokens is None:
            return spatial_in
        global_tokens = self.global_spatial_tokens.expand(spatial_in.size(0), -1, -1).to(dtype=spatial_in.dtype)
        return torch.cat([global_tokens, spatial_in], dim=1)

    def _prime_spatial_kv_with_global_tokens(
        self,
        batch_size: int,
        spatial_kv: list[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> list[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.global_spatial_tokens is None:
            return spatial_kv

        global_h = self.global_spatial_tokens.expand(batch_size, -1, -1).to(
            dtype=self._activation_dtype(self.global_spatial_tokens.device)
        )
        _, spatial_kv = run_transformer_blocks(
            global_h,
            self.spatial_blocks,
            None,
            kv_cache=spatial_kv,
        )
        return spatial_kv

    def _prepare_depth_inputs(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B, T, D], got {tuple(tokens.shape)}")
        B, T, D = tokens.shape
        if D != self.rollout_depth:
            raise ValueError(
                f"Expected rollout tokens with depth {self.rollout_depth}, got {D}"
            )
        device = tokens.device
        act_dtype = self._activation_dtype(device)
        tokens = _private_long_tensor(tokens, device=device)

        if self.real_valued_coeffs and self.autoregressive_coeffs:
            if coeffs is None:
                raise ValueError("coeffs must be provided when real_valued_coeffs=True")
            coeffs = self._clamp_conditioning_coeffs(coeffs.to(device=device, dtype=act_dtype))
            if tuple(coeffs.shape) != (B, T, D):
                raise ValueError(
                    f"Expected coeffs with shape {(B, T, D)}, got {tuple(coeffs.shape)}"
                )
        else:
            coeffs = None

        return tokens, coeffs, act_dtype

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        act_dtype = self._activation_dtype(tokens.device)
        return self.token_emb(tokens).to(dtype=act_dtype)

    def _split_quantized_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.real_valued_coeffs:
            raise RuntimeError("_split_quantized_tokens is only valid when real_valued_coeffs=False")
        if tokens.size(-1) != self.output_depth:
            raise ValueError(f"Expected token depth {self.output_depth}, got {tokens.size(-1)}")
        atom_tokens = tokens[..., 0::2].to(torch.long)
        coeff_tokens = tokens[..., 1::2].to(torch.long) - self.atom_vocab_size
        return atom_tokens, coeff_tokens

    def _assemble_quantized_support_logits(
        self,
        atom_logits: torch.Tensor,
        coeff_logits: torch.Tensor,
    ) -> torch.Tensor:
        if atom_logits.shape[:3] != coeff_logits.shape[:3]:
            raise ValueError(
                "atom_logits and coeff_logits shape mismatch: "
                f"{tuple(atom_logits.shape)} vs {tuple(coeff_logits.shape)}"
            )
        B, T, D, _ = atom_logits.shape
        if D != self.rollout_depth:
            raise ValueError(f"Expected rollout depth {self.rollout_depth}, got {D}")
        if coeff_logits.size(-1) != self.coeff_vocab_size:
            raise ValueError(
                f"Expected coeff vocab size {self.coeff_vocab_size}, got {coeff_logits.size(-1)}"
            )
        full = atom_logits.new_full((B, T, self.output_depth, self.cfg.vocab_size), float("-inf"))
        full[:, :, 0::2, :] = atom_logits
        full[:, :, 1::2, self.atom_vocab_size:self.atom_vocab_size + self.coeff_vocab_size] = coeff_logits
        return full

    def _pack_quantized_tokens(
        self,
        atom_ids: torch.Tensor,
        coeff_bins: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(atom_ids.shape) != tuple(coeff_bins.shape):
            raise ValueError(f"atom_ids and coeff_bins shape mismatch: {atom_ids.shape} vs {coeff_bins.shape}")
        if atom_ids.size(-1) != self.rollout_depth:
            raise ValueError(f"Expected rollout depth {self.rollout_depth}, got {atom_ids.size(-1)}")
        tokens = torch.empty(
            *atom_ids.shape[:-1],
            self.output_depth,
            device=atom_ids.device,
            dtype=torch.long,
        )
        tokens[..., 0::2] = atom_ids.to(torch.long)
        tokens[..., 1::2] = coeff_bins.to(torch.long) + self.atom_vocab_size
        return tokens

    def _masked_training_logits(
        self,
        logits: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected [B, T, D, V] logits, got {tuple(logits.shape)}")
        masked = logits
        if tokens is not None:
            if tokens.shape != logits.shape[:3]:
                raise ValueError(
                    f"Expected tokens with shape {tuple(logits.shape[:3])}, got {tuple(tokens.shape)}"
                )
            tokens = _private_long_tensor(tokens, device=logits.device)
            if self.real_valued_coeffs or not self.autoregressive_coeffs:
                depth_range = range(1, self.rollout_depth)
                prev_atom_slice = lambda depth_idx: tokens[:, :, :depth_idx]
            else:
                depth_range = range(2, self.rollout_depth, 2)
                prev_atom_slice = lambda depth_idx: tokens[:, :, :depth_idx:2]
            for depth_idx in depth_range:
                prev_atoms = prev_atom_slice(depth_idx).to(torch.long)
                masked[:, :, depth_idx, :].scatter_(2, prev_atoms, float("-inf"))
        return masked

    def _forward_spatial_hidden(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens, coeffs, act_dtype = self._prepare_depth_inputs(tokens, coeffs)
        B, T, D = tokens.shape
        d_model = self.cfg.d_model
        device = tokens.device

        spatial_in = torch.zeros(B, T, d_model, device=device, dtype=act_dtype)
        spatial_in[:, 0] = self.start_emb.squeeze(1)
        if T > 1:
            prev_token_emb = self._embed_tokens(tokens[:, :-1])
            if self.real_valued_coeffs and self.autoregressive_coeffs:
                if self.coeff_proj is None:
                    raise RuntimeError("Missing coeff_proj for autoregressive real-valued coefficient conditioning")
                prev_token_emb = prev_token_emb + self.coeff_proj(coeffs[:, :-1].unsqueeze(-1))
            prev_flat = prev_token_emb.reshape(B, T - 1, D * d_model)
            spatial_in[:, 1:] = self.spatial_fuse(prev_flat)

        spatial_in = spatial_in + self._spatial_pos(T, device, dtype=act_dtype).unsqueeze(0)
        spatial_h = self._prepend_global_spatial_tokens(spatial_in)
        spatial_h, _ = run_transformer_blocks(
            spatial_h,
            self.spatial_blocks,
            self.spatial_ln,
        )
        if self.n_global_spatial_tokens > 0:
            spatial_h = spatial_h[:, self.n_global_spatial_tokens:]
        return spatial_h

    def _forward_depth_hidden_from_spatial(
        self,
        spatial_h: torch.Tensor,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens, coeffs, act_dtype = self._prepare_depth_inputs(tokens, coeffs)
        if spatial_h.ndim != 3:
            raise ValueError(
                f"Expected spatial_h with shape [B, T, d_model], got {tuple(spatial_h.shape)}"
            )

        B, T, D = tokens.shape
        d_model = self.cfg.d_model
        expected_spatial_shape = (B, T, d_model)
        if tuple(spatial_h.shape) != expected_spatial_shape:
            raise ValueError(
                f"Expected spatial_h with shape {expected_spatial_shape}, got {tuple(spatial_h.shape)}"
            )

        device = tokens.device
        spatial_h = spatial_h.to(device=device, dtype=act_dtype)
        bt = B * T
        spatial_h_flat = spatial_h.reshape(bt, d_model)
        depth_in = torch.zeros(bt, D, d_model, device=device, dtype=act_dtype)
        depth_in[:, 0] = spatial_h_flat
        if D > 1:
            prev_depth_tokens = tokens[:, :, :-1].reshape(bt, D - 1)
            depth_in[:, 1:] = self._embed_tokens(prev_depth_tokens)
            if self.real_valued_coeffs and self.autoregressive_coeffs:
                if self.coeff_proj is None:
                    raise RuntimeError("Missing coeff_proj for autoregressive real-valued coefficient conditioning")
                prev_depth_coeffs = coeffs[:, :, :-1].reshape(bt, D - 1, 1)
                depth_in[:, 1:] = depth_in[:, 1:] + self.coeff_proj(prev_depth_coeffs)

        depth_pos = self.depth_emb(torch.arange(D, device=device)).to(dtype=act_dtype)
        depth_in = depth_in + depth_pos.unsqueeze(0)

        depth_h, _ = run_transformer_blocks(
            depth_in,
            self.depth_blocks,
            self.depth_ln,
        )
        return depth_h.reshape(B, T, D, d_model)

    def _forward_depth_hidden(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spatial_h = self._forward_spatial_hidden(tokens, coeffs)
        return self._forward_depth_hidden_from_spatial(spatial_h, tokens, coeffs)

    def _predict_coeff_distribution_from_concat(
        self,
        coeff_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        coeff_mu = soft_clamp(
            self.coeff_head(coeff_in).squeeze(-1),
            self.cfg.coeff_max,
        )
        if not self.gaussian_coeffs or self.coeff_logvar_head is None:
            return coeff_mu, None
        coeff_std = self.cfg.coeff_min_std + F.softplus(self.coeff_logvar_head(coeff_in).squeeze(-1))
        coeff_std = coeff_std.clamp_max(max(float(self.cfg.coeff_max), float(self.cfg.coeff_min_std)))
        coeff_logvar = 2.0 * torch.log(coeff_std.clamp_min(1e-6))
        return coeff_mu, coeff_logvar

    def predict_coeff_distribution_for_atoms(
        self,
        depth_h: torch.Tensor,
        atom_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.real_valued_coeffs or self.atom_coeff_emb is None or self.coeff_head is None:
            raise RuntimeError("predict_coeff_distribution_for_atoms is only valid when real_valued_coeffs=True")
        if depth_h.ndim != 4:
            raise ValueError(f"Expected depth_h with shape [B, T, D, d_model], got {tuple(depth_h.shape)}")
        if tuple(atom_ids.shape) != tuple(depth_h.shape[:3]):
            raise ValueError(
                f"Expected atom_ids with shape {tuple(depth_h.shape[:3])}, got {tuple(atom_ids.shape)}"
            )

        B, T, D, d_model = depth_h.shape
        depth_h_flat = depth_h.reshape(B * T, D, d_model)
        atom_ids_flat = _private_long_tensor(
            atom_ids.reshape(B * T, D),
            device=depth_h.device,
        )
        coeff_emb = self.atom_coeff_emb(atom_ids_flat).to(dtype=depth_h_flat.dtype)
        coeff_in = torch.cat([depth_h_flat, coeff_emb], dim=-1)
        coeff_mu, coeff_logvar = self._predict_coeff_distribution_from_concat(coeff_in)
        coeff_mu = coeff_mu.reshape(B, T, D)
        if coeff_logvar is None:
            return coeff_mu, None
        return coeff_mu, coeff_logvar.reshape(B, T, D)

    def predict_coeffs_for_atoms(
        self,
        depth_h: torch.Tensor,
        atom_ids: torch.Tensor,
    ) -> torch.Tensor:
        coeff_mu, _ = self.predict_coeff_distribution_for_atoms(depth_h, atom_ids)
        return coeff_mu

    def predict_quantized_coeff_logits_for_atoms(
        self,
        depth_h: torch.Tensor,
        atom_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.real_valued_coeffs or self.coeff_token_head is None or self.atom_coeff_emb is None:
            raise RuntimeError(
                "predict_quantized_coeff_logits_for_atoms is only valid for support-only quantized priors"
            )
        if depth_h.ndim != 4:
            raise ValueError(f"Expected depth_h with shape [B, T, D, d_model], got {tuple(depth_h.shape)}")
        if tuple(atom_ids.shape) != tuple(depth_h.shape[:3]):
            raise ValueError(
                f"Expected atom_ids with shape {tuple(depth_h.shape[:3])}, got {tuple(atom_ids.shape)}"
            )

        B, T, D, d_model = depth_h.shape
        depth_h_flat = depth_h.reshape(B * T, D, d_model)
        atom_ids_flat = _private_long_tensor(
            atom_ids.reshape(B * T, D),
            device=depth_h.device,
        )
        coeff_emb = self.atom_coeff_emb(atom_ids_flat).to(dtype=depth_h_flat.dtype)
        coeff_in = torch.cat([depth_h_flat, coeff_emb], dim=-1)
        coeff_logits = self.coeff_token_head(coeff_in)
        return coeff_logits.reshape(B, T, D, self.coeff_vocab_size)

    def forward_features(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
        mask_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_tokens is None:
            mask_tokens = tokens
        depth_h = self._forward_depth_hidden(tokens, coeffs)
        token_logits = self.token_head(depth_h)
        token_logits = self._masked_training_logits(token_logits, tokens=mask_tokens)
        return token_logits, depth_h

    def _mask_generation_logits(
        self,
        logits: torch.Tensor,
        depth_idx: int,
        prev_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prev_atoms = None
        if prev_tokens is not None and depth_idx > 0:
            if self.real_valued_coeffs or not self.autoregressive_coeffs:
                prev_atoms = prev_tokens[:, :depth_idx]
            elif (depth_idx % 2) == 0:
                prev_atoms = prev_tokens[:, :depth_idx:2]
        if prev_atoms is not None:
            if prev_atoms.numel() > 0:
                prev_atoms = _private_long_tensor(prev_atoms, device=logits.device)
                logits.scatter_(1, prev_atoms, float("-inf"))
        return logits

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
            v, ix = torch.topk(logits, k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, ix, v)
            logits = mask
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(
            probs.view(-1, probs.size(-1)),
            1,
        ).view(*probs.shape[:-1])

    def forward(
        self,
        tokens: torch.Tensor,
        coeffs: Optional[torch.Tensor] = None,
        return_features: bool = False,
        mask_tokens: Optional[torch.Tensor] = None,
    ):
        """Training forward with teacher forcing.

        Args:
            tokens: [B, H*W, D] ground-truth atom ids in real-valued mode or
                interleaved atom/bin tokens in quantized mode.
            coeffs: [B, H*W, D] ground-truth coefficients for real-valued mode.
            mask_tokens: Optional [B, H*W, D] token sequence used for support
                masking. Defaults to ``tokens``. This lets scheduled-sampling
                training condition on mixed histories while still scoring
                ground-truth targets under the target support constraints.

        Returns:
            real-valued mode: (atom_logits [B, H*W, D, vocab], coeff_mean [B, H*W, D], coeff_logvar [B, H*W, D] or None)
            quantized mode:   token_logits [B, H*W, D, vocab]
        """
        token_logits, depth_h = self.forward_features(tokens, coeffs, mask_tokens=mask_tokens)
        if not self.real_valued_coeffs:
            if return_features:
                return token_logits, depth_h
            return token_logits

        coeff_pred, coeff_logvar = self.predict_coeff_distribution_for_atoms(depth_h, tokens)
        if return_features:
            if coeff_logvar is None:
                return token_logits, coeff_pred, depth_h
            return token_logits, coeff_pred, coeff_logvar, depth_h
        if coeff_logvar is None:
            return token_logits, coeff_pred
        return token_logits, coeff_pred, coeff_logvar

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
        prompt_tokens = _private_long_tensor(prompt_tokens, device=device)

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
        prompt_coeffs = self._clamp_conditioning_coeffs(
            prompt_coeffs.to(device=device, dtype=self._activation_dtype(device))
        )
        return prompt_tokens, prompt_coeffs, prompt_mask

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unconditional generation with KV-cached spatial transformer.

        Returns:
            real-valued mode: (atom_ids [B, H*W, D], coeffs [B, H*W, D])
            quantized mode:   tokens [B, H*W, D]
        """
        device = next(self.parameters()).device
        coeff_sample_mode = str(coeff_sample_mode).strip().lower()
        if coeff_sample_mode not in {"gaussian", "mean"}:
            raise ValueError(
                f"coeff_sample_mode must be 'gaussian' or 'mean', got {coeff_sample_mode!r}"
            )
        if coeff_temperature is None:
            resolved_coeff_temperature = float(temperature)
        else:
            resolved_coeff_temperature = float(coeff_temperature)
        if resolved_coeff_temperature <= 0.0:
            raise ValueError(
                f"coeff_temperature must be > 0, got {resolved_coeff_temperature}"
            )
        cfg = self.cfg
        T = cfg.H * cfg.W
        D = self.output_depth
        prompt_tokens, prompt_coeffs, prompt_mask = self._prepare_prompt_inputs(
            batch_size=batch_size,
            T=T,
            D=D,
            device=device,
            prompt_tokens=prompt_tokens,
            prompt_coeffs=prompt_coeffs,
            prompt_mask=prompt_mask,
        )

        rollout_prompt_tokens = prompt_tokens
        rollout_prompt_mask = prompt_mask
        prompt_coeff_mask = prompt_mask if (self.real_valued_coeffs and not self.autoregressive_coeffs) else None

        rollout_D = self.rollout_depth
        act_dtype = self._activation_dtype(device)
        tokens = torch.zeros(batch_size, T, rollout_D, dtype=torch.long, device=device)
        coeffs = (
            torch.zeros(batch_size, T, rollout_D, device=device, dtype=act_dtype)
            if (self.real_valued_coeffs and self.autoregressive_coeffs)
            else None
        )
        deferred_depth_h = (
            torch.zeros(batch_size, T, rollout_D, cfg.d_model, device=device, dtype=act_dtype)
            if not self.autoregressive_coeffs
            else None
        )

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
                x_new = self.start_emb.expand(batch_size, -1, -1).to(dtype=act_dtype)
            else:
                prev_emb = self._embed_tokens(tokens[:, t - 1])    # [B, D, d]
                if self.real_valued_coeffs and self.autoregressive_coeffs:
                    if self.coeff_proj is None:
                        raise RuntimeError("Missing coeff_proj for autoregressive real-valued coefficient conditioning")
                    prev_emb = prev_emb + self.coeff_proj(coeffs[:, t - 1].unsqueeze(-1))
                x_new = self.spatial_fuse(
                    prev_emb.reshape(batch_size, -1),
                ).unsqueeze(1)                                   # [B, 1, d]

            x_new = x_new + (
                self.row_emb(self._rows[t]) + self.col_emb(self._cols[t])
            ).to(dtype=x_new.dtype)

            spatial_h = x_new
            spatial_h, spatial_kv = run_transformer_blocks(
                spatial_h,
                self.spatial_blocks,
                self.spatial_ln,
                kv_cache=spatial_kv,
            )
            h_t = spatial_h.squeeze(1)                           # [B, d]

            # -- depth stage: autoregressive, no KV cache (D is small) --
            depth_seq: list[torch.Tensor] = []
            for d in range(rollout_D):
                if d == 0:
                    step_in = h_t.unsqueeze(1)
                else:
                    step_in = self._embed_tokens(tokens[:, t, d - 1]).unsqueeze(1)
                    if self.real_valued_coeffs and self.autoregressive_coeffs:
                        if self.coeff_proj is None:
                            raise RuntimeError("Missing coeff_proj for autoregressive real-valued coefficient conditioning")
                        step_in = step_in + self.coeff_proj(coeffs[:, t, d - 1].view(batch_size, 1, 1))
                step_in = step_in + self.depth_emb.weight[d].to(dtype=step_in.dtype)
                depth_seq.append(step_in)

                depth_h = torch.cat(depth_seq, dim=1)            # [B, d+1, dm]
                depth_h, _ = run_transformer_blocks(
                    depth_h,
                    self.depth_blocks,
                    self.depth_ln,
                )
                last_h = depth_h[:, -1]                          # [B, dm]

                logits = self.token_head(last_h)
                logits = self._mask_generation_logits(logits, d, prev_tokens=tokens[:, t, :d])
                sampled = self._sample_from_logits(
                    logits,
                    temperature=float(temperature),
                    top_k=top_k,
                )
                if rollout_prompt_mask is not None:
                    clamp_now = rollout_prompt_mask[:, t, d]
                    if clamp_now.any():
                        sampled = torch.where(clamp_now, rollout_prompt_tokens[:, t, d], sampled)
                tokens[:, t, d] = sampled
                if deferred_depth_h is not None:
                    deferred_depth_h[:, t, d] = last_h

                if self.real_valued_coeffs and self.autoregressive_coeffs:
                    c_emb = self.atom_coeff_emb(sampled).to(dtype=last_h.dtype)
                    c_in = torch.cat([last_h, c_emb], dim=-1)
                    c_mean, c_logvar = self._predict_coeff_distribution_from_concat(c_in)
                    if c_logvar is None or coeff_sample_mode == "mean":
                        c_val = c_mean
                    else:
                        c_std = (0.5 * c_logvar).exp()
                        c_val = soft_clamp(
                            c_mean + max(float(resolved_coeff_temperature), 1e-8) * c_std * torch.randn_like(c_std),
                            cfg.coeff_max,
                        )
                    if rollout_prompt_mask is not None:
                        clamp_now = rollout_prompt_mask[:, t, d]
                        if clamp_now.any():
                            c_val = torch.where(clamp_now, prompt_coeffs[:, t, d], c_val)
                    coeffs[:, t, d] = c_val

        if self.real_valued_coeffs:
            if not self.autoregressive_coeffs:
                if deferred_depth_h is None:
                    raise RuntimeError("Missing deferred depth features for support-only coeff prediction")
                coeff_mean, coeff_logvar = self.predict_coeff_distribution_for_atoms(deferred_depth_h, tokens)
                if coeff_logvar is None or coeff_sample_mode == "mean":
                    coeffs = coeff_mean
                else:
                    coeff_std = (0.5 * coeff_logvar).exp()
                    coeffs = soft_clamp(
                        coeff_mean + max(float(resolved_coeff_temperature), 1e-8) * coeff_std * torch.randn_like(coeff_std),
                        cfg.coeff_max,
                    )
                if prompt_coeff_mask is not None and prompt_coeffs is not None:
                    coeffs = torch.where(prompt_coeff_mask, prompt_coeffs, coeffs)
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
