"""
Structure-aware auxiliary losses for the LASER stage-2 prior.

Three complementary losses that inject geometric and metric structure
from the dictionary / coefficient space into the prior's training signal:

1. **Soft-target atom CE** — replaces hard one-hot atom targets with soft
   targets derived from the dictionary's cosine similarity matrix.
   Predicting a nearby atom is penalised less than a distant one.

2. **Ordinal coefficient regression** — treats coefficient bin prediction
   as both a classification *and* a regression problem.  Minimises the
   gap between the expected coefficient value under the predicted
   distribution and the ground-truth value.

3. **Reconstruction contrastive (InfoNCE)** — a projection head on the
   depth-transformer hidden states, trained so that positions whose
   sparse codes reconstruct similar feature vectors have similar
   representations.

All three are auxiliary losses added to the existing CE.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Soft-target atom CE
# ---------------------------------------------------------------------------

class DictionarySimCache(nn.Module):
    """Stores the frozen dictionary similarity matrix and produces soft
    target distributions on demand.

    The similarity matrix ``S[i, j] = d_i^T d_j`` is precomputed once
    from the column-normalised dictionary.  Soft targets for a given
    ground-truth atom *i* are ``softmax(S[i] / tau)``.
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        tau: float = 0.07,
    ):
        super().__init__()
        # dictionary: [C, K] with unit-norm columns
        D = F.normalize(dictionary.detach().float(), p=2, dim=0)
        sim = D.t() @ D                                      # [K, K]
        self.register_buffer("_sim", sim, persistent=False)
        self.tau = tau

    @torch.no_grad()
    def soft_targets(self, atom_ids: torch.Tensor) -> torch.Tensor:
        """Return soft target distributions ``[*, K]`` for atom ids ``[*]``."""
        flat = atom_ids.reshape(-1).long()
        sim_rows = self._sim[flat]                            # [N, K]
        return F.softmax(sim_rows / self.tau, dim=-1).view(
            *atom_ids.shape, self._sim.size(0),
        )


def soft_atom_ce(
    atom_logits: torch.Tensor,
    atom_ids: torch.Tensor,
    sim_cache: DictionarySimCache,
    hard_mix: float = 0.5,
) -> torch.Tensor:
    """Cross-entropy with dictionary-similarity label smoothing.

    Targets are a blend of one-hot (hard) and dictionary-similarity (soft):
    ``target = hard_mix * one_hot(gt) + (1 - hard_mix) * sim_softmax(gt)``

    With ``hard_mix=1.0`` this is standard CE.  With ``hard_mix=0.0``
    this is pure soft-target CE.  The default ``0.5`` gives a single
    consistent gradient that rewards both the exact atom *and*
    dictionary-similar atoms, avoiding the conflict between separate
    hard CE and soft CE losses.

    Args:
        atom_logits: ``[..., K]`` logits over the atom vocabulary.
        atom_ids:    ``[...]`` ground-truth atom indices in ``[0, K)``.
        sim_cache:   :class:`DictionarySimCache` instance.
        hard_mix:    blend weight for one-hot targets (0–1).

    Returns:
        Scalar loss (mean over all elements).
    """
    K = atom_logits.size(-1)
    N = atom_ids.numel()
    flat_logits = atom_logits.reshape(-1, K)
    flat_ids = atom_ids.reshape(-1).long()

    # One-hot targets
    one_hot = torch.zeros_like(flat_logits)
    one_hot.scatter_(1, flat_ids.unsqueeze(1), 1.0)

    # Soft similarity targets
    soft = sim_cache.soft_targets(atom_ids).to(
        device=flat_logits.device, dtype=flat_logits.dtype,
    ).reshape(-1, K)

    # Blend
    target = hard_mix * one_hot + (1.0 - hard_mix) * soft

    # Zero out where logits are masked (-inf) to avoid inf loss.
    valid = flat_logits > float("-inf")
    target = target * valid.float()
    row_sum = target.sum(-1, keepdim=True).clamp(min=1e-8)
    target = target / row_sum

    log_probs = F.log_softmax(flat_logits, dim=-1)
    per_sample = -(target * log_probs).sum(-1)
    return per_sample[per_sample.isfinite()].mean()


# ---------------------------------------------------------------------------
# 2.  Ordinal coefficient regression
# ---------------------------------------------------------------------------

def ordinal_coeff_loss(
    coeff_logits: torch.Tensor,
    coeff_ids: torch.Tensor,
    bin_values: torch.Tensor,
    loss_type: str = "huber",
    huber_delta: float = 0.5,
    magnitude_weighted: bool = False,
    zero_drift_margin: float = 0.0,
    zero_drift_threshold: float = 0.0,
) -> torch.Tensor:
    """Expected-value regression loss for quantised coefficient bins.

    Computes ``E[v] = sum_j softmax(logits)_j * bin_values_j`` and
    penalises the gap to the ground-truth bin value.

    When ``magnitude_weighted=True``, each element's loss is scaled by
    ``max(|gt_value|, 0.1)`` so that meaningful (large) coefficients
    receive proportionally more gradient than near-zero ones.

    When ``zero_drift_margin > 0``, an additional penalty is added for
    coefficients whose ground-truth magnitude is below
    ``zero_drift_threshold`` but whose predicted magnitude exceeds
    ``zero_drift_margin``:  ``relu(|pred| - margin)``.

    Args:
        coeff_logits:         ``[..., n_bins]`` logits over coefficient bins.
        coeff_ids:            ``[...]`` ground-truth bin indices in ``[0, n_bins)``.
        bin_values:           ``[n_bins]`` dequantised coefficient values per bin.
        loss_type:            ``"huber"`` or ``"l1"``.
        huber_delta:          delta for Huber loss.
        magnitude_weighted:   weight each element by ``max(|gt|, 0.1)``.
        zero_drift_margin:    penalise ``|pred| > margin`` when ``|gt| < threshold``.
        zero_drift_threshold: GT magnitude below which zero-drift penalty applies.

    Returns:
        Scalar loss.
    """
    n_bins = coeff_logits.size(-1)
    flat_logits = coeff_logits.reshape(-1, n_bins)
    probs = F.softmax(flat_logits, dim=-1)                    # [N, n_bins]
    bv = bin_values.to(device=probs.device, dtype=probs.dtype)
    expected = (probs * bv.unsqueeze(0)).sum(-1)              # [N]

    gt = bv[coeff_ids.reshape(-1).long().clamp(0, n_bins - 1)]

    # ---- per-element regression loss ----
    if loss_type == "l1":
        per_elem = (expected - gt).abs()
    else:
        per_elem = F.huber_loss(expected, gt, reduction="none", delta=huber_delta)

    # ---- magnitude weighting ----
    if magnitude_weighted:
        w = gt.abs().clamp(min=0.1)
        w = w / w.mean()                                     # normalise to mean 1
        per_elem = per_elem * w

    loss = per_elem.mean()

    # ---- zero-drift penalty ----
    if zero_drift_margin > 0.0 and zero_drift_threshold > 0.0:
        near_zero = gt.abs() < zero_drift_threshold
        if near_zero.any():
            drift = F.relu(expected[near_zero].abs() - zero_drift_margin)
            loss = loss + drift.mean()

    return loss


# ---------------------------------------------------------------------------
# 3.  Reconstruction contrastive (InfoNCE)
# ---------------------------------------------------------------------------

class ReconContrastiveHead(nn.Module):
    """Small projection head mapping depth hidden states and reconstruction
    vectors to a shared contrastive embedding space.

    Parameters
    ----------
    d_model : int
        Dimension of the depth-transformer hidden states.
    recon_dim : int
        Dimension of the sparse-code reconstruction vector (= embedding_dim C).
    proj_dim : int
        Dimension of the contrastive embedding.
    """

    def __init__(self, d_model: int, recon_dim: int, proj_dim: int = 128):
        super().__init__()
        self.h_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.z_proj = nn.Sequential(
            nn.Linear(recon_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(
        self,
        depth_h: torch.Tensor,
        z_recon: torch.Tensor,
        tau: float = 0.07,
        max_pairs: int = 4096,
    ) -> torch.Tensor:
        """Compute InfoNCE loss between depth hidden states and
        their corresponding reconstruction vectors.

        Args:
            depth_h: ``[B, L, D, d_model]`` depth-transformer hidden states.
            z_recon: ``[B, L, C]`` ground-truth reconstruction at each
                     spatial position.
            tau:     Temperature for InfoNCE.
            max_pairs: Subsample to this many pairs to limit memory.

        Returns:
            Scalar InfoNCE loss.
        """
        B, L, D, d = depth_h.shape
        # mean-pool across depth to get one vector per position
        h_pos = depth_h.mean(dim=2).reshape(B * L, d)        # [BL, d]
        z_pos = z_recon.reshape(B * L, -1)                    # [BL, C]

        # subsample if needed
        N = h_pos.size(0)
        if N > max_pairs:
            idx = torch.randperm(N, device=h_pos.device)[:max_pairs]
            h_pos = h_pos[idx]
            z_pos = z_pos[idx]
            N = max_pairs

        h_e = F.normalize(self.h_proj(h_pos), dim=-1)        # [N, proj]
        z_e = F.normalize(self.z_proj(z_pos), dim=-1)        # [N, proj]

        sim = h_e @ z_e.t() / tau                             # [N, N]
        labels = torch.arange(N, device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


# ---------------------------------------------------------------------------
# Convenience: combined loss computation for the quantised path
# ---------------------------------------------------------------------------

def compute_contrastive_sparse_losses(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    atom_vocab_size: int,
    coeff_vocab_size: int,
    sim_cache: DictionarySimCache,
    bin_values: torch.Tensor,
    depth_h: Optional[torch.Tensor] = None,
    z_recon: Optional[torch.Tensor] = None,
    contrastive_head: Optional[ReconContrastiveHead] = None,
    ordinal_loss_type: str = "huber",
    ordinal_huber_delta: float = 0.5,
    ordinal_magnitude_weighted: bool = False,
    ordinal_zero_drift_margin: float = 0.0,
    ordinal_zero_drift_threshold: float = 0.0,
    soft_atom_hard_mix: float = 0.5,
    infonce_tau: float = 0.07,
    infonce_max_pairs: int = 4096,
) -> dict:
    """Compute all three auxiliary losses from the prior's logits.

    This operates on the **quantised** (interleaved atom / coeff-bin)
    token format only.

    Args:
        logits:            ``[B, L, D, V]`` full-vocab logits from the prior.
        tokens:            ``[B, L, D]`` ground-truth interleaved tokens.
        atom_vocab_size:   K (number of dictionary atoms).
        coeff_vocab_size:  number of coefficient bins.
        sim_cache:         :class:`DictionarySimCache` for soft atom targets.
        bin_values:        ``[n_bins]`` dequantised bin centres.
        depth_h:           ``[B, L, D, d_model]`` depth hidden states
                           (needed only for InfoNCE).
        z_recon:           ``[B, L, C]`` ground-truth reconstruction
                           (needed only for InfoNCE).
        contrastive_head:  :class:`ReconContrastiveHead` (needed only for
                           InfoNCE).

    Returns:
        Dictionary with keys ``"soft_atom_ce"``, ``"ordinal_coeff"``,
        and optionally ``"infonce"``.  All values are scalar tensors.
    """
    out: dict[str, torch.Tensor] = {}

    # --- atom logits / targets (even depth positions) ---
    atom_logits = logits[:, :, 0::2, :atom_vocab_size]          # [B, L, s, K]
    atom_ids = tokens[:, :, 0::2]                               # [B, L, s]

    out["soft_atom_ce"] = soft_atom_ce(atom_logits, atom_ids, sim_cache, hard_mix=soft_atom_hard_mix)

    # --- coeff logits / targets (odd depth positions) ---
    coeff_logits = logits[
        :, :, 1::2, atom_vocab_size : atom_vocab_size + coeff_vocab_size
    ]                                                           # [B, L, s, n_bins]
    coeff_ids = tokens[:, :, 1::2] - atom_vocab_size            # [B, L, s]

    out["ordinal_coeff"] = ordinal_coeff_loss(
        coeff_logits,
        coeff_ids,
        bin_values,
        loss_type=ordinal_loss_type,
        huber_delta=ordinal_huber_delta,
        magnitude_weighted=ordinal_magnitude_weighted,
        zero_drift_margin=ordinal_zero_drift_margin,
        zero_drift_threshold=ordinal_zero_drift_threshold,
    )

    # --- reconstruction contrastive (optional) ---
    if depth_h is not None and z_recon is not None and contrastive_head is not None:
        out["infonce"] = contrastive_head(
            depth_h, z_recon, tau=infonce_tau, max_pairs=infonce_max_pairs,
        )

    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    K, n_bins, C, d_model = 1024, 16, 16, 256
    B, L, s = 4, 64, 8
    D = 2 * s
    V = K + n_bins + 2

    dictionary = F.normalize(torch.randn(C, K), p=2, dim=0)
    bin_values = torch.linspace(-3.0, 3.0, n_bins)

    # --- test DictionarySimCache ---
    sim_cache = DictionarySimCache(dictionary, tau=0.07)
    ids = torch.randint(0, K, (B, L, s))
    soft = sim_cache.soft_targets(ids)
    assert soft.shape == (B, L, s, K)
    assert torch.allclose(soft.sum(-1), torch.ones(B, L, s), atol=1e-5)
    print(f"DictionarySimCache OK  soft_targets {tuple(soft.shape)}")

    # --- test soft_atom_ce ---
    atom_logits = torch.randn(B, L, s, K)
    loss_soft = soft_atom_ce(atom_logits, ids, sim_cache)
    assert loss_soft.shape == ()
    print(f"soft_atom_ce = {loss_soft.item():.4f}")

    # --- test ordinal_coeff_loss ---
    coeff_logits = torch.randn(B, L, s, n_bins)
    coeff_ids = torch.randint(0, n_bins, (B, L, s))
    loss_ord = ordinal_coeff_loss(coeff_logits, coeff_ids, bin_values)
    assert loss_ord.shape == ()
    print(f"ordinal_coeff = {loss_ord.item():.4f}")

    # --- test ReconContrastiveHead ---
    head = ReconContrastiveHead(d_model, C, proj_dim=128)
    depth_h = torch.randn(B, L, D, d_model)
    z_recon = torch.randn(B, L, C)
    loss_nce = head(depth_h, z_recon, tau=0.07, max_pairs=256)
    assert loss_nce.shape == ()
    print(f"infonce = {loss_nce.item():.4f}")

    # --- test combined ---
    logits = torch.randn(B, L, D, V)
    tokens = torch.zeros(B, L, D, dtype=torch.long)
    tokens[:, :, 0::2] = torch.randint(0, K, (B, L, s))
    tokens[:, :, 1::2] = torch.randint(0, n_bins, (B, L, s)) + K

    losses = compute_contrastive_sparse_losses(
        logits=logits,
        tokens=tokens,
        atom_vocab_size=K,
        coeff_vocab_size=n_bins,
        sim_cache=sim_cache,
        bin_values=bin_values,
        depth_h=depth_h,
        z_recon=z_recon,
        contrastive_head=head,
    )
    for name, val in losses.items():
        print(f"  {name} = {val.item():.4f}")

    print("Smoke test passed.")
