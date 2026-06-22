"""Anti-collapse stage-2 levers for the real-valued spatial_depth atom head.

Covers the two gated knobs added to SparseTokenPriorModule:
  * atom_label_smoothing -> passed to the atom-id cross-entropy
  * atom_coverage_weight -> rewards a high-entropy batch-marginal atom
    distribution (fights generation collapse onto a few atoms)
plus the always-on collapse diagnostics.
"""

import math

import torch
import torch.nn.functional as F

from src.models.sparse_token_prior import SparseTokenPriorModule


class _UniformRealPrior(torch.nn.Module):
    """Real-valued prior stub that emits uniform atom logits (vocab=3)."""

    real_valued_coeffs = True
    gaussian_coeffs = False

    def __init__(self):
        super().__init__()
        self.cfg = type("Cfg", (), {"H": 1, "W": 1, "D": 2, "vocab_size": 3})()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tok_grid, coeff_grid, mask_tokens=None, return_features=False):
        bsz, steps, depth = tok_grid.shape
        logits = torch.zeros(bsz, steps, depth, int(self.cfg.vocab_size))
        coeff_pred = torch.zeros_like(coeff_grid) + self.anchor
        return logits, coeff_pred


class _MaskedRealPrior(torch.nn.Module):
    """Real-valued prior stub that masks invalid atoms to -inf (vocab=4).

    Mirrors the spatial_depth prior, which sets invalid / already-selected atom
    logits to -inf. Targets stay within the valid {0,1} classes.
    """

    real_valued_coeffs = True
    gaussian_coeffs = False

    def __init__(self):
        super().__init__()
        self.cfg = type("Cfg", (), {"H": 1, "W": 1, "D": 2, "vocab_size": 4})()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tok_grid, coeff_grid, mask_tokens=None, return_features=False):
        bsz, steps, depth = tok_grid.shape
        logits = torch.zeros(bsz, steps, depth, int(self.cfg.vocab_size))
        logits[..., 2:] = float("-inf")  # classes 2,3 are invalid -> masked
        coeff_pred = torch.zeros_like(coeff_grid) + self.anchor
        return logits, coeff_pred


def _batch():
    return (
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.tensor([[1.0, -1.0]], dtype=torch.float32),
    )


def test_defaults_are_off_and_match_legacy_loss():
    mod = SparseTokenPriorModule(prior=_UniformRealPrior(), coeff_loss_type="mse")
    assert mod.atom_label_smoothing == 0.0
    assert mod.atom_coverage_weight == 0.0
    mod.log = lambda *a, **k: None
    loss = mod._real_valued_shared_step(_batch(), "train")
    # uniform vocab=3 CE = log(3); coeff MSE(0 vs [1,-1]) = 1
    assert torch.isclose(loss, torch.log(torch.tensor(3.0)) + 1.0)


def test_coverage_weight_rewards_marginal_entropy():
    base = SparseTokenPriorModule(prior=_UniformRealPrior(), coeff_loss_type="mse")
    cov = SparseTokenPriorModule(
        prior=_UniformRealPrior(), coeff_loss_type="mse", atom_coverage_weight=0.5
    )
    base.log = lambda *a, **k: None
    cov.log = lambda *a, **k: None
    loss_base = base._real_valued_shared_step(_batch(), "train")
    loss_cov = cov._real_valued_shared_step(_batch(), "train")
    # uniform marginal entropy = log(3); loss is reduced by weight * entropy
    assert torch.isclose(loss_base - loss_cov, torch.tensor(0.5 * math.log(3.0)))


def test_collapse_diagnostics_are_logged():
    logged = {}
    mod = SparseTokenPriorModule(prior=_UniformRealPrior(), coeff_loss_type="mse")
    mod.log = lambda name, value, *a, **k: logged.__setitem__(
        name, float(value) if torch.is_tensor(value) else value
    )
    mod._real_valued_shared_step(_batch(), "train")
    for name in (
        "train/atom_marginal_entropy",
        "train/atom_coverage_frac",
        "train/atom_pred_unique_frac",
    ):
        assert name in logged, f"missing diagnostic {name}"
    # uniform 3-way prediction -> full marginal coverage, single argmax atom (id 0)
    assert math.isclose(logged["train/atom_coverage_frac"], 1.0, abs_tol=1e-5)
    assert math.isclose(logged["train/atom_pred_unique_frac"], 1.0 / 3.0, abs_tol=1e-6)


def test_label_smoothing_stays_finite_under_masked_logits():
    """Regression: -inf-masked atom logits must not make the smoothed CE blow up."""
    # plain F.cross_entropy(label_smoothing>0) returns +inf here; the masked-safe
    # path smooths only over the valid classes and stays finite.
    mod = SparseTokenPriorModule(
        prior=_MaskedRealPrior(), coeff_loss_type="mse", atom_label_smoothing=0.1
    )
    mod.log = lambda *a, **k: None
    loss = mod._real_valued_shared_step(_batch(), "train")
    assert torch.isfinite(loss), f"smoothed CE went non-finite under masking: {loss}"
    # and it still trains: a backward yields finite gradients on the prior
    loss.backward()
    assert torch.isfinite(mod.prior.anchor.grad)


def test_coverage_gradient_increases_marginal_entropy():
    """A few ascent steps on the coverage objective spread a peaked distribution."""
    torch.manual_seed(0)
    vocab = 8
    logits = torch.zeros(64, vocab, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 6.0  # start collapsed onto atom 0

    def marginal_entropy(lg):
        p = F.softmax(lg, dim=-1).mean(dim=0).clamp_min(1e-12)
        return -(p * p.log()).sum()

    h0 = float(marginal_entropy(logits))
    opt = torch.optim.Adam([logits], lr=0.5)
    for _ in range(200):
        opt.zero_grad()
        (-marginal_entropy(logits)).backward()  # minimize -H == maximize H
        opt.step()
    h1 = float(marginal_entropy(logits))
    # collapsed init H0 ~ 0.12 nats; ascent should approach the max log(8) ~ 2.08
    assert h1 > h0 + 1.0, f"coverage objective failed to raise entropy: {h0:.3f}->{h1:.3f}"


def test_label_smoothed_ce_is_finite_with_gradients():
    torch.manual_seed(0)
    logits = torch.randn(16, 5, requires_grad=True)
    target = torch.randint(0, 5, (16,))
    loss = F.cross_entropy(logits, target, label_smoothing=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
