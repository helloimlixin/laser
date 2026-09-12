import torch
from torch.nn import functional as F

from src.sign_pattern_prior import (
    SignPatternPrior, pack_signs, pattern_log_probabilities, sign_patterns, sign_metrics,
)


def tiny_model(mode="joint"):
    torch.manual_seed(17)
    return SignPatternPrior(F.normalize(torch.randn(6, 12), dim=0), torch.ones(4),
                            sites=3, width=16, layers=2, heads=2, dropout=0, mode=mode).eval()


def inputs():
    atoms = torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]])
    coefficients = torch.tensor([[[1., -2, 3, -4], [-1, 2, -3, 4], [2, -3, 4, -1]]])
    return atoms, coefficients


def test_all_16_patterns_round_trip_and_normalize():
    patterns = sign_patterns(4)
    assert torch.equal(pack_signs(patterns), torch.arange(16))
    for mode, size in [("joint", 16), ("independent", 4)]:
        logp = pattern_log_probabilities(torch.randn(3, size), mode, 4)
        torch.testing.assert_close(logp.exp().sum(-1), torch.ones(3))


def test_current_and_future_signs_cannot_leak_to_predictions():
    atoms, coefficients = inputs()
    for mode in ["joint", "independent"]:
        model = tiny_model(mode)
        baseline = model(atoms, coefficients.abs(), coefficients)
        changed = coefficients.clone()
        changed[:, 1:] *= -1
        altered = model(atoms, coefficients.abs(), changed)
        torch.testing.assert_close(baseline[:, :2], altered[:, :2], rtol=0, atol=0)
        assert not torch.allclose(baseline[:, 2], altered[:, 2])


def test_later_oracle_support_and_magnitude_do_not_leak_backward():
    atoms, coefficients = inputs()
    model = tiny_model()
    baseline = model(atoms, coefficients.abs(), coefficients)
    other_atoms = atoms.clone()
    other_atoms[:, 2] = other_atoms[:, 2].flip(-1)
    magnitudes = coefficients.abs().clone()
    magnitudes[:, 2] *= 3
    other = model(other_atoms, magnitudes, coefficients)
    torch.testing.assert_close(baseline[:, :2], other[:, :2], rtol=0, atol=0)


def test_both_arms_start_with_identical_backbones_and_observations():
    joint, independent = tiny_model(), tiny_model("independent")
    for name, value in joint.state_dict().items():
        if not name.startswith("head."):
            torch.testing.assert_close(value, independent.state_dict()[name], rtol=0, atol=0)


def test_joint_head_can_represent_correlated_signs_without_impossible_tuples():
    # Equal mass on ---- and ++++ has uncertain marginals but perfect correlation.
    joint = torch.full((1, 16), -100.)
    joint[:, [0, 15]] = 0
    joint_p = pattern_log_probabilities(joint, "joint", 4).exp()
    independent_p = pattern_log_probabilities(torch.zeros(1, 4), "independent", 4).exp()
    assert joint_p[:, 1:15].sum() < 1e-6
    torch.testing.assert_close(independent_p[:, 1:15].sum(), torch.tensor(0.875))


def test_latent_error_accounts_for_cancellation_and_magnitude():
    dictionary = torch.tensor([[1., 1.]])
    atoms = torch.tensor([[[0, 1]]])
    coefficients = torch.tensor([[[2., -2.]]])
    # Flip both signs: coefficient error is large but the zero latent is unchanged.
    logits = torch.full((1, 1, 4), -100.)
    logits[..., 2] = 100
    metrics = sign_metrics(logits, coefficients, atoms, dictionary, "joint")
    assert metrics["sign_accuracy"].item() == 0
    assert metrics["physical_coefficient_mse"].item() == 16
    assert metrics["latent_mse"].item() == 0


def test_rollout_uses_generated_signs_without_observing_targets():
    model = tiny_model()
    atoms, coefficients = inputs()
    generated = model.rollout(atoms, coefficients.abs())
    torch.testing.assert_close(generated.abs(), coefficients.abs())
    history = torch.zeros_like(coefficients)
    for site in range(3):
        logits = model(atoms[:, :site + 1], coefficients[:, :site + 1].abs(), history[:, :site + 1])[:, -1]
        signs = sign_patterns(4)[logits.argmax(-1)].float() * 2 - 1
        history[:, site] = coefficients[:, site].abs() * signs
    torch.testing.assert_close(generated, history)
