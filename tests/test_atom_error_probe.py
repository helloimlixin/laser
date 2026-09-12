import torch

from archive.scripts.probe_church_atom_error_propagation import (
    exclude_site_atoms, scalar_projection_coefficient, select_coefficients, restore_component,
)


def test_replacement_target_accounts_for_rescaling_and_sign():
    old = torch.tensor([[1., 2.], [1., 2.]])
    new = torch.tensor([[2., 4.], [-1., -2.]])
    coefficient = torch.tensor([3., 3.])
    adjusted = scalar_projection_coefficient(old, new, coefficient)
    torch.testing.assert_close(adjusted, torch.tensor([1.5, -3.]))
    torch.testing.assert_close(new * adjusted[:, None], old * coefficient[:, None])


def test_projection_minimizes_error_with_other_contributions_fixed():
    old = torch.tensor([[1., 1., 0.]])
    new = torch.tensor([[1., 0., 1.]])
    c = scalar_projection_coefficient(old, new, torch.tensor([-2.]))
    residual = new * c[:, None] - old * -2
    torch.testing.assert_close((residual * new).sum(-1), torch.zeros(1))
    for delta in [-.5, .5]:
        assert ((new * (c[:, None] + delta) - old * -2).square().sum() > residual.square().sum())


def test_replacements_exclude_current_and_remaining_site_supports():
    scores = torch.tensor([[4., 3., 2., 1., 0.]])
    masked = exclude_site_atoms(scores, torch.tensor([[0, 1, 2, 4]]))
    assert masked.argmax(-1).item() == 3
    assert torch.isfinite(scores).all()


def test_common_uniforms_couple_identical_coefficient_distributions():
    logits = torch.tensor([[0., 0., 0., 0.]]).repeat(4, 1)
    u = torch.tensor([0.1, 0.3, 0.6, 0.9])
    assert torch.equal(select_coefficients(logits, u), torch.arange(4))
    assert torch.equal(select_coefficients(logits, u), select_coefficients(logits + 10, u))


def test_oracles_preserve_the_unmodified_physical_component():
    bins = torch.linspace(-20, 20, 2048)
    prediction = torch.arange(2048)
    truth = torch.roll(prediction, 731)
    signed = restore_component(prediction, truth, "sign")
    magnitude = restore_component(prediction, truth, "magnitude")
    assert torch.equal(signed >= 1024, truth >= 1024)
    assert torch.equal(magnitude >= 1024, prediction >= 1024)
    torch.testing.assert_close(bins[signed].abs(), bins[prediction].abs())
    torch.testing.assert_close(bins[magnitude].abs(), bins[truth].abs())
