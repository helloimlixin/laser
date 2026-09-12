import torch

from scripts.audit_church_noise_scale import expected_latent_error


def test_expected_error_matches_biased_discrete_sampling_with_correlated_atoms():
    # Nonorthogonal supports and asymmetric discrete draws exercise both the
    # cross terms and truncation bias that a simple sum(sigma²) misses.
    dictionary = torch.tensor([[1., .8], [0., .6]], dtype=torch.float64)
    gram = dictionary.t() @ dictionary
    values = torch.tensor([[-.2, .3, .7], [-.4, .1, .6]], dtype=torch.float64)
    probs = torch.tensor([[.1, .6, .3], [.5, .4, .1]], dtype=torch.float64)
    mean = (values*probs).sum(-1)
    var = (probs*(values-mean[:, None]).square()).sum(-1)
    expected = expected_latent_error(gram, mean, var)
    generator = torch.Generator().manual_seed(93)
    draws = torch.multinomial(probs, 200000, replacement=True, generator=generator)
    errors = values.gather(-1, draws)
    actual = (dictionary @ errors).square().sum(0).mean()
    torch.testing.assert_close(actual, expected, rtol=.012, atol=0.)


def test_inactive_dictionary_columns_do_not_change_coefficient_noise_budget():
    torch.manual_seed(2)
    dictionary = torch.nn.functional.normalize(torch.randn(16, 32), dim=0)
    expanded = torch.cat([dictionary, torch.randn(16, 224)], dim=1)
    support = torch.tensor([3, 9, 15, 2])
    errors = torch.full((4,), .1875**2)
    def energy(d):
        active = d[:, support]
        return expected_latent_error(active.t() @ active, torch.zeros(4), errors)
    torch.testing.assert_close(energy(dictionary), energy(expanded), rtol=0, atol=0)
    torch.testing.assert_close(energy(dictionary), torch.tensor(4*.1875**2))


def test_sparsity_changes_total_noise_and_sqrt_depth_scaling_restores_budget():
    torch.manual_seed(3)
    dictionary = torch.nn.functional.normalize(torch.randn(16, 8), dim=0)
    def energy(depth, sigma):
        d = dictionary[:, :depth]
        return expected_latent_error(d.t() @ d, torch.zeros(depth), torch.full((depth,), sigma**2))
    torch.testing.assert_close(energy(4, .2), 2*energy(2, .2))
    torch.testing.assert_close(energy(4, .2/2**.5), energy(2, .2))
