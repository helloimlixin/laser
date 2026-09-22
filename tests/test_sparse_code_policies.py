import torch
import pytest

from src.training.sparse_code_policies import coefficient_targets, DepthAtomSampler


def test_original_center_measure_is_exact_and_replays_samples():
    values = torch.tensor([[-2.3, .4, 1.7, 5.]])
    centers = torch.tensor([-5., -1., -.2, 0., .1, .3, 1., 6.])
    expected = (-(values[..., None] - centers).square() / .125).softmax(-1)
    generator = torch.Generator().manual_seed(32)
    state = generator.get_state()
    ids, probabilities = coefficient_targets(values, centers, .125, generator=generator)
    torch.testing.assert_close(probabilities, expected, rtol=0, atol=0)
    generator.set_state(state)
    assert torch.equal(ids, torch.multinomial(expected.reshape(-1, len(centers)), 1,
                                            generator=generator).reshape(values.shape))


def test_cell_measure_removes_dense_center_bias_and_handles_tails():
    # Deliberately place many more centers on one side of zero.
    centers = torch.cat((torch.linspace(-3, -.01, 100), torch.linspace(0, 3, 601)))
    values = torch.tensor([0., -100., 100.])
    _, old = coefficient_targets(values, centers, .125, stochastic=False)
    _, corrected = coefficient_targets(values, centers, .125, measure='cells', stochastic=False)
    assert abs(float((old[0] * centers).sum())) > .05
    assert abs(float((corrected[0] * centers).sum())) < .002
    torch.testing.assert_close(corrected.sum(-1), torch.ones(3))
    assert corrected[1, 0] == 1 and corrected[2, -1] == 1
    assert torch.isfinite(corrected).all() and (corrected >= 0).all()


def test_temperature_reduction_reduces_physical_variance():
    centers = torch.linspace(-3, 3, 2048)
    variances = []
    for temperature in [.125, .03125]:
        _, p = coefficient_targets(torch.zeros(4), centers, temperature, measure='cells')
        variances.append(float((p * centers.square()).sum(-1).mean()))
    assert variances[1] / variances[0] == pytest.approx(.25, abs=.001)


def test_depth_cutoffs_preserve_coefficients_and_reset_between_batches():
    seen = []
    def sample(logits, **kwargs):
        seen.append(kwargs['top_k'])
        return torch.zeros(len(logits), dtype=torch.long)
    sampler = DepthAtomSampler(sample, 16, [2, 4, 8, 16])
    for _ in range(8):
        sampler(torch.zeros(3, 16), top_k=2)
        sampler(torch.zeros(3, 5), top_k=5)
    assert seen == [2, 5, 4, 5, 8, 5, 16, 5] * 2
    sampler.reset()
    sampler(torch.zeros(1, 16), top_k=99)
    assert seen[-1] == 2


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        coefficient_targets(torch.zeros(1), torch.tensor([0., 1.]), -1)
    with pytest.raises(ValueError):
        DepthAtomSampler(lambda *a, **k: None, 16, [0, 17])
