import pytest
import torch

from src.training.joint_integer_sampling_audit import retained_probability


@pytest.mark.parametrize('top_k', [None, 1, 3, 7])
@pytest.mark.parametrize('top_p', [.5, .95, 1.])
def test_retained_mass_matches_upstream_filtering_including_ties(top_k, top_p):
    from src import rq_upstream
    from rqvae.utils.utils import top_k_logits, top_p_probs
    logits = torch.tensor([[3., 3., 1., .4, -.2, -.8, -2.],
                           [1., .9, .8, .7, .6, .5, .4]])
    cropped = logits if top_k is None else top_k_logits(logits, top_k)
    final = top_p_probs(cropped.softmax(-1), top_p)
    mass, count = retained_probability(logits, top_k=top_k, top_p=top_p)
    expected = logits.softmax(-1) * (final > 0)
    torch.testing.assert_close(mass, expected.sum(-1))
    assert torch.equal(count, (final > 0).sum(-1))


def test_nucleus_probability_is_relative_to_the_top_k_distribution():
    # top-p .95 after top-k can retain far less than .95 of original mass.
    logits = torch.zeros(1, 100)
    logits[0, :2] = 1.
    mass, count = retained_probability(logits, top_k=2, top_p=.95)
    assert int(count) == 2
    assert float(mass) < .06
