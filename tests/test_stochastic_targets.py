import math

import pytest
import torch

from src.compact_rq_training import TrainingAdaptiveScaledAtomRQ, DepthAdaptiveScaledAtomRQ
from src.training.stochastic_targets import (compact_soft_codes, depth_temperatures,
    calibrate_depth_temperatures, install_compact_target_policy)


def make_quantizer(depth_specific=False):
    torch.manual_seed(271)
    dictionary = torch.randn(5, 9)
    levels = torch.rand(9, 2).add(.2) * torch.tensor([-1., 1.])
    if depth_specific:
        return DepthAdaptiveScaledAtomRQ(dictionary, torch.stack([levels * s for s in [1., .8, .5, .2]]))
    return TrainingAdaptiveScaledAtomRQ(dictionary, levels, depth=4)


@pytest.mark.parametrize('depth_specific', [False, True])
def test_depth_temperatures_match_dense_codeword_distances_on_sampled_prefixes(depth_specific):
    q = make_quantizer(depth_specific)
    x = torch.randn(2, 2, 2, 5)
    temperatures = [.15, .4, .8, 1.3]
    torch.manual_seed(719)
    probability, codes = q.get_soft_codes(x, temp=temperatures, chunk_size=128)
    residual = x.reshape(-1, 5).clone()
    for depth, temperature in enumerate(temperatures):
        book = q.codebooks[depth] if depth_specific else q
        entries = book.expanded_codebook()
        dense = -((residual[:, None] - entries[None]) ** 2).sum(-1)
        expected = (dense / temperature).softmax(-1).reshape(2, 2, 2, -1)
        torch.testing.assert_close(probability[..., depth, :], expected, atol=3e-6, rtol=3e-5)
        residual -= book.embed(codes[..., depth].reshape(-1))
    assert not probability.requires_grad


def test_changing_later_temperature_cannot_change_earlier_targets_or_sampled_prefix():
    q = make_quantizer()
    x = torch.randn(3, 2, 2, 5)
    torch.manual_seed(28)
    p, c = q.get_soft_codes(x, temp=[.2, .3, .4, .5])
    torch.manual_seed(28)
    other_p, other_c = q.get_soft_codes(x, temp=[.2, .3, 1., 2.])
    torch.testing.assert_close(p[..., :2, :], other_p[..., :2, :], rtol=0, atol=0)
    assert torch.equal(c[..., :2], other_c[..., :2])
    assert not torch.allclose(p[..., 2, :], other_p[..., 2, :])


def test_scalar_and_repeated_depth_temperatures_preserve_rng_and_frozen_state():
    q = make_quantizer()
    x = torch.randn(3, 2, 2, 5)
    before = {k:v.clone() for k,v in q.state_dict().items()}
    torch.manual_seed(119)
    p, c = q.get_soft_codes(x, temp=.25, chunk_size=5)
    rng = torch.get_rng_state()
    install_compact_target_policy(q)
    torch.manual_seed(119)
    p2, c2 = q.get_soft_codes(x, temp=[.25]*4, chunk_size=5)
    torch.testing.assert_close(p, p2, rtol=0, atol=0)
    assert torch.equal(c, c2) and torch.equal(torch.get_rng_state(), rng)
    for k,v in q.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_target_geometry_remains_fp32_under_ambient_autocast():
    q = make_quantizer()
    x = torch.randn(3, 2, 2, 5)
    expected = q.get_soft_codes(x, temp=[.15,.2,.3,.5], stochastic=False)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = q.get_soft_codes(x, temp=[.15,.2,.3,.5], stochastic=False)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[0].dtype == torch.float32 and torch.equal(actual[1], expected[1])


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf'), True,
    [.1,.2], [.1,.2,.3,0], [.1,.2,float('nan'),.4], [.1,.2,False,.4]])
def test_invalid_temperature_fails_before_sampling(temperature):
    q = make_quantizer()
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match='temperature|Temperature'):
        q.get_soft_codes(torch.ones(1,5), temp=temperature)
    assert torch.equal(rng, torch.get_rng_state())


def test_calibration_matches_entropy_on_stochastic_prefixes_and_preserves_rng():
    q = make_quantizer()
    x = torch.randn(64,5)
    goals = [.25,.6,.9,1.2]
    rng = torch.get_rng_state()
    result = calibrate_depth_temperatures(q,x,goals,seed=221,max_temperature=20.,tolerance=.001)
    assert torch.equal(rng, torch.get_rng_state())
    torch.manual_seed(221)
    p, _ = q.get_soft_codes(x,temp=result['selected_temperature'])
    measured = -(p*p.clamp_min(1e-30).log()).sum(-1).mean(0)
    torch.testing.assert_close(measured, torch.tensor(goals), atol=.0011, rtol=0)


def test_unreachable_entropy_rejected_instead_of_silently_clipped():
    q = make_quantizer()
    with pytest.raises(ValueError, match='Unreachable entropy'):
        calibrate_depth_temperatures(q,torch.ones(32,5),[math.log(q.vocab_size)-.01]*4,
                                     min_temperature=1e-5,max_temperature=1e-4)

