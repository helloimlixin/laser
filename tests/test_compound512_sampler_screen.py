"""Check the diagnostic score against the kernel definition, without GPUs."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location('sampler_sweep',
    Path(__file__).resolve().parents[1]/'scripts/tools/sweep_compound512_samplers.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_moment_score_equals_explicit_polynomial_kernel():
    rng = np.random.default_rng(148)
    real, fake = rng.normal(size=(37, 7)), rng.normal(size=(19, 7))+.2
    center, scale = real.mean(0), real.std(0, ddof=1)
    real, fake = (real-center)/scale, (fake-center)/scale
    dimension, count = fake.shape[1], len(fake)
    fake_kernel = (1+fake@fake.T/dimension)**2
    real_kernel = (1+real@real.T/dimension)**2
    cross_kernel = (1+fake@real.T/dimension)**2
    expected = ((fake_kernel.sum()-fake_kernel.trace())/(count*(count-1))
        + real_kernel.mean()-2*cross_kernel.mean())
    actual = sweep.moment_mmd(fake.sum(0), fake.T@fake,
        np.square(np.square(fake).sum(-1)).sum(), count, real.T@real/len(real))
    np.testing.assert_allclose(actual, expected, rtol=1.e-11, atol=1.e-12)


def test_screen_keeps_ancestral_factor_sampling_and_no_atom_topk():
    policies = sweep.candidates()
    assert len(policies) == 11
    assert len({tuple(sorted(p.items())) for p in policies.values()}) == len(policies)
    for policy in policies.values():
        assert policy['atom_top_k'] == 0
        assert policy['atom_temperature'] > 0 and policy['coefficient_temperature'] > 0
        assert 0 < policy['atom_top_p'] <= 1 and 0 < policy['coefficient_top_p'] <= 1
        assert set(policy) == {'atom_temperature','atom_top_k','atom_top_p',
            'coefficient_temperature','coefficient_top_k','coefficient_top_p'}
