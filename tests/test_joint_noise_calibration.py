import math

import numpy as np
import pytest
import torch

from src.training.joint_noise_calibration import (
    align_noise_budgets, fit_gap_temperature, hard_trajectory, measure_noise,
)


def test_budget_transfer_preserves_total_and_splits_added_depth():
    result = align_noise_budgets([256, 64, 16, 4, 1], [.1, .2, .3, .4],
                                [256, 64, 16, 4, 2, 1])
    np.testing.assert_allclose(result['budgets'], [.1, .2, .3, .2, .2])
    assert sum(result['budgets']) == pytest.approx(1.)
    # Changing the units of either latent space cannot change the mapping.
    rescaled = align_noise_budgets([2560, 640, 160, 40, 10], [.1, .2, .3, .4],
                                  [512, 128, 32, 8, 4, 2])
    np.testing.assert_allclose(result['budgets'], rescaled['budgets'])


def test_two_code_temperature_has_known_closed_form():
    # p(second)=1/4 for a distance gap of 2 implies T=2/log(3).
    temperature, actual = fit_gap_temperature(torch.tensor([[0., 2.]]), .5)
    assert temperature == pytest.approx(2/math.log(3), rel=1.e-6)
    assert actual == pytest.approx(.5, rel=1.e-6)


def test_unattainable_noise_rejected():
    with pytest.raises(ValueError, match='attainable'):
        fit_gap_temperature(torch.tensor([[0., 2.]]), 1.5)


class ToyBook:
    depth = 2
    book = torch.tensor([[0.], [1.]])

    def scores(self, residual, depth):
        return -(residual-self.book.T).square()

    def embed_depth(self, codes, depth):
        return self.book[codes]


def test_stochastic_prefix_profile_matches_decoded_total_error():
    x = torch.linspace(.05, 1.95, 512).reshape(8, 8, 8, 1)
    quantizer = ToyBook()
    hard = hard_trajectory(quantizer, x)
    profile = measure_noise(quantizer, x, [.2, .1], [3, 5], hard, chunk_size=64)
    hard_decoded = quantizer.book[hard['codes']].sum(-2)
    hard_error = (x-hard_decoded).square().mean()
    for codes, ratio in zip(profile['sampled_codes'], profile['ratio_by_seed']):
        decoded = quantizer.book[codes].sum(-2)
        assert ratio == pytest.approx(float((x-decoded).square().mean()/hard_error), rel=1.e-6)
    assert all(row['extra_squared_error_per_site'] > 0 for row in profile['profiles'])
