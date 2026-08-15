import numpy as np
import pytest
import torch

from src.rqvae_metrics import (
    DistributedOriginalRQVAEMetrics,
    load_reference_statistics,
)


def test_load_reference_statistics_accepts_upstream_mu_sigma(tmp_path):
    path = tmp_path / "stats.npz"
    expected_mu = np.array([1.0, 2.0], dtype=np.float32)
    expected_sigma = np.eye(2, dtype=np.float64)
    np.savez(path, mu=expected_mu, sigma=expected_sigma)

    mu, sigma = load_reference_statistics(path, expected_dimension=2)

    np.testing.assert_array_equal(mu, expected_mu)
    np.testing.assert_array_equal(sigma, expected_sigma)


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"mu": np.zeros(2), "sigma": np.eye(2), "acts": np.zeros((1, 2))},
         "exactly mu and sigma"),
        ({"mu": np.zeros(3), "sigma": np.eye(3)}, "expected"),
        ({"mu": np.array([np.nan, 0.0]), "sigma": np.eye(2)}, "non-finite"),
    ],
)
def test_load_reference_statistics_rejects_incompatible_files(
    tmp_path, payload, match
):
    path = tmp_path / "bad_stats.npz"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=match):
        load_reference_statistics(path, expected_dimension=2)


def test_distributed_metric_uses_precomputed_reference_instead_of_real_state(
    tmp_path,
):
    path = tmp_path / "stats.npz"
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    np.savez(path, mu=np.zeros(2, dtype=np.float32), sigma=covariance)

    # Avoid constructing the heavyweight Inception network: compute() only
    # needs accumulated sufficient statistics at this point.
    metric = DistributedOriginalRQVAEMetrics.__new__(
        DistributedOriginalRQVAEMetrics
    )
    metric.device = torch.device("cpu")
    metric.feature_dim = 2
    metric.compute_inception_score = False
    metric.reference_stats_path = path
    metric.fake_sum = torch.zeros(2, dtype=torch.float64)
    metric.fake_cross = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    metric.fake_count = torch.tensor(3, dtype=torch.long)
    # These deliberately invalid online-real statistics must be ignored.
    metric.real_sum = torch.full((2,), torch.nan, dtype=torch.float64)
    metric.real_cross = torch.full((2, 2), torch.nan, dtype=torch.float64)
    metric.real_count = torch.tensor(0, dtype=torch.long)

    fid, inception_mean, inception_std = metric.compute()

    assert fid == pytest.approx(0.0, abs=1e-10)
    assert inception_mean is None
    assert inception_std is None
