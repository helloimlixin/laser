"""Metrics compatible with the vendored KakaoBrain RQ-VAE implementation.

The upstream project does not use TorchMetrics' Inception network.  It uses
the TensorFlow-FID-compatible weights vendored in ``rqvae/metrics/inception.py``
for both FID features and Inception Score logits.  This module keeps that
backend and the upstream equations while adding distributed sufficient-stat
aggregation so a 50k-image evaluation does not gather images or activations.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy import linalg


ROOT = Path(__file__).resolve().parents[1]
INCEPTION_PATH = (
    ROOT
    / "third_party"
    / "rq-vae-transformer"
    / "rqvae"
    / "metrics"
    / "inception.py"
)


def _load_upstream_inception_module():
    """Load only upstream Inception, bypassing rqvae.metrics' CLIP import."""
    spec = importlib.util.spec_from_file_location(
        "laser_original_rqvae_inception", INCEPTION_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream RQ-VAE Inception from {INCEPTION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OriginalRQVAEInception(torch.nn.Module):
    """The exact feature/logit network used by the upstream RQ-VAE metrics."""

    def __init__(self):
        super().__init__()
        module = _load_upstream_inception_module()
        block = module.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = module.InceptionV3([block])

    def forward(self, images: torch.Tensor, *, return_logits: bool = False):
        outputs = self.model(images, return_logits=return_logits)
        if return_logits:
            feature_maps, logits = outputs
        else:
            feature_maps, logits = outputs, None
        features = feature_maps[0]
        if features.size(2) != 1 or features.size(3) != 1:
            features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
        features = features.reshape(features.shape[0], -1)
        return features, logits


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Upstream RQ-VAE/pytorch-fid Frechet equation, including safeguards."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape:
        raise ValueError("training and generated mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("training and generated covariances have different dimensions")

    diff = mu1 - mu2
    covariance_product = sigma1.dot(sigma2)
    # SciPy 1.18 removed ``disp``; older versions return (sqrt, error).
    try:
        sqrtm_result = linalg.sqrtm(covariance_product, disp=False)
    except TypeError:
        sqrtm_result = linalg.sqrtm(covariance_product)
    covmean = sqrtm_result[0] if isinstance(sqrtm_result, tuple) else sqrtm_result
    if not np.isfinite(covmean).all():
        logging.warning("FID covariance product is singular; adding %s to diagonals", eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary FID component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covmean)
    )


def _mean_covariance(total, cross, count: int):
    if count < 2:
        raise ValueError("FID requires at least two samples")
    total = total.cpu()
    cross = cross.cpu()
    mean = total / float(count)
    covariance = (
        cross - torch.outer(total, total) / float(count)
    ) / float(count - 1)
    return mean.numpy(), covariance.numpy()


def inception_score(probabilities: torch.Tensor, splits: int = 10):
    """Exact 10-split equation from ``rqvae/metrics/IS.py``."""
    scores = []
    num_samples = probabilities.shape[0]
    if num_samples < splits:
        raise ValueError(f"Inception Score needs at least {splits} samples")
    for split in range(splits):
        part = probabilities[
            split * num_samples // splits:(split + 1) * num_samples // splits
        ]
        marginal = torch.mean(part, dim=0, keepdim=True)
        kl = part * (torch.log(part) - torch.log(marginal))
        scores.append(torch.exp(torch.mean(torch.sum(kl, dim=1))).unsqueeze(0))
    scores = torch.cat(scores)
    # torch.std defaults to the same sample-standard-deviation correction used
    # by the original implementation.
    return float(torch.mean(scores).item()), float(torch.std(scores).item())


def load_reference_statistics(path, *, expected_dimension: int = 2048):
    """Load and validate an upstream RQ-VAE ``mu``/``sigma`` statistics file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FID reference statistics do not exist: {path}")
    with np.load(path) as payload:
        if set(payload.files) != {"mu", "sigma"}:
            raise ValueError(
                f"FID statistics {path} need exactly mu and sigma; got {payload.files}"
            )
        mean = np.asarray(payload["mu"])
        covariance = np.asarray(payload["sigma"])
    expected_mean = (int(expected_dimension),)
    expected_covariance = (int(expected_dimension), int(expected_dimension))
    if mean.shape != expected_mean or covariance.shape != expected_covariance:
        raise ValueError(
            f"FID statistics {path} have mu={mean.shape}, sigma={covariance.shape}; "
            f"expected {expected_mean} and {expected_covariance}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError(f"FID statistics contain non-finite values: {path}")
    return mean, covariance


class DistributedOriginalRQVAEMetrics:
    """Accumulate original RQ-VAE FID and optional ImageNet IS metrics."""

    feature_dim = 2048
    num_logits = 1008

    def __init__(self, device: torch.device, *, compute_inception_score: bool = False,
                 reference_stats_path=None):
        self.device = torch.device(device)
        self.inception = OriginalRQVAEInception().to(self.device).eval()
        self.compute_inception_score = bool(compute_inception_score)
        self.reference_stats_path = (
            None if reference_stats_path is None else Path(reference_stats_path)
        )
        if self.reference_stats_path is not None and not self.reference_stats_path.is_file():
            raise FileNotFoundError(
                f"FID reference statistics do not exist: {self.reference_stats_path}"
            )
        self.real_sum = torch.zeros(self.feature_dim, dtype=torch.float64, device=self.device)
        self.fake_sum = torch.zeros_like(self.real_sum)
        self.real_cross = torch.zeros(
            self.feature_dim, self.feature_dim, dtype=torch.float64, device=self.device
        )
        self.fake_cross = torch.zeros_like(self.real_cross)
        self.real_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.fake_count = torch.zeros_like(self.real_count)
        self.fake_probabilities = []

    @torch.no_grad()
    def update(self, images: torch.Tensor, *, real: bool):
        images = images.detach().to(device=self.device, dtype=torch.float32)
        features, logits = self.inception(
            images, return_logits=(self.compute_inception_score and not real)
        )
        features = features.to(dtype=torch.float64)
        if real:
            self.real_sum.add_(features.sum(dim=0))
            self.real_cross.addmm_(features.t(), features)
            self.real_count.add_(features.shape[0])
        else:
            self.fake_sum.add_(features.sum(dim=0))
            self.fake_cross.addmm_(features.t(), features)
            self.fake_count.add_(features.shape[0])
            if self.compute_inception_score:
                self.fake_probabilities.append(logits.softmax(dim=1).cpu())

    def _reduce_fid_state(self):
        values = [self.fake_sum, self.fake_cross, self.fake_count]
        if self.reference_stats_path is None:
            values[:0] = [self.real_sum, self.real_cross, self.real_count]
        if dist.is_initialized():
            for value in values:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

    def _gather_probabilities(self):
        local = (
            torch.cat(self.fake_probabilities, dim=0)
            if self.fake_probabilities
            else torch.empty(0, self.num_logits)
        ).to(self.device)
        if not dist.is_initialized():
            return local

        world = dist.get_world_size()
        process_rank = dist.get_rank()
        local_size = torch.tensor([local.shape[0]], dtype=torch.long, device=self.device)
        sizes = [torch.zeros_like(local_size) for _ in range(world)]
        dist.all_gather(sizes, local_size)
        lengths = [int(size.item()) for size in sizes]
        max_length = max(lengths)
        padded = torch.zeros(
            max_length, self.num_logits, dtype=local.dtype, device=self.device
        )
        padded[:local.shape[0]].copy_(local)
        gathered = [torch.empty_like(padded) for _ in range(world)]
        dist.all_gather(gathered, padded)
        if process_rank != 0:
            return None
        return torch.cat(
            [rank_values[:length] for rank_values, length in zip(gathered, lengths)], dim=0
        )

    @torch.no_grad()
    def compute(self, *, shuffle_for_inception: bool = True):
        """Return ``(fid, is_mean, is_std)`` on every distributed rank."""
        self._reduce_fid_state()
        process_rank = dist.get_rank() if dist.is_initialized() else 0
        result = torch.zeros(3, dtype=torch.float64, device=self.device)
        if process_rank == 0:
            fake_count = int(self.fake_count.item())
            if self.reference_stats_path is None:
                real_count = int(self.real_count.item())
                mu_real, covariance_real = _mean_covariance(
                    self.real_sum, self.real_cross, real_count
                )
            else:
                mu_real, covariance_real = load_reference_statistics(
                    self.reference_stats_path, expected_dimension=self.feature_dim
                )
            mu_fake, covariance_fake = _mean_covariance(
                self.fake_sum, self.fake_cross, fake_count
            )
            result[0] = frechet_distance(
                mu_real, covariance_real, mu_fake, covariance_fake
            )

        probabilities = None
        if self.compute_inception_score:
            probabilities = self._gather_probabilities()
            if process_rank == 0:
                if shuffle_for_inception:
                    # Upstream uses DataLoader(..., shuffle=True) before its
                    # ten contiguous splits.
                    probabilities = probabilities[
                        torch.randperm(probabilities.shape[0], device=self.device)
                    ]
                result[1], result[2] = inception_score(probabilities, splits=10)

        if dist.is_initialized():
            dist.broadcast(result, src=0)
        fid = float(result[0].item())
        if not self.compute_inception_score:
            return fid, None, None
        return fid, float(result[1].item()), float(result[2].item())
