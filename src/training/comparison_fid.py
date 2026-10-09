"""Score the same generated pixels against official and historical references."""
from pathlib import Path

import torch
import torch.distributed as dist
from torchmetrics.image.fid import FrechetInceptionDistance

from src.rqvae_metrics import (
    DistributedOriginalRQVAEMetrics, _mean_covariance,
    frechet_distance, load_reference_statistics,
)
from src.training.fid_reference import seed_torchmetrics_reference


class ComparisonFIDMetrics(DistributedOriginalRQVAEMetrics):
    """Score both protocols and explicitly select the checkpoint-ranking FID."""

    def __init__(self, *args, validation_reference, torchmetrics_reference,
                 on_comparison=None, selection_metric='original_train', **kwargs):
        if selection_metric not in ('original_train', 'torchmetrics_val50k'):
            raise ValueError('Unknown checkpoint-selection FID metric')
        super().__init__(*args, **kwargs)
        self.selection_metric = selection_metric
        self.validation_reference = Path(validation_reference)
        self.on_comparison = on_comparison
        self.companion = FrechetInceptionDistance(feature=2048, normalize=True,
            sync_on_compute=dist.is_initialized()).to(self.device).eval()
        seed_torchmetrics_reference(self.companion, torchmetrics_reference,
            rank=dist.get_rank() if dist.is_initialized() else 0)

    @torch.no_grad()
    def update(self, images, *, real):
        super().update(images, real=real)
        if not real:
            self.companion.update(images, real=False)

    def compute(self, **kwargs):
        # The parent synchronizes original fake moments exactly once.
        primary = super().compute(**kwargs)
        historical = float(self.companion.compute())
        if not dist.is_initialized() or dist.get_rank() == 0:
            real_mean, real_cov = load_reference_statistics(self.validation_reference)
            fake_mean, fake_cov = _mean_covariance(self.fake_sum, self.fake_cross, int(self.fake_count))
            original_validation = frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
            result = {'eval/fid_original_train50k': primary[0],
                      'eval/fid_original_val50k': original_validation,
                      'eval/fid_torchmetrics_val50k': historical,
                      'eval/comparison_real_images': 50000,
                      'eval/comparison_generated_images': int(self.fake_count)}
            if self.on_comparison is not None:
                self.on_comparison(result)
        if self.selection_metric == 'torchmetrics_val50k':
            return historical, primary[1], primary[2]
        return primary
