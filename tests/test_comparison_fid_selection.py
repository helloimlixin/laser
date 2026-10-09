from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training import comparison_fid


@pytest.mark.parametrize('selection,expected', [
    ('original_train', 12.5), ('torchmetrics_val50k', 8.75),
])
def test_checkpoint_selection_preserves_all_scores_and_inception(monkeypatch, selection, expected):
    # Exercise selection after feature accumulation without downloading or
    # instantiating an Inception network in this CPU test.
    monkeypatch.setattr(comparison_fid.DistributedOriginalRQVAEMetrics, 'compute',
                        lambda self, **kwargs: (12.5, 55., 2.))
    monkeypatch.setattr(comparison_fid, 'load_reference_statistics',
                        lambda path: (np.zeros(2), np.eye(2)))
    monkeypatch.setattr(comparison_fid, '_mean_covariance',
                        lambda *args: (np.ones(2), np.eye(2)))
    monkeypatch.setattr(comparison_fid, 'frechet_distance', lambda *args: 9.)
    logged = []
    metric = object.__new__(comparison_fid.ComparisonFIDMetrics)
    metric.selection_metric = selection
    metric.companion = SimpleNamespace(compute=lambda: torch.tensor(8.75))
    metric.validation_reference = 'unused.npz'
    metric.fake_sum = torch.zeros(2)
    metric.fake_cross = torch.zeros(2, 2)
    metric.fake_count = torch.tensor(50000)
    metric.on_comparison = logged.append

    assert metric.compute() == (expected, 55., 2.)
    assert len(logged) == 1
    assert logged[0]['eval/fid_original_train50k'] == 12.5
    assert logged[0]['eval/fid_original_val50k'] == 9.
    assert logged[0]['eval/fid_torchmetrics_val50k'] == 8.75
    assert logged[0]['eval/comparison_generated_images'] == 50000


def test_unknown_checkpoint_selection_is_rejected_before_model_loading():
    with pytest.raises(ValueError, match='checkpoint-selection'):
        comparison_fid.ComparisonFIDMetrics(selection_metric='unknown',
            validation_reference='unused.npz', torchmetrics_reference='unused.pt')
