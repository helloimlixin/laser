from types import SimpleNamespace

import pytest
import torch

from src.training import rqtransformer as training
from src import rqvae_metrics


@pytest.mark.parametrize('compound', [True, False])
def test_large_generation_batch_is_decoded_in_bounded_chunks_without_reordering(monkeypatch, compound):
    sampled, decoded, features = [], [], []

    class Compound(training.CompoundLaserRQTransformer):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.cursor = 0
        def sample_compound(self, count, aux, **kwargs):
            sampled.append(count)
            ids = torch.arange(self.cursor, self.cursor+count).reshape(count, 1, 1, 1)
            self.cursor += count
            return ids, ids + 10

    class Sparse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.cursor = 0
        def sample_sparse(self, count, aux, **kwargs):
            sampled.append(count)
            ids = torch.arange(self.cursor, self.cursor+count).reshape(count, 1, 1, 1)
            self.cursor += count
            return ids

    def decode(ids, coefficients=None):
        assert len(ids) <= 64, 'decode must be chunked BEFORE allocating image activations'
        if coefficients is not None:
            assert torch.equal(coefficients, ids+10)
        decoded.append(len(ids))
        return ids.float().expand(-1, 3, 2, 2)/1000 - 1

    class Metrics:
        def __init__(self, *args, **kwargs): pass
        def update(self, images, *, real):
            assert not real
            features.append(images.clone())
        def compute(self, **kwargs): return 3., None, None

    monkeypatch.setattr(rqvae_metrics, 'DistributedOriginalRQVAEMetrics', Metrics)
    aux = SimpleNamespace(coeff_vocab_size=2048, num_atoms=16384,
                          decode_compound=decode, decode_tokens=decode)
    model = (Compound() if compound else Sparse()).train()
    result = training.evaluate_generation_metrics(model, aux, None, 1033,
        batch_size=1000, num_condition_classes=1, compute_inception_score=False,
        fid_reference_stats='existing-reference.npz')
    assert sampled == [1000, 33]
    assert decoded == [64]*15 + [40, 33]
    expected = ((torch.arange(1033).float()/1000-1)+1)/2
    torch.testing.assert_close(torch.cat(features)[:, 0, 0, 0], expected, atol=0, rtol=0)
    assert result == (3., None, None) and model.training


def test_checkpoint_snapshot_fallback_does_not_copy_mount_metadata(tmp_path, monkeypatch):
    source, target = tmp_path/'last.pt', tmp_path/'best.pt'
    source.write_bytes(b'complete checkpoint')
    def cannot_link(*args): raise OSError('hard links unavailable')
    def forbidden_metadata(*args): raise AssertionError('copy2 can fail on the workspace mount')
    monkeypatch.setattr(training.os, 'link', cannot_link)
    monkeypatch.setattr(training.shutil, 'copy2', forbidden_metadata)
    training.snapshot_checkpoint(source, target)
    assert source.read_bytes() == target.read_bytes()
