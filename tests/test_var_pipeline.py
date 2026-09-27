from contextlib import nullcontext
import json
import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from src.training import var_laser as driver


def test_evaluation_preserves_training_rng_even_when_it_fails():
    before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    with pytest.raises(RuntimeError):
        with driver.evaluation_rng():
            torch.rand(7)
            np.random.rand(7)
            random.random()
            raise RuntimeError('evaluation failure')
    assert torch.equal(torch.get_rng_state(), before[0])
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert random.getstate() == before[2]


@pytest.mark.parametrize('supplied_reference', [True, False])
def test_unconditional_generation_labels_and_metric_counts(tmp_path, monkeypatch, supplied_reference):
    monkeypatch.setattr(driver.dist, 'barrier', lambda: None)
    monkeypatch.setattr(driver.dist, 'broadcast_object_list', lambda *a, **kw: None)
    monkeypatch.setattr(driver, 'frechet_distance', lambda *args: 42.)
    monkeypatch.setattr(driver, 'save_image', lambda *args, **kwargs: None)
    class Moments:
        def __init__(self, device): self.count = 0
        def update(self, values): self.count += len(values)
        def finish(self): return self.count, np.zeros(3), np.eye(3)
    monkeypatch.setattr(driver, 'FeatureMoments', Moments)
    experiment = driver.Experiment.__new__(driver.Experiment)
    experiment.device = torch.device('cpu')
    experiment.rank, experiment.world, experiment.num_classes = 0, 1, 1
    experiment.out, experiment.run, experiment.kind = tmp_path, None, 'laser'
    experiment.inception = lambda x: x.mean((2, 3))
    experiment.vae = SimpleNamespace(fhat_to_img=lambda z: z)
    experiment.amp = nullcontext
    experiment.cfg = OmegaConf.create(dict(prior=dict(cfg=1.5, top_k=900, top_p=.96),
        evaluation=dict(batch_size=3, adm_evaluator=None)))
    reference = tmp_path/'reference.npz'
    np.savez(reference, mu=np.zeros(3), sigma=np.eye(3))
    if supplied_reference:
        experiment.fid_reference = lambda: reference
    else:
        # Cached datasets and CompoundExperiment use real-image shards without
        # the manifest fields initialized by the base Experiment constructor.
        experiment.val = list(range(8))
        experiment.loader = lambda dataset, batch, indices: [(torch.zeros(len(indices), 3, 256, 256), None)]
    records = []
    experiment.log = lambda phase, **values: records.append((phase, values))
    seen_labels = []
    class Model:
        def eval(self): pass
        def sample(self, labels, **kwargs):
            seen_labels.extend(labels.tolist())
            return torch.zeros(len(labels), 3, 256, 256)
    assert experiment.generate(Model(), 2, 8, official=True) == 42.
    assert seen_labels == [0]*8
    assert records == [('generation', dict(epoch=2, count=8, fid_8=42.))]
    assert np.load(tmp_path/'samples-epoch002-8.npy').shape == (8,256,256,3)
    assert 'pytorch_fid_diagnostic' not in records[0][1]
    receipt = json.loads((tmp_path/'generation-epoch002-8.json').read_text())
    assert receipt['real_reference_images'] == (None if supplied_reference else 8)


def test_best_checkpoint_survives_later_atomic_last_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.dist, 'barrier', lambda: None)
    experiment = driver.Experiment.__new__(driver.Experiment)
    experiment.rank, experiment.out = 0, tmp_path
    experiment.log = lambda *args, **kwargs: None
    last = tmp_path/'prior-last.pt'
    last.write_bytes(b'epoch-2')
    experiment.retain_best_prior(52., 2, 10, 2000)
    next_file = tmp_path/'next.tmp'
    next_file.write_bytes(b'epoch-3')
    next_file.replace(last)
    experiment.retain_best_prior(60., 3, 20, 2000)
    assert (tmp_path/'prior-best-fid.pt').read_bytes() == b'epoch-2'
    assert json.loads((tmp_path/'best-prior.json').read_text())['epoch'] == 2
    with pytest.raises(ValueError, match='different FID sample counts'):
        experiment.retain_best_prior(40., 3, 20, 128)


def test_checkpoint_verification_allows_only_intentional_mask_infinities():
    from scripts.tools.verify_celebahq_smoke import finite_tree
    finite_tree({'model': {'attn_bias_for_masking': torch.tensor([0., -torch.inf]),
                          'weight': torch.ones(2)}})
    for invalid in [torch.inf, torch.nan]:
        with pytest.raises(AssertionError):
            finite_tree({'model': {'attn_bias_for_masking': torch.tensor([invalid])}})
    with pytest.raises(AssertionError):
        finite_tree({'model': {'weight': torch.tensor([-torch.inf])}})
