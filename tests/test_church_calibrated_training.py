import io
from types import SimpleNamespace

import lmdb
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.train_official_rqtransformer_laser_stage2 import LaserAux
from src.church_calibrated_training import (
    physical_targets, augmented_view, AugmentedChurch, PendingEpochBatches,
    HeldoutStop, calibrated_prior, optimizer_groups,
)
from src.coefficient_history_training import EpochStream


def fake_aux():
    aux = SimpleNamespace(soft_target_physical=True, coeff_scales=torch.tensor([7.4, 4.16, 2.46, 1.65]),
                          coeff_bins=torch.linspace(-3, 3, 2048), coeff_vocab_size=2048, sparsity_level=4)
    aux.compound_coeff_ids = lambda *a, **kw: LaserAux.compound_coeff_ids(aux, *a, **kw)
    return aux


def test_target_width_is_physical_and_does_not_expand_with_depth_scale():
    aux = fake_aux()
    truth = torch.tensor([[[[2., -2., 1., -1.]]]])
    packed, probs = physical_targets(aux, torch.zeros_like(truth).long(), truth, 'soft', .125, False)
    values = aux.coeff_bins * aux.coeff_scales[:, None]
    mean = (probs * values).sum(-1)
    std = (probs * (values - mean[..., None]).square()).sum(-1).sqrt()
    torch.testing.assert_close(mean, truth, atol=1e-5, rtol=0)
    torch.testing.assert_close(std, torch.full_like(truth, .125), atol=1e-6, rtol=0)
    assert packed.shape == truth.shape
    aux.soft_target_physical = False
    with pytest.raises(ValueError, match='physical'):
        physical_targets(aux, packed, truth, 'soft', .125)


def test_hard_targets_have_no_stochastic_perturbation():
    aux = fake_aux()
    truth = torch.tensor([[[[2., -2., 1., -1.]]]])
    atoms = torch.tensor([[[[1, 2, 3, 4]]]])
    torch.manual_seed(12)
    a, p = physical_targets(aux, atoms, truth, 'hard', .125)
    torch.manual_seed(999)
    b, q = physical_targets(aux, atoms, truth, 'hard', .125)
    assert torch.equal(a, b) and torch.equal(p, q)
    assert torch.equal(p.sum(-1), torch.ones_like(truth))
    assert int(torch.count_nonzero(p)) == 4
    assert torch.equal(a // 2048, atoms)


def test_views_reproduce_by_image_and_epoch_independent_of_global_rng():
    pixels = np.random.default_rng(12).integers(0, 255, (270, 380, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    a = augmented_view(image, 8701, 14, 0)
    torch.manual_seed(123)
    b = augmented_view(image, 8701, 14, 0)
    c = augmented_view(image, 8701, 14, 1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (3, 256, 256) and a.min() >= -1 and a.max() <= 1


def test_prefetch_and_resume_preserve_exact_image_views_and_partial_batches(tmp_path):
    path = tmp_path / 'images'
    rng = np.random.default_rng(7)
    keys = [f'image-{i}' for i in range(7)]
    with lmdb.open(str(path), map_size=8 * 1024 * 1024) as env:
        with env.begin(write=True) as transaction:
            for key in keys:
                image = Image.fromarray(rng.integers(0, 255, (270, 380, 3), dtype=np.uint8))
                data = io.BytesIO()
                image.save(data, format='PNG')
                transaction.put(key.encode('ascii'), data.getvalue())
    stream = EpochStream(7, 11)
    dataset = AugmentedChurch(path, keys, 17)
    loader = DataLoader(dataset, batch_sampler=PendingEpochBatches(stream, 3), num_workers=2)
    iterator = iter(loader)
    _, indices = next(iterator)
    assert stream.position == 0  # DataLoader may already have prefetched the entire epoch.
    committed, _, _ = stream.next(3)
    assert torch.equal(indices, committed)
    saved = stream.state_dict()
    expected = list(iterator)
    resumed = EpochStream(7, 999)
    resumed.load_state_dict(saved)
    actual = list(DataLoader(AugmentedChurch(path, keys, 17),
                            batch_sampler=PendingEpochBatches(resumed, 3), num_workers=0))
    assert [len(x[1]) for x in actual] == [3, 1]
    for (a, ai), (b, bi) in zip(expected, actual):
        assert torch.equal(a, b) and torch.equal(ai, bi)
    assert resumed.position == 3


def test_early_stop_survives_resume_and_new_improvement_resets_counter():
    monitor = HeldoutStop(patience=3, minimum_epoch=8)
    assert monitor.observe(5., 1) == (True, False)
    assert monitor.observe(4., 2) == (True, False)
    assert monitor.observe(4.1, 4) == (False, False)
    assert monitor.observe(4.2, 6) == (False, False)
    restored = HeldoutStop(**monitor.state_dict())
    assert restored.observe(4.3, 8) == (False, True)
    assert restored.observe(3.9, 10) == (True, False)
    assert restored.bad_checks == 0 and restored.best_epoch == 10
    with pytest.raises(FloatingPointError):
        restored.observe(float('nan'), 12)


def test_optimizer_has_every_parameter_once_and_preserves_calibrated_capacity():
    with torch.device('meta'):
        model = calibrated_prior()
    assert sum(p.numel() for p in model.parameters()) == 218802176
    groups = optimizer_groups(model, .05)
    ids = [id(p) for group in groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert model.body_transformer.blocks[0].attn.resid_drop.p == .15
