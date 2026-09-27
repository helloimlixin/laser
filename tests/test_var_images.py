import json

import numpy as np
from PIL import Image
import pytest
import torch

from src.data.var_images import Images, load_manifests


def manifests(tmp_path, train=None, val=None, classes=None):
    for split, rows in [('train', train or [['female/a.png', 0]]),
                        ('val', val or [['male/b.png', 0]])]:
        (tmp_path/f'{split}-manifest.json').write_text(json.dumps(
            dict(classes=classes or ['face'], samples=rows)))


def test_faces_are_unconditional_and_disjoint(tmp_path):
    manifests(tmp_path)
    result = load_manifests(tmp_path, 'celebahq')
    assert result['train']['classes'] == ['face']
    manifests(tmp_path, classes=['female', 'male'])
    with pytest.raises(ValueError, match='unconditional'):
        load_manifests(tmp_path, 'celebahq')
    manifests(tmp_path, val=[['male/a.png', 0]])
    with pytest.raises(ValueError, match='overlap'):
        load_manifests(tmp_path, 'celebahq')


@pytest.mark.parametrize('bad', [[['x.png', 1]], [['../x.png', 0]],
                                [['x.png', 0], ['x.png', 0]]])
def test_invalid_manifests_fail_before_training(tmp_path, bad):
    manifests(tmp_path, train=bad)
    with pytest.raises(ValueError):
        load_manifests(tmp_path, 'celebahq')


def test_face_pixels_match_validation_and_do_not_reset_cuda_rng(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path/'face.png')
    manifest = dict(samples=[['face.png', 0]])
    train = Images(tmp_path, manifest, True, 7, 'celebahq')
    val = Images(tmp_path, manifest, False, 7, 'celebahq')
    cuda_calls = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: cuda_calls.append(seed))
    before = torch.get_rng_state().clone()
    first, label = train[0]
    train.epoch = 9
    torch.testing.assert_close(first, train[0][0])
    torch.testing.assert_close(first, val[0][0])
    torch.testing.assert_close(first, val[np.int64(0)][0])
    expected = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()/127.5-1
    torch.testing.assert_close(first, expected)
    assert label == 0 and not cuda_calls
    assert torch.equal(before, torch.get_rng_state())


def test_corrupt_images_raise_instead_of_replacing_the_sample(tmp_path):
    (tmp_path/'bad.png').write_text('not an image')
    data = Images(tmp_path, dict(samples=[['bad.png', 0]]), True, 0, 'celebahq')
    with pytest.raises(OSError):
        data[0]


def test_training_flip_is_reproducible_and_validation_is_unflipped(tmp_path):
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[:, :64, 0] = 255
    Image.fromarray(pixels).save(tmp_path/'face.png')
    manifest = dict(samples=[['face.png', 0]])
    train = Images(tmp_path, manifest, True, 7, 'celebahq', horizontal_flip=True)
    val = Images(tmp_path, manifest, False, 7, 'celebahq', horizontal_flip=True)
    original = val[0][0]
    orientations = set()
    for epoch in range(20):
        train.epoch = epoch
        pixels = train[0][0]
        torch.testing.assert_close(pixels, train[0][0])
        orientations.add(torch.equal(pixels, original))
        assert torch.equal(pixels, original) or torch.equal(pixels, original.flip(2))
        torch.testing.assert_close(val[0][0], original)
    assert orientations == {True, False}
