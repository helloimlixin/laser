import json
import io
import tarfile

import numpy as np
from PIL import Image
import pytest
import torch

from src.data.var_images import Images, IndexedImageNet, load_manifests


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


@pytest.mark.parametrize('split', ['train', 'val'])
def test_archive_pixels_labels_and_epoch_crops_match_folders(tmp_path, monkeypatch, split):
    pixels = np.random.default_rng(7).integers(0, 256, (321, 379, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path/'image.JPEG')
    payload = (tmp_path/'image.JPEG').read_bytes()
    with tarfile.open(tmp_path/f'ILSVRC2012_img_{split}.tar', 'w') as archive:
        member = tarfile.TarInfo('image.JPEG')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(tmp_path/f'ILSVRC2012_img_{split}.tar') as archive:
        offset = archive.getmember('image.JPEG').offset_data
    np.savez(tmp_path/f'{split}-index.npz', offsets=np.array([offset]),
             sizes=np.array([len(payload)]), labels=np.array([0]), classes=np.array(['n00000001']))
    folder = Images(tmp_path, dict(samples=[['image.JPEG', 0]]), split == 'train', 13)
    indexed = IndexedImageNet(tmp_path, split, 13)
    cuda_seeds = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: cuda_seeds.append(seed))
    before = torch.get_rng_state().clone()
    views = []
    for epoch in (0, 1, 7):
        folder.epoch = indexed.epoch = epoch
        image, label = indexed[np.int64(0)]
        torch.testing.assert_close(image, folder[0][0], atol=0, rtol=0)
        assert label == folder[0][1] == 0
        views.append(image)
    assert not cuda_seeds and torch.equal(before, torch.get_rng_state())
    assert torch.equal(views[0], views[1]) == (split == 'val')
    # Truncated archives must fail instead of silently replacing a sample.
    with indexed.archive.open('r+b') as archive:
        archive.truncate(offset + len(payload) - 1)
    with pytest.raises(OSError, match='Truncated'):
        indexed[0]
    with pytest.raises(ValueError, match='archive index'):
        IndexedImageNet(tmp_path, split, 13)
