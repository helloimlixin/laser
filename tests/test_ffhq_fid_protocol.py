from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from scripts.tools.build_ffhq_fid_references import read_ids, real_pixels
from src.original_rq_training import file_sha256
from src.training.ffhq_fid_protocol import validate_matched_reference, PROTOCOL_VERSION, INCEPTION_PROCESSING, GENERATED_PROCESSING


def test_reference_ids_reject_duplicates_missing_rows_and_invalid_ids(tmp_path):
    path = tmp_path / 'ids.txt'
    path.write_text('00007.png\n00002.png\n')
    np.testing.assert_array_equal(read_ids(path, 2), [2, 7])
    for text in ['00007.png\n00007.png\n', '00007.png\n', '70000.png\n00002.png\n']:
        path.write_text(text)
        with pytest.raises(ValueError):
            read_ids(path, 2)


def test_reference_pixels_match_released_normalization_without_augmentation():
    rng = np.random.default_rng(67)
    pixels = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    published = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([.5]*3, [.5]*3)])(image).mul(.5).add(.5).clamp(0, 1)
    actual = real_pixels(image)
    torch.testing.assert_close(actual, published, rtol=0, atol=0)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(real_pixels(image), actual, rtol=0, atol=0)
    with pytest.raises(ValueError, match='256x256 RGB'):
        real_pixels(image.resize((128, 128)))
    with pytest.raises(ValueError, match='256x256 RGB'):
        real_pixels(image.convert('L'))


def test_matched_reference_rejects_changed_data_pixels_and_statistics(tmp_path):
    import json
    reference = tmp_path/'reference.npz'
    np.savez(reference, mu=np.zeros(3), sigma=np.eye(3))
    data_root = tmp_path/'data'; data_root.mkdir()
    (data_root/'manifest.json').write_text('{}')
    ids = tmp_path/'laser-train-ids.npy'
    np.save(ids, np.arange(60000))
    source = tmp_path/'inception.py'; source.write_text('original source')
    weights = tmp_path/'weights.pt'; weights.write_bytes(b'original weights')
    manifest = tmp_path/'reference.json'
    fingerprints = dict(train='train-cache', validation='validation-cache')
    record = dict(protocol_version=PROTOCOL_VERSION, count=60000, uses_exact_laser_training_images=True,
        inception_processing=INCEPTION_PROCESSING, generated_pixel_processing=GENERATED_PROCESSING,
        source_preprocessing='Actual lossless cached RGB 256x256 images; original downsampling PIL LANCZOS',
        cache_files_verified=True, statistics_sha256=file_sha256(reference),
        dataset_manifest_sha256=file_sha256(data_root/'manifest.json'), dataset_fingerprints=fingerprints,
        image_ids_sha256=file_sha256(ids), inception_source_sha256=file_sha256(source),
        inception_weights_sha256=file_sha256(weights))
    def verify():
        return validate_matched_reference(reference,manifest,data_root,fingerprints,
                                          inception_source=source,inception_weights=weights)
    manifest.write_text(json.dumps(record))
    assert verify()['count'] == 60000
    for key, value, error in [('count', 10000, '60000'),
                              ('uses_exact_laser_training_images', False, '60000'),
                              ('inception_processing', 'nearest resize', 'preprocessing'),
                              ('dataset_fingerprints', dict(train='other'), 'fingerprints')]:
        manifest.write_text(json.dumps(dict(record, **{key:value})))
        with pytest.raises(ValueError, match=error): verify()
    manifest.write_text(json.dumps(record))
    np.savez(reference, mu=np.ones(3), sigma=np.eye(3))
    with pytest.raises(ValueError, match='statistics checksum'): verify()
