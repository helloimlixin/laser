"""Fail-closed provenance checks for FID against the exact FFHQ training images."""
import json
import os
from pathlib import Path

import numpy as np

from src.original_rq_training import file_sha256


PROTOCOL_VERSION = 'ffhq-fid-fp32-continuous-v1'
INCEPTION_PROCESSING = 'bilinear 299x299, align_corners=False, antialias=False; 2*x-1; FID Inception 2048D'
GENERATED_PROCESSING = 'FP32 decoder -> clamp((x+1)/2,0,1); no image-file roundtrip'


def validate_matched_reference(reference, manifest, data_root, dataset_fingerprints, *,
                               inception_source=None, inception_weights=None):
    """Reject a different split, transform, cached dataset, or feature extractor."""
    reference, manifest, data_root = Path(reference), Path(manifest), Path(data_root)
    record = json.loads(manifest.read_text())
    if record.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('FID reference protocol version mismatch')
    if record.get('uses_exact_laser_training_images') is not True or record.get('count') != 60000:
        raise ValueError('FID reference must use all 60000 exact training images')
    if record.get('inception_processing') != INCEPTION_PROCESSING or record.get('generated_pixel_processing') != GENERATED_PROCESSING:
        raise ValueError('FID preprocessing contract mismatch')
    expected_source = 'Actual lossless cached RGB 256x256 images; original downsampling PIL LANCZOS'
    if record.get('source_preprocessing') != expected_source or not record.get('cache_files_verified'):
        raise ValueError('FID reference does not document the verified training image cache')
    if record.get('statistics_sha256') != file_sha256(reference):
        raise ValueError('FID reference statistics checksum mismatch')
    if record.get('dataset_manifest_sha256') != file_sha256(data_root/'manifest.json'):
        raise ValueError('FID reference dataset manifest mismatch')
    if record.get('dataset_fingerprints') != dataset_fingerprints:
        raise ValueError('FID reference dataset fingerprints mismatch')
    ids_path = manifest.parent/'laser-train-ids.npy'
    if record.get('image_ids_sha256') != file_sha256(ids_path):
        raise ValueError('FID reference image IDs checksum mismatch')
    if not np.array_equal(np.load(ids_path, allow_pickle=False), np.arange(60000)):
        raise ValueError('FID reference image IDs do not match the full training split')
    root = Path(__file__).resolve().parents[2]
    inception_source = inception_source or root/'third_party/rq-vae-transformer/rqvae/metrics/inception.py'
    inception_weights = inception_weights or Path(os.environ['TORCH_HOME'])/'hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'
    if record.get('inception_source_sha256') != file_sha256(inception_source):
        raise ValueError('FID Inception preprocessing implementation mismatch')
    if record.get('inception_weights_sha256') != file_sha256(inception_weights):
        raise ValueError('FID Inception weights mismatch')
    return record
