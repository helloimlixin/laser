#!/usr/bin/env python3
"""Rebuild the full Church FID reference with the actual fine-tuning transform.

The released statistics/FID functions are used unchanged. A forward hook only
records the real features and progress. Previously generated images are held
fixed by reusing their saved feature statistics.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import train_church_laser_three_epoch_official_fid as recipe
import numpy as np
import PIL
import torch
import torchvision
from omegaconf import OmegaConf
from rqvae.metrics.fid import compute_statistics_dataset, get_inception_model, frechet_distance
from church_official_fid import verify_official_sources

ROOT = recipe.ROOT
PIPELINE = ROOT / 'outputs/church-laser-three-epoch-20260914'
HISTORICAL = PIPELINE / 'stage1-source/third_party/rq-vae-transformer'
OFFICIAL_REFERENCE = ROOT / 'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz'


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    def status(phase, **kwargs):
        row = dict(phase=phase, elapsed_seconds=time.time()-started,
                   updated_unix=time.time(), **kwargs)
        recipe.atomic_json(out / 'status.json', row)
        print(json.dumps(row), flush=True)

    status('loading_exact_finetune_transform')
    metric_sources = verify_official_sources()
    manifest = json.loads((PIPELINE / 'official-fid-source-manifest.json').read_text())
    for name, expected in manifest.items():
        assert recipe.file_sha256(ROOT / name) == expected, name
    stage1 = json.loads((PIPELINE / 'stage1/complete.json').read_text())
    assert stage1['epoch'] == 3 and not stage1['smoke_only']
    run_directory = Path(json.loads((PIPELINE / 'stage1/run-directory.json').read_text())['path'])
    config_path = run_directory / 'config.yaml'
    config = OmegaConf.load(config_path)
    transforms_path = HISTORICAL / 'rqvae/img_datasets/transforms.py'
    dataset_path = HISTORICAL / 'rqvae/img_datasets/lsun.py'
    transforms_module = load_file_module('actual_stage1_transforms', transforms_path)
    dataset_module = load_file_module('actual_stage1_lsun', dataset_path)
    transform = transforms_module.create_transforms(config.dataset, split='train')
    assert [type(item).__name__ for item in transform.transforms] == ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']
    dataset = dataset_module.LSUNClass(config.dataset.root, 'church', transform=transform)
    assert len(dataset) == 126227
    # Use every unique real image once; no DistributedSampler padding.
    key_digest = hashlib.sha256()
    for key in dataset.keys:
        key_digest.update(len(key).to_bytes(8, 'little'))
        key_digest.update(key)
    inception = get_inception_model().to(device).eval().requires_grad_(False)
    assert inception.resize_input and inception.normalize_input
    specification = dict(real_images=len(dataset), real_population='all Church training LMDB entries, once each',
        padded_or_dropped_images=0, real_transform=str(transform),
        actual_finetune_config=str(config_path), actual_finetune_config_sha256=recipe.file_sha256(config_path),
        transform_source=str(transforms_path), transform_source_sha256=recipe.file_sha256(transforms_path),
        dataset_source=str(dataset_path), dataset_source_sha256=recipe.file_sha256(dataset_path),
        dataset_key_order_sha256=key_digest.hexdigest(),
        dataset_root=str(config.dataset.root), stage1_checkpoint_sha256=stage1['checkpoint_sha256'],
        torchvision_version=torchvision.__version__, pillow_version=PIL.__version__,
        real_pixel_path='Exact fine-tune PIL Resize(short side=256, bilinear), CenterCrop(256), ToTensor, Normalize[-1,1]; official evaluator maps back to [0,1]',
        common_feature_preprocessing='Released Inception: bilinear resize 256x256 -> 299x299, align_corners=False, for real and generated inputs',
        metric_sources=metric_sources, reference_batch_size=args.batch_size,
        precision='FP32 Inception, TF32 off; released NumPy mean/covariance',
        generated_images_resampled=False, training_modified=False)
    recipe.atomic_json(out / 'specification.json', specification)
    features_path = out / 'real-features.npy'
    features = np.lib.format.open_memmap(features_path, mode='w+', dtype=np.float32,
                                        shape=(len(dataset), 2048))
    offset = 0

    def record_features(module, inputs, output):
        nonlocal offset
        values = output.detach().cpu().numpy()
        assert values.dtype == np.float32 and values.shape[1:] == (2048,)
        assert np.isfinite(values).all() and offset + len(values) <= len(dataset)
        features[offset:offset+len(values)] = values
        offset += len(values)
        if offset == len(values) or offset % 5000 == 0 or offset == len(dataset):
            status('extracting_full_real_reference', images_done=offset, images_total=len(dataset))

    hook = inception.register_forward_hook(record_features)
    mu, sigma, unused_mu, unused_sigma = compute_statistics_dataset(
        dataset, batch_size=args.batch_size, inception_model=inception, device=device)
    hook.remove()
    features.flush()
    assert offset == len(dataset) and unused_mu is None and unused_sigma is None
    dataset.env.close()
    reference_path = out / 'lsun-church-stage1-transform-full.npz'
    np.savez(reference_path, mu=mu, sigma=sigma)
    status('scoring_fixed_generated_samples', real_images=offset)
    official = np.load(OFFICIAL_REFERENCE)
    reference_difference = float(frechet_distance(mu, sigma, official['mu'], official['sigma']))
    candidates = [
        ('laser_three_epoch_stage2_epoch50', PIPELINE / 'stage2/official-fid/fid-50000-epoch050/acts.npz',
         50000, 11.535296932071873),
        ('laser_one_epoch_stage2_epoch50', ROOT / 'outputs/church-laser-original-recipe-20260913/train/statistics-50000-epoch050.npz',
         50000, 13.884514101156697),
        ('released_church_rq_pair', ROOT / 'outputs/church-published-audit-20260913/published-saved-50k/statistics.npz',
         50000, 7.670965861813414),
    ]
    results = []
    for name, path, count, recorded in candidates:
        with np.load(path) as fake:
            if 'acts' in fake:
                assert fake['acts'].shape == (count, 2048)
            elif 'samples' in fake:
                assert int(fake['samples']) == count
            rebuilt_score = float(frechet_distance(mu, sigma, fake['mu'], fake['sigma']))
            official_score = float(frechet_distance(official['mu'], official['sigma'], fake['mu'], fake['sigma']))
        assert abs(official_score-recorded) < 1e-5, (name, official_score, recorded)
        row = dict(model=name, generated_images=count, generated_statistics=str(path),
            generated_statistics_sha256=recipe.file_sha256(path), fid_stage1_transform_reference=rebuilt_score,
            fid_official_reference=official_score, difference=rebuilt_score-official_score)
        results.append(row)
        status('scored', **row)
    report = dict(real_images=offset, reference=str(reference_path),
        reference_sha256=recipe.file_sha256(reference_path),
        original_reference=str(OFFICIAL_REFERENCE), original_reference_sha256=recipe.file_sha256(OFFICIAL_REFERENCE),
        real_reference_vs_official_fid=reference_difference,
        real_mean_max_abs_difference=float(np.max(np.abs(mu-official['mu']))),
        real_covariance_max_abs_difference=float(np.max(np.abs(sigma-official['sigma']))),
        comparisons=results, generated_samples_held_fixed=True,
        same_inception_resize_on_both_sides=True, finished_unix=time.time())
    recipe.atomic_json(out / 'result.json', report)
    status('complete', real_images=offset, results=results, reference_vs_official_fid=reference_difference)


if __name__ == '__main__':
    main()
