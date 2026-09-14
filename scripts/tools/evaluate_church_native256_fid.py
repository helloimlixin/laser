#!/usr/bin/env python3
"""Evaluate native-256 Inception FID with spatial input resizing disabled.

Real images keep the exact stage-1 preparation. Both real and generated images
enter Inception at 256x256. This produces a separate, nonstandard FID metric.
"""
import argparse
import json
from pathlib import Path
import time

from recompute_church_fid_stage1_reference import (
    ROOT, PIPELINE, HISTORICAL, load_file_module, recipe)
import numpy as np
import torch
from omegaconf import OmegaConf
from rqvae.metrics.fid import InceptionWrapper, compute_statistics_dataset, mean_covar_numpy, frechet_distance
from rqvae.metrics.inception import InceptionV3
from rqvae.models import create_model
from rqvae.utils.config import load_config, augment_arch_defaults
from church_official_fid import verify_official_sources


def native_inception(device):
    model = InceptionWrapper([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=False)
    model = model.to(device).eval().requires_grad_(False)
    assert not model.resize_input and model.normalize_input
    observed = dict(calls=0, spatial_shape=None)

    def verify_stem(module, arguments):
        value = arguments[0]
        assert tuple(value.shape[1:]) == (3, 256, 256), tuple(value.shape)
        observed['calls'] += 1
        observed['spatial_shape'] = [256, 256]

    model.blocks[0].register_forward_pre_hook(verify_stem)
    return model, observed


def setup(args, name):
    folder = args.output.resolve() / name
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def status(phase, **values):
        row = dict(phase=phase, elapsed_seconds=time.time()-started,
                   updated_unix=time.time(), **values)
        recipe.atomic_json(folder / 'status.json', row)
        print(json.dumps(row), flush=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    sources = verify_official_sources()
    manifest = json.loads((PIPELINE / 'official-fid-source-manifest.json').read_text())
    for name, expected in manifest.items():
        assert recipe.file_sha256(ROOT / name) == expected, name
    recipe.atomic_json(folder / 'metric.json', dict(metric='fid_native256',
        inception_resize_input=False, inception_spatial_input=[256, 256],
        feature_dimensions=2048, feature_precision='float32, TF32 disabled',
        statistics='released mean_covar_numpy', distance='released frechet_distance',
        upstream_metric_sources=sources, script_sha256=recipe.file_sha256(__file__),
        standard_published_fid_comparable=False))
    return folder, device, status


@torch.no_grad()
def real_reference(args):
    out, device, status = setup(args, 'real')
    status('loading_stage1_prepared_images')
    directory = Path(json.loads((PIPELINE / 'stage1/run-directory.json').read_text())['path'])
    config = OmegaConf.load(directory / 'config.yaml')
    transform_module = load_file_module('native256_stage1_transforms', HISTORICAL / 'rqvae/img_datasets/transforms.py')
    dataset_module = load_file_module('native256_stage1_lsun', HISTORICAL / 'rqvae/img_datasets/lsun.py')
    transform = transform_module.create_transforms(config.dataset, split='train')
    dataset = dataset_module.LSUNClass(config.dataset.root, 'church', transform=transform)
    assert len(dataset) == len(dataset.keys) == len(set(dataset.keys)) == 126227
    inception, observed = native_inception(device)
    features = np.lib.format.open_memmap(out / 'features.npy', mode='w+', dtype=np.float32,
                                        shape=(len(dataset), 2048))
    offset = 0

    def record(module, inputs, output):
        nonlocal offset
        values = output.cpu().numpy()
        assert values.shape[1:] == (2048,) and np.isfinite(values).all()
        features[offset:offset+len(values)] = values
        offset += len(values)
        if offset == len(values) or offset % 10000 == 0 or offset == len(dataset):
            status('extracting_real_features_without_inception_resize', images_done=offset, images_total=len(dataset))

    hook = inception.register_forward_hook(record)
    mu, sigma, _, _ = compute_statistics_dataset(dataset, batch_size=100,
        inception_model=inception, device=device)
    hook.remove()
    features.flush()
    dataset.env.close()
    assert offset == 126227 and observed['calls'] == 1263
    np.savez(out / 'statistics.npz', mu=mu, sigma=sigma, samples=offset,
             inception_resize_input=False, inception_input_size=256)
    recipe.atomic_json(out / 'result.json', dict(real_images=offset,
        real_transform=str(transform), stage1_preparation_retained=True,
        config=str(directory / 'config.yaml'), config_sha256=recipe.file_sha256(directory / 'config.yaml'),
        inception_resize_input=False, observed_stem_inputs=observed,
        statistics=str(out / 'statistics.npz'), statistics_sha256=recipe.file_sha256(out / 'statistics.npz')))
    status('complete', real_images=offset, observed_stem_inputs=observed)


@torch.no_grad()
def generated(args):
    out, device, status = setup(args, args.model)
    status('loading_frozen_tokenizer_and_saved_codes')
    if args.model == 'laser':
        code_folder = PIPELINE / 'stage2/official-fid/fid-50000-epoch050'
        source = json.loads((PIPELINE / 'stage2/cache-provenance.json').read_text())
        assert recipe.file_sha256(source['checkpoint']) == source['checkpoint_sha256']
        assert recipe.file_sha256(source['codebook']) == source['codebook_sha256']
        tokenizer = recipe.FrozenCompactTokenizer(source['checkpoint'], source['codebook'])
        expected_tokenizer_state = source['frozen_state_sha256']
    else:
        code_folder = ROOT / 'outputs/church-published-audit-20260913/published-saved-50k'
        source = json.loads((code_folder / 'specification.json').read_text())
        assert recipe.file_sha256(source['vqvae']) == source['vqvae_sha256']
        assert recipe.file_sha256(source['vqvae_config']) == source['vqvae_config_sha256']
        config = load_config(source['vqvae_config'])
        tokenizer, _ = create_model(augment_arch_defaults(config.arch), ema=False)
        state = torch.load(source['vqvae'], map_location='cpu', weights_only=True, mmap=True)
        tokenizer.load_state_dict(state['state_dict'], strict=True)
        del state
        expected_tokenizer_state = source['initial_tokenizer_state_sha256']
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    assert recipe.state_sha256(tokenizer) == expected_tokenizer_state
    inception, observed = native_inception(device)
    arrays = [np.load(code_folder / f'codes-rank{rank}.npy', mmap_mode='r') for rank in range(2)]
    assert all(a.shape == (25000, 8, 8, 4) for a in arrays)
    features = np.lib.format.open_memmap(out / 'features.npy', mode='w+', dtype=np.float32,
                                        shape=(50000, 2048))
    recipe.atomic_json(out / 'provenance.json', dict(model=args.model,
        source_tokenizer=source, code_folder=str(code_folder), generated_images=50000,
        code_file_sha256={f'codes-rank{rank}.npy':recipe.file_sha256(code_folder / f'codes-rank{rank}.npy') for rank in range(2)},
        autoregressive_resampling=False, decoder_batch_size=args.batch_size,
        decoding='FP32, full 256x256 images, clamp RGB to [0,1], no resize',
        inception_resize_input=False, tokenizer_state_sha256=expected_tokenizer_state))
    offset = 0
    for array in arrays:
        for start in range(0, len(array), args.batch_size):
            codes = torch.from_numpy(np.array(array[start:start+args.batch_size], copy=True)).long().to(device)
            pixels = tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0, 1)
            assert pixels.dtype == torch.float32 and tuple(pixels.shape[1:]) == (3, 256, 256)
            assert torch.isfinite(pixels).all()
            values = inception(pixels).cpu().numpy()
            assert values.shape == (len(codes), 2048) and np.isfinite(values).all()
            features[offset:offset+len(values)] = values
            offset += len(values)
            if start == 0 or start % (args.batch_size*100) == 0 or start+len(values) == len(array):
                status('decoding_and_extracting_without_resize', images_done=offset, images_total=50000)
    features.flush()
    assert offset == 50000 and recipe.state_sha256(tokenizer) == expected_tokenizer_state
    mu, sigma = mean_covar_numpy(features)
    np.savez(out / 'statistics.npz', mu=mu, sigma=sigma, samples=offset,
             inception_resize_input=False, inception_input_size=256)
    recipe.atomic_json(out / 'result.json', dict(generated_images=offset,
        tokenizer_unchanged=True, autoregressive_codes_held_fixed=True,
        inception_resize_input=False, observed_stem_inputs=observed,
        statistics=str(out / 'statistics.npz'), statistics_sha256=recipe.file_sha256(out / 'statistics.npz')))
    status('complete', generated_images=offset, observed_stem_inputs=observed)


def compare(args):
    out = args.output.resolve()
    real = np.load(out / 'real/statistics.npz')
    assert int(real['samples']) == 126227 and not bool(real['inception_resize_input'])
    rows = []
    for name in ('laser', 'published'):
        fake = np.load(out / name / 'statistics.npz')
        assert int(fake['samples']) == 50000 and not bool(fake['inception_resize_input'])
        score = float(frechet_distance(real['mu'], real['sigma'], fake['mu'], fake['sigma']))
        assert np.isfinite(score)
        rows.append(dict(model=name, fid_native256_50000=score, generated_images=50000))
    report = dict(metric='fid_native256', real_images=126227, comparisons=rows,
        inception_resize_input=False, real_and_fake_inception_spatial_input=[256,256],
        stage1_real_image_preparation_retained=True, generated_codes_held_fixed=True,
        standard_published_fid_comparable=False, finished_unix=time.time())
    recipe.atomic_json(out / 'result.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['real', 'generated', 'compare'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', choices=['laser', 'published'], default='laser')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    {'real':real_reference, 'generated':generated, 'compare':compare}[args.mode](args)
