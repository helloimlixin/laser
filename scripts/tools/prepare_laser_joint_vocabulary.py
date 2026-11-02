#!/usr/bin/env python3
"""Fit and compare finite LASER pair vocabularies using frozen ImageNet latents."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0, str(ROOT))
from src import rq_upstream
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.scaled_atom_rq import continuous_matching_pursuit
from src.training.laser_joint_vocabulary import (
    fit_coefficient_candidates, allocate_coefficients, pack_codebook,
)
from src.training.rq_imagenet_runtime import TarImageNet, write_json, sha256

SOURCE = Path('/tmp/laser-rq-clean-scratch-imagenet-20261015')
BASE = Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
OUT = ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015'


def tokenizer(device):
    from omegaconf import OmegaConf
    spec = OmegaConf.load(SOURCE/'inputs/recipe.yaml').tokenizer
    return FrozenLaserTokenizer(SOURCE/'inputs/stage1-model.pt', num_atoms=spec.num_atoms,
        coeff_vocab_size=spec.coefficient_bins, coeff_max=spec.coefficient_max,
        coeff_scales=spec.coefficient_scales, sparsity_level=spec.depth).to(device)


def status(phase, **kwargs):
    value = dict(phase=phase, time=time.time(), **kwargs)
    write_json(OUT/'status.json', value)
    print(json.dumps(value), flush=True)


@torch.inference_mode()
def cache(device, fit_images, screen_images):
    OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    config = load_config(SOURCE/'inputs/stage2-training-config.yaml')
    aux = tokenizer(device)
    indices = np.random.default_rng(26101565537).permutation(1281167)[:fit_images + 128]
    manifest = dict(fit_indices=indices[:fit_images].tolist(),
        temperature_indices=indices[fit_images:].tolist(),
        screen_indices=np.random.default_rng(261015).choice(50000, screen_images, replace=False).tolist(),
        stage1_sha256=sha256(SOURCE/'inputs/stage1-model.pt'),
        fit_on_training_images_only=True, frozen_encoder_decoder_dictionary=True)
    for split, name, selected in [('train', 'fit', manifest['fit_indices']),
            ('train', 'temperature', manifest['temperature_indices']),
            ('val', 'screen', manifest['screen_indices'])]:
        data = TarImageNet(SOURCE/'data', split, create_transforms(config.dataset, split='val'))
        loader = DataLoader(Subset(data, selected), batch_size=8, num_workers=4, pin_memory=True)
        destination = BASE/f'{name}-latents.npy'
        values = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32,
                                          shape=(len(selected), 8, 8, 256))
        first = 0
        for images, _ in loader:
            latent = aux.quant_conv(aux.encoder(images.to(device).float())).permute(0, 2, 3, 1)
            values[first:first + len(images)] = latent.cpu().numpy()
            first += len(images)
            if first % 256 == 0: status('cache', split=name, images=first, total=len(selected))
        values.flush()
        manifest[name + '_latents_sha256'] = sha256(destination)
    torch.save(aux.dictionary.cpu(), BASE/'dictionary.pt')
    write_json(OUT/'calibration-data.json', manifest)
    status('cache_complete', fit_images=fit_images, screen_images=screen_images)


@torch.inference_mode()
def fit(device, passes):
    manifest = json.loads((OUT/'calibration-data.json').read_text())
    latents = np.load(BASE/'fit-latents.npy', mmap_mode='r')
    dictionary = torch.load(BASE/'dictionary.pt', weights_only=True).to(device)
    atoms, coefficients = [], []
    for first in range(0, len(latents), 8):
        x = torch.from_numpy(latents[first:first + 8].copy()).to(device)
        result = continuous_matching_pursuit(x, dictionary, depth=4)
        atoms.append(result['atoms']); coefficients.append(result['coefficients'])
    atoms, coefficients = torch.cat(atoms), torch.cat(coefficients)
    depth_books, adaptive = [], []
    for d in range(4):
        books, error = fit_coefficient_candidates(atoms[..., d], coefficients[..., d], dictionary.shape[1])
        allocation = allocate_coefficients(error, dictionary.shape[1] * 4)
        depth_books.append(books); adaptive.append(allocation)
        status('initial_coefficients', depth=d, allocation_counts={str(k): allocation.count(k) for k in books})
    del atoms, coefficients
    for name, allocation in [('uniform64k', [[4] * dictionary.shape[1]] * 4),
                              ('adaptive64k', adaptive),
                              ('uniform128k', [[8] * dictionary.shape[1]] * 4)]:
        spec = pack_codebook(dictionary, depth_books, allocation)
        q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)
        prior = q.coefficients.clone()
        trace = []
        for iteration in range(passes):
            counts = torch.zeros_like(q.coefficients, dtype=torch.float64)
            sums = torch.zeros_like(counts)
            error = 0.
            for first in range(0, len(latents), 8):
                x = torch.from_numpy(latents[first:first + 8].copy()).to(device)
                result = q.quantize(x, chunk_size=128, return_projections=True)
                for d in range(4):
                    codes = result['codes'][..., d].flatten()
                    counts[d] += torch.bincount(codes, minlength=q.vocab_size)
                    sums[d] += torch.bincount(codes, weights=result['projections'][..., d].flatten().double(),
                                             minlength=q.vocab_size)
                error += (x - result['quantized']).square().sum().item()
            updated = ((sums + 4. * prior) / (counts + 4.)).float()
            updated[:, 0] = 0.
            assert ((updated[:, 1:] * prior[:, 1:]) > 0).all()
            q.coefficients.copy_(updated)
            row = dict(iteration=iteration + 1, latent_mse_before_update=error / latents.size,
                       used_entries_by_depth=(counts > 0).sum(-1).tolist())
            trace.append(row)
            status('lloyd', candidate=name, **row)
        spec.update(coefficients=q.coefficients.cpu(),
            source_stage1_checkpoint_sha256=manifest['stage1_sha256'],
            fitting_images=len(latents), fitting_split='train', allocation=name)
        torch.save(spec, OUT/f'{name}-codebook.pt')
        write_json(OUT/f'{name}-fit.json', dict(trace=trace, training_only=True, passes=passes,
            prior_weight=4., allocation_counts=[{str(k): a.count(k) for k in [2, 4, 6, 8]} for a in allocation],
            codebook_sha256=sha256(OUT/f'{name}-codebook.pt')))
    status('fit_complete')


@torch.inference_mode()
def screen(device):
    from src import rqvae_metrics as metrics
    metrics.INCEPTION_PATH = SOURCE/'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    # The original backend classifies decoded tensors after [0,1] conversion.
    aux = tokenizer(device)
    config = load_config(SOURCE/'inputs/stage2-training-config.yaml')
    manifest = json.loads((OUT/'calibration-data.json').read_text())
    data = TarImageNet(SOURCE/'data', 'val', create_transforms(config.dataset, split='val'))
    loader = DataLoader(Subset(data, manifest['screen_indices']), batch_size=8, num_workers=4, pin_memory=True)
    real = []
    for images, _ in loader:
        value = inception(images.to(device).mul(.5).add(.5).clamp(0, 1))
        real.append((value[0] if isinstance(value, tuple) else value).cpu().numpy())
    real = np.concatenate(real)
    np.save(BASE/'screen-real-features.npy', real)
    latents = np.load(BASE/'screen-latents.npy', mmap_mode='r')
    results = {}
    from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients, ordered_support_basis
    from src.training.laser_hard_ar import sparse_encoding, nearest_coefficient_ids
    gram = aux.dictionary.T @ aux.dictionary
    variants = [('source_sparse', None, 1.)]
    for name in ['uniform64k', 'adaptive64k', 'uniform128k']:
        for gain in [1., 1.1]:
            variants.append((name if gain == 1. else name + '-gain110', name, gain))
    for name, source, gain in variants:
        q = None
        if name != 'source_sparse':
            spec = torch.load(OUT/f'{source}-codebook.pt', weights_only=True)
            spec['coefficients'] = spec['coefficients'] * gain
            spec['coefficient_gain'] = gain
            if gain != 1.:
                torch.save(spec, OUT/f'{name}-codebook.pt')
            q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)
        fake, error = [], 0.
        for first in range(0, len(latents), 8):
            x = torch.from_numpy(latents[first:first + 8].copy()).to(device)
            if q is None:
                a, c = sparse_encoding(x, aux.dictionary, 4, gram=gram)
                vectors = aux.dictionary.T[a]
                gamma, _ = dictionary_to_orthogonal_coefficients(vectors, c)
                ids = nearest_coefficient_ids(gamma / aux.coeff_scales, aux.coeff_bins)
                basis, _ = ordered_support_basis(vectors)
                reconstructed = (basis * (aux.coeff_bins[ids] * aux.coeff_scales)[..., None]).sum(-2)
            else:
                reconstructed = q.quantize(x, chunk_size=128)['quantized']
            images = aux.decoder(aux.post_quant_conv(reconstructed.permute(0, 3, 1, 2).contiguous()))
            value = inception(images.mul(.5).add(.5).clamp(0, 1))
            fake.append((value[0] if isinstance(value, tuple) else value).cpu().numpy())
            error += (x - reconstructed).square().sum().item()
            if first % 256 == 0: status('screen_reconstruction', candidate=name, images=first, total=len(latents))
        fake = np.concatenate(fake)
        fid = metrics.frechet_distance(fake.mean(0), np.cov(fake, rowvar=False), real.mean(0), np.cov(real, rowvar=False))
        results[name] = dict(matched_rfid=float(fid), images=len(latents),
                            latent_mse=error / latents.size, vocabulary=None if q is None else q.vocab_size)
        write_json(OUT/'screen-results.json', dict(results=results, selection_only=True,
            reference='same validation images; subset rFID is not full-50k rFID'))
        status('screen_result', candidate=name, **results[name])
    status('screen_complete', results=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['cache', 'fit', 'screen', 'all'])
    p.add_argument('--device', default='cuda:3')
    p.add_argument('--fit-images', type=int, default=8192)
    p.add_argument('--screen-images', type=int, default=2048)
    p.add_argument('--passes', type=int, default=4)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); BASE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.18, args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault('TORCH_HOME', str(SOURCE/'torch-cache'))
    if args.action in ('cache', 'all'): cache(args.device, args.fit_images, args.screen_images)
    if args.action in ('fit', 'all'): fit(args.device, args.passes)
    if args.action in ('screen', 'all'): screen(args.device)


if __name__ == '__main__': main()
