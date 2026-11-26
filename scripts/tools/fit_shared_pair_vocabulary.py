#!/usr/bin/env python3
"""Fit one shared integer-to-(atom, physical coefficient) vocabulary.

Example: python scripts/tools/fit_shared_pair_vocabulary.py --config
configs/research/imagenet-shared-pair24577-20261112.yaml
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, SharedAtomCoefficientRQ
from src.training.laser_joint_vocabulary import (
    allocate_coefficients, fit_coefficient_candidates, pack_codebook,
)
from src.training.laser_shared_joint_vocabulary import pack_shared_codebook, refit_shared_book
from src.training.rq_imagenet_runtime import TarImageNet, sha256, write_json


def cache_latents(config, out, device):
    from src import rq_upstream
    from rqvae.img_datasets.transforms import create_transforms
    from omegaconf import OmegaConf
    aux = FrozenJointLaserTokenizer(config['checkpoint'], config['source_codebook']).to(device)
    data_cfg = OmegaConf.create(dict(transforms=dict(type='imagenet256x256')))
    data = TarImageNet(config['data_directory'], 'train', create_transforms(data_cfg, split='train'),
                      seed=config['seed'])
    selected = np.random.default_rng(config['seed']).permutation(len(data))[
        :config['fitting_images'] + config['heldout_training_images']]
    assert len(selected) == config['fitting_images'] + config['heldout_training_images']
    paths = []
    for split, ids in [('fit', selected[:config['fitting_images']]),
                       ('held', selected[config['fitting_images']:])]:
        path = out / f'{split}-latents.npy'
        identity = dict(checkpoint_sha256=config['stage1_checkpoint_sha256'],
            source_codebook_sha256=sha256(config['source_codebook']),
            image_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
            seed=config['seed'], augmentation='imagenet256x256 training crop and flip, epoch 0')
        receipt = path.with_suffix('.json')
        if not path.exists() or not receipt.exists() or json.loads(receipt.read_text()) != identity:
            loader = DataLoader(Subset(data, ids.tolist()), batch_size=config['encoder_batch_size'],
                                num_workers=config['workers'], pin_memory=True)
            cached = np.lib.format.open_memmap(path.with_suffix('.pending.npy'), mode='w+',
                dtype='float32', shape=(len(ids), 8, 8, 256))
            count = 0
            with torch.inference_mode():
                for images, _ in loader:
                    latent = aux.encode(images.to(device)).cpu().numpy()
                    cached[count:count + len(images)] = latent
                    count += len(images)
                    write_json(out / 'status.json', dict(phase='encoding_training_latents',
                        split=split, images=count, target=len(ids)))
            cached.flush()
            del cached
            path.with_suffix('.pending.npy').replace(path)
            write_json(receipt, identity)
        paths.append(path)
    np.save(out / 'fit-image-ids.npy', selected[:config['fitting_images']])
    np.save(out / 'held-image-ids.npy', selected[config['fitting_images']:])
    del aux
    torch.cuda.empty_cache()
    return paths


@torch.no_grad()
def fit(config, paths, out, device):
    source = torch.load(config['source_codebook'], map_location='cpu', weights_only=True)
    depth = config['depth']
    prior = SharedAtomCoefficientRQ(source['dictionary'], source['atom_ids'],
                                    source['coefficients'], depth).to(device)
    x, held = [torch.from_numpy(np.load(p)).reshape(-1, prior.dictionary.shape[0]).to(device)
               for p in paths]
    encoded = prior.quantize(x, chunk_size=config['chunk_size'], return_projections=True)
    atoms, _ = prior.pairs_from_ids(encoded['codes'])
    observed = encoded['codes'] != 0
    # Pool PHYSICAL coefficient projections over every residual depth.
    books, errors = fit_coefficient_candidates(atoms[observed], encoded['projections'][observed],
        prior.dictionary.shape[1], counts=tuple(config['coefficient_candidates']))
    allocation = allocate_coefficients(errors, config['vocabulary'] - 1)
    expanded = pack_codebook(prior.dictionary, [books], [allocation])
    q = SharedAtomCoefficientRQ(expanded['dictionary'], expanded['atom_ids'][0],
                                expanded['coefficients'][0], depth).to(device)
    del encoded, atoms, observed, books, errors, prior
    trace, best, best_error = [], None, float('inf')
    baseline = SharedAtomCoefficientRQ(source['dictionary'], source['atom_ids'],
                                      source['coefficients'], depth).to(device)
    baseline_error = float((held - baseline.quantize(held, chunk_size=config['chunk_size'])['quantized']).square().mean())
    del baseline
    for iteration in range(config['fitting_passes'] + 1):
        encoded = q.quantize(x, chunk_size=config['chunk_size'])
        held_codes = q.quantize(held, chunk_size=config['chunk_size'])
        error = float((held - held_codes['quantized']).square().mean())
        row = dict(iteration=iteration, fit_latent_mse=float((x - encoded['quantized']).square().mean()),
                   heldout_training_latent_mse=error)
        if error < best_error:
            best_error = error
            best = pack_shared_codebook(q.dictionary, q.atom_ids, q.coefficients, depth,
                source_stage1_checkpoint_sha256=config['stage1_checkpoint_sha256'],
                learned_compact_dictionary=True, encoder_decoder_frozen=True,
                fitting_split='train', fitting_views=config['fitting_images'],
                validation_used_for_fitting=False, selected_iteration=iteration,
                source_codebook_sha256=sha256(config['source_codebook']))
            torch.save(best, out / 'joint-codebook.pending.pt')
            (out / 'joint-codebook.pending.pt').replace(out / 'joint-codebook.pt')
        if iteration < config['fitting_passes']:
            row.update(refit_shared_book(q, x, encoded['codes']))
        trace.append(row)
        write_json(out / 'fit.json', dict(completed=iteration == config['fitting_passes'],
            trace=trace, baseline_shared_vocabulary=source['vocabulary'],
            baseline_heldout_training_latent_mse=baseline_error,
            best_heldout_training_latent_mse=best_error, vocabulary=config['vocabulary'],
            depth=depth, unique_physical_banks=1))
        print(json.dumps(row), flush=True)
    # Verify the saved mapping, including exact reconstruction and reload.
    restored = SharedAtomCoefficientRQ(best['dictionary'], best['atom_ids'], best['coefficients'], depth).to(device)
    result = restored.quantize(held[:128], chunk_size=config['chunk_size'])
    assert torch.equal(result['quantized'], restored.embed(result['codes']).sum(-2))
    write_json(out / 'verification.json', dict(passed=True, unique_physical_banks=1,
        vocabulary=restored.vocab_size, depth=depth, tokens_per_image=64 * depth,
        shared_pair_tables=True, separate_coefficient_tokens=False,
        codebook_sha256=sha256(out / 'joint-codebook.pt'),
        heldout_latent_improved=best_error < baseline_error,
        full_validation_rfid_measured=False, stage2_training_started=False))
    write_json(out / 'status.json', dict(phase='fitted', codebook=str(out / 'joint-codebook.pt'),
        heldout_latent_improved=best_error < baseline_error,
        next_step='Measure full validation reconstruction and calibrate teacher before fresh stage2'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    out = Path(config['output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'status.json', dict(phase='verifying_source_checkpoint'))
    torch.set_num_threads(config.get('cpu_threads', 4))
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(config.get('device', 'cuda:0'))
    if sha256(config['checkpoint']) != config['stage1_checkpoint_sha256']:
        raise ValueError('Frozen source checkpoint SHA256 mismatch')
    write_json(out / 'recipe.json', config)
    paths = cache_latents(config, out, device)
    write_json(out / 'status.json', dict(phase='fitting_shared_pair_vocabulary'))
    fit(config, paths, out, device)


if __name__ == '__main__':
    main()
