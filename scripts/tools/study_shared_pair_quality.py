#!/usr/bin/env python3
"""Training-only selection of shared-pair tokenizer quality improvements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import rq_upstream
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, SharedAtomCoefficientRQ
from src.rqvae_metrics import OriginalRQVAEInception, frechet_distance
from src.training.laser_shared_joint_vocabulary import pack_shared_codebook
from src.training.rq_imagenet_runtime import TarImageNet, sha256, write_json
from scripts.tools.fit_shared_pair_vocabulary import fit
from rqvae.img_datasets.transforms import create_transforms

SOURCE = ROOT / 'outputs/imagenet-shared-pair24577-20261112'
OUT = ROOT / 'outputs/imagenet-shared-pair24577-improve-20261112'
WORK = Path('/tmp/laser-shared-pair-quality')
CHECKPOINT = Path('/tmp/laser-shared-pair-rfid/stage1.pt')


def load_q(path, device):
    spec = torch.load(path, map_location='cpu', weights_only=True)
    return SharedAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients'],
                                   spec['depth']).to(device)


def variants():
    values = []
    for name, path in [('original8192', SOURCE / 'joint-codebook.pt'),
                       ('layout4096', OUT / 'layout4096/joint-codebook.pt'),
                       ('layout6144', OUT / 'layout6144/joint-codebook.pt')]:
        if not path.exists():
            continue
        for depth in ([4, 5] if name == 'original8192' else [4]):
            gains = [.975, 1., 1.025, 1.05, 1.075] if depth == 4 and name == 'original8192' else [1., 1.025, 1.05]
            for gain in gains:
                values.append(dict(name=f'{name}_d{depth}_g{gain}', source=str(path), depth=depth, gain=gain))
    return values


@torch.no_grad()
def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['fit-latents.npy', 'held-latents.npy', 'fit-image-ids.npy', 'held-image-ids.npy']:
        if not (WORK / name).exists():
            shutil.copyfile(SOURCE / name, WORK / name)
    fit_ids, held_ids = [np.load(WORK / name) for name in ['fit-image-ids.npy', 'held-image-ids.npy']]
    assert len(held_ids) == 1024 and not np.intersect1d(fit_ids, held_ids).size
    aux = FrozenJointLaserTokenizer(CHECKPOINT, SOURCE / 'joint-codebook.pt').cuda()
    config = json.loads((SOURCE / 'recipe.json').read_text())
    data = TarImageNet(config['data_directory'], 'train', create_transforms(
        OmegaConf.create(dict(transforms=dict(type='imagenet256x256'))), split='train'), seed=config['seed'])
    network = OriginalRQVAEInception().cuda().eval().requires_grad_(False)
    images_path, features_path = WORK / 'held-images.npy', WORK / 'real-features.npy'
    images = np.lib.format.open_memmap(images_path, mode='w+', dtype='float32', shape=(1024, 3, 256, 256))
    features = np.lib.format.open_memmap(features_path, mode='w+', dtype='float32', shape=(1024, 2048))
    latents = np.load(WORK / 'held-latents.npy', mmap_mode='r')
    screen_latents = np.lib.format.open_memmap(WORK / 'screen-held-latents.npy', mode='w+',
        dtype='float32', shape=(1024, 8, 8, 256))
    loader = DataLoader(Subset(data, held_ids.tolist()), batch_size=16, num_workers=16, pin_memory=True)
    seen = 0
    for batch, _ in loader:
        images[seen:seen + len(batch)] = batch.numpy()
        batch = batch.cuda(non_blocking=True)
        if not seen:
            # The original fitter encoded batches of eight with cuDNN TF32.
            # Check image/crop identity in that precision, then screen using
            # the same full precision as the final validation evaluation.
            torch.backends.cudnn.allow_tf32 = True
            reference = torch.cat([aux.encode(part) for part in batch.split(8)])
            torch.backends.cudnn.allow_tf32 = False
            torch.testing.assert_close(reference, torch.tensor(np.array(latents[:len(batch)]), device='cuda'),
                                       rtol=1.e-4, atol=2.e-5)
        screen_latents[seen:seen + len(batch)] = aux.encode(batch).cpu().numpy()
        value, _ = network(batch.mul(.5).add(.5).clamp(0, 1))
        features[seen:seen + len(batch)] = value.cpu().numpy()
        seen += len(batch)
    images.flush(); features.flush(); screen_latents.flush()
    write_json(OUT / 'selection-data.json', dict(training_images=1024, fitting_images=8192,
        fitting_and_selection_images_disjoint=True, image_ids_sha256=hashlib.sha256(held_ids.tobytes()).hexdigest(),
        validation_used_for_selection=False, source=str(SOURCE), checkpoint_sha256=sha256(CHECKPOINT),
        screening_convolutions_tf32=False, cached_training_crop_identity_verified=True))
    print('Held-out training images and real features ready', flush=True)


@torch.no_grad()
def fit_layout(directions, device):
    directory = OUT / f'layout{directions}'
    directory.mkdir(parents=True, exist_ok=True)
    source = torch.load(SOURCE / 'joint-codebook.pt', map_location='cpu', weights_only=True)
    q = load_q(SOURCE / 'joint-codebook.pt', device)
    x = torch.from_numpy(np.load(WORK / 'fit-latents.npy')).reshape(-1, 256).to(device)
    codes = q.quantize(x, chunk_size=512)['codes']
    atoms, _ = q.pairs_from_ids(codes)
    counts = torch.bincount(atoms[codes != 0], minlength=q.dictionary.shape[1])
    selected = torch.argsort(counts, descending=True, stable=True)[:directions].sort().values
    remap = torch.full_like(counts, -1)
    remap[selected] = torch.arange(directions, device=device)
    keep = remap[q.shared_atom_ids[1:]] >= 0
    new_atoms = torch.cat((torch.zeros(1, dtype=torch.long, device=device), remap[q.shared_atom_ids[1:][keep]]))
    new_values = torch.cat((torch.zeros(1, device=device), q.shared_coefficients[1:][keep]))
    prior = pack_shared_codebook(q.dictionary[:, selected], new_atoms, new_values, 4,
        learned_compact_dictionary=True, encoder_decoder_frozen=True,
        source_stage1_checkpoint_sha256=source['source_stage1_checkpoint_sha256'])
    torch.save(prior, directory / 'initial-book.pt')
    del x, q, codes, atoms
    torch.cuda.empty_cache()
    config = dict(source_codebook=str(directory / 'initial-book.pt'), depth=4, vocabulary=24577,
        chunk_size=512, coefficient_candidates=[2, 4, 6, 8], fitting_passes=12,
        stage1_checkpoint_sha256=source['source_stage1_checkpoint_sha256'], fitting_images=8192)
    write_json(directory / 'recipe.json', config)
    fit(config, [WORK / 'fit-latents.npy', WORK / 'held-latents.npy'], directory, torch.device(device))


@torch.no_grad()
def screen(rank, world):
    choices = variants()
    network = OriginalRQVAEInception().cuda().eval().requires_grad_(False)
    aux = FrozenJointLaserTokenizer(CHECKPOINT, SOURCE / 'joint-codebook.pt').cuda()
    images = np.load(WORK / 'held-images.npy', mmap_mode='r')
    latents = torch.from_numpy(np.load(WORK / 'screen-held-latents.npy')).cuda()
    for candidate in choices[rank::world]:
        name = candidate['name']
        spec = torch.load(candidate['source'], map_location='cpu', weights_only=True)
        spec.update(depth=candidate['depth'], coefficients=spec['coefficients'] * candidate['gain'],
                    global_coefficient_gain=candidate['gain'], quality_study_source=candidate['source'])
        path = OUT / 'candidates' / f'{name}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(spec, path)
        q = load_q(path, 'cuda')
        features = np.lib.format.open_memmap(WORK / f'{name}-features.npy', mode='w+', dtype='float32', shape=(1024, 2048))
        error = 0.
        for first in range(0, 1024, 16):
            x = latents[first:first + 16]
            result = q.quantize(x, chunk_size=512)
            error += float((x - result['quantized']).double().square().sum())
            reconstructed = aux.decode(result['quantized']).mul(.5).add(.5).clamp(0, 1)
            value, _ = network(reconstructed)
            features[first:first + len(x)] = value.cpu().numpy()
        features.flush()
        write_json(OUT / 'candidates' / f'{name}.json', dict(candidate,
            path=str(path), vocabulary=q.vocab_size, directions=q.dictionary.shape[1],
            tokens_per_image=64 * q.depth, selection_images=1024,
            heldout_latent_mse=error / latents.numel(), features=str(WORK / f'{name}-features.npy')))
        print('Screened', name, flush=True)


def feature_fid(real, fake):
    """Exact empirical FID via the smaller sample cross matrix.

    The trace of the covariance geometric mean is the nuclear norm of
    centered R @ F.T / sqrt((nr-1)*(nf-1)). This avoids a singular 2048-D
    covariance square root when screening just 512 or 1024 images.
    """
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    mean_r, mean_f = real.mean(0), fake.mean(0)
    r, f = real - mean_r, fake - mean_f
    trace = np.linalg.svd(r @ f.T, compute_uv=False).sum() / np.sqrt((len(r)-1)*(len(f)-1))
    return float(max(0., np.square(mean_r-mean_f).sum() + np.square(r).sum()/(len(r)-1)
                     + np.square(f).sum()/(len(f)-1) - 2*trace))


def select():
    real = np.load(WORK / 'real-features.npy')
    choices = variants()
    def score(candidate):
        path = OUT / 'candidates' / f"{candidate['name']}.json"
        result = json.loads(path.read_text())
        fake = np.load(result['features'])
        result['training_holdout_rfid'] = feature_fid(real, fake)
        result['first_half_rfid'] = feature_fid(real[:512], fake[:512])
        result['second_half_rfid'] = feature_fid(real[512:], fake[512:])
        write_json(path, result)
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(score, choices))
    selected = {}
    for depth in [4, 5]:
        candidates = [row for row in results if row['depth'] == depth]
        best = min(candidates, key=lambda row: row['training_holdout_rfid'])
        selected[str(depth)] = best
        shutil.copyfile(best['path'], OUT / f'selected-depth{depth}.pt')
    report = dict(results=results, selected=selected, validation_used_for_selection=False,
                  selection_split='held-out training images, 1024', metric='original RQ-VAE Inception',
                  full_validation_rfid_requires_measurement=True)
    write_json(OUT / 'selection.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'fit-layout', 'screen', 'select'])
    parser.add_argument('--directions', type=int, default=4096)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.phase == 'prepare': prepare()
    elif args.phase == 'fit-layout': fit_layout(args.directions, args.device)
    elif args.phase == 'screen': screen(args.rank, args.world)
    else: select()


if __name__ == '__main__':
    main()
