#!/usr/bin/env python3
"""Fit and measure compact joint dictionaries before any transformer training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
SOURCE = Path('/tmp/laser-rq-native-joint-depthtemp-imagenet-20261016')
CACHE = Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
BASE = Path('/tmp/laser-compact-joint-vocabulary-20261016')
OUT = ROOT/'outputs/imagenet-laser-compact-joint-vocabulary-20261016'
os.environ['RQ_UPSTREAM_ROOT'] = str(SOURCE/'upstream')
os.environ.setdefault('TORCH_HOME', str(SOURCE/'torch-cache'))
sys.path[:0] = [str(ROOT), str(SOURCE/'upstream'), str(ROOT/'runtime')]
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, JointAtomCoefficientRQ
from src.training.laser_compact_joint_vocabulary import refit_compact_book
from src.training.rq_imagenet_runtime import TarImageNet, write_json, sha256


def progress(name, **values):
    value = dict(phase=name, time=time.time(), **values)
    write_json(OUT/(values.get('candidate', 'cache')+'-progress.json'), value)
    print(json.dumps(value), flush=True)


def build_quantizer(path, device):
    spec = torch.load(path, map_location='cpu', weights_only=True)
    return JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)


@torch.inference_mode()
def cache(device):
    manifest_path = ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015/calibration-data.json'
    original = json.loads(manifest_path.read_text())
    assert sha256(CACHE/'fit-latents.npy') == original['fit_latents_sha256']
    assert sha256(SOURCE/'inputs/stage1-model.pt') == original['stage1_sha256']
    config = load_config(SOURCE/'inputs/stage2-training-config.yaml')
    aux = FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt', SOURCE/'inputs/joint-codebook.pt').to(device)
    indices = original['fit_indices'][:4096]
    data = TarImageNet(SOURCE/'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    data.set_epoch(0)
    loader = DataLoader(Subset(data, indices), batch_size=16, num_workers=4, pin_memory=True)
    path = BASE/'fit-latents.npy'
    values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(8192, 8, 8, 256))
    old = np.load(CACHE/'fit-latents.npy', mmap_mode='r')
    values[:4096] = old[:4096]
    first = 4096
    for images, _ in loader:
        values[first:first+len(images)] = aux.encode(images.to(device)).cpu().numpy()
        first += len(images)
        if first%512 == 0:progress('cache_training_views', images=first-4096)
    assert first == len(values)
    values.flush()
    x = torch.from_numpy(np.asarray(values).copy()).to(device)
    encoded = aux.quantizer.quantize(x, chunk_size=2048)
    importance = torch.zeros(aux.quantizer.dictionary.shape[1], device=device)
    for depth in range(4):
        codes = encoded['codes'][..., depth].flatten()
        atom = aux.quantizer.atom_ids[depth, codes]
        coefficient = aux.quantizer.coefficients[depth, codes]
        importance += torch.bincount(atom, weights=coefficient.square(), minlength=len(importance))
    ranking = importance.argsort(descending=True).cpu()
    torch.save(ranking, BASE/'atom-ranking.pt')
    write_json(OUT/'calibration-data.json', dict(training_images=4096, training_views=8192,
        fit_indices=indices, views=['center crop', 'original RandomCrop/RandomHorizontalFlip epoch0 seed261001'],
        source_stage1_sha256=original['stage1_sha256'], source_codebook_sha256=sha256(SOURCE/'inputs/joint-codebook.pt'),
        fit_latents_sha256=sha256(path), screen_latents_sha256=original['screen_latents_sha256'],
        source_data_manifest=str(manifest_path), fitting_training_only=True,
        encoder_decoder_frozen=True, refitted_dictionary=True,
        old_greedy_latent_mse_on_fit=float((x-encoded['quantized']).square().mean())))
    progress('cache_complete')


@torch.inference_mode()
def fit(device, count, passes, resume_from=None, candidate_name=None):
    candidate = candidate_name or f'compact{count}'
    path = OUT/(candidate+'-codebook.pt')
    if path.exists():raise RuntimeError('Refusing to overwrite fitted candidate')
    calibration = json.loads((OUT/'calibration-data.json').read_text())
    assert sha256(BASE/'fit-latents.npy') == calibration['fit_latents_sha256']
    prior = torch.load(SOURCE/'inputs/joint-codebook.pt', map_location='cpu', weights_only=True)
    selected = torch.load(BASE/'atom-ranking.pt', weights_only=True)[:count]
    coefficients = torch.cat((prior['coefficients'][:, :1],
        prior['coefficients'][:, 1:].view(4, 16384, 4)[:, selected].flatten(1)), -1)
    atoms = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(count).repeat_interleave(4))).expand(4, -1)
    q = (build_quantizer(resume_from, device) if resume_from else
         JointAtomCoefficientRQ(prior['dictionary'][:, selected], atoms, coefficients).to(device))
    assert q.dictionary.shape[1] == count
    x = torch.from_numpy(np.load(BASE/'fit-latents.npy').reshape(-1, 256)).to(device)
    heldout = torch.from_numpy(np.load(CACHE/'temperature-latents.npy').reshape(-1, 256)).to(device)
    best, best_error, trace = None, float('inf'), []
    for iteration in range(passes+1):
        result = q.quantize(x, chunk_size=2048)
        error = float((x-result['quantized']).square().mean())
        validation_error = float((heldout-q.quantize(heldout, chunk_size=2048)['quantized']).square().mean())
        row = dict(iteration=iteration, fresh_greedy_latent_mse=error, heldout_training_latent_mse=validation_error)
        if validation_error < best_error:
            best_error = validation_error
            best = {key:getattr(q, key).detach().cpu().clone() for key in ('dictionary', 'atom_ids', 'coefficients')}
        if iteration < passes:
            row.update(refit_compact_book(q, x, result['codes']))
        trace.append(row);progress('fit', candidate=candidate, **row)
        write_json(OUT/(candidate+'-fit-progress.json'), dict(trace=trace, best_mse=best_error))
        del result
    spec = dict(best, kind='laser_joint_atom_coefficient_rq', format_version=2,
        source_stage1_checkpoint_sha256=calibration['source_stage1_sha256'],
        source_joint_codebook_sha256=calibration['source_codebook_sha256'],
        learned_compact_dictionary=True, encoder_decoder_frozen=True,
        fitting_views=8192, fitting_split='train', coefficient_gain=1.,
        offline_fit_resume_from=str(resume_from) if resume_from else None,
        selection='MSE on128 separate training images; no validation images used during fitting')
    torch.save(spec, path)
    torch.save(dict(spec, coefficients=spec['coefficients']*1.1, coefficient_gain=1.1),
        OUT/(candidate+'-gain110-codebook.pt'))
    write_json(OUT/(candidate+'-fit-final.json'), dict(completed=True, trace=trace, best_mse=best_error,
        vocabulary=1+4*count, atoms=count, codebook_sha256=sha256(path), stage2_launched=False,
        selection='heldout_training_latent_mse'))


@torch.inference_mode()
def screen(device, count, candidate_name=None):
    from src import rqvae_metrics as metrics
    metrics.INCEPTION_PATH = SOURCE/'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    aux = FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt', SOURCE/'inputs/joint-codebook.pt').to(device)
    real = np.load(CACHE/'screen-real-features.npy')
    latents = np.load(CACHE/'screen-latents.npy', mmap_mode='r')
    assert real.shape == (2048, 2048) and len(latents) == 2048
    candidate = candidate_name or f'compact{count}'
    variants = [('baseline', SOURCE/'inputs/joint-codebook.pt')] if count==0 else [
        (candidate, OUT/f'{candidate}-codebook.pt'),
        (candidate+'-gain110', OUT/f'{candidate}-gain110-codebook.pt')]
    results = {}
    for name, path in variants:
        q = build_quantizer(path, device)
        fake, error = [], 0.
        for first in range(0, len(latents), 16):
            x = torch.from_numpy(latents[first:first+16].copy()).to(device)
            latent = q.quantize(x, chunk_size=1024)['quantized']
            decoded = aux.decode(latent)
            value = inception(decoded.mul(.5).add(.5).clamp(0, 1))
            fake.append((value[0] if isinstance(value, tuple) else value).cpu().numpy())
            error += float((x-latent).square().sum())
            if first%512==0:progress('screen', candidate=name, images=first+len(x))
        fake = np.concatenate(fake)
        fid = float(metrics.frechet_distance(fake.mean(0), np.cov(fake, rowvar=False), real.mean(0), np.cov(real, rowvar=False)))
        results[name] = dict(matched_subset_rfid=fid, images=2048, latent_mse=error/latents.size,
            vocabulary=q.vocab_size, codebook_path=str(path), codebook_sha256=sha256(path))
        progress('screen_result', candidate=name, **results[name])
    write_json(OUT/f'screen-{candidate if count else 0}-final.json', dict(results=results, selection_only=True,
        full_50000_image_rfid_required=True, stage2_launched=False))


@torch.inference_mode()
def full(device, path):
    import torch.distributed as dist
    from src import rqvae_metrics as metrics
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    dist.init_process_group('nccl')
    metrics.INCEPTION_PATH = SOURCE/'upstream/rqvae/metrics/inception.py'
    aux = FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt', SOURCE/'inputs/joint-codebook.pt').to(device)
    q = build_quantizer(path, device)
    config = load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(SOURCE/'data', 'val', create_transforms(config.dataset, split='val'))
    assert len(data) == 50000
    loader = DataLoader(Subset(data, range(rank, len(data), world)), batch_size=16, num_workers=4, pin_memory=True)
    metric = metrics.DistributedOriginalRQVAEMetrics(device)
    count = 0
    name = path.name.removesuffix('-codebook.pt')
    for images, _ in loader:
        images = images.to(device)
        latent = aux.encode(images)
        result = q.quantize(latent, chunk_size=1024)
        recon = aux.decode(result['quantized'])
        metric.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
        metric.update(recon.mul(.5).add(.5).clamp(0, 1), real=False)
        count += len(images)
        if rank==0 and count%512==0:progress('full_rfid', candidate=name, images_on_rank=count,
            total_on_rank=len(range(rank, len(data), world)))
    assert count == len(range(rank, len(data), world))
    assert int(metric.real_count) == int(metric.fake_count) == count
    value, _, _ = metric.compute()
    assert int(metric.real_count) == int(metric.fake_count) == 50000
    if rank==0:
        write_json(OUT/(name+'-full-rfid-final.json'), dict(completed=True, passed=value<=4.73,
            measured_rfid=value, maximum_rfid=4.73, real_images=50000, reconstructed_images=50000,
            reference='full ImageNet validation, same original RQ Inception', encoder_decoder_frozen=True,
            learned_compact_dictionary=True, vocabulary=q.vocab_size, codebook_sha256=sha256(path),
            codebook_path=str(path), stage2_launched=False, time=time.time()))
        print(json.dumps(dict(candidate=name, full_rfid=value, gate_passed=value<=4.73)), flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['cache', 'fit', 'screen', 'full'])
    parser.add_argument('--atoms', type=int, choices=[0,4096,8192], default=8192)
    parser.add_argument('--passes', type=int, default=6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--codebook', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--candidate-name')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True);BASE.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if args.action=='full' else torch.device(args.device)
    torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='cache':cache(device)
    elif args.action=='fit':fit(device, args.atoms, args.passes, args.resume_from, args.candidate_name)
    elif args.action=='screen':screen(device, args.atoms, args.candidate_name)
    else:full(device, args.codebook)


if __name__=='__main__':main()
