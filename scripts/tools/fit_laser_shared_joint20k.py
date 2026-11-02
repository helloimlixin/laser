#!/usr/bin/env python3
"""Fit and validate one shared sparse integer book on all four GPUs."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
SOURCE = Path('/tmp/laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018')
LOCAL = Path('/tmp/laser-sharedjoint20k-depth5-20261018')
OUT = ROOT / 'outputs/imagenet-laser-sharedjoint20k-depth5-codebook-20261018'
os.environ['RQ_UPSTREAM_ROOT'] = str(SOURCE / 'upstream')
os.environ['TORCH_HOME'] = str(SOURCE / 'torch-cache')
sys.path[:0] = [str(ROOT), str(SOURCE / 'upstream'), str(ROOT / 'runtime')]
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, JointAtomCoefficientRQ
from src.training.laser_joint_vocabulary import fit_coefficient_candidates, allocate_coefficients, pack_codebook
from src.training.laser_shared_joint_vocabulary import assert_shared_codewords, refit_shared_book
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    pending = p.with_name(p.name + f'.{os.getpid()}.pending')
    pending.write_text(json.dumps(dict(value, time=time.time()), indent=2, allow_nan=False) + '\n')
    pending.replace(p)


def progress(phase, **values):
    if dist.get_rank() == 0:
        write('progress.json', dict(phase=phase, **values))
        print(json.dumps(dict(phase=phase, **values)), flush=True)


@torch.inference_mode()
def cache(device):
    rank, world = dist.get_rank(), dist.get_world_size()
    indices = np.random.default_rng(26101820481).permutation(1281167)[:9216].tolist()
    fit_ids, held_ids = indices[:8192], indices[8192:]
    assert set(fit_ids).isdisjoint(held_ids)
    if rank == 0:
        write('calibration-data.json', dict(fitting_split='train', fit_images=8192,
            heldout_training_images=1024, fit_indices=fit_ids, heldout_training_indices=held_ids,
            augmentation='Original training crop and flip; deterministic image seed 261001, epoch 0',
            validation_images_used_for_fitting=0,
            source_stage1_sha256=digest(SOURCE / 'inputs/stage1-model.pt'),
            source_codebook_sha256=digest(SOURCE / 'inputs/joint-codebook.pt')))
    config = load_config(SOURCE / 'inputs/stage2-training-config.yaml')
    aux = FrozenJointLaserTokenizer(SOURCE / 'inputs/stage1-model.pt', SOURCE / 'inputs/joint-codebook.pt').to(device)
    data = TarImageNet(SOURCE / 'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    data.set_epoch(0)
    for split, selected in [('fit', fit_ids), ('held', held_ids)]:
        path = LOCAL / f'{split}-rank{rank}.npy'
        if path.exists():
            assert np.load(path, mmap_mode='r').shape == (len(selected[rank::world]), 8, 8, 256)
            continue
        loader = DataLoader(Subset(data, selected[rank::world]), batch_size=16, num_workers=4, pin_memory=True)
        chunks, count = [], 0
        for images, _ in loader:
            chunks.append(aux.encode(images.to(device)).cpu().numpy())
            count += len(images)
            progress('encoding_training_latents', split=split, images_on_rank=count,
                     target_on_rank=len(selected[rank::world]))
        pending = path.with_suffix('.pending.npy')
        np.save(pending, np.concatenate(chunks))
        pending.replace(path)
    del aux
    torch.cuda.empty_cache()
    dist.barrier()


def global_mse(q, signals):
    error = (signals - q.quantize(signals, chunk_size=512)['quantized']).double().square().sum()
    values = torch.stack([error, error.new_tensor(signals.numel())])
    dist.all_reduce(values)
    return float(values[0] / values[1])


@torch.inference_mode()
def usage(device):
    reports = {}
    rank = dist.get_rank()
    for name, path in [('control', SOURCE / 'inputs/joint-codebook.pt'),
                       ('shared', OUT / 'sharedjoint20k-depth5-codebook.pt')]:
        spec = torch.load(path, map_location='cpu', weights_only=True)
        q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)
        counts = []
        for split in ('fit', 'held'):
            x = torch.from_numpy(np.load(LOCAL / f'{split}-rank{rank}.npy')).reshape(-1,256).to(device)
            codes = q.quantize(x, chunk_size=512)['codes']
            histogram = torch.stack([torch.bincount(codes[:,d], minlength=q.vocab_size) for d in range(5)]).double()
            dist.all_reduce(histogram)
            counts.append(histogram)
        fit_counts, held_counts = counts
        probability = fit_counts / fit_counts.sum(-1,keepdim=True)
        fitted = (fit_counts + .1) / (fit_counts.sum(-1,keepdim=True) + .1*q.vocab_size)
        reports[name] = dict(fit_marginal_entropy_nats=(-(probability*probability.clamp_min(1.e-30).log()).sum(-1)).tolist(),
            heldout_unigram_cross_entropy_nats=(-(held_counts*fitted.log()).sum(-1)/held_counts.sum(-1)).tolist(),
            fit_active_ids_by_depth=(fit_counts>0).sum(-1).tolist(),
            fit_active_ids_pooled=int((fit_counts.sum(0)>0).sum()),
            null_probability_by_depth=probability[:,0].tolist())
    if rank == 0:
        write('code-usage.json', dict(completed=True, fit_images=8192, heldout_training_images=1024,
            same_images_and_latents=True, cases=reports,
            interpretation='Marginal token statistics only; no conditional predictability or generation claim'))


@torch.inference_mode()
def fit(args, device):
    cache(device)
    rank, world = dist.get_rank(), dist.get_world_size()
    x = torch.from_numpy(np.load(LOCAL / f'fit-rank{rank}.npy')).reshape(-1, 256).to(device)
    held = torch.from_numpy(np.load(LOCAL / f'held-rank{rank}.npy')).reshape(-1, 256).to(device)
    prior_spec = torch.load(SOURCE / 'inputs/joint-codebook.pt', map_location='cpu', weights_only=True)
    prior = JointAtomCoefficientRQ(prior_spec['dictionary'], prior_spec['atom_ids'], prior_spec['coefficients']).to(device)
    encoded = prior.quantize(x, chunk_size=512, return_projections=True)
    codes = encoded['codes']
    atoms = torch.stack([prior.atom_ids[d, codes[:, d]] for d in range(5)], -1)
    observed = codes != 0
    local = (atoms[observed].cpu(), encoded['projections'][observed].cpu())
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    packed = [None]
    if rank == 0:
        progress('fitting_one_pooled_coefficient_table', observed_codes=sum(len(v[0]) for v in gathered))
        all_atoms = torch.cat([v[0] for v in gathered]).to(device)
        all_values = torch.cat([v[1] for v in gathered]).to(device)
        books, errors = fit_coefficient_candidates(all_atoms, all_values, 8192, counts=(2, 4, 6, 8))
        allocation = allocate_coefficients(errors, 20480)
        spec = pack_codebook(prior.dictionary, [books] * 5, [allocation] * 5)
        packed[0] = spec
        write('allocation.json', dict(pooled_across_depths=True, physical_vectors_shared=True,
            allocation_histogram={str(n):allocation.count(n) for n in (2, 4, 6, 8)},
            atoms=8192, vocabulary=20481, depth=5))
        del all_atoms, all_values, books, errors
    dist.broadcast_object_list(packed, 0)
    spec = packed[0]
    q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)
    assert_shared_codewords(q)
    del encoded, codes, atoms, local, gathered, packed, spec, prior
    torch.cuda.empty_cache()
    best_error, best, trace = float('inf'), None, []
    for iteration in range(args.passes + 1):
        encoded = q.quantize(x, chunk_size=512)
        v = torch.stack([(x - encoded['quantized']).double().square().sum(), x.new_tensor(x.numel(), dtype=torch.float64)])
        dist.all_reduce(v)
        fit_error = float(v[0] / v[1])
        held_error = global_mse(q, held)
        row = dict(iteration=iteration, fit_latent_mse=fit_error, heldout_training_latent_mse=held_error)
        if held_error < best_error:
            best_error = held_error
            best = {key:getattr(q, key).cpu().clone() for key in ('dictionary', 'atom_ids', 'coefficients')}
        if iteration < args.passes:
            row.update(refit_shared_book(q, x, encoded['codes']))
        trace.append(row)
        assert_shared_codewords(q)
        if rank == 0:
            payload = dict(prior_spec, **best, shared_codebook_across_depths=True,
                first_four_depths_unchanged=False, fitting_split='train', fitting_views=8192,
                shared_coefficient_values=True, fitting_objective='Final latent error with exact sharing across depths',
                source_joint_codebook_sha256=digest(SOURCE / 'inputs/joint-codebook.pt'))
            pending = OUT / 'sharedjoint20k-depth5-codebook.pending.pt'
            torch.save(payload, pending)
            pending.replace(OUT / 'sharedjoint20k-depth5-codebook.pt')
            write('fit.json', dict(completed=iteration == args.passes, trace=trace,
                best_heldout_training_latent_mse=best_error, encoder_decoder_frozen=True,
                sharing_verified_after_every_iteration=True, coefficient_fitting_pooled_across_depths=True,
                vocabulary=20481, depth=5, world_size=world))
        progress('refitting_shared_codewords', **row)
        del encoded
    dist.barrier()
    if rank == 0:
        write('selected.json', dict(candidate='shared8192', codebook_path=str(OUT / 'sharedjoint20k-depth5-codebook.pt'),
            codebook_sha256=digest(OUT / 'sharedjoint20k-depth5-codebook.pt'), vocabulary=20481,
            depth=5, shared_codebook_across_depths=True, heldout_training_latent_mse=best_error,
            full_reconstruction_validation_required=True))


@torch.no_grad()
def full(device):
    from src import rqvae_metrics as metrics
    rank, world = dist.get_rank(), dist.get_world_size()
    chosen = json.loads((OUT / 'selected.json').read_text())
    path = Path(chosen['codebook_path'])
    assert digest(path) == chosen['codebook_sha256']
    aux = FrozenJointLaserTokenizer(SOURCE / 'inputs/stage1-model.pt', path).to(device)
    assert_shared_codewords(aux.quantizer)
    versions = [value._version for value in aux.state_dict().values()]
    metrics.INCEPTION_PATH = SOURCE / 'upstream/rqvae/metrics/inception.py'
    metric = metrics.DistributedOriginalRQVAEMetrics(device)
    config = load_config(SOURCE / 'inputs/stage2-training-config.yaml')
    data = TarImageNet(SOURCE / 'data', 'val', create_transforms(config.dataset, split='val'))
    assert len(data) == 50000
    loader = DataLoader(Subset(data, range(rank, len(data), world)), batch_size=16, num_workers=4, pin_memory=True)
    count = 0
    for images, _ in loader:
        images = images.to(device)
        reconstruction, _, _ = aux(images)
        metric.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
        metric.update(reconstruction.mul(.5).add(.5).clamp(0, 1), real=False)
        count += len(images)
        if count % 512 == 0:
            progress('full_reconstruction_fid', images_on_rank=count, target_on_rank=12500)
    fid, _, _ = metric.compute()
    assert int(metric.real_count) == int(metric.fake_count) == 50000
    assert versions == [value._version for value in aux.state_dict().values()]
    assert_shared_codewords(aux.quantizer)
    if rank == 0:
        write('full-rfid.json', dict(completed=True, measured_rfid=fid, maximum_rfid=5.25, passed=fid <= 5.25,
            real_images=50000, reconstructed_images=50000, encoder_decoder_frozen=True,
            physical_vectors_exactly_shared=True, codebook_path=str(path), codebook_sha256=chosen['codebook_sha256'],
            vocabulary=20481, depth=5, reference='Full ImageNet validation; original RQ Inception',
            prior_generation_quality_not_measured=True))
        progress('reconstruction_validation_completed', measured_rfid=fid, passed=fid <= 5.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'full', 'usage'))
    parser.add_argument('--passes', type=int, default=24)
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(16 * 2**30 / torch.cuda.get_device_properties(device).total_memory, device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=60))
    assert dist.get_world_size() == 4
    torch.manual_seed(26101820481)
    {'fit':lambda:fit(args, device), 'full':lambda:full(device), 'usage':lambda:usage(device)}[args.phase]()
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
