#!/usr/bin/env python3
"""Fit smaller per-OMP-level dictionaries before training an integer-token prior."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
CODE_ROOT = Path(os.environ.get('LASER_DEPTH_SOURCE_ROOT', ROOT))
sys.path.insert(0, str(CODE_ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from src.models.laser_depth_integer_tokenizer import DepthIntegerCodebooks
from src.training.laser_depth_codebook_fit import initialize_depth_books, refit_depth_books
from src.training.laser_hard_ar import nearest_coefficient_ids, sparse_encoding

OUT = ROOT/'outputs/imagenet-laser-depth-pair-codebooks-20261017'
CACHE = Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
FIT = Path('/tmp/laser-compact-joint-vocabulary-20261016/fit-latents.npy')
SOURCE = Path('/tmp/laser-rq-native-joint-depthtemp-imagenet-20261016')
SCALES = [8.203365325927734, 4.265638828277588, 3.0662174224853516, 1.8273425102233887]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def progress(phase, candidate, **values):
    row = dict(values, phase=phase, candidate=candidate, time=time.time())
    write(OUT/(candidate+'-progress.json'), row)
    print(json.dumps(row), flush=True)


def save(path, spec):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(spec, tmp)
    tmp.replace(path)


def build(path, device):
    spec = torch.load(path, map_location='cpu', weights_only=True)
    return DepthIntegerCodebooks(spec['dictionaries'], spec['scales'],
        coefficient_bins=spec['coefficient_bins'], coefficient_max=spec['coefficient_max'],
        ridge=spec['ridge']).to(device)


@torch.inference_mode()
def validate_inputs(device):
    meta = json.loads((ROOT/'outputs/imagenet-laser-compact-joint-vocabulary-20261016/calibration-data.json').read_text())
    old = json.loads((ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015/calibration-data.json').read_text())
    assert meta['fitting_training_only'] and set(meta['fit_indices']).isdisjoint(old['temperature_indices'])
    assert sha(FIT) == meta['fit_latents_sha256']
    assert sha(CACHE/'temperature-latents.npy') == old['temperature_latents_sha256']
    assert sha(SOURCE/'inputs/stage1-model.pt') == meta['source_stage1_sha256']
    source = torch.load(CACHE/'dictionary.pt', map_location='cpu', weights_only=True)
    state = torch.load(SOURCE/'inputs/stage1-model.pt', map_location='cpu', mmap=True, weights_only=False)['state_dict']
    key = 'quantizer.dictionary' if 'quantizer.dictionary' in state else 'bottleneck.dictionary'
    torch.testing.assert_close(source, F.normalize(state[key].float(), dim=0), atol=0, rtol=0)
    del state
    write(OUT/'input-verification.json', dict(fit_latents_sha256=meta['fit_latents_sha256'],
        heldout_latents_sha256=old['temperature_latents_sha256'], dictionary_sha256=sha(CACHE/'dictionary.pt'),
        source_stage1_sha256=meta['source_stage1_sha256'], fit_views=8192, unique_fit_images=4096,
        heldout_training_images=128, disjoint=True, fitting_split='train', encoder_decoder_frozen=True))
    return source.to(device), meta


@torch.inference_mode()
def source_energy(x, held, source, device):
    path = OUT/'source-depth-energy.pt'
    if path.exists():
        spec = torch.load(path, map_location=device, weights_only=True)
        assert spec['fit_latents_sha256'] == sha(FIT)
        assert spec['dictionary_sha256'] == sha(CACHE/'dictionary.pt')
        return spec['energy']
    energy = x.new_zeros(4, source.shape[1])
    gram = (source.T @ source).contiguous()
    for first in range(0, len(x), 512):
        atoms, values = sparse_encoding(x[first:first+512], source, 4, gram=gram)
        for d in range(4):
            energy[d].index_add_(0, atoms[:, d], values[:, d].square())
        if first % 32768 == 0:
            progress('source_depth_statistics', 'source', latent_sites=first+len(atoms), total_sites=len(x))
    scales = x.new_tensor(SCALES)
    bins = torch.linspace(-3., 3., 512, device=device)
    continuous, quantized = 0., 0.
    clipped = x.new_zeros(4)
    for chunk in held.split(512):
        a, c = sparse_encoding(chunk, source, 4, gram=gram)
        continuous += float((chunk-(source.T[a]*c[..., None]).sum(-2)).double().square().sum())
        ids = nearest_coefficient_ids(c/scales, bins)
        quantized += float((chunk-(source.T[a]*(bins[ids]*scales)[..., None]).sum(-2)).double().square().sum())
        clipped += (c.abs() > scales*3.).sum(0)
    write(OUT/'source-heldout-training-baseline.json', dict(continuous_latent_mse=continuous/held.numel(),
        quantized_latent_mse=quantized/held.numel(), clipped_coefficient_fraction=(clipped/len(held)).tolist(),
        atoms=16384, dictionary_shared_across_depth=True, coefficient_bins=512,
        heldout_training_images=128, reconstruction_fid_not_recomputed=True))
    save(path, dict(energy=energy.cpu(), fit_latents_sha256=sha(FIT), dictionary_sha256=sha(CACHE/'dictionary.pt')))
    return energy


@torch.inference_mode()
def measure(q, x, *, return_codes=False):
    atoms, values = q.encode_components(x)
    continuous, quantized = 0., 0.
    clip = torch.zeros(q.depth, device=x.device, dtype=torch.long)
    for first in range(0, len(x), 4096):
        stop = first+4096
        a, c = atoms[first:stop], values[first:stop]
        continuous += float((x[first:stop]-q.reconstruct(a, c)).double().square().sum())
        normalized = c/q.scales
        bins = nearest_coefficient_ids(normalized, q.bins)
        quantized += float((x[first:stop]-q.reconstruct(a, q.bins[bins]*q.scales)).double().square().sum())
        clip += (normalized.abs() > float(q.bins[-1])).sum(0)
    row = dict(continuous_latent_mse=continuous/x.numel(), quantized_latent_mse=quantized/x.numel(),
        clipped_coefficient_fraction=(clip.double()/len(x)).tolist(),
        used_atoms=[int(atoms[:, d].unique().numel()) for d in range(q.depth)])
    return (row, atoms, values) if return_codes else row


@torch.inference_mode()
def fit(args, device):
    candidate = f'depth4-atoms{args.atoms}-bins512'
    final_path = OUT/(candidate+'-codebook.pt')
    if final_path.exists():
        raise RuntimeError('Refusing to overwrite fitted depth codebooks')
    source, meta = validate_inputs(device)
    x = torch.from_numpy(np.load(FIT).reshape(-1, 256)).to(device)
    held = torch.from_numpy(np.load(CACHE/'temperature-latents.npy').reshape(-1, 256)).to(device)
    energy = source_energy(x, held, source, device)
    progress('initializing_signed_ray_clusters', candidate, atoms_per_book=args.atoms, books=4)
    books = initialize_depth_books(source, energy, args.atoms, iterations=8)
    del source, energy
    torch.cuda.empty_cache()
    q = DepthIntegerCodebooks(books, SCALES).to(device)
    spec = dict(kind='laser_depth_integer_codebooks', format_version=1,
        dictionaries=q.dictionaries.cpu().clone(), scales=q.scales.cpu().clone(),
        coefficient_bins=512, coefficient_max=3., ridge=q.ridge, source_atoms=16384,
        source_stage1_sha256=meta['source_stage1_sha256'], encoder_decoder_frozen=True,
        fit_split='train', fit_views=8192, atoms_per_book=args.atoms, total_atoms=args.atoms*4,
        pair_id='local_atom * 512 + coefficient_bin',
        encoding='one atom per depth-specific book, deterministic grouped OMP with final LS',
        physical_embedding='dictionary[depth,:,local_atom] * scales[depth] * bins[coefficient_bin]',
        selection='quantized latent MSE on 128 disjoint held-out training images',
        fitting_source_sha256={str(p.relative_to(CODE_ROOT)):sha(p) for p in [Path(__file__),
            CODE_ROOT/'src/models/laser_depth_integer_tokenizer.py', CODE_ROOT/'src/training/laser_depth_codebook_fit.py']})
    trace, best = [], float('inf')
    for iteration in range(args.passes+1):
        started = time.time()
        train, atoms, values = measure(q, x, return_codes=True)
        heldout = measure(q, held)
        row = dict(iteration=iteration, training=train, heldout_training=heldout)
        if heldout['quantized_latent_mse'] < best:
            best = heldout['quantized_latent_mse']
            spec.update(dictionaries=q.dictionaries.cpu().clone(), best_iteration=iteration,
                        heldout_quantized_latent_mse=best)
            save(OUT/(candidate+'-best-codebook.pt'), spec)
        if iteration < args.passes:
            row.update(refit_depth_books(q, x, atoms, values))
        row['elapsed_seconds'] = time.time()-started
        trace.append(row)
        progress('dictionary_refitting', candidate, best_heldout_quantized_mse=best, **row)
        write(OUT/(candidate+'-fit-trace.json'), dict(trace=trace, best_heldout_quantized_mse=best))
    save(final_path, spec)
    final = dict(completed=True, atoms_per_book=args.atoms, total_atoms=4*args.atoms, books=4,
        coefficient_bins=512, vocab_per_book=q.vocab_size, total_class_rows=4*q.vocab_size,
        best_iteration=spec['best_iteration'], heldout_quantized_latent_mse=best,
        codebook_path=str(final_path), codebook_sha256=sha(final_path),
        fitting_split='train', encoder_decoder_frozen=True, stage2_training_launched=False,
        full_reconstruction_fid_validation_required=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
    write(OUT/(candidate+'-fit-final.json'), final)
    progress('fit_completed', candidate, **final)


@torch.inference_mode()
def evaluate(args, device):
    os.environ['RQ_UPSTREAM_ROOT'] = str(SOURCE/'upstream')
    os.environ.setdefault('TORCH_HOME', str(SOURCE/'torch-cache'))
    sys.path[:0] = [str(SOURCE/'upstream'), str(CODE_ROOT/'runtime')]
    from src.models.laser_depth_integer_tokenizer import FrozenDepthIntegerLaserTokenizer
    from src import rqvae_metrics as metrics
    from torch.utils.data import DataLoader, Subset
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config
    from src.training.rq_imagenet_runtime import TarImageNet
    import torch.distributed as dist
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        dist.init_process_group('nccl')
    metrics.INCEPTION_PATH = SOURCE/'upstream/rqvae/metrics/inception.py'
    aux = FrozenDepthIntegerLaserTokenizer(SOURCE/'inputs/stage1-model.pt', args.codebook).to(device)
    metric = metrics.DistributedOriginalRQVAEMetrics(device, compute_inception_score=False)
    candidate = args.codebook.name.removesuffix('-codebook.pt')
    config = load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(SOURCE/'data', 'val', create_transforms(config.dataset, split='val'))
    assert len(data) == 50000
    indices = list(range(rank, len(data), world))
    if args.action == 'screen':
        meta = json.loads((ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015/calibration-data.json').read_text())
        indices = meta['screen_indices'][rank::world]
    loader = DataLoader(Subset(data, indices), batch_size=args.batch_size, num_workers=2, pin_memory=True)
    count, latent_error = 0, 0.
    for images, _ in loader:
        images = images.to(device)
        latent = aux.encode(images)
        result = aux.quantizer.quantize(latent, chunk_size=512)
        recon = aux.decode(result['quantized'])
        metric.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
        metric.update(recon.mul(.5).add(.5).clamp(0, 1), real=False)
        latent_error += float((latent-result['quantized']).double().square().sum())
        count += len(images)
        if rank == 0 and (count % 256 == 0 or count == len(indices)):
            progress(args.action+'_reconstruction_fid', candidate, images_on_rank=count,
                target_on_rank=len(indices), world_size=world)
    assert count == len(indices)
    fid, _, _ = metric.compute()
    error = torch.tensor(latent_error, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(error)
    total = 50000 if args.action == 'full' else 2048
    assert int(metric.fake_count) == int(metric.real_count) == total
    if rank == 0:
        report = dict(completed=True, measured_rfid=fid, real_images=total, reconstructed_images=total,
            full_50000_reconstruction_fid=args.action=='full', selection_only=args.action!='full',
            latent_mse=float(error)/(total*8*8*256),
            reference='matched ImageNet validation images, original RQ Inception',
            codebook_sha256=sha(args.codebook), encoder_decoder_frozen=True,
            evaluator_sha256=sha(Path(__file__)),
            vocabulary_per_depth=aux.vocab_size, total_atoms=aux.quantizer.depth*aux.quantizer.atoms_per_book,
            stage2_training_launched=False, time=time.time(),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
        write(OUT/(candidate+'-'+args.action+'-rfid.json'), report)
        progress(args.action+'_completed', candidate, **report)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fit', 'screen', 'full'])
    parser.add_argument('--atoms', type=int, choices=[1024, 2048, 4096], default=2048)
    parser.add_argument('--passes', type=int, default=12)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--codebook', type=Path)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--memory-fraction', type=float, default=.05)
    args = parser.parse_args()
    if args.action != 'fit' and args.codebook is None:
        parser.error('--codebook is required for reconstruction evaluation')
    if args.passes < 1 or args.batch_size < 1:
        parser.error('Positive fitting passes and batch size required')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if 'LOCAL_RANK' in os.environ else torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == 'fit':
        fit(args, device)
    else:
        evaluate(args, device)


if __name__ == '__main__':
    main()
