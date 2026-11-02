#!/usr/bin/env python3
"""Calibrate compact integer pair catalogs without changing source OMP support."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
CODE = Path(os.environ.get('LASER_CATALOG_SOURCE_ROOT', ROOT))
sys.path.insert(0, str(CODE))
import numpy as np
import torch

from src.training.laser_catalog_calibration import fit_catalog_bins
from src.training.laser_hard_ar import nearest_coefficient_ids, sparse_encoding

OUT = ROOT/'outputs/imagenet-laser-catalog512-vocabulary-20261017'
BASE = Path('/tmp/laser-catalog512-vocabulary-20261017')
SOURCE = Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3')
CACHE = Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
FIT = Path('/tmp/laser-compact-joint-vocabulary-20261016/fit-latents.npy')
SCALES = [8.203365325927734, 4.265638828277588, 3.0662174224853516, 1.8273425102233887]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, data):
    path = OUT/name
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def progress(phase, **values):
    value = dict(values, phase=phase, time=time.time())
    write('progress.json', value)
    print(json.dumps(value), flush=True)


@torch.inference_mode()
def components(x, dictionary, gram):
    result = [sparse_encoding(chunk, dictionary, 4, gram=gram) for chunk in x.split(512)]
    return tuple(torch.cat([row[index] for row in result]) for index in (0, 1))


@torch.inference_mode()
def latent_error(x, dictionary, atoms, physical_values):
    total = 0.
    for first in range(0, len(x), 2048):
        a, c = atoms[first:first+2048], physical_values[first:first+2048]
        total += float((x[first:first+2048]-(dictionary.T[a]*c[..., None]).sum(-2)).double().square().sum())
    return total/x.numel()


@torch.inference_mode()
def fit(device):
    manifest = json.loads((ROOT/'outputs/imagenet-laser-compact-joint-vocabulary-20261016/calibration-data.json').read_text())
    held_meta = json.loads((ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015/calibration-data.json').read_text())
    assert sha(FIT) == manifest['fit_latents_sha256']
    assert sha(CACHE/'temperature-latents.npy') == held_meta['temperature_latents_sha256']
    assert set(manifest['fit_indices']).isdisjoint(held_meta['temperature_indices'])
    assert sha(SOURCE/'inputs/stage1-model.pt') == manifest['source_stage1_sha256']
    dictionary = torch.load(CACHE/'dictionary.pt', map_location=device, weights_only=True)
    scales = dictionary.new_tensor(SCALES)
    bins = torch.linspace(-3., 3., 512, device=device)
    x = torch.from_numpy(np.load(FIT).reshape(-1, 256)).to(device)
    held = torch.from_numpy(np.load(CACHE/'temperature-latents.npy').reshape(-1, 256)).to(device)
    progress('source_omp_encoding', training_latent_sites=len(x), training_views=8192)
    gram = dictionary.T @ dictionary
    atoms, values = components(x, dictionary, gram)
    ha, hv = components(held, dictionary, gram)
    baseline = latent_error(held, dictionary, ha, bins[nearest_coefficient_ids(hv/scales, bins)]*scales)
    del gram
    trace = []
    for levels in (8, 16):
        progress('fitting_joint_integer_catalog', levels_per_atom=levels, vocabulary=16384*levels)
        ids = fit_catalog_bins(atoms, values/scales, scales, 16384, bins, levels)
        spec = dict(kind='laser_catalog_pair_integer', format_version=1,
            bin_ids=ids.cpu(), source_stage1_sha256=manifest['source_stage1_sha256'],
            source_dictionary_sha256=sha(CACHE/'dictionary.pt'), source_atoms=16384,
            coefficient_bins=512, coefficient_max=3., coefficient_scales=SCALES,
            depth=4, levels_per_atom=levels, vocab_size=16384*levels,
            fitting_split='train', fitting_views=8192, unique_fitting_images=4096,
            fitting_latents_sha256=manifest['fit_latents_sha256'],
            heldout_training_latents_sha256=held_meta['temperature_latents_sha256'],
            source_omp_support_unchanged=True, source_final_ls_coefficients=True,
            encoder_decoder_dictionary_frozen=True, id_mapping='atom * levels_per_atom + catalog_slot',
            same_atom_and_coefficient_bin_at_every_depth=True,
            full_cartesian_vocabulary=False, exact_pair_lookup='bin_ids[atom, catalog_slot]',
            calibration_source_sha256={str(p.relative_to(CODE)):sha(p) for p in [Path(__file__),
                CODE/'src/training/laser_catalog_calibration.py']})
        candidates = bins[ids[ha]]
        slot = (candidates-(hv/scales)[..., None]).abs().argmin(-1)
        quantized = candidates.gather(-1, slot[..., None]).squeeze(-1)*scales
        error = latent_error(held, dictionary, ha, quantized)
        path = OUT/f'catalog{16384*levels}-grid512.pt'
        assert not path.exists(), 'Refusing to overwrite a calibrated catalog'
        torch.save(spec, path)
        row = dict(levels_per_atom=levels, vocabulary=16384*levels,
            source_heldout_latent_mse=baseline, catalog_heldout_latent_mse=error,
            relative_latent_error_increase=error/baseline-1., catalog_path=str(path), catalog_sha256=sha(path))
        trace.append(row)
        progress('catalog_fit_complete', **row)
    selected = min((row for row in trace if row['relative_latent_error_increase'] <= .10),
                   key=lambda row:row['vocabulary'], default=trace[-1])
    write('fit-report.json', dict(completed=True, candidates=trace, selected=selected,
        source_heldout_latent_mse=baseline, full_50000_reconstruction_fid_required=True,
        fitting_split='train', stage2_training_launched=False,
        source_omp_support_unchanged=True, coefficient_grid_bins=512,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30))


@torch.inference_mode()
def evaluate(args, device):
    os.environ['RQ_UPSTREAM_ROOT'] = str(SOURCE/'upstream')
    os.environ.setdefault('TORCH_HOME', str(SOURCE/'torch-cache'))
    sys.path[:0] = [str(SOURCE/'upstream'), str(CODE/'runtime')]
    from torch.utils.data import DataLoader, Subset
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config
    from src.models.laser_catalog_integer_tokenizer import FrozenCatalogIntegerLaserTokenizer
    from src.training.rq_imagenet_runtime import TarImageNet
    from src import rqvae_metrics as metrics
    import torch.distributed as dist
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        dist.init_process_group('nccl')
    aux = FrozenCatalogIntegerLaserTokenizer(SOURCE/'inputs/stage1-model.pt', args.catalog).to(device)
    metrics.INCEPTION_PATH = SOURCE/'upstream/rqvae/metrics/inception.py'
    metric = metrics.DistributedOriginalRQVAEMetrics(device)
    config = load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(SOURCE/'data', 'val', create_transforms(config.dataset, split='val'))
    assert len(data) == 50000
    indices = list(range(rank, len(data), world))
    if args.action == 'screen':
        meta = json.loads((ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015/calibration-data.json').read_text())
        indices = meta['screen_indices'][rank::world]
    loader = DataLoader(Subset(data, indices), batch_size=args.batch_size, num_workers=2, pin_memory=True)
    count = 0
    for images, _ in loader:
        images = images.to(device)
        recon, _, codes = aux(images)
        metric.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
        metric.update(recon.mul(.5).add(.5).clamp(0, 1), real=False)
        count += len(images)
        if rank == 0 and (count % 512 == 0 or count == len(indices)):
            progress(args.action+'_reconstruction_fid', images_on_rank=count, target_on_rank=len(indices),
                world_size=world, vocabulary=aux.vocab_size)
    assert count == len(indices)
    fid, _, _ = metric.compute()
    total = 50000 if args.action == 'full' else 2048
    assert int(metric.real_count) == int(metric.fake_count) == total
    if rank == 0:
        report = dict(completed=True, measured_rfid=fid, real_images=total, reconstructed_images=total,
            full_50000_reconstruction_fid=args.action=='full', selection_only=args.action!='full',
            catalog_sha256=sha(args.catalog), catalog_path=str(args.catalog), vocabulary=aux.vocab_size,
            coefficient_bins=512, levels_per_atom=aux.levels_per_atom,
            source_omp_support_unchanged=True, encoder_decoder_dictionary_frozen=True,
            source_full_rfid=4.214186495529759, source_rfid_not_recomputed_here=True,
            rfid_drift=fid-4.214186495529759 if args.action=='full' else None,
            reference='matched full validation, original RQ Inception' if args.action=='full' else 'matched2048 subset, selection only',
            stage2_training_launched=False, evaluator_sha256=sha(Path(__file__)),time=time.time())
        write(f'catalog{aux.vocab_size}-{args.action}-rfid.json', report)
        progress(args.action+'_completed', **report)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['fit', 'screen', 'full'])
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--memory-fraction', type=float, default=.05)
    args = parser.parse_args()
    if args.action != 'fit' and args.catalog is None:
        parser.error('--catalog required')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if 'LOCAL_RANK' in os.environ else torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    if args.action == 'fit':
        fit(device)
    else:
        evaluate(args, device)


if __name__ == '__main__':
    main()
