"""Fit free vector codewords on training data, then measure native RQ quality."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from environment import BASE, SOURCE, QUALITY, VOCABULARY

sys.path[:0] = [str(BASE / 'source'), str(BASE)]
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src import rq_upstream, rqvae_metrics as metrics
from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config
from runtime.tokenizer import NativeVectorTokenizer, file_digest, module_digest


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temp.replace(path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    torch.save(value, temp)
    temp.replace(path)


def progress(phase, rank, **kw):
    row = dict(phase=phase, rank=rank, time=time.time(), **kw)
    write(BASE / 'quality' / f'{phase}-rank{rank}.json', row)
    if rank == 0:
        write(QUALITY / 'progress.json', row)
        print(json.dumps(row), flush=True)


def source_aux(device):
    return RFIDPreservedJointTokenizer(SOURCE / 'inputs/stage1-model.pt',
                                      SOURCE / 'inputs/joint-codebook.pt',
                                      SOURCE / 'inputs/reconstruction-codebook.pt').to(device)


def dataset(split, transform_split):
    config = load_config(BASE / 'inputs/stage2-training-config.yaml')
    return TarImageNet(BASE / 'data', split, create_transforms(config.dataset, split=transform_split),
                      seed=26110824576)


@torch.inference_mode()
def cache(rank, world, device):
    aux = source_aux(device)
    training = dataset('train', 'train')
    rng = np.random.default_rng(26110824576)
    fitting, heldout = [], []
    for label in range(1000):
        selected = rng.choice(np.flatnonzero(training.labels == label), 60, replace=False)
        fitting.extend(selected[:50].tolist())
        heldout.extend(selected[50:].tolist())
    rng.shuffle(fitting)
    rng.shuffle(heldout)
    assert len(set(fitting)) == 50000 and len(set(heldout)) == 10000
    assert set(fitting).isdisjoint(heldout)
    if rank == 0:
        np.save(QUALITY / 'fit-indices.npy', fitting)
        np.save(QUALITY / 'heldout-training-indices.npy', heldout)
        save(BASE / 'quality/source-vectors.pt', torch.stack([
            aux.physical_book.expanded_codebook(d).cpu() for d in range(4)]))
    metrics.INCEPTION_PATH = BASE / 'upstream/rqvae/metrics/inception.py'
    inception = None
    for name, indices, data in [('fit', fitting, training), ('held', heldout, dataset('train', 'val'))]:
        part = indices[rank::world]
        path = BASE / 'cache' / f'{name}-latents-rank{rank}.npy'
        path.parent.mkdir(exist_ok=True)
        values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                           shape=(len(part), 8, 8, 256))
        features = None
        if name == 'held':
            inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
            features = np.lib.format.open_memmap(BASE / 'cache' / f'held-features-rank{rank}.npy',
                                                 mode='w+', dtype=np.float32, shape=(len(part), 2048))
        loader = DataLoader(Subset(data, part), batch_size=32, num_workers=4,
                            pin_memory=True, persistent_workers=True, prefetch_factor=1,
                            multiprocessing_context='spawn')
        first = 0
        for images, _ in loader:
            images = images.to(device, non_blocking=True).float()
            latent = aux.encode(images)
            values[first:first + len(images)] = latent.cpu().numpy()
            if features is not None:
                value, _ = inception(images.mul(.5).add(.5).clamp(0, 1))
                features[first:first + len(images)] = value.cpu().numpy()
            first += len(images)
            if first % 512 == 0 or first == len(part):
                progress('cache', rank, split=name, images_on_rank=first, target_on_rank=len(part))
        assert first == len(part)
        values.flush()
        if features is not None:
            features.flush()
        del values, features, loader
    dist.barrier()
    if rank == 0:
        report = dict(passed=True, fitting_images=50000, heldout_training_images=10000,
                      classes=1000, fit_images_per_class=50, heldout_images_per_class=10,
                      disjoint=True, validation_used_for_fitting=False,
                      fitting_views='RandomCrop256 and horizontal flip',
                      heldout_views='official RQ evaluation transforms on disjoint training images',
                      seed=26110824576, source_cnn_digest=module_digest(aux.backbone),
                      stage1_checkpoint_sha256=file_digest(BASE / 'inputs/stage1-model.pt'),
                      source_codebook_sha256=file_digest(SOURCE / 'inputs/joint-codebook.pt'),
                      world_size=world, beam_search_used=False)
        write(QUALITY / 'cache-verification.json', report)


@torch.no_grad()
def greedy(x, vectors, chunk=4096):
    residual = x.reshape(-1, 256).clone()
    codes = torch.empty(len(residual), 4, device=x.device, dtype=torch.long)
    for depth in range(4):
        table = vectors[depth]
        norms = table.square().sum(-1)
        for first in range(0, len(residual), chunk):
            r = residual[first:first + chunk]
            # Identical distance expression to upstream VQEmbedding.
            distance = torch.addmm(r.square().sum(-1, keepdim=True) + norms,
                                   r, table.T, alpha=-2.)
            chosen = distance.argmin(-1)
            codes[first:first + len(r), depth] = chosen
            r.sub_(table[chosen])
    reconstruction = torch.stack([vectors[d, codes[:, d]] for d in range(4)], 1).sum(1)
    return codes, reconstruction


@torch.inference_mode()
def fit(rank, world, device, passes):
    x = torch.from_numpy(np.load(BASE / 'cache' / f'fit-latents-rank{rank}.npy')).to(device).reshape(-1, 256)
    held = torch.from_numpy(np.load(BASE / 'cache' / f'held-latents-rank{rank}.npy')).to(device).reshape(-1, 256)
    source = torch.load(BASE / 'quality/source-vectors.pt', map_location=device, weights_only=True)
    assert source.shape == (4, 16384, 256)
    vectors = x.new_zeros(4, VOCABULARY, 256)
    vectors[:, 1:16385] = source
    residual = x.clone()
    generator = torch.Generator(device=device).manual_seed(26110824576 + rank)
    for depth in range(4):
        old = source[depth]
        norms = old.square().sum(-1)
        errors = torch.empty(len(residual), device=device)
        choices = torch.empty(len(residual), device=device, dtype=torch.long)
        for first in range(0, len(residual), 4096):
            r = residual[first:first + 4096]
            distance = torch.addmm(r.square().sum(-1, keepdim=True) + norms, r, old.T, alpha=-2.)
            errors[first:first + len(r)], choices[first:first + len(r)] = distance.min(-1)
        # Extra prototypes cover residuals poorly represented by the source vectors.
        count = (VOCABULARY - 16385 + world - 1) // world
        ids = torch.multinomial(errors.clamp_min(1e-8), count, replacement=False, generator=generator)
        proposed = residual[ids].contiguous()
        gathered = [torch.empty_like(proposed) for _ in range(world)]
        dist.all_gather(gathered, proposed)
        vectors[depth, 16385:] = torch.cat(gathered)[:VOCABULARY - 16385]
        residual.sub_(old[choices])
    del source, residual, errors, choices, gathered
    snapshots = [0, 2, 4, 8, 16, passes]
    trace = []
    started = time.monotonic()
    for iteration in range(passes + 1):
        codes, reconstructed = greedy(x, vectors)
        _, measured = greedy(held, vectors)
        values = torch.tensor([(x - reconstructed).square().sum(), x.numel(),
                               (held - measured).square().sum(), held.numel()],
                              device=device, dtype=torch.float64)
        dist.all_reduce(values)
        row = dict(iteration=iteration, fit_latent_mse=float(values[0] / values[1]),
                   heldout_latent_mse=float(values[2] / values[3]), elapsed_seconds=time.monotonic() - started)
        usage = torch.stack([torch.bincount(codes[:, d], minlength=VOCABULARY) for d in range(4)])
        dist.all_reduce(usage)
        row['used_codewords_by_depth'] = (usage > 0).sum(-1).cpu().tolist()
        if rank == 0 and iteration in snapshots:
            save(BASE / 'quality/books' / f'pass-{iteration:03d}.pt', vectors.cpu())
        trace.append(row)
        if rank == 0:
            write(QUALITY / 'fit-trace.json', dict(trace=trace, snapshots=snapshots,
                                                  vocabulary=VOCABULARY, passes=passes))
        progress('fit', rank, **row)
        if iteration == passes:
            break
        error = x - reconstructed
        for depth in range(4):
            old = vectors[depth].clone()
            index = codes[:, depth]
            counts = torch.bincount(index, minlength=VOCABULARY).float()
            sums = torch.zeros_like(old)
            for first in range(0, len(x), 8192):
                ids = index[first:first + 8192]
                sums.index_add_(0, ids, error[first:first + len(ids)] + old[ids])
            dist.all_reduce(counts)
            dist.all_reduce(sums)
            proposal = (sums + 8. * old) / (counts[:, None] + 8.)
            proposal[0].zero_()
            unused = torch.nonzero(counts == 0).flatten()
            unused = unused[unused != 0]
            if len(unused):
                # Recycle unused IDs into training residuals. This leaves every
                # currently assigned reconstruction unchanged at fixed codes.
                per_rank = (len(unused) + world - 1) // world
                ids = torch.multinomial(error.square().sum(-1).clamp_min(1e-8), per_rank,
                                        replacement=False, generator=generator)
                candidates = (error[ids] + old[index[ids]]).contiguous()
                parts = [torch.empty_like(candidates) for _ in range(world)]
                dist.all_gather(parts, candidates)
                proposal[unused] = torch.cat(parts)[:len(unused)]
            assert torch.isfinite(proposal).all()
            vectors[depth].copy_(proposal)
            for first in range(0, len(x), 8192):
                ids = index[first:first + 8192]
                error[first:first + len(ids)].sub_(proposal[ids] - old[ids])
        del reconstructed, measured, error, codes
    dist.barrier()


def export_book(vectors, candidate):
    manifest = read(QUALITY / 'cache-verification.json')
    records = []
    for depth in range(4):
        path = BASE / 'inputs' / f'vectors-depth{depth}.pt'
        save(path, vectors[depth].cpu())
        records.append(dict(name=path.name, sha256=file_digest(path)))
    spec = dict(kind='native_rq24576_vectors', vocabulary=VOCABULARY, depth=4,
                channels=256, vector_shards=records, source_cnn_digest=manifest['source_cnn_digest'],
                stage1_checkpoint_sha256=manifest['stage1_checkpoint_sha256'],
                fitting_images=50000, heldout_training_images=10000, fitting_split='train',
                candidate=candidate, beam_search_used=False, mixture_teacher=False)
    save(BASE / 'inputs/joint-codebook.pt', spec)


@torch.inference_mode()
def screen(rank, world, device):
    aux = source_aux(device)
    latents = np.load(BASE / 'cache' / f'held-latents-rank{rank}.npy', mmap_mode='r')
    real_features = torch.from_numpy(np.load(BASE / 'cache' / f'held-features-rank{rank}.npy')).to(device).double()
    metrics.INCEPTION_PATH = BASE / 'upstream/rqvae/metrics/inception.py'
    network = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    candidates = []
    results = {}
    for path in sorted((BASE / 'quality/books').glob('pass-*.pt')):
        for gain in [1., 1.025, 1.05]:
            candidates.append((path, gain))
    for path, gain in candidates:
        name = f'{path.stem}-gain{gain:g}'
        vectors = torch.load(path, map_location=device, weights_only=True) * gain
        metric = metrics.DistributedOriginalRQVAEMetrics(device, inception=network)
        metric.real_sum.copy_(real_features.sum(0))
        metric.real_cross.copy_(real_features.T @ real_features)
        metric.real_count.fill_(len(latents))
        error = torch.zeros(2, device=device, dtype=torch.float64)
        for first in range(0, len(latents), 16):
            x = torch.from_numpy(latents[first:first + 16].copy()).to(device)
            _, quantized = greedy(x, vectors)
            quantized = quantized.reshape_as(x)
            fake = aux.decode(quantized).float().mul(.5).add(.5).clamp(0, 1)
            metric.update(fake, real=False)
            error[0] += (x - quantized).square().sum().double()
            error[1] += x.numel()
        fid, _, _ = metric.compute()
        dist.all_reduce(error)
        results[name] = dict(rfid=fid, images=10000, latent_mse=float(error[0] / error[1]),
                             snapshot=str(path), gain=gain)
        progress('screen', rank, candidate=name, **results[name])
        if rank == 0:
            write(QUALITY / 'screen-results.json', dict(results=results, validation_used=False))
    selected = min(results, key=lambda name: results[name]['rfid'])
    if rank == 0:
        row = results[selected]
        vectors = torch.load(row['snapshot'], map_location='cpu', weights_only=True) * row['gain']
        export_book(vectors, selected)
        write(QUALITY / 'selected.json', dict(candidate=selected, **row, vocabulary=VOCABULARY,
                                             selection_split='heldout training', full_validation_pending=True))
    dist.barrier()


@torch.inference_mode()
def full(rank, world, device):
    aux = NativeVectorTokenizer(BASE / 'inputs/stage1-model.pt', BASE / 'inputs/joint-codebook.pt').to(device)
    assert module_digest(aux.backbone) == read(QUALITY / 'cache-verification.json')['source_cnn_digest']
    data = dataset('val', 'val')
    assert len(data) == 50000 and np.array_equal(np.bincount(data.labels), np.full(1000, 50))
    loader = DataLoader(Subset(data, list(range(rank, len(data), world))), batch_size=8, num_workers=4,
                        pin_memory=True, persistent_workers=True, prefetch_factor=1,
                        multiprocessing_context='spawn')
    metrics.INCEPTION_PATH = BASE / 'upstream/rqvae/metrics/inception.py'
    network = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    accum = {name: metrics.DistributedOriginalRQVAEMetrics(device, inception=network)
             for name in ['greedy', 'native_soft05']}
    stats = torch.zeros(6, dtype=torch.float64, device=device)
    seen = 0
    before = module_digest(aux)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        latent = aux.encode(images.float())
        hard = aux.quantize_reconstruction(latent)['codes']
        targets, sampled = aux.quantizer.get_soft_codes(latent, temp=.5, stochastic=True)
        flat = targets.reshape(-1, VOCABULARY)
        entropy = torch.cat([-(p * p.clamp_min(1e-30).log()).sum(-1) for p in flat.split(256)])
        stats[:4] += entropy.reshape(-1, 4).sum(0).double()
        stats[4] += entropy.numel() / 4
        stats[5] += len(images)
        if seen == 0:
            assert targets.shape == (*sampled.shape, VOCABULARY)
            torch.testing.assert_close(targets.sum(-1), torch.ones_like(targets[..., 0]), rtol=2e-5, atol=2e-5)
            # Native stochastic labels and prefixes come directly from the unmodified method.
            assert type(aux.quantizer).__name__ == 'RQBottleneck'
            write(QUALITY / f'parity-rank{rank}.json', dict(passed=True, vocabulary=VOCABULARY,
                  native_RQBottleneck=True, stochastic_prefixes=True, soft_targets=True,
                  beam_search=False, mixture_teacher=False, source_CNN_unchanged=True))
        del targets, flat, entropy
        real_features, _ = network(images.mul(.5).add(.5).clamp(0, 1))
        real_features = real_features.double()
        for name, codes in [('greedy', hard), ('native_soft05', sampled)]:
            metric = accum[name]
            metric.real_sum.add_(real_features.sum(0))
            metric.real_cross.add_(real_features.T @ real_features)
            metric.real_count.add_(len(images))
            fake = aux.decode_code(codes).float().mul(.5).add(.5).clamp(0, 1)
            metric.update(fake, real=False)
        seen += len(images)
        if seen % 200 == 0 or seen == len(images):
            progress('full_validation', rank, images_on_rank=seen, target_on_rank=6250)
    dist.all_reduce(stats)
    results = {name: metric.compute()[0] for name, metric in accum.items()}
    assert int(stats[5]) == 50000 and module_digest(aux) == before
    if rank == 0:
        report = dict(passed=results['greedy'] < 4.926736693144846,
                      greedy_rfid=results['greedy'], stochastic_teacher_rfid=results['native_soft05'],
                      native_temperature=.5, mean_entropy=float(stats[:4].sum() / (stats[4] * 4)),
                      entropy_by_depth=(stats[:4] / stats[4]).tolist(),
                      images=50000, split='official ImageNet2012 validation', vocabulary=VOCABULARY,
                      source_CNN_unchanged=True, quantizer_frozen_after_fit=True,
                      beam_search=False, mixture_teacher=False, reference_greedy16k_rfid=4.926736693144846,
                      reference_beam16_rfid=4.514300868542591, reference_original_OMP_rfid=4.213210725808494,
                      selection_split='disjoint10000 training images', validation_used_for_fitting=False,
                      codebook_sha256=file_digest(BASE / 'inputs/joint-codebook.pt'))
        write(QUALITY / 'full-validation.json', report)
        print(json.dumps(report), flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['cache', 'fit', 'screen', 'full'])
    parser.add_argument('--passes', type=int, default=24)
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(26110824576 + rank)
    torch.cuda.manual_seed(26110824576 + rank)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=45))
    QUALITY.mkdir(parents=True, exist_ok=True)
    (BASE / 'quality').mkdir(exist_ok=True)
    if args.phase == 'cache': cache(rank, world, device)
    elif args.phase == 'fit': fit(rank, world, device, args.passes)
    elif args.phase == 'screen': screen(rank, world, device)
    else: full(rank, world, device)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
