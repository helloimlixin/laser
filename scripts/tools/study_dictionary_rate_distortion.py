#!/usr/bin/env python3
"""Fit entropy-constrained LASER dictionaries and measure matched 50k rFID.

Example: prepare; torchrun --nproc-per-node=8 ... cache; fit on one GPU per
(vocabulary, beta, sweeps) candidate; torchrun ... evaluate; report.
All fit/selection images come from training; validation is reporting only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from src.models.entropy_constrained_dictionary import EntropyConstrainedBook, coding_statistics
from src.models.joint_rate_distortion import JointVectorBook
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def progress(args, phase, **data):
    row = dict(phase=phase, time=time.time(), **data)
    write(args.base / f'{phase}-rank{args.rank}.json', row)
    print(json.dumps(row), flush=True)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_local(source, target):
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source_fd = os.open(source, os.O_RDONLY)
    target_fd = os.open(target.with_suffix('.partial'), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    size = os.fstat(source_fd).st_size
    os.ftruncate(target_fd, size)
    def copy_segment(first):
        end = min(first + 32 * 1024**2, size)
        while first < end:
            data = os.pread(source_fd, min(4 * 1024**2, end - first), first)
            if not data:
                raise OSError('Truncated source asset')
            view = memoryview(data)
            while view:
                n = os.pwrite(target_fd, view, first)
                if n <= 0:
                    raise OSError('Incomplete asset write')
                first += n; view = view[n:]
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(copy_segment, range(0, size, 32 * 1024**2)))
    finally:
        os.close(source_fd); os.close(target_fd)
    target.with_suffix('.partial').replace(target)


def prepare(args):
    data = args.base / 'data'; data.mkdir(parents=True, exist_ok=True)
    for source, name in [(args.stage1, 'stage1.pt'), (args.codebook, 'source-book.pt')]:
        progress(args, 'prepare', asset=name)
        copy_local(source, args.base / name)
    expected = 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
    if sha256(args.base / 'stage1.pt') != expected:
        raise ValueError('Source checkpoint differs from the 4.21 LASER checkpoint')
    training = args.archives / 'ILSVRC2012_img_train.verified-20261005.tar'
    if not training.exists():
        raise FileNotFoundError('Expected the verified training archive')
    link = data / 'ILSVRC2012_img_train.tar'
    if not link.exists():
        link.symlink_to(training)
    manifest = args.base / 'calibration-data.json'
    if not manifest.exists():
        with tarfile.open(training, 'r:') as stream:
            outer = sorted([m for m in stream if m.isfile()], key=lambda m: m.name)
        if len(outer) != 1000:
            raise ValueError('Expected 1000 training class archives')
        fd = os.open(training, os.O_RDONLY)
        def prefix_rows(item):
            label, member = item
            # Prefix sampling avoids a fresh 148GB archive scan on a new host.
            # Its exact scope is recorded; it is not a full uniform train sample.
            for size in (4, 8, 16, 32):
                raw = os.pread(fd, min(size * 1024**2, member.size), member.offset_data)
                rows = []; offset = 0
                while offset + 512 <= len(raw):
                    header = raw[offset:offset + 512]
                    if not header.strip(b'\0'):
                        break
                    info = tarfile.TarInfo.frombuf(header, encoding='utf-8', errors='strict')
                    if info.isfile() and info.name.lower().endswith(('.jpeg', '.jpg')):
                        rows.append((member.offset_data + offset + 512, info.size, label, info.name))
                    offset += 512 + ((info.size + 511) // 512) * 512
                    if len(rows) >= args.prefix_images:
                        return rows
            raise ValueError(f'Could not find the requested prefix in {member.name}')
        try:
            with ThreadPoolExecutor(max_workers=24) as pool:
                classes = list(pool.map(prefix_rows, enumerate(outer)))
        finally:
            os.close(fd)
        rows = [r for group in classes for r in group]
        np.savez(data / 'train-index.npz', offsets=[r[0] for r in rows], sizes=[r[1] for r in rows],
                 labels=[r[2] for r in rows], classes=[m.name.removesuffix('.tar') for m in outer])
        rng = np.random.default_rng(26102417)
        fit_ids, held_ids = [], []
        for label in range(1000):
            selected = label * args.prefix_images + rng.permutation(args.prefix_images)
            fit_ids.extend(selected[:args.fit_per_class].tolist())
            held_ids.extend(selected[args.fit_per_class:args.fit_per_class + args.held_per_class].tolist())
        write(manifest, dict(fit_indices=fit_ids, heldout_training_indices=held_ids,
            fit_images=len(fit_ids), heldout_training_images=len(held_ids),
            sampling='Class-stratified random selection within each class archive prefix',
            prefix_images_per_class=args.prefix_images, seed=26102417,
            full_training_uniform_sample=False, validation_used_for_fitting=False,
            selected_fit_jpegs=[rows[i][3] for i in fit_ids],
            selected_heldout_jpegs=[rows[i][3] for i in held_ids],
            source_stage1_sha256=expected, source_codebook_sha256=sha256(args.base/'source-book.pt'),
            tokens_per_image=256, code_shape=[8, 8, 4]))
    progress(args, 'prepare', asset='validation_archive')
    copy_local(args.archives / 'ILSVRC2012_img_val.tar', data / 'ILSVRC2012_img_val.tar')
    if not (data / 'val-index.npz').exists():
        with tarfile.open(data / 'ILSVRC2012_img_val.tar', 'r:') as stream:
            members = sorted([m for m in stream if m.isfile()], key=lambda m: m.name)
        if len(members) != 50000:
            raise ValueError('Expected all 50000 validation images')
        np.savez(data / 'val-index.npz', offsets=[m.offset_data for m in members],
                 sizes=[m.size for m in members], labels=np.zeros(50000, dtype=np.int64),
                 classes=np.array(['unused_for_reconstruction_metrics']))
    progress(args, 'prepare', completed=True)


def setup_runtime(args):
    from src import rq_upstream
    from src.training.rq_imagenet_runtime import TarImageNet
    from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
    from torchvision import transforms
    def dataset(split):
        transform = transforms.Compose([transforms.Resize(256),
            transforms.RandomCrop(256) if split == 'train' else transforms.CenterCrop(256),
            *([transforms.RandomHorizontalFlip()] if split == 'train' else []),
            transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
        return TarImageNet(args.base / 'data', split, transform, seed=261001)
    return FrozenJointLaserTokenizer(args.base / 'stage1.pt', args.base / 'source-book.pt'), dataset


@torch.inference_mode()
def cache(args):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from src.rqvae_metrics import OriginalRQVAEInception
    aux, dataset = setup_runtime(args); aux = aux.to(args.device).eval()
    manifest = json.loads((args.base / 'calibration-data.json').read_text())
    net = OriginalRQVAEInception().to(args.device).eval().requires_grad_(False)
    for split, indices in [('fit', manifest['fit_indices']), ('held', manifest['heldout_training_indices']),
                           ('validation', list(range(50000)))]:
        path = args.base / f'cache/{split}-rank{args.rank}.pt'
        if path.exists():
            continue
        selected = indices[args.rank::args.world]
        loader = DataLoader(Subset(dataset('val' if split == 'validation' else 'train'), selected),
            batch_size=args.batch_size, num_workers=4, pin_memory=True)
        latents, features, codes = [], [], []
        for i, (images, _) in enumerate(loader):
            images = images.to(args.device)
            z = aux.encode(images)
            latents.append(z.cpu())
            if split != 'validation':
                codes.append(aux.quantizer.quantize(z, chunk_size=512)['codes'].cpu())
            if split == 'validation':
                features.append(net(images.mul(.5).add(.5).clamp(0, 1))[0].cpu())
            if i % 25 == 0:
                progress(args, 'cache', split=split, local_images=min((i+1)*args.batch_size, len(selected)))
        path.parent.mkdir(exist_ok=True)
        value = dict(latents=torch.cat(latents), indices=torch.tensor(selected))
        if features: value['features'] = torch.cat(features)
        if codes: value['source_codes'] = torch.cat(codes)
        torch.save(value, path)
    dist.barrier()
    if args.rank == 0:
        write(args.base/'cache-complete.json', dict(completed=True, world_size=args.world))


def load_cache(args, split):
    paths = sorted((args.base/'cache').glob(f'{split}-rank*.pt'))
    if not paths: raise FileNotFoundError(f'No {split} cache')
    parts = [torch.load(p, weights_only=True, map_location='cpu') for p in paths]
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


def counts(codes, vocabulary):
    return torch.stack([torch.bincount(codes[..., d].flatten(), minlength=vocabulary) for d in range(4)])


def candidate_name(args):
    return f'ecvq{args.vocabulary}-beta{args.beta:g}-sweeps{args.sweeps}'


@torch.no_grad()
def fit(args):
    fit_data, held_data = load_cache(args, 'fit'), load_cache(args, 'held')
    source = torch.load(args.base/'source-book.pt', weights_only=True, map_location='cpu')
    prior = JointAtomCoefficientRQ(source['dictionary'], source['atom_ids'], source['coefficients'])
    training_counts = counts(fit_data['source_codes'], prior.vocab_size)
    selected = torch.cat([torch.zeros(1, dtype=torch.long), training_counts.sum(0)[1:].topk(args.vocabulary-1).indices+1])
    vectors = torch.stack([prior.embed_depth(selected, d) for d in range(4)])
    q = EntropyConstrainedBook(vectors).to(args.device)
    x, held = fit_data['latents'].to(args.device), held_data['latents'].to(args.device)
    best, best_objective, trace = None, math.inf, []
    name = candidate_name(args)
    target = args.base/f'books/{name}.pt'
    if target.exists(): raise FileExistsError(target)
    for iteration in range(args.passes + 1):
        encoded = q.quantize(x, rate_weight=args.beta, sweeps=args.sweeps)
        q.update_probabilities(encoded['codes'])
        # Reassign under the probability model that is actually saved.
        encoded = q.quantize(x, rate_weight=args.beta, sweeps=args.sweeps)
        measured = q.quantize(held, rate_weight=args.beta, sweeps=args.sweeps)
        rates = coding_statistics(q.code_counts(encoded['codes']), q.code_counts(measured['codes']))
        objective = q.objective(held, measured['codes'], rate_weight=args.beta)
        row = dict(iteration=iteration, fit_latent_mse=float((x-encoded['quantized']).square().mean()),
            heldout_training_latent_mse=float((held-measured['quantized']).square().mean()),
            heldout_selection_objective=objective, **rates)
        if objective < best_objective:
            best_objective = objective
            best = dict(vectors=q.vectors.cpu().clone(), probabilities=q.probabilities.cpu().clone(),
                training_counts=q.code_counts(encoded['codes']).cpu())
        if iteration < args.passes:
            row.update(q.refit(x, encoded['codes'], prior_weight=8.))
        trace.append(row)
        progress(args, 'fit', candidate=name, iteration=iteration,
                 heldout_mse=row['heldout_training_latent_mse'], heldout_bits=rates['heldout_cross_entropy_bits_per_token'])
        write(args.base/f'books/{name}-trace.json', trace)
    best.update(name=name, vocabulary=args.vocabulary, beta=args.beta, sweeps=args.sweeps,
        source_codebook_sha256=sha256(args.base/'source-book.pt'), tokens_per_image=256,
        image_encoder_decoder_frozen=True, validation_used_for_fitting=False,
        selection='Heldout training latent MSE + beta * model bits per latent site',
        selected_objective=best_objective, trace=trace)
    target.parent.mkdir(exist_ok=True)
    torch.save(best, target)
    progress(args, 'fit', candidate=name, completed=True)


@torch.inference_mode()
def evaluate(args):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from src import rqvae_metrics as metrics
    aux, dataset = setup_runtime(args); aux = aux.to(args.device).eval()
    inception = metrics.OriginalRQVAEInception().to(args.device).eval().requires_grad_(False)
    local = torch.load(args.base/f'cache/validation-rank{args.rank}.pt', weights_only=True, map_location='cpu')
    training = load_cache(args, 'fit')
    candidates = [args.base/'source-book.pt'] + sorted((args.base/'books').glob('ecvq*.pt'))
    loader = DataLoader(Subset(dataset('val'), local['indices'].tolist()), batch_size=args.batch_size,
        num_workers=4, pin_memory=True, persistent_workers=True)
    results = []
    for path in candidates:
        spec = torch.load(path, weights_only=True, map_location='cpu')
        if path.name == 'source-book.pt':
            q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(args.device)
            name, beta, sweeps = 'source49153', 0., 0
            train_counts = counts(training['source_codes'], q.vocab_size)
        else:
            q = EntropyConstrainedBook(spec['vectors'], spec['probabilities']).to(args.device)
            name, beta, sweeps = spec['name'], spec['beta'], spec['sweeps']
            train_counts = spec['training_counts']
        state = metrics.DistributedOriginalRQVAEMetrics(args.device, inception=inception, compute_inception_score=False)
        real = local['features'].to(args.device, dtype=torch.float64)
        state.real_sum.copy_(real.sum(0)); state.real_cross.addmm_(real.T, real)
        state.real_count.fill_(len(real))
        histogram = torch.zeros(4, q.vocab_size, device=args.device, dtype=torch.float64)
        errors = torch.zeros(4, device=args.device, dtype=torch.float64)
        begin = time.monotonic()
        for batch, (images, _) in enumerate(loader):
            first = batch * args.batch_size
            x = local['latents'][first:first+len(images)].to(args.device)
            result = q.quantize(x, chunk_size=512, **(dict(rate_weight=beta, sweeps=sweeps) if beta or sweeps else {}))
            codes, z = result['codes'], result['quantized']
            image = aux.decode(z).mul(.5).add(.5).clamp(0, 1)
            reference = images.to(args.device).mul(.5).add(.5).clamp(0, 1)
            state.update(image, real=False)
            histogram.add_(counts(codes, q.vocab_size))
            errors.add_(torch.stack([(x-z).double().square().sum(), x.new_tensor(x.numel(), dtype=torch.float64),
                (image-reference).double().square().sum(), x.new_tensor(reference.numel(), dtype=torch.float64)]))
            if batch % 50 == 0:
                progress(args, 'evaluate', candidate=name, local_images=first+len(images))
        dist.all_reduce(histogram); dist.all_reduce(errors)
        rfid, _, _ = state.compute()
        if int(state.real_count) != 50000 or int(state.fake_count) != 50000:
            raise RuntimeError('Incomplete validation evaluation')
        rate = coding_statistics(train_counts, histogram)
        # The trained assignment prior can differ from the final histogram refit.
        assignment_model_bits = None
        if hasattr(q, 'probabilities'):
            assignment_model_bits = float((histogram * -q.probabilities.log2()).sum() / histogram.sum())
        row = dict(name=name, rfid=rfid, latent_mse=float(errors[0]/errors[1]),
            pixel_mse=float(errors[2]/errors[3]), psnr_db=float(-10*torch.log10(errors[2]/errors[3])),
            beta=beta, coordinate_sweeps=sweeps, validation_images=50000,
            tokens_per_image=256, source_codebook_sha256=sha256(path),
            assignment_model_bits_per_token=assignment_model_bits,
            metric_backend='original-rqvae', evaluation_split='full ImageNet validation',
            seconds=time.monotonic()-begin, **{k:v for k,v in rate.items() if k != 'tokens_per_image'})
        results.append(row)
        if args.rank == 0:
            torch.save(dict(validation_counts=histogram.cpu(), training_counts=train_counts), args.base/f'{name}-counts.pt')
            write(args.base/'results.json', dict(completed=len(results)==len(candidates), results=results))
            print(json.dumps(row), flush=True)
        del state, q
    dist.barrier()


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads((args.base/'results.json').read_text())
    rows = data['results']
    keys = ['name','vocabulary','tokens_per_image','rfid','latent_mse','psnr_db',
        'fixed_width_bpp','marginal_entropy_bits_per_token','depth_conditional_entropy_bits_per_token',
        'depth_conditional_entropy_bpp','heldout_cross_entropy_bits_per_token','heldout_model_bpp']
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'rate-distortion.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, key, label in zip(axes, ['vocabulary','depth_conditional_entropy_bpp','heldout_model_bpp'],
            ['Vocabulary','Empirical H(token | depth), bpp','Train-fitted model cost on validation, bpp']):
        for r in rows:
            ax.scatter(r[key], r['rfid']); ax.annotate(r['name'], (r[key],r['rfid']), fontsize=6)
        ax.axhline(4.210914, color='grey', linestyle='--', label='Reported original stage-1 rFID 4.21')
        ax.set_xlabel(label);ax.set_ylabel('Matched full-validation rFID');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('LASER dictionary bottleneck: fixed 256 integer tokens per image')
    for suffix in ['png','pdf']:fig.savefig(args.out/f'rate-distortion.{suffix}', dpi=180)
    data['empirical_entropy_is_actual_bitstream_size'] = False
    data['original_stage1_rfid_is_historical_reference'] = True
    data['fit_manifest'] = json.loads((args.base/'calibration-data.json').read_text())
    data['pareto_by_heldout_rate_and_rfid'] = [r['name'] for r in rows if not any(
        s['heldout_model_bpp'] <= r['heldout_model_bpp'] and s['rfid'] <= r['rfid'] and
        (s['heldout_model_bpp'] < r['heldout_model_bpp'] or s['rfid'] < r['rfid']) for s in rows)]
    write(args.out/'rate-distortion.json', data)
    for p in (args.base/'books').glob('*'):
        if p.is_file(): shutil.copyfile(p, args.out/p.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare','cache','fit','evaluate','report'])
    p.add_argument('--base', type=Path, default=Path('/dev/shm/laser-entropy-rd-20261024'))
    p.add_argument('--out', type=Path, default=ROOT/'outputs/imagenet-laser-entropy-rd-20261024')
    p.add_argument('--archives', type=Path, default=ROOT.parent/'data/imagenet/archives')
    p.add_argument('--stage1', type=Path, default=ROOT/'outputs/imagenet-rfid421-combination-soft-20260927/assets/laser-tokenizer.pt')
    p.add_argument('--codebook', type=Path, default=ROOT/'outputs/imagenet-laser-joint49153-rfid4436-scratch20-8h100-20261023/reproducibility/joint-codebook.pt')
    p.add_argument('--prefix-images', type=int, default=16)
    p.add_argument('--fit-per-class', type=int, default=4)
    p.add_argument('--held-per-class', type=int, default=1)
    p.add_argument('--vocabulary', type=int, default=16384)
    p.add_argument('--beta', type=float, default=0.)
    p.add_argument('--sweeps', type=int, default=2)
    p.add_argument('--passes', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    if args.fit_per_class + args.held_per_class > args.prefix_images:
        p.error('Training and heldout samples must fit within the prefix')
    args.rank = int(os.environ.get('RANK',0)); args.world = int(os.environ.get('WORLD_SIZE',1))
    args.base.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.phase not in ('prepare','report'):
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
        args.device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        os.environ.setdefault('TORCH_HOME', str(args.base/'torch-cache'))
    if args.phase in ('cache','evaluate'):
        import torch.distributed as dist
        dist.init_process_group('nccl', device_id=args.device)
    globals()[args.phase](args)
    if args.phase in ('cache','evaluate'):dist.destroy_process_group()


if __name__ == '__main__':
    main()
