#!/usr/bin/env python3
"""Fit exact shared four-depth joint books; select on training data, then audit rFID."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src import rq_upstream
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, JointAtomCoefficientRQ
from src.training.laser_shared_joint_vocabulary import assert_shared_codewords, refit_shared_book
from src.training.laser_joint_vocabulary import fit_coefficient_candidates, allocate_coefficients, pack_codebook
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.img_datasets.transforms import create_transforms
from omegaconf import OmegaConf

RUN = 'imagenet-laser-sharedjoint20k-depth4-refit-20261022'
BASE = Path('/dev/shm/laser-joint20k-depth4-refit-20261022')
EVALUATION = Path('/dev/shm/laser-joint20k-depth4-20261022')
BASELINE = ROOT / 'outputs/imagenet-laser-sharedjoint20k-depth4-codebook-20261022/joint20k-depth4-codebook.pt'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value, time=time.time()), indent=2, allow_nan=False) + '\n'
    if str(path).startswith('/workspace/'):
        path.write_text(payload)
    else:
        pending = path.with_name(path.name + f'.{os.getpid()}.pending')
        pending.write_text(payload)
        pending.replace(path)


def progress(args, phase, **values):
    if dist.get_rank() == 0:
        row = dict(phase=phase, **values)
        write(args.out / 'progress.json', row)
        print(json.dumps(row), flush=True)


def build(path, device, gain=1.):
    spec = torch.load(path, map_location='cpu', weights_only=True)
    q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients'] * gain).to(device)
    assert q.depth == 4 and q.vocab_size == 20481
    assert_shared_codewords(q)
    return q


def calibration(args):
    excluded = set()
    for relative in [
        'imagenet-laser-sharedjoint20k-depth5-codebook-20261018/calibration-data.json',
        'imagenet-laser-adaptive-joint20k-vocabulary-20261017/calibration-data.json',
    ]:
        record = json.loads((ROOT / 'outputs' / relative).read_text())
        for key in ('fit_indices', 'heldout_training_indices'):
            excluded.update(record.get(key, []))
    selected = [int(i) for i in np.random.default_rng(26102220484).permutation(1281167)
                if int(i) not in excluded][:args.fit_images + args.held_images]
    fit_ids, held_ids = selected[:args.fit_images], selected[args.fit_images:]
    assert set(fit_ids).isdisjoint(held_ids) and set(held_ids).isdisjoint(excluded)
    return dict(fitting_split='train', fit_images=len(fit_ids), heldout_training_images=len(held_ids),
        fit_indices=fit_ids, heldout_training_indices=held_ids,
        previous_fit_and_holdout_images_excluded=len(excluded),
        heldout_disjoint_from_source_fitting=True, validation_images_used_for_selection=0,
        augmentation='Upstream training resize/crop/flip; deterministic seed 261001, epoch zero',
        source_codebook_sha256=digest(args.baseline))


@torch.inference_mode()
def cache(args, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    metadata = calibration(args)
    if rank == 0:
        write(args.out / 'calibration-data.json', metadata)
    required = [args.base / f'{split}-rank{rank}.npy' for split in ('fit', 'held')]
    if all(p.exists() for p in required) and (args.base / f'held-real-features-rank{rank}.npy').exists():
        dist.barrier()
        return
    from src.rqvae_metrics import OriginalRQVAEInception
    aux = FrozenJointLaserTokenizer(args.evaluation / 'stage1.pt', args.baseline).to(device)
    config = OmegaConf.create(dict(transforms=dict(type='imagenet256x256')))
    data = TarImageNet(args.base / 'data', 'train', create_transforms(config, split='train'), seed=261001)
    assert len(data) == 1281167
    data.set_epoch(0)
    for split, selected in [('fit', metadata['fit_indices']), ('held', metadata['heldout_training_indices'])]:
        target = args.base / f'{split}-rank{rank}.npy'
        feature_target = args.base / f'held-real-features-rank{rank}.npy'
        if target.exists() and (split == 'fit' or feature_target.exists()):
            continue
        net = OriginalRQVAEInception().eval().requires_grad_(False).to(device) if split == 'held' else None
        loader = DataLoader(Subset(data, selected[rank::world]), batch_size=32, num_workers=2, pin_memory=True)
        chunks, features, count = [], [], 0
        for images, _ in loader:
            images = images.to(device)
            chunks.append(aux.encode(images).cpu().numpy())
            if net is not None:
                features.append(net(images.mul(.5).add(.5).clamp(0, 1))[0].cpu().numpy())
            count += len(images)
            if count % 256 == 0:
                progress(args, 'encoding_training_images', split=split, images_on_rank=count,
                         target_on_rank=len(selected[rank::world]))
        np.save(target, np.concatenate(chunks))
        if net is not None:
            np.save(feature_target, np.concatenate(features))
        del net, chunks, features
    del aux
    torch.cuda.empty_cache()
    dist.barrier()


def global_mse(q, x):
    residual = x - q.quantize(x, chunk_size=1024)['quantized']
    values = torch.stack([residual.double().square().sum(), x.new_tensor(x.numel(), dtype=torch.float64)])
    dist.all_reduce(values)
    return float(values[0] / values[1])


def save_book(args, template, q, name, **metadata):
    path = args.out / (name + '-codebook.pt')
    if dist.get_rank() == 0:
        spec = dict(template)
        for key in ('first_four_depths_exactly_unchanged', 'first_four_depths_unchanged',
                    'dictionary_encoder_decoder_frozen'):
            spec.pop(key, None)
        spec.update({key:getattr(q, key).cpu().clone() for key in ('dictionary', 'atom_ids', 'coefficients')})
        spec.update(residual_depth=4, vocabulary=20481, source_prefix_preserved=False,
            refitted_for_four_depths=True, dictionary_refitted_for_depth=4,
            shared_codebook_across_depths=True, shared_coefficient_values=True,
            learned_compact_dictionary=True, encoder_decoder_frozen=True,
            source_joint_codebook_sha256=digest(args.baseline), fitting_split='train',
            fitting_views=args.fit_images, fitting_objective='Four-depth final latent error',
            validation_images_used_for_fitting=0, **metadata)
        torch.save(spec, path)
    return path


@torch.inference_mode()
def fit_one(args, name, q, template, x, held, run):
    best, best_error, trace = None, float('inf'), []
    for iteration in range(args.passes + 1):
        encoded = q.quantize(x, chunk_size=1024)
        values = torch.stack([(x - encoded['quantized']).double().square().sum(),
                              x.new_tensor(x.numel(), dtype=torch.float64)])
        dist.all_reduce(values)
        fit_error, held_error = float(values[0] / values[1]), global_mse(q, held)
        row = dict(iteration=iteration, fit_latent_mse=fit_error, heldout_training_latent_mse=held_error)
        if held_error < best_error:
            best_error = held_error
            best = {key:getattr(q, key).clone() for key in ('dictionary', 'atom_ids', 'coefficients')}
        if iteration < args.passes:
            row.update(refit_shared_book(q, x, encoded['codes'], ridge_fraction=args.ridge))
        assert_shared_codewords(q)
        trace.append(row)
        if dist.get_rank() == 0:
            write(args.out / (name + '-fit.json'), dict(completed=iteration == args.passes,
                candidate=name, trace=trace, best_heldout_training_latent_mse=best_error,
                atoms=q.dictionary.shape[1], vocabulary=q.vocab_size, depth=q.depth,
                sharing_verified_after_every_iteration=True, encoder_decoder_frozen=True))
            if run:
                run.log({f'fit/{name}/heldout_mse': held_error, f'fit/{name}/training_mse': fit_error,
                         f'fit/{name}/iteration': iteration})
        progress(args, 'fitting_four_depth_dictionary', candidate=name, **row)
        del encoded
    for key, value in best.items():
        getattr(q, key).copy_(value)
    q.norms.copy_(q.dictionary.square().sum(0))
    path = save_book(args, template, q, name, heldout_training_latent_mse=best_error)
    dist.barrier()
    return path


@torch.inference_mode()
def reallocate(args, prior, x, atoms, uniform, device):
    encoded = prior.quantize(x, chunk_size=1024, return_projections=True)
    codes = encoded['codes']
    ids = prior.atom_ids[0, codes]
    observed = codes != 0
    local = (ids[observed].cpu(), encoded['projections'][observed].cpu())
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    packed = [None]
    if dist.get_rank() == 0:
        all_atoms = torch.cat([v[0] for v in gathered]).to(device)
        all_values = torch.cat([v[1] for v in gathered]).to(device)
        importance = torch.bincount(all_atoms, weights=all_values.square(), minlength=prior.dictionary.shape[1])
        selected = importance.argsort(descending=True)[:atoms].sort().values
        mapping = torch.full((prior.dictionary.shape[1],), -1, dtype=torch.long, device=device)
        mapping[selected] = torch.arange(atoms, device=device)
        retained = mapping[all_atoms] >= 0
        counts = (4,) if uniform else (2, 4, 6, 8)
        books, errors = fit_coefficient_candidates(mapping[all_atoms[retained]], all_values[retained], atoms, counts=counts)
        allocation = [4] * atoms if uniform else allocate_coefficients(errors, 20480)
        assert sum(allocation) == 20480
        packed[0] = pack_codebook(prior.dictionary[:, selected], [books] * 4, [allocation] * 4)
        write(args.out / f'allocation-{atoms}.json', dict(atoms=atoms, vocabulary=20481,
            pooled_across_four_depths=True, validation_images_used=0,
            allocation_histogram={str(n):allocation.count(n) for n in sorted(set(allocation))}))
    dist.broadcast_object_list(packed, 0)
    spec = packed[0]
    q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)
    assert_shared_codewords(q)
    return q


@torch.inference_mode()
def fit(args, device, run):
    cache(args, device)
    rank = dist.get_rank()
    x = torch.from_numpy(np.load(args.base / f'fit-rank{rank}.npy')).reshape(-1, 256).to(device)
    held = torch.from_numpy(np.load(args.base / f'held-rank{rank}.npy')).reshape(-1, 256).to(device)
    template = torch.load(args.baseline, map_location='cpu', weights_only=True)
    warm = build(args.baseline, device)
    warm_path = fit_one(args, 'warm8192', warm, template, x, held, run)
    for atoms, uniform in [(8192, False), (6144, False), (5120, True)]:
        name = f'reallocated{atoms}'
        q = reallocate(args, warm, x, atoms, uniform, device)
        fit_one(args, name, q, template, x, held, run)
        del q
        torch.cuda.empty_cache()
    if rank == 0:
        write(args.out / 'fit-completed.json', dict(completed=True, vocabulary=20481, depth=4,
            fit_images=args.fit_images, heldout_training_images=args.held_images,
            candidates=['warm8192', 'reallocated8192', 'reallocated6144', 'reallocated5120'],
            encoder_decoder_frozen=True, validation_images_used_for_fitting=0))
    dist.barrier()


@torch.inference_mode()
def screen(args, device, run):
    from src.rqvae_metrics import OriginalRQVAEInception, DistributedOriginalRQVAEMetrics
    rank = dist.get_rank()
    latent = torch.from_numpy(np.load(args.base / f'held-rank{rank}.npy')).to(device)
    real = torch.from_numpy(np.load(args.base / f'held-real-features-rank{rank}.npy')).double().to(device)
    aux = FrozenJointLaserTokenizer(args.evaluation / 'stage1.pt', args.baseline).to(device)
    net = OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    rows = []
    cases = [('baseline', args.baseline, [1.])]
    cases += [(name, args.out / (name + '-codebook.pt'), [.9, .95, 1., 1.05, 1.1])
              for name in ['warm8192', 'reallocated8192', 'reallocated6144', 'reallocated5120']]
    for name, path, gains in cases:
        for gain in gains:
            q = build(path, device, gain)
            metric = DistributedOriginalRQVAEMetrics(device, inception=net)
            metric.real_sum.copy_(real.sum(0))
            metric.real_cross.copy_(real.T @ real)
            metric.real_count.fill_(len(real))
            squared_error = latent.new_zeros((), dtype=torch.float64)
            for first in range(0, len(latent), args.batch_size):
                x = latent[first:first + args.batch_size]
                z = q.quantize(x, chunk_size=1024)['quantized']
                squared_error += (x - z).double().square().sum()
                metric.update(aux.decode(z).mul(.5).add(.5).clamp(0, 1), real=False)
            dist.all_reduce(squared_error)
            fid, _, _ = metric.compute()
            assert int(metric.real_count) == int(metric.fake_count) == args.held_images
            row = dict(candidate=name, gain=gain, heldout_training_rfid=fid,
                heldout_training_latent_mse=float(squared_error / (args.held_images * 64 * 256)),
                source_codebook_path=str(path), atoms=q.dictionary.shape[1])
            rows.append(row)
            if rank == 0:
                write(args.out / 'screen-results.json', dict(completed=False, results=rows,
                    images=args.held_images, split='heldout train', validation_images_used_for_selection=0))
                if run:
                    key = f'screen/{name}/gain{gain:g}'
                    run.log({key + '/rfid': fid, key + '/latent_mse': row['heldout_training_latent_mse']})
            progress(args, 'screening_on_heldout_training_images', **row)
            del q, metric
    chosen = min(rows, key=lambda row: row['heldout_training_rfid'])
    if rank == 0:
        spec = torch.load(chosen['source_codebook_path'], map_location='cpu', weights_only=True)
        spec['coefficients'] = spec['coefficients'] * chosen['gain']
        spec['coefficient_gain'] = float(spec.get('coefficient_gain', 1.)) * chosen['gain']
        path = args.out / 'joint20k-depth4-codebook.pt'
        torch.save(spec, path)
        write(args.out / 'selected.json', dict(chosen, codebook_path=str(path), codebook_sha256=digest(path),
            depth=4, vocabulary=20481, tokens_per_image=256,
            shared_codebook_across_depths=True, selected_using_heldout_training_rfid=True,
            validation_images_used_for_selection=0, full_reconstruction_validation_required=True))
        write(args.out / 'screen-results.json', dict(completed=True, results=rows,
            selected=chosen, images=args.held_images, split='heldout train', validation_images_used_for_selection=0))
        if run:
            run.summary['selection/candidate'] = chosen['candidate']
            run.summary['selection/gain'] = chosen['gain']
            run.summary['selection/heldout_training_rfid'] = chosen['heldout_training_rfid']
    progress(args, 'heldout_training_selection_completed', selected=chosen)
    dist.barrier()


@torch.no_grad()
def full(args, device, run):
    sys.path.insert(0, str(ROOT / 'scripts/tools'))
    from evaluate_laser_joint20k_depth4 import ValidationImages
    from src.rqvae_metrics import OriginalRQVAEInception, DistributedOriginalRQVAEMetrics
    from rqvae.models.rqvae.quantizations import RQBottleneck
    rank, world = dist.get_rank(), dist.get_world_size()
    selected = json.loads((args.out / 'selected.json').read_text())
    path = Path(selected['codebook_path'])
    assert digest(path) == selected['codebook_sha256']
    aux = FrozenJointLaserTokenizer(args.evaluation / 'stage1.pt', path).to(device)
    assert_shared_codewords(aux.quantizer)
    versions = [v._version for v in aux.state_dict().values()]
    baseline = build(args.baseline, device)
    net = OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    old = DistributedOriginalRQVAEMetrics(device, inception=net)
    new = DistributedOriginalRQVAEMetrics(device, inception=net)
    data = ValidationImages(args.evaluation)
    assert len(data) == 50000
    loader = DataLoader(Subset(data, range(rank, len(data), world)),
                        batch_size=args.batch_size, num_workers=2, pin_memory=True)
    count, started = 0, time.monotonic()
    for images in loader:
        images = images.to(device)
        x = aux.encode(images)
        encoded = aux.quantizer.quantize(x, chunk_size=1024)
        if count == 0:
            native = RQBottleneck([8, 8, 256], [8, 8, 4], 20481,
                shared_codebook=True, restart_unused_codes=False).eval().to(device)
            for d, book in enumerate(native.codebooks):
                book.weight[:-1].copy_(aux.quantizer.expanded_codebook(d))
            native_z, _, native_codes = native(x[:2])
            assert torch.equal(native_codes, encoded['codes'][:2])
            torch.testing.assert_close(native_z, encoded['quantized'][:2], rtol=1.e-4, atol=2.e-5)
            del native, native_z, native_codes
        old.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
        for metric, z in [(new, encoded['quantized']), (old, baseline.quantize(x, chunk_size=1024)['quantized'])]:
            metric.update(aux.decode(z).mul(.5).add(.5).clamp(0, 1), real=False)
        count += len(images)
        if count % 256 == 0 or count == len(range(rank, len(data), world)):
            progress(args, 'full_validation_comparison', estimated_total_images=min(count * world, 50000),
                images_on_rank=count, elapsed_seconds=time.monotonic() - started)
    new.real_sum.copy_(old.real_sum)
    new.real_cross.copy_(old.real_cross)
    new.real_count.copy_(old.real_count)
    baseline_fid, _, _ = old.compute()
    candidate_fid, _, _ = new.compute()
    assert int(old.real_count) == int(old.fake_count) == int(new.fake_count) == 50000
    assert abs(baseline_fid - 4.96756247210908) < .001
    assert versions == [v._version for v in aux.state_dict().values()]
    assert_shared_codewords(aux.quantizer)
    if rank == 0:
        report = dict(completed=True, measured_rfid=candidate_fid, baseline_rfid=baseline_fid,
            improvement=baseline_fid - candidate_fid, improved=candidate_fid < baseline_fid,
            selected_candidate=selected['candidate'], selected_gain=selected['gain'],
            atoms=aux.quantizer.dictionary.shape[1], vocabulary=20481, depth=4, tokens_per_image=256,
            real_images=50000, reconstructed_images_per_candidate=50000,
            codebook_path=str(path), codebook_sha256=selected['codebook_sha256'],
            physical_vectors_exactly_shared=True, encoder_decoder_frozen=True,
            native_shared_quantizer_verified=True, native_hard_codes_exact=True,
            fitting_images=args.fit_images, heldout_training_selection_images=args.held_images,
            validation_images_used_for_selection=0, stage2_training_launched=False,
            reference='Matched full ImageNet validation; exact original RQ Inception',
            elapsed_seconds=time.monotonic() - started)
        write(args.out / 'full-rfid.json', report)
        print(json.dumps(report), flush=True)
        if run:
            run.log({'rfid/refitted_depth4': candidate_fid, 'rfid/truncated_depth4': baseline_fid,
                     'rfid/improvement': baseline_fid - candidate_fid})
            import wandb
            artifact = wandb.Artifact(RUN, type='tokenizer')
            for file in sorted(args.out.iterdir()):
                if file.suffix in ('.json', '.pt', '.py'):
                    artifact.add_file(str(file))
            run.log_artifact(artifact).wait()
        destination = ROOT / 'outputs' / RUN
        destination.mkdir(parents=True, exist_ok=True)
        for file in sorted(args.out.iterdir()):
            if file.is_file() and file.suffix in ('.json', '.pt', '.py'):
                (destination / file.name).write_bytes(file.read_bytes())
        assert digest(destination / 'joint20k-depth4-codebook.pt') == selected['codebook_sha256']
        write(destination / 'verification.json', dict(passed=True, result=report,
            saved_codebook_path=str(destination / 'joint20k-depth4-codebook.pt'),
            stage1_sha256=json.loads((args.evaluation / 'stage1.verified.json').read_text())['sha256'],
            validation_md5=json.loads((args.evaluation / 'ILSVRC2012_img_val.verified.json').read_text())['md5']))
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'cache', 'fit', 'screen', 'full'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--out', type=Path, default=BASE / 'output')
    parser.add_argument('--evaluation', type=Path, default=EVALUATION)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--fit-images', type=int, default=16384)
    parser.add_argument('--held-images', type=int, default=2048)
    parser.add_argument('--passes', type=int, default=32)
    parser.add_argument('--ridge', type=float, default=.01)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('TORCH_HOME', str(args.evaluation / 'torch-cache'))
    args.base.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(26102220484)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=60))
    run, succeeded = None, False
    try:
        if dist.get_rank() == 0:
            shutil.copyfile(__file__, args.out / Path(__file__).name)
            if args.wandb:
                import wandb
                os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
                os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
                run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN,
                    name='LASER exact shared joint20k | dedicated four-depth refit',
                    job_type='tokenizer-fitting', dir=str(args.base), resume='allow',
                    config=dict(depth=4, vocabulary=20481, fit_images=args.fit_images,
                        heldout_training_images=args.held_images, fitting_passes=args.passes,
                        candidate_atoms=[8192, 6144, 5120], encoder_decoder_frozen=True,
                        shared_codewords_across_all_depths=True, validation_used_for_selection=False,
                        baseline_rfid=4.96756247210908, tokens_per_image=256))
                write(args.out / 'wandb.json', dict(run_id=run.id, url=run.url))
        if args.phase in ('cache',):
            cache(args, device)
        if args.phase in ('run', 'fit'):
            fit(args, device, run)
        if args.phase in ('run', 'screen'):
            screen(args, device, run)
        if args.phase in ('run', 'full'):
            full(args, device, run)
        succeeded = True
    finally:
        if run:
            run.finish(exit_code=0 if succeeded else 1)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
