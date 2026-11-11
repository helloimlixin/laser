"""Disposable real-image speed trials and full validation of the compound128 codec."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import statistics
import tarfile
import time
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf

from src.training.compound128_recipe import Compound128Recipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training.laser_hard_ar import nearest_coefficient_ids
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config


def benchmark(base, args):
    from src.training import rq_imagenet_shared as shared
    from runtime.amp_replay import recoverable_amp_step
    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device); torch.set_num_threads(4)
    profile = Compound128Recipe(base)
    profile.recipe.execution.attention_backend = args.attention
    shared.BASE, shared.PROFILE = base, profile
    shared.seed_all(int(profile.recipe.seed))
    report = dict(passed=False, gpu=args.gpu, microbatch=args.microbatch,
        attention_backend=args.attention, global_batch=2048, weights_discarded=True,
        frozen_encoder_precision='fp32', includes_ddp=False, includes_jpeg_decode=False)
    try:
        aux, _ = profile.tokenizer(device)
        before = profile.digest(aux)
        shared.seed_all(int(profile.recipe.seed))
        model, optimizer, config = shared.fresh_model(device)
        dataset = TarImageNet(base / 'data', 'train', create_transforms(config.dataset, split='train'))
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(261001))[:512]
        rows = [dataset[int(i)] for i in indices]
        images = torch.stack([r[0] for r in rows]).pin_memory()
        labels = torch.tensor([r[1] for r in rows]).pin_memory()
        accumulation = 512 // args.microbatch
        assert accumulation * args.microbatch == 512
        scaler = torch.amp.GradScaler('cuda')
        model.train(); optimizer.zero_grad(set_to_none=True)
        timings, losses, retries = [], [], []
        torch.cuda.reset_peak_memory_stats(device)
        updates = int(profile.recipe.execution.benchmark_updates)
        for update in range(updates):
            torch.cuda.synchronize(device); started = time.monotonic()
            def backward():
                value = 0.
                for first in range(0, 512, args.microbatch):
                    loss = shared.training_backward(profile, model, aux,
                        images[first:first+args.microbatch].to(device, non_blocking=True),
                        labels[first:first+args.microbatch].to(device, non_blocking=True),
                        scaler, 1. / accumulation, amp=True)
                    value += float(loss) / accumulation
                return value
            loss = backward()
            norm, retry = recoverable_amp_step(model.parameters(), optimizer, scaler,
                max_norm=1., replay_backward=backward)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            timings.append(time.monotonic() - started); losses.append(loss); retries.append(retry)
        assert profile.digest(aux) == before
        assert all(int(s['step']) == updates for s in optimizer.state.values())
        assert len(optimizer.state) == len(list(model.parameters()))
        report.update(passed=True, updates=updates, warmup_updates=2, seconds=timings, losses=losses,
            images_per_second_per_gpu=512. / statistics.median(timings[2:]),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
            optimizer_state_count=len(optimizer.state), maximum_amp_retries=max(retries),
            frozen_tokenizer_unchanged=True, parameter_count=sum(p.numel() for p in model.parameters()))
    except torch.OutOfMemoryError as error:
        report.update(reason='CUDA out of memory', error=str(error))
    except Exception as error:
        report.update(reason='error', error=str(error), traceback=traceback.format_exc())
    write_json(base / f'benchmark-gpu{args.gpu}.json', report)
    print(json.dumps(report), flush=True)
    if report.get('reason') == 'error':
        raise RuntimeError(report['error'])


def prepare_validation(base):
    import fcntl
    with (base / 'data/validation-reconstruction.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        _prepare_validation(base)


def _prepare_validation(base):
    """Validation reconstruction uses images only; labels are never consumed."""
    data = base / 'data'
    target = data / 'ILSVRC2012_img_val.tar'
    index = data / 'val-index.npz'
    if target.is_file() and index.is_file():
        return
    source = Path('/workspace/Projects/data/imagenet/archives/ILSVRC2012_img_val.tar')
    # Bounded independent range reads avoid serial object-store readahead stalls.
    from concurrent.futures import ThreadPoolExecutor
    size = source.stat().st_size
    target_fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(target_fd, size)
    def copy_range(first):
        fd = os.open(source, os.O_RDONLY)
        try:
            end = min(size, first + 128 * 1024**2)
            while first < end:
                value = os.pread(fd, min(8 * 1024**2, end-first), first)
                if not value: raise OSError('Incomplete validation archive')
                view = memoryview(value)
                while view:
                    count = os.pwrite(target_fd, view, first)
                    if not count: raise OSError('Incomplete local validation archive')
                    first += count; view = view[count:]
        finally: os.close(fd)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy_range, range(0, size, 128 * 1024**2)))
        os.fsync(target_fd)
    finally: os.close(target_fd)
    with tarfile.open(target, 'r:') as archive:
        members = sorted([m for m in archive if m.isfile() and m.name.endswith('.JPEG')], key=lambda m:m.name)
    if len(members) != 50000 or len({m.name for m in members}) != 50000:
        raise ValueError('Expected exactly 50000 distinct ImageNet validation images')
    np.savez(index, offsets=np.array([m.offset_data for m in members], dtype=np.int64),
        sizes=np.array([m.size for m in members], dtype=np.int32), labels=np.zeros(50000,dtype=np.int16),
        classes=np.array(['unused_for_reconstruction']))
    write_json(data / 'validation-reconstruction-index.json', dict(images=50000,
        source=str(source), bytes=size, labels_unused=True, purpose='unconditioned codec reconstruction only'))


@torch.no_grad()
def quality(base):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from torchvision.utils import save_image
    from src import rqvae_metrics as metrics
    rank, world, local = [int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']]
    device = torch.device('cuda', local); torch.cuda.set_device(device); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device)
    profile = Compound128Recipe(base)
    split = str(profile.recipe.verification.get('reconstruction_split', 'val'))
    if split == 'val' and rank == 0: prepare_validation(base)
    dist.barrier()
    aux, _ = profile.tokenizer(device); before = profile.digest(aux)
    config = load_config(base / 'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(base / 'data', 'train' if split == 'train_balanced50k' else 'val',
        create_transforms(config.dataset, split='val'))
    if split == 'train_balanced50k':
        rng = np.random.default_rng(261107128)
        indices = np.stack([rng.choice(np.flatnonzero(dataset.labels == label),50,replace=False)
            for label in range(1000)]).T.reshape(-1)
        assert len(np.unique(indices)) == 50000
        if rank == 0:
            np.save(base / 'quality-indices.npy',indices)
    else:
        assert len(dataset) == 50000
        indices = np.arange(50000)
    chunk = int(profile.recipe.targets.encoder_batch_size)
    loader = DataLoader(Subset(dataset, indices[rank::world].tolist()), batch_size=chunk,
        num_workers=4, pin_memory=True, prefetch_factor=2)
    metrics.INCEPTION_PATH = base / 'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    accum = {name:metrics.DistributedOriginalRQVAEMetrics(device, inception=inception)
             for name in ['continuous', 'compound128']}
    error = torch.zeros(3,dtype=torch.float64,device=device)
    seen = 0; started = time.monotonic()
    output = Path(profile.recipe.output_dir).resolve()
    for index, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        atoms, values = profile.clean_components(aux, images)
        ids = nearest_coefficient_ids(values, aux.coeff_bins)
        vectors = aux.dictionary.T[atoms]
        real = images.mul(.5).add(.5).clamp(0,1)
        features, _ = inception(real); features = features.double()
        for name, coefficients in [('continuous', values), ('compound128', aux.coeff_bins[ids])]:
            latent = (vectors * (coefficients * aux.coeff_scales)[...,None]).sum(-2).permute(0,3,1,2).contiguous()
            decoded = aux.decoder(aux.post_quant_conv(latent)).clamp(-1,1)
            if index == 0 and name == 'compound128':
                codes = torch.stack((atoms, ids + 16384), -1).flatten(-2)
                actual, context = profile.targets(aux, images)
                torch.testing.assert_close(actual, codes, rtol=0, atol=0)
                torch.testing.assert_close(profile.decode(aux, context), decoded, rtol=0, atol=0)
                assert profile.verify_targets(aux, actual, context)['training_and_evaluation_codec_identical']
                if rank == 0:
                    path = output / 'samples/tokenizer-reconstructions.png'; path.parent.mkdir(exist_ok=True)
                    save_image(torch.stack((real, decoded.mul(.5).add(.5)), 1).flatten(0,1), path,nrow=4)
            metric = accum[name]
            metric.real_sum.add_(features.sum(0)); metric.real_cross.add_(features.T @ features)
            metric.real_count.add_(len(images)); metric.update(decoded.mul(.5).add(.5).clamp(0,1), real=False)
        error[0] += (aux.coeff_bins[ids] - values).square().sum()
        error[1] += (values.abs() > float(profile.recipe.tokenizer.coefficient_max)).sum()
        error[2] += values.numel(); seen += len(images)
        if rank == 0 and index % 25 == 0:
            row = dict(images_on_rank=seen, target_on_rank=50000 // world,
                seconds=time.monotonic()-started,time=time.time())
            write_json(base / 'quality/progress.json', row); print(json.dumps(row),flush=True)
    results = {name:float(metric.compute()[0]) for name,metric in accum.items()}
    dist.all_reduce(error)
    assert profile.digest(aux) == before
    delta = results['compound128'] - results['continuous']
    passed = (results['compound128'] < float(profile.recipe.verification.maximum_reconstruction_rfid)
        and delta < float(profile.recipe.verification.maximum_rfid_increase))
    if rank == 0:
        report = dict(passed=passed, measured_reconstruction_rfid=results['compound128'],results=results,
            delta_vs_continuous=delta, real_images=50000, reconstructed_images=50000,
            reference_split=split, original_encoder_dictionary_decoder_frozen=True,
            reconstruction_transform='original RQ validation resize/center crop',
            validation_archive_available=(split == 'val'),
            limitation=('Class-balanced training-image codec diagnostic, not validation rFID'
                if split == 'train_balanced50k' else None),
            frozen_weights_unchanged=True, atoms=16384,depth=4,coefficient_bins=128,coefficient_bits=7,
            coefficient_range=[-3.,3.],coefficient_scales=list(profile.recipe.tokenizer.coefficient_scales),
            coefficient_noise=False,training_and_evaluation_codec_identical=True,
            measured_codec='actual deterministic compound training targets and prefixes',
            coefficient_mse=float(error[0]/error[2]),coefficient_clipping_fraction=float(error[1]/error[2]),
            source_stage1_checkpoint_sha256=str(profile.recipe.tokenizer.sha256),training_updates_added=0,
            seconds=time.monotonic()-started)
        write_json(base / 'tokenizer-verification.json',report); print(json.dumps(report),flush=True)
    dist.barrier();dist.destroy_process_group()
    if not passed: raise RuntimeError('128-bin tokenizer failed the reconstruction gate')


def main(base):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['benchmark','quality'])
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--microbatch',type=int,default=128)
    parser.add_argument('--attention',choices=['native','sdpa_spatial'],default='native')
    args = parser.parse_args()
    if args.action == 'benchmark': benchmark(Path(base),args)
    else: quality(Path(base))
