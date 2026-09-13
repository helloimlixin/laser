"""Runtime utilities for the released KakaoBrain RQ models.

Training algorithms live in the upstream package. This module supplies the
unreleased stage-2 driver with batching, data caching, and evaluation utilities.
"""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(checkpoint, config_path, device):
    from rqvae.models import create_model
    from rqvae.utils.config import load_config, augment_arch_defaults
    config = load_config(config_path)
    model, _ = create_model(augment_arch_defaults(config.arch), ema=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    model.requires_grad_(False).eval().to(device)
    assert list(model.code_shape) == [8, 8, 4]
    assert model.quantizer.shared_codebook
    assert model.quantizer.codebooks[0].weight.shape == (16385, 256)
    return model, config


def load_stage2_config(upstream, tokenizer_checkpoint=None):
    from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
    config = load_config(Path(upstream) / 'configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml')
    config.arch = augment_arch_defaults(config.arch)
    config.arch.vocab_size = config.dataset.vocab_size
    config.optimizer = augment_optimizer_defaults(config.optimizer)
    config.seed = 0
    if tokenizer_checkpoint:
        config.vqvae = {'ckpt': str(tokenizer_checkpoint)}
    # The upstream sampling CLI reads "sampling", training YAML uses experiment.sample.
    config.sampling = {'temp': 1.0, 'top_k': config.experiment.sample.top_k,
                       'top_p': config.experiment.sample.top_p}
    return config


def fresh_transformer(config, device='cpu'):
    from rqvae.models import create_model
    from rqvae.optimizer.optimizer import create_resnet_optimizer
    model, _ = create_model(config.arch, ema=False)
    # Use the initializer supplied by the released Stage2Model interface.
    # The unpublished driver cannot be consulted for optimizer parameter grouping.
    model.apply(model._init_weights)
    model.to(device)
    optimizer = create_resnet_optimizer(model, config.optimizer)
    assert not optimizer.state
    return model, optimizer


class CachedLatents(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.values = np.load(path, mmap_mode='r')
        assert self.values.dtype == np.float32 and self.values.shape[1:] == (8, 8, 256)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return torch.from_numpy(self.values[index].copy())


class IndexedImages(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0], index


class FeatureMoments:
    def __init__(self, device):
        self.count = torch.zeros((), device=device, dtype=torch.float64)
        self.total = torch.zeros(2048, device=device, dtype=torch.float64)
        self.cross = torch.zeros(2048, 2048, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, features):
        features = features.double()
        self.count += len(features)
        self.total += features.sum(0)
        self.cross.add_(features.T @ features)

    def finish(self):
        for value in (self.count, self.total, self.cross):
            dist.all_reduce(value)
        n = int(self.count.item())
        assert n > 1
        mean = self.total / n
        covariance = (self.cross - n * torch.outer(mean, mean)) / (n - 1)
        return n, mean.cpu().numpy(), covariance.cpu().numpy()


def fid_from_moments(moments, reference):
    from rqvae.metrics.fid import frechet_distance
    _, mu, covariance = moments
    real = np.load(reference)
    score = float(frechet_distance(mu, covariance, real['mu'], real['sigma']))
    assert math.isfinite(score)
    return score


@torch.no_grad()
def build_latent_cache(tokenizer, config, output, device, reference, rank, world):
    from rqvae.img_datasets.lsun import LSUNClass
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.metrics.fid import get_inception_model
    transform = create_transforms(config.dataset, split='train')
    images = LSUNClass('/tmp/laser-sign-data', category_name='church', transform=transform)
    assert len(images) == 126227
    output = Path(output)
    path = output / 'latents-fp32.npy'
    if rank == 0:
        assert not path.exists(), 'Refusing an unverified latent cache'
        values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                          shape=(len(images), 8, 8, 256))
        del values
    dist.barrier()
    values = np.load(path, mmap_mode='r+')
    loader = DataLoader(Subset(IndexedImages(images), list(range(rank, len(images), world))),
                        batch_size=64, num_workers=8, pin_memory=True, shuffle=False)
    inception = get_inception_model().eval().requires_grad_(False).to(device)
    reconstructed = FeatureMoments(device)
    originals = FeatureMoments(device)
    started = time.time()
    before = state_sha256(tokenizer.quantizer)
    for batch_index, (xs, indices) in enumerate(loader):
        xs = xs.to(device, non_blocking=True)
        z = tokenizer.encode(xs)
        assert z.dtype == torch.float32 and torch.isfinite(z).all()
        values[indices.numpy()] = z.cpu().numpy()
        chosen = indices < 50000
        if chosen.any():
            selected_z = z[chosen.to(device)]
            quantized, _, _ = tokenizer.quantizer(selected_z)
            decoded = tokenizer.decode(quantized).mul(.5).add(.5).clamp(0, 1)
            reconstructed.update(inception(decoded))
            originals.update(inception(xs[chosen.to(device)].mul(.5).add(.5).clamp(0, 1)))
        if rank == 0 and (batch_index % 20 == 0 or batch_index + 1 == len(loader)):
            atomic_json(output / 'status.json', {'phase': 'caching_frozen_fp32_latents',
                'batch': batch_index + 1, 'total_batches': len(loader),
                'elapsed_seconds': time.time() - started, 'updated_unix': time.time()})
    values.flush()
    del values
    assert before == state_sha256(tokenizer.quantizer), 'Frozen codebook changed during caching'
    reconstruction_stats = reconstructed.finish()
    original_stats = originals.finish()
    assert reconstruction_stats[0] == original_stats[0] == 50000
    if rank == 0:
        from rqvae.metrics.fid import frechet_distance
        matched_rfid = float(frechet_distance(reconstruction_stats[1], reconstruction_stats[2],
                                            original_stats[1], original_stats[2]))
        rfid_reference = fid_from_moments(reconstruction_stats, reference)
        report = {'phase': 'cache_complete', 'images': len(images), 'dtype': 'float32',
            'shape': [len(images), 8, 8, 256], 'hard_codes_cached': False,
            'stochastic_quantization_recomputed_each_visit': True,
            'reconstruction_samples': 50000, 'rfid_matched_50k': matched_rfid,
            'rfid_against_published_reference_50k': rfid_reference,
            'reference': str(reference), 'cache_sha256': file_sha256(path),
            'elapsed_seconds': time.time() - started}
        atomic_json(output / 'cache.json', report)
        np.savez(output / 'reconstruction-50k-statistics.npz', mu=reconstruction_stats[1],
                 sigma=reconstruction_stats[2], n=50000)
    dist.barrier()
    images.env.close()
    del inception, reconstructed, originals, images, loader
    torch.cuda.empty_cache()
    return CachedLatents(path)


def accumulated_update(ddp, tokenizer, optimizer, scaler, batches, max_gn=1.0):
    """Each depth target is conditioned on the sampled prefix from upstream RQ."""
    total = sum(len(z) for z in batches)
    device = next(ddp.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = torch.zeros((), device=device)
    for index, z in enumerate(batches):
        with (ddp.no_sync() if index + 1 < len(batches) else nullcontext()):
            with torch.no_grad():
                targets, codes = tokenizer.quantizer.get_soft_codes(
                    z.to(device, non_blocking=True), temp=0.5, stochastic=True)
            logits = ddp(codes, model_aux=tokenizer, amp=True)
            # FP32 softmax/cross entropy, original probability targets and reduction.
            loss = ddp.module.compute_loss(logits.float(), targets, use_soft_target=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite original RQ soft-target cross entropy')
            weight = len(z) / total
            scaler.scale(loss * weight).backward()
            weighted_loss += loss.detach() * weight
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), max_gn)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    success = scaler.get_scale() >= old_scale
    metrics = torch.stack([weighted_loss, norm.float()])
    dist.all_reduce(metrics)
    metrics /= dist.get_world_size()
    return {'loss': float(metrics[0]), 'gradient_norm': float(metrics[1]),
            'amp_scale': scaler.get_scale(), 'optimizer_updated': success,
            'global_images': total * dist.get_world_size()}


@torch.no_grad()
def validation_latents(tokenizer, config, device, rank, world):
    from rqvae.img_datasets.lsun import LSUNClass
    from rqvae.img_datasets.transforms import create_transforms
    # Original LSUNClass lists the val path but omits it from its accepted names.
    if 'church_val' not in LSUNClass.valid_categories:
        LSUNClass.valid_categories.append('church_val')
    images = LSUNClass('/tmp/laser-sign-data', category_name='church_val',
                      transform=create_transforms(config.dataset, split='val'))
    assert len(images) == 300
    loader = DataLoader(Subset(images, list(range(rank, 300, world))), batch_size=32,
                        num_workers=4, pin_memory=True)
    values = [tokenizer.encode(xs.to(device)).cpu() for xs, _ in loader]
    images.env.close()
    return torch.cat(values)


@torch.no_grad()
def evaluate_validation(model, tokenizer, latents, device, rank):
    model.eval()
    result = torch.zeros(7, device=device, dtype=torch.float64)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(61000 + rank)
        for z in latents.split(32):
            z = z.to(device)
            targets, codes = tokenizer.quantizer.get_soft_codes(z, temp=.5, stochastic=True)
            logits = model(codes, model_aux=tokenizer, amp=True).float()
            soft_loss = model.compute_loss(logits, targets, use_soft_target=True)
            _, _, hard_codes = tokenizer.quantizer(z)
            hard_logits = model(hard_codes, model_aux=tokenizer, amp=True).float()
            depth_nll = model.compute_codebook_loss(hard_logits, hard_codes)
            result[0] += soft_loss * len(z)
            result[1] += depth_nll.mean() * len(z)
            result[2:6] += depth_nll.double() * len(z)
            result[6] += len(z)
    dist.all_reduce(result)
    assert result[6].item() == 300 and torch.isfinite(result).all()
    result[:6] /= result[6]
    model.train()
    return {'soft_ce': float(result[0]), 'hard_code_nll': float(result[1]),
            **{f'depth_{i}_nll': float(result[2+i]) for i in range(4)}, 'images': 300}


@torch.no_grad()
def evaluate_samples(model, tokenizer, inception, n, epoch, output, reference, device, rank, world):
    from torchvision.utils import save_image
    model.eval()
    moments = FeatureMoments(device)
    # Isolate evaluation RNG so sampling does not change training dropout or codes.
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(71000 + rank)
        local_total = len(range(rank, n, world))
        for offset in range(0, local_total, 100):
            size = min(100, local_total - offset)
            codes = model.sample(torch.zeros(size, 8, 8, 4, dtype=torch.long, device=device),
                model_aux=tokenizer, temperature=1.0, top_k=250, top_p=1.0, amp=True,
                cached=True, is_tqdm=False)
            # Released sampling code decodes in FP32 and keeps continuous [0,1] pixels.
            images = tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0, 1)
            moments.update(inception(images))
            if rank == 0 and offset == 0:
                save_image(images[:64], Path(output) / f'samples-epoch{epoch:03d}.png', nrow=8)
            if rank == 0:
                atomic_json(Path(output) / 'status.json', {'phase': 'evaluating_generation',
                    'epoch': epoch, 'samples_per_rank_done': offset + size,
                    'samples_total': n, 'updated_unix': time.time()})
    statistics = moments.finish()
    assert statistics[0] == n
    score = None
    if rank == 0:
        score = fid_from_moments(statistics, reference)
        atomic_json(Path(output) / f'fid-{n}-epoch{epoch:03d}.json',
                    {'epoch': epoch, 'n': n, 'fid': score, 'temperature': 1.,
                     'top_k': 250, 'top_p': 1., 'reference': str(reference)})
    dist.barrier()
    model.train()
    return score
