#!/usr/bin/env python3
"""Audit saved Church FID statistics and independently re-read LASER images.

Uses isolated, recorded sources. Does not change a training process or W&B run.
Run `statistics` on CPU; run `images` with two torchrun workers.
"""
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'outputs/church-compact-rq-stage2-20260913/source-snapshot'
UPSTREAM = SNAPSHOT / 'outputs/church-rq-baseline-scratch-20260912/upstream-source'
STAGE1 = ROOT / 'outputs/church-laser-three-epoch-20260914/stage1-source'
RUN = ROOT / 'outputs/church-laser-original-recipe-20260913/train'
PUBLISHED = ROOT / 'outputs/church-published-audit-20260913/published-saved-50k'
REFERENCE = ROOT / 'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz'
sys.path[:0] = [str(UPSTREAM), str(SNAPSHOT), str(ROOT)]
import src
src.__path__ = [str(SNAPSHOT / 'src')]
os.environ.setdefault('TORCH_HOME', '/workspace/tmp/official-rqvae-eval-cache')

import numpy as np
from scipy import linalg
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from rqvae.metrics.fid import (get_inception_model, frechet_distance,
                               compute_statistics_from_files)
from rqvae.models import create_model
from src.compact_rq_training import FrozenCompactTokenizer
from src.original_rq_training import (atomic_json, file_sha256, state_sha256,
                                      FeatureMoments, fid_from_moments)


def historical_metrics():
    path = STAGE1 / 'src/rqvae_metrics.py'
    spec = importlib.util.spec_from_file_location('church_historical_metrics_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def statistics(out):
    """Independent symmetric-PSD equation, using no scipy.sqrtm call."""
    out.mkdir(parents=True, exist_ok=False)
    assert file_sha256(REFERENCE) == '809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6'
    reference = np.load(REFERENCE)
    mu_ref, cov_ref = reference['mu'], reference['sigma']
    eigenvalues, vectors = linalg.eigh(cov_ref)
    assert eigenvalues.min() > -1e-8
    root_ref = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T

    def score(mu, covariance):
        sandwich = root_ref @ covariance @ root_ref
        values = linalg.eigvalsh((sandwich + sandwich.T) * .5)
        assert values.min() > -1e-8, values.min()
        return float(np.square(mu - mu_ref).sum() + np.trace(covariance)
                     + np.trace(cov_ref) - 2 * np.sqrt(np.maximum(values, 0)).sum())

    records = []
    for epoch in (50, 100):
        path = RUN / f'statistics-50000-epoch{epoch:03d}.npz'
        saved = np.load(path)
        logged = json.loads((RUN / f'fid-50000-epoch{epoch:03d}.json').read_text())
        value = score(saved['mu'], saved['sigma'])
        record = dict(epoch=epoch, samples=int(saved['samples']), logged=logged['fid'],
                      independent_psd_fid=value, error=abs(value - logged['fid']),
                      statistics=str(path), statistics_sha256=file_sha256(path))
        assert record['samples'] == 50000 and record['error'] < 1e-5, record
        records.append(record)
        print(json.dumps(record), flush=True)

    features = np.concatenate([np.load(PUBLISHED / f'features-rank{rank}.npy')
                               for rank in range(2)]).astype(np.float64)
    assert features.shape == (50000, 2048) and np.isfinite(features).all()
    full = score(features.mean(0), np.cov(features, rowvar=False))
    published_logged = json.loads((PUBLISHED / 'result.json').read_text())['fid']
    assert abs(full - published_logged) < 1e-5
    subsets = []
    for seed in range(4):
        indices = np.random.default_rng(seed).choice(len(features), 4096, replace=False)
        np.save(out / f'published-subset-indices-seed{seed}.npy', indices)
        part = features[indices]
        value = score(part.mean(0), np.cov(part, rowvar=False))
        subsets.append(dict(seed=seed, samples=len(part), fid=value))
        print(json.dumps(subsets[-1]), flush=True)
    # No feature extraction or model changes between full and subset scores.
    result = dict(reference=str(REFERENCE), reference_sha256=file_sha256(REFERENCE),
                  equation='symmetric covariance sandwich; scipy.linalg.eigh/eigvalsh, no sqrtm',
                  laser_saved_fid50000=records,
                  published_fid50000=full, published_logged_fid50000=published_logged,
                  published_fid4096_subsets=subsets,
                  subset_mean=float(np.mean([r['fid'] for r in subsets])),
                  sample_count_caveat='Subset variation demonstrates sample-count effects; it is not a universal FID correction.',
                  finished_unix=time.time())
    atomic_json(out / 'result.json', result)


def reconstruction_aggregation(out):
    """Exercise the active stage-1 accumulator on recorded matched features."""
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 2
    torch.set_num_threads(8)
    dist.init_process_group('gloo', timeout=timedelta(minutes=10))
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    module = historical_metrics()

    class RecordedFeatures(torch.nn.Module):
        def forward(self, values, *, return_logits=False):
            assert not return_logits
            return values, None

    base = ROOT / 'outputs/church-compact-scaled-rq-20260913/reconstruction'
    real = np.load(base / 'original-features.npy')
    fake = np.load(base / 'adaptive2-features.npy')
    assert real.shape == fake.shape == (4096, 2048)
    metric = module.DistributedOriginalRQVAEMetrics('cpu', inception=RecordedFeatures())
    for real_flag, array in [(True, real), (False, fake)]:
        local = array[rank::world]
        for offset in range(0, len(local), 127):
            metric.update(torch.from_numpy(local[offset:offset+127]), real=real_flag)
    actual, _, _ = metric.compute()
    assert int(metric.real_count) == int(metric.fake_count) == 4096
    if rank == 0:
        mu_real, cov_real = real.astype(np.float64).mean(0), np.cov(real, rowvar=False)
        mu_fake, cov_fake = fake.astype(np.float64).mean(0), np.cov(fake, rowvar=False)
        expected = float(frechet_distance(mu_real, cov_real, mu_fake, cov_fake))
        identity = float(frechet_distance(mu_real, cov_real, mu_real, cov_real))
        assert abs(actual - expected) < 1e-6 and abs(identity) < 1e-6
        result = dict(samples_real=4096, samples_reconstructed=4096, world_size=world,
            historical_distributed_rfid=actual, independent_dense_rfid=expected,
            error=abs(actual - expected), identical_distribution_fid=identity,
            accumulator_source=str(STAGE1 / 'src/rqvae_metrics.py'),
            accumulator_source_sha256=file_sha256(STAGE1 / 'src/rqvae_metrics.py'),
            scope='Two-rank accumulation and reduction using recorded features; live feature extraction checked separately.',
            finished_unix=time.time())
        atomic_json(out / 'result.json', result)
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


@torch.no_grad()
def images(out):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 2
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)
    dist.init_process_group('nccl', timeout=timedelta(hours=1))
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'samples').mkdir()
    dist.barrier()
    started = time.time()

    def status(phase, **kwargs):
        if rank == 0:
            record = dict(phase=phase, elapsed_seconds=time.time() - started,
                          updated_unix=time.time(), **kwargs)
            atomic_json(out / 'status.json', record)
            print(json.dumps(record), flush=True)

    status('loading')
    manifest = json.loads((SNAPSHOT.parent / 'source-manifest.json').read_text())
    for name, digest in manifest.items():
        assert file_sha256(SNAPSHOT / name) == digest, name
    config = OmegaConf.load(RUN / 'config.yaml')
    model, _ = create_model(config.arch, ema=False)
    checkpoint = RUN / 'epoch100_model.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert payload['epoch'] == 100
    model.load_state_dict(payload['state_dict'], strict=True)
    del payload
    model = model.to(device).eval().requires_grad_(False)
    tokenizer = FrozenCompactTokenizer(config.vqvae.ckpt, config.vqvae.codebook).to(device).eval()
    inception = get_inception_model().to(device).eval().requires_grad_(False)
    historical = historical_metrics().OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    old_state, new_state = historical.model.state_dict(), inception.state_dict()
    assert old_state.keys() == new_state.keys()
    assert all(torch.equal(old_state[key], new_state[key]) for key in old_state)
    initial_hashes = [state_sha256(model), state_sha256(tokenizer)]
    if rank == 0:
        atomic_json(out / 'specification.json', dict(checkpoint=str(checkpoint),
            checkpoint_sha256=file_sha256(checkpoint), tokenizer=str(config.vqvae.ckpt),
            tokenizer_sha256=file_sha256(config.vqvae.ckpt), codebook=str(config.vqvae.codebook),
            codebook_sha256=file_sha256(config.vqvae.codebook), reference=str(REFERENCE),
            reference_sha256=file_sha256(REFERENCE), samples=4096, seed=71000,
            top_k=1400, top_p=1., temperature=1., batch_per_gpu=100, world_size=2,
            precision='FP16 AR; FP32 decoder/Inception; FP64 moments; TF32 off',
            pixels='float32 RGB [0,1]; upstream pickle format; no PNG quantization',
            stage1_and_stage2_inception_weights_identical=True, source_files_verified=len(manifest)))
    features, codes_all = [], []
    moments = FeatureMoments(device)
    # All model initialization is above this scope, so it cannot perturb sampling.
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(71000 + rank)
        local_total = len(range(rank, 4096, world))
        for offset in range(0, local_total, 100):
            size = min(100, local_total - offset)
            codes = model.sample(torch.zeros(size, 8, 8, 4, dtype=torch.long, device=device),
                                 model_aux=tokenizer, temperature=1., top_k=1400, top_p=1.,
                                 amp=True, cached=True, is_tqdm=False)
            pixels = tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0, 1)
            assert pixels.dtype == torch.float32 and torch.isfinite(pixels).all()
            values = inception(pixels)
            moments.update(values)
            features.append(values.cpu().numpy())
            codes_all.append(codes.cpu().numpy())
            with (out / 'samples' / f'samples-rank{rank}-{offset:05d}.pkl').open('wb') as stream:
                pickle.dump(pixels.cpu().numpy(), stream, protocol=4)
            if offset == 0:
                historical_values = historical(pixels[:16])[0]
                separate_values = inception(pixels[:16])
                backend_error = float((historical_values - separate_values).abs().max())
                assert backend_error == 0., backend_error
                serial = torch.cat([tokenizer.decode_code(codes[i:i+1]) for i in range(2)])
                serial = serial.mul(.5).add(.5).clamp(0, 1)
                decoder_mse = float((serial - pixels[:2]).square().mean())
                assert decoder_mse < 1e-8, decoder_mse
                atomic_json(out / f'path-checks-rank{rank}.json', dict(
                    stage1_vs_stage2_features_max_error=backend_error,
                    serial_vs_batch_decode_mse=decoder_mse))
            status('generating_and_saving', samples_done=(offset + size) * world, samples=4096)
    n, mu, cov = moments.finish()
    assert n == 4096
    np.save(out / f'features-rank{rank}.npy', np.concatenate(features))
    np.save(out / f'codes-rank{rank}.npy', np.concatenate(codes_all))
    assert initial_hashes == [state_sha256(model), state_sha256(tokenizer)]
    if rank == 0:
        streamed = fid_from_moments((n, mu, cov), REFERENCE)
        np.savez(out / 'statistics.npz', mu=mu, sigma=cov, samples=n)
        logged = json.loads((RUN / 'fid-4096-epoch100.json').read_text())['fid']
        atomic_json(out / 'streamed-result.json', dict(fid=streamed, samples=n,
                                                     historical_logged_fid=logged,
                                                     replay_difference=abs(streamed - logged)))
    dist.barrier()
    dist.destroy_process_group()
    if rank != 0:
        return
    # Exercise released loading, feature extraction, dense aggregation and FID
    # on the saved RGB samples. A fresh directory prevents stale acts.npz reuse.
    del model, tokenizer, historical, moments
    torch.cuda.empty_cache()
    status('official_evaluator_reading_saved_rgb')
    file_mu, file_cov, acts = compute_statistics_from_files(
        str(out / 'samples'), batch_size=64, inception_model=inception,
        device=device, return_acts=True)
    assert tuple(acts.shape) == (4096, 2048)
    reference = np.load(REFERENCE)
    official = float(frechet_distance(file_mu, file_cov, reference['mu'], reference['sigma']))
    np.savez(out / 'official-statistics.npz', mu=file_mu, sigma=file_cov, samples=len(acts))
    result = dict(samples=len(acts), streamed_fid=streamed, official_files_fid=official,
                  official_vs_streamed_error=abs(official - streamed),
                  historical_logged_fid=logged, replay_difference=abs(streamed - logged),
                  mean_max_error=float(np.max(np.abs(file_mu - mu))),
                  covariance_max_error=float(np.max(np.abs(file_cov - cov))),
                  frozen_states_unchanged=True, finished_unix=time.time())
    assert result['official_vs_streamed_error'] < 1e-3, result
    atomic_json(out / 'result.json', result)
    status('complete', **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['statistics', 'images', 'reconstruction'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'statistics':
        statistics(args.output.resolve())
    elif args.mode == 'reconstruction':
        reconstruction_aggregation(args.output.resolve())
    else:
        images(args.output.resolve())
