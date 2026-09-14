"""Use released RQ-VAE file-based FID directly after distributed sampling.

Callers must bind the verified RQ-VAE source snapshot before importing this file.
Only sampling, artifact management and rank synchronization are local code.
Feature extraction, dense statistics and FID are the released implementation.
"""
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image
from rqvae.metrics import fid as upstream_fid
from src.original_rq_training import atomic_json, file_sha256

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'outputs/church-compact-rq-stage2-20260913/source-snapshot'
UPSTREAM = SNAPSHOT / 'outputs/church-rq-baseline-scratch-20260912/upstream-source'
REFERENCE_SHA256 = '809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6'


def verify_official_sources():
    assert Path(upstream_fid.__file__).resolve() == UPSTREAM / 'rqvae/metrics/fid.py'
    manifest = json.loads((SNAPSHOT.parent / 'source-manifest.json').read_text())
    paths = [UPSTREAM / 'rqvae/metrics/fid.py', UPSTREAM / 'rqvae/metrics/inception.py']
    hashes = {str(path.relative_to(SNAPSHOT)): file_sha256(path) for path in paths}
    for name, digest in hashes.items():
        assert manifest[name] == digest, name
    return hashes


def save_rgb(folder, pixels, rank, offset):
    assert pixels.dtype == torch.float32 and pixels.ndim == 4
    assert tuple(pixels.shape[1:]) == (3, 256, 256)
    assert torch.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 1
    name = f'samples-rank{rank}-{offset:06d}.pkl'
    data = pickle.dumps(pixels.detach().cpu().numpy(), protocol=4)
    with (folder / name).open('xb') as stream:
        stream.write(data)
    return dict(file=name, images=len(pixels), sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data), rank=rank, offset=offset)


def score_saved_rgb(folder, records, expected_count, reference, device):
    """Fresh upstream compute_fid call; never consume an existing acts cache."""
    sources = verify_official_sources()
    assert file_sha256(reference) == REFERENCE_SHA256
    assert not (folder / 'acts.npz').exists(), 'Refusing a pre-existing FID activation cache'
    assert sum(row['images'] for row in records) == expected_count
    assert len({row['file'] for row in records}) == len(records)
    assert {path.name for path in folder.glob('samples*.pkl')} == {row['file'] for row in records}
    # This is the unmodified released function, including Inception creation,
    # reading float32 pickles, mean_covar_numpy and frechet_distance.
    value = float(upstream_fid.compute_fid(str(folder), str(reference), batch_size=100, device=device))
    assert np.isfinite(value)
    with np.load(folder / 'acts.npz') as saved:
        assert saved['acts'].shape == (expected_count, 2048)
        assert np.isfinite(saved['acts']).all()
        mu, sigma = saved['mu'], saved['sigma']
    receipt = dict(fid=value, n=expected_count, evaluator='rqvae.metrics.fid.compute_fid',
        evaluator_sources=sources, reference=str(reference), reference_sha256=REFERENCE_SHA256,
        inception_batch_size=100, pixels='float32 RGB [0,1], released pickle format',
        activation_cache='acts.npz', preexisting_activation_cache=False,
        sample_files=records, generated_rgb_bytes=sum(row['bytes'] for row in records),
        completed_unix=time.time())
    atomic_json(folder / 'result.json', receipt)
    return value, mu, sigma


@torch.no_grad()
def evaluate_samples(model, tokenizer, inception, n, epoch, output, reference, device, rank, world):
    """Training-driver interface; `inception` is unused by upstream compute_fid."""
    assert n > 1 and world == dist.get_world_size() and rank == dist.get_rank()
    folder = Path(output) / 'official-fid' / f'fid-{n}-epoch{epoch:03d}'
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    was_training = model.training
    model.eval()
    try:
        # Includes evaluation model initialization and the upstream DataLoader,
        # not just sampling, to keep the training RNG sequence unchanged.
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(71000 + rank)
            local_total = len(range(rank, n, world))
            records, all_codes = [], []
            for offset in range(0, local_total, 100):
                size = min(100, local_total - offset)
                codes = model.sample(torch.zeros(size, 8, 8, 4, dtype=torch.long, device=device),
                    model_aux=tokenizer, temperature=1., top_k=1400, top_p=1., amp=True,
                    cached=True, is_tqdm=False)
                pixels = tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0, 1)
                records.append(save_rgb(folder, pixels, rank, offset))
                all_codes.append(codes.cpu().numpy())
                if rank == 0:
                    if offset == 0:
                        save_image(pixels[:64], Path(output) / f'samples-{n}-epoch{epoch:03d}.png', nrow=8)
                    atomic_json(Path(output) / 'status.json', dict(phase='saving_generated_rgb_for_official_fid',
                        epoch=epoch, samples_per_rank_done=offset+size, samples_total=n,
                        updated_unix=time.time()))
            np.save(folder / f'codes-rank{rank}.npy', np.concatenate(all_codes) if all_codes
                    else np.empty((0, 8, 8, 4), dtype=np.int64))
            gathered = [None] * world
            dist.all_gather_object(gathered, records)
            result = [None]
            if rank == 0:
                try:
                    records = [row for shard in gathered for row in shard]
                    atomic_json(Path(output) / 'status.json', dict(phase='official_file_based_fid',
                        epoch=epoch, samples_total=n, updated_unix=time.time()))
                    value, mu, sigma = score_saved_rgb(folder, records, n, reference, device)
                    np.savez(Path(output) / f'statistics-{n}-epoch{epoch:03d}.npz', mu=mu, sigma=sigma, samples=n)
                    atomic_json(Path(output) / f'fid-{n}-epoch{epoch:03d}.json', dict(epoch=epoch,
                        n=n, fid=value, temperature=1., top_k=1400, top_p=1., reference=str(reference),
                        evaluator='rqvae.metrics.fid.compute_fid', artifacts=str(folder)))
                    # Keep codes, all Inception features/statistics, and RGB file
                    # hashes. Raw float32 RGB is temporary (39 GB at 50k).
                    for row in records:
                        (folder / row['file']).unlink()
                    atomic_json(folder / 'rgb-cleanup.json', dict(completed=True,
                        deleted_files=len(records), retained='codes, acts.npz, result with RGB hashes'))
                    result[0] = dict(ok=True, fid=value)
                except BaseException as error:
                    result[0] = dict(ok=False, error_type=type(error).__name__, error=str(error))
                    atomic_json(folder / 'failure.json', result[0])
            dist.broadcast_object_list(result, src=0, device=device)
            if not result[0]['ok']:
                raise RuntimeError(f"Official FID failed: {result[0]}")
            return result[0]['fid']
    finally:
        model.train(was_training)
