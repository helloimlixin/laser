"""Controlled free-vector vocabulary sweep on the existing training-only cache."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
import yaml


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.no_grad()
def greedy(x, vectors, chunk=4096):
    residual = x.reshape(-1, 256).clone()
    codes = torch.empty(len(residual), 4, device=x.device, dtype=torch.long)
    for depth, table in enumerate(vectors):
        norms = table.square().sum(-1)
        for first in range(0, len(residual), chunk):
            r = residual[first:first + chunk]
            distance = torch.addmm(r.square().sum(-1, keepdim=True) + norms, r, table.T, alpha=-2.)
            chosen = distance.argmin(-1)
            codes[first:first + len(r), depth] = chosen
            r.sub_(table[chosen])
    reconstruction = torch.stack([vectors[d, codes[:, d]] for d in range(4)], 1).sum(1)
    return codes, reconstruction


@torch.inference_mode()
def initialize(x, source, vocabulary, generator, world):
    vectors = x.new_zeros(4, vocabulary, 256)
    residual = x.clone()
    source_count = source.shape[1]
    for depth in range(4):
        old = source[depth]
        norms = old.square().sum(-1)
        errors = torch.empty(len(residual), device=x.device)
        choices = torch.empty(len(residual), device=x.device, dtype=torch.long)
        for first in range(0, len(residual), 4096):
            r = residual[first:first + 4096]
            distance = torch.addmm(r.square().sum(-1, keepdim=True) + norms, r, old.T, alpha=-2.)
            errors[first:first + len(r)], choices[first:first + len(r)] = distance.min(-1)
        if vocabulary - 1 < source_count:
            # Select source prototypes using training frequencies, with stable
            # ID ordering to break ties. No validation images enter this rule.
            counts = torch.bincount(choices, minlength=source_count)
            dist.all_reduce(counts)
            selected = torch.argsort(counts, descending=True, stable=True)[:vocabulary - 1]
            vectors[depth, 1:] = old[selected]
        else:
            vectors[depth, 1:source_count + 1] = old
            extra = vocabulary - source_count - 1
            if extra:
                per_rank = (extra + world - 1) // world
                ids = torch.multinomial(errors.clamp_min(1e-8), per_rank, replacement=False, generator=generator)
                proposed = residual[ids].contiguous()
                gathered = [torch.empty_like(proposed) for _ in range(world)]
                dist.all_gather(gathered, proposed)
                vectors[depth, source_count + 1:] = torch.cat(gathered)[:extra]
        residual.sub_(old[choices])
    return vectors


@torch.inference_mode()
def fit_one(x, held, source, vocabulary, config, rank, world, device):
    generator = torch.Generator(device=device).manual_seed(config['seed'] + rank)
    vectors = initialize(x, source, vocabulary, generator, world)
    trace = []
    started = time.monotonic()
    output = Path(config['output_directory'])
    for iteration in range(config['fitting_passes'] + 1):
        codes, reconstructed = greedy(x, vectors)
        _, measured = greedy(held, vectors)
        values = torch.tensor([(x - reconstructed).square().sum(), x.numel(),
                               (held - measured).square().sum(), held.numel()],
                              dtype=torch.float64, device=device)
        dist.all_reduce(values)
        usage = torch.stack([torch.bincount(codes[:, d], minlength=vocabulary) for d in range(4)])
        dist.all_reduce(usage)
        row = dict(iteration=iteration, fit_latent_mse=float(values[0] / values[1]),
                   heldout_latent_mse=float(values[2] / values[3]),
                   used_vectors_by_depth=(usage > 0).sum(-1).cpu().tolist(),
                   seconds=time.monotonic() - started)
        trace.append(row)
        if rank == 0:
            write(output / 'fit' / f'T{vocabulary}.json', dict(vocabulary=vocabulary, trace=trace))
            if iteration % 4 == 0:
                print(json.dumps(dict(vocabulary=vocabulary, **row)), flush=True)
        if iteration == config['fitting_passes']:
            break
        error = x - reconstructed
        for depth in range(4):
            old = vectors[depth].clone()
            index = codes[:, depth]
            counts = torch.bincount(index, minlength=vocabulary).float()
            sums = torch.zeros_like(old)
            for first in range(0, len(x), 8192):
                ids = index[first:first + 8192]
                sums.index_add_(0, ids, error[first:first + len(ids)] + old[ids])
            dist.all_reduce(counts)
            dist.all_reduce(sums)
            prior = config['centroid_prior_weight']
            proposal = (sums + prior * old) / (counts[:, None] + prior)
            proposal[0].zero_()
            unused = torch.nonzero(counts == 0).flatten()
            unused = unused[unused != 0]
            if len(unused):
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
        del codes, reconstructed, measured, error
    if rank == 0:
        path = Path(config['work_directory']) / 'books' / f'T{vocabulary}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        final = vectors.cpu() * config['vector_gain']
        assert torch.count_nonzero(final[:, 0]) == 0
        torch.save(final, path)
        durable = output / 'books' / path.name
        durable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, durable)
        record = dict(vocabulary=vocabulary, depth=4, channels=256,
                      shape=list(final.shape), path=str(path), sha256=digest(path),
                      passes=config['fitting_passes'], vector_gain=config['vector_gain'],
                      validation_used_for_fitting=False)
        if vocabulary == 24576:
            base = Path(config['source_run_directory'])
            previous = torch.stack([torch.load(base / 'inputs' / f'vectors-depth{d}.pt', map_location='cpu', weights_only=True) for d in range(4)])
            record['max_abs_difference_from_original_24576'] = float((final - previous).abs().max())
            record['mean_squared_difference_from_original_24576'] = float((final - previous).square().mean())
        write(output / 'books' / f'T{vocabulary}.json', record)
        print(json.dumps(dict(fit_complete=True, **record)), flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 8, 'Use the original eight cache shards unchanged.'
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(config['evaluation']['memory_gib_per_gpu'] * 1024**3 / torch.cuda.get_device_properties(device).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
    base = Path(config['source_run_directory'])
    cache_manifest = json.loads((Path('/workspace/Projects/laser/outputs/imagenet-rq24576-tokenizer-20261108/cache-verification.json')).read_text())
    assert cache_manifest['stage1_checkpoint_sha256'] == config['stage1_checkpoint_sha256']
    assert cache_manifest['validation_used_for_fitting'] is False
    x = torch.from_numpy(np.load(base / 'cache' / f'fit-latents-rank{rank}.npy')).to(device).reshape(-1, 256)
    held = torch.from_numpy(np.load(base / 'cache' / f'held-latents-rank{rank}.npy')).to(device).reshape(-1, 256)
    assert x.shape == (400000, 256) and held.shape == (80000, 256)
    source = torch.load(base / 'quality/source-vectors.pt', map_location=device, weights_only=True)
    assert source.shape == (4, 16384, 256)
    if rank == 0:
        output = Path(config['output_directory'])
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, output / 'recipe.yaml')
        write(output / 'cache-verification.json', cache_manifest)
    for vocabulary in config['vocabularies']:
        fit_one(x, held, source, vocabulary, config, rank, world, device)
        torch.cuda.empty_cache()
    if rank == 0:
        write(Path(config['output_directory']) / 'fit-complete.json', dict(passed=True, vocabularies=config['vocabularies'], world_size=world))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
