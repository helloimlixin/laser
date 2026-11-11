"""Fit two seeds per vocabulary; select by disjoint held-out training latent error."""
from common import BASE, ROOT, distributed, sha, write
import math
import time

import torch
import torch.distributed as dist
from coefficient_pattern_codec import assign_coefficient_patterns, fit_coefficient_patterns


@torch.inference_mode()
def main():
    rank, world, device = distributed()
    assert world == 4
    sizes = [1024, 4096, 1024, 4096]
    size, seed = sizes[rank], 261107500+rank//2
    shards = []
    for source in range(world):
        path = ROOT/f'training-samples-rank{source}.pt'
        import json
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == metadata['sha256'] and metadata['frozen_tokenizer_unchanged']
        shards.append(torch.load(path, map_location='cpu', mmap=True, weights_only=False))
    values = torch.cat([s['coefficients'] for s in shards]).to(device)
    grams = torch.cat([s['grams'] for s in shards]).to(device)
    mask = torch.cat([s['fit'] for s in shards]).to(device)
    fitting, fitting_grams = values[mask], grams[mask]
    heldout, heldout_grams = values[~mask], grams[~mask]
    del shards, values, grams, mask
    assert len(fitting) == 1024000 and len(heldout) == 128000
    started = time.monotonic()
    history = []

    def progress(iteration, error, empty):
        row = dict(phase='fit_patterns', rank=rank, patterns=size, seed=seed, iteration=iteration,
            iterations=40, fitting_mean_squared_latent_error=error, empty_clusters=empty,
            elapsed_seconds=time.monotonic()-started, time=time.time())
        history.append(row)
        write(ROOT/f'fit-progress-rank{rank}.json', row)
        print(row, flush=True)

    patterns = fit_coefficient_patterns(fitting, num_patterns=size, grams=fitting_grams,
        iterations=40, seed=seed, chunk_size=2048, progress=progress)
    assert patterns.shape == (size, 4) and torch.isfinite(patterns).all()
    ids, errors = assign_coefficient_patterns(heldout, patterns, grams=heldout_grams, chunk_size=2048)
    counts = torch.bincount(ids, minlength=size)
    probability = counts.double()/len(ids)
    entropy = -(probability*probability.clamp_min(1e-300).log2()).sum()
    row = dict(patterns=size, seed=seed, rank=rank, fitting_images=16000, fitting_sites=len(fitting),
        heldout_training_images=2000, heldout_training_sites=len(heldout),
        heldout_training_mean_squared_latent_error=float(errors.mean()),
        heldout_pattern_marginal_entropy_bits=float(entropy),
        heldout_patterns_used=int((counts>0).sum()), iterations=40,
        coefficient_space='unclipped final least-squares physical coefficients',
        assignment_metric='exact selected-support latent squared error',
        validation_used_for_fit_or_seed_selection=False, seconds=time.monotonic()-started)
    path = ROOT/f'patterns-{size}-seed{seed}.pt'
    torch.save(dict(patterns=patterns.cpu(), metadata=row, heldout_counts=counts.cpu()), path)
    row.update(path=str(path), sha256=sha(path))
    write(ROOT/f'fit-result-rank{rank}.json', row)
    write(ROOT/f'fit-history-rank{rank}.json', history)
    dist.barrier()
    if rank == 0:
        import json
        rows = [json.loads((ROOT/f'fit-result-rank{r}.json').read_text()) for r in range(world)]
        selected = {}
        for size in [1024, 4096]:
            best = min([r for r in rows if r['patterns'] == size],
                key=lambda r:r['heldout_training_mean_squared_latent_error'])
            selected[str(size)] = best
        write(ROOT/'fit-results.json', dict(passed=True, candidates=rows, selected=selected,
            selection_metric='held-out training decoded-latent error', validation_used_for_fitting=False,
            decisions_per_site=5, original_compound_decisions_per_site=8,
            frozen_encoder_dictionary_decoder=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
