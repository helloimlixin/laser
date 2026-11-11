"""Extract disjoint fitting/selection samples from ImageNet training images only."""
from common import BASE, ROOT, distributed, sha, write
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src import rq_upstream
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from runtime.compound_recipe import Compound128Recipe
from coefficient_pattern_codec import selected_support_grams


@torch.inference_mode()
def main():
    rank, world, device = distributed()
    profile = Compound128Recipe(BASE)
    assert sha(BASE/'inputs/stage1-model.pt') == str(profile.recipe.tokenizer.sha256)
    aux, _ = profile.tokenizer(device)
    before = profile.digest(aux)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(BASE/'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    rng = np.random.default_rng(261107426)
    fit, heldout = [], []
    for label in range(1000):
        chosen = rng.choice(np.flatnonzero(dataset.labels == label), 18, replace=False).tolist()
        fit.extend(chosen[:16])
        heldout.extend(chosen[16:])
    assert len(fit) == 16000 and len(heldout) == 2000 and not set(fit).intersection(heldout)
    if rank == 0:
        write(ROOT/'training-indices.json', dict(fit=fit, heldout=heldout, split='train',
            samples_per_class_fit=16, samples_per_class_heldout=2, seed=261107426,
            augmentation='original deterministic seeded training crop and flip, epoch zero',
            validation_used_for_fitting=False))
    all_indices = fit+heldout
    local_indices = all_indices[rank::world]
    fit_set = set(fit)
    loader = DataLoader(Subset(dataset, local_indices), batch_size=8, num_workers=2, pin_memory=True,
        prefetch_factor=1, multiprocessing_context='spawn')
    parts = dict(coefficients=[], grams=[], atoms=[], fit=[])
    seen = 0
    started = time.monotonic()
    for batch, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        atoms, normalized = profile.clean_components(aux, images)
        physical = normalized*aux.coeff_scales
        grams = selected_support_grams(atoms.reshape(-1, 4), aux.dictionary.T)
        flags = torch.tensor([index in fit_set for index in local_indices[seen:seen+len(images)]])
        parts['coefficients'].append(physical.reshape(-1, 4).cpu())
        parts['grams'].append(grams.cpu())
        parts['atoms'].append(atoms.reshape(-1, 4).to(dtype=torch.int16, device='cpu'))
        parts['fit'].append(flags[:, None].expand(-1, 64).reshape(-1))
        seen += len(images)
        if batch % 25 == 0:
            row = dict(phase='extract_training', rank=rank, images_on_rank=seen,
                target_on_rank=len(local_indices), elapsed_seconds=time.monotonic()-started, time=time.time())
            write(ROOT/f'extraction-rank{rank}.json', row)
            if rank == 0:
                print(row, flush=True)
    assert profile.digest(aux) == before
    payload = {key: torch.cat(value) for key, value in parts.items()}
    assert int(payload['fit'].sum()) == 1024000//world
    assert int((~payload['fit']).sum()) == 128000//world
    target = ROOT/f'training-samples-rank{rank}.pt'
    temporary = target.with_suffix('.partial')
    torch.save(payload, temporary)
    temporary.replace(target)
    write(ROOT/f'training-samples-rank{rank}.json', dict(images=seen,
        fit_sites=int(payload['fit'].sum()), heldout_sites=int((~payload['fit']).sum()),
        sha256=sha(target), frozen_tokenizer_unchanged=True,
        coefficient_space='unclipped final least-squares physical coefficients',
        support_gram='D_S transpose times D_S', validation_used=False))
    dist.barrier()
    if rank == 0:
        write(ROOT/'extraction-verification.json', dict(passed=True, fitting_images=16000,
            fitting_sites=1024000, heldout_training_images=2000, heldout_training_sites=128000,
            train_images_only=True, validation_used_for_fitting=False, frozen_tokenizer_unchanged=True))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
