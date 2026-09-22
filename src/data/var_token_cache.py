"""Complete cached multiscale trajectories; never mix sites across scales."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FORMAT = 'laser_var_complete_trajectory_v1'


class VARTokenCache(Dataset):
    def __init__(self, directory, split, seed=0):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest['format'] != FORMAT:
            raise ValueError('Not a complete-trajectory VAR token cache')
        self.split, self.seed, self.epoch = split, int(seed), 0
        self.arrays = {name: np.load(self.directory / f'{split}-{name}.npy', mmap_mode='r')
                       for name in ('atoms', 'coefficients', 'physical_coefficients', 'labels')}
        shape = self.arrays['atoms'].shape
        if shape != self.arrays['coefficients'].shape or shape != self.arrays['physical_coefficients'].shape:
            raise ValueError('Cached atoms, coefficient IDs, and physical targets must be paired')
        if len(shape) != 5 or shape[0] != len(self.arrays['labels']):
            raise ValueError('Expected [image, flip, trajectory, position, sparse_depth]')
        self.views, self.variants = shape[1:3]

    def __len__(self):
        return len(self.arrays['labels'])

    def __getitem__(self, index):
        index = int(index)
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 10000019 + index)
        # Matches the image loader's seeded RandomHorizontalFlip, without decoding JPEGs.
        view = int(torch.rand((), generator=generator) < .5) if self.views == 2 else 0
        variant = int(torch.randint(self.variants, (), generator=generator)) if self.variants > 1 else 0
        selection = (index, view, variant)
        return dict(atoms=torch.tensor(self.arrays['atoms'][selection].astype(np.int64)),
                    coefficients=torch.tensor(self.arrays['coefficients'][selection].astype(np.int64)),
                    physical_coefficients=torch.tensor(self.arrays['physical_coefficients'][selection]),
                    labels=int(self.arrays['labels'][index]), index=index, view=view, variant=variant)


@torch.no_grad()
def restore_cached_codes(quantizer, batch, coefficient_temperatures=None):
    """Reconstruct exact teacher contexts and soft targets from paired cache fields."""
    device = quantizer.coefficient_max.device
    atoms = batch['atoms'].to(device, non_blocking=True)
    ids = batch['coefficients'].to(device, non_blocking=True)
    physical = batch['physical_coefficients'].to(device, non_blocking=True)
    latent, inputs = quantizer.from_codes(atoms, ids)
    probabilities, offset = [], 0
    for scale, pn in enumerate(quantizer.v_patch_nums):
        end = offset + pn * pn
        temperature = coefficient_temperatures[scale] if coefficient_temperatures else 0.
        if temperature:
            grid = quantizer.coefficient_values(torch.arange(quantizer.coefficient_bins, device=device), scale)
            p = (-(physical[:, offset:end, :, None] - grid).square() / temperature).softmax(-1)
        else:
            p = torch.nn.functional.one_hot(ids[:, offset:end], quantizer.coefficient_bins).float()
        probabilities.append(p)
        offset = end
    return dict(atoms=atoms, coefficients=ids, physical_coefficients=physical,
                coefficient_probabilities=torch.cat(probabilities, 1), inputs=inputs, latent=latent)
