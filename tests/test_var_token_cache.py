import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.var_token_cache import FORMAT, VARTokenCache, restore_cached_codes
from src.models.multiscale_laser_var import MultiScaleLaser
from src.models.compound_var import compound_decompose


def test_cache_roundtrip_preserves_multiscale_context_and_soft_targets(tmp_path):
    torch.manual_seed(12)
    q = MultiScaleLaser(channels=4, atoms=8, sparsity=2, patch_nums=(1, 2, 3),
                        coefficient_bins=33, coefficient_max=3.).eval()
    z = torch.randn(2, 4, 3, 3)
    temperatures = [.03] * 3
    codes = compound_decompose(q, z, stochastic=True, atom_temperatures=[.01]*3,
                              coefficient_temperatures=temperatures)
    serialized = {name: torch.tensor(codes[name].cpu().numpy().astype(dtype)).to(target)
                  for name, dtype, target in [('atoms', 'uint16', torch.long),
                                              ('coefficients', 'uint16', torch.long),
                                              ('physical_coefficients', 'float32', torch.float32)]}
    restored = restore_cached_codes(q, serialized, temperatures)
    for name in ('atoms', 'coefficients', 'inputs', 'latent', 'coefficient_probabilities'):
        torch.testing.assert_close(restored[name], codes[name], rtol=0, atol=0)


def test_cache_selects_one_paired_trajectory_per_image_and_replays_on_resume(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(format=FORMAT)))
    values = np.broadcast_to(np.arange(16)[None,None,:,None,None], (2,2,16,14,2)).copy()
    for name, offset, dtype in [('atoms',0,'uint16'), ('coefficients',20,'uint16'), ('physical_coefficients',40,'float32')]:
        np.save(tmp_path/f'train-{name}.npy', (values + offset).astype(dtype))
    np.save(tmp_path/'train-labels.npy', np.array([0,1]))
    dataset = VARTokenCache(tmp_path, 'train', seed=7)
    choices = set()
    for epoch in range(20):
        dataset.epoch = epoch
        item = dataset[1]
        choices.add(item['variant'])
        assert (item['atoms'] == item['variant']).all()
        assert (item['coefficients'] == item['atoms'] + 20).all()
        assert (item['physical_coefficients'] == item['atoms'] + 40).all()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7 + epoch * 10000019 + 1)
            assert item['view'] == int(torch.rand(1) < .5)
        again = VARTokenCache(tmp_path, 'train', seed=7)
        again.epoch = epoch
        assert again[1]['variant'] == item['variant']
    assert len(choices) > 1


def test_cache_rejects_unpaired_fields(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(format=FORMAT)))
    for name in ('atoms', 'coefficients', 'physical_coefficients'):
        shape = (2,2,4 if name == 'atoms' else 3,14,2)
        np.save(tmp_path/f'train-{name}.npy', np.zeros(shape))
    np.save(tmp_path/'train-labels.npy', np.zeros(2))
    with pytest.raises(ValueError, match='paired'):
        VARTokenCache(tmp_path, 'train')
