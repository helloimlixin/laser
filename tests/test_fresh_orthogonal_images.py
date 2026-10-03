import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.training.fresh_images import EpochImageFolder, encode_orthogonal_images, encode_dictionary_images
from src.training.rqtransformer import image_transform, ResumableDistributedSampler
from tests.test_orthogonal_compound_tokens import tiny_aux


def make_dataset(tmp_path):
    target = tmp_path / 'n00000001'
    target.mkdir()
    rng = np.random.default_rng(40)
    for i in range(4):
        Image.fromarray(rng.integers(0, 256, (280, 400, 3), dtype=np.uint8)).save(target/f'{i}.png')
    return EpochImageFolder(tmp_path, transform=image_transform(), augmentation_seed=99)


def test_fresh_views_change_by_epoch_and_resume_independent_of_workers(tmp_path):
    dataset = make_dataset(tmp_path)
    sampler = ResumableDistributedSampler(dataset, num_replicas=1, rank=0, seed=83)
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=2, persistent_workers=True)
    sampler.set_epoch(0)
    first = [x.clone() for x, _ in loader]
    sampler.set_epoch(1)
    second = [x.clone() for x, _ in loader]
    assert any(not torch.equal(a,b) for a,b in zip(first,second))
    sampler.set_epoch(1)
    sampler.set_start_index(2)
    resumed = [x for x, _ in loader]
    for actual, expected in zip(resumed, second[2:]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    state = torch.random.get_rng_state()
    dataset[0]
    assert torch.equal(torch.random.get_rng_state(), state)


def test_online_orthogonal_conversion_preserves_reconstruction_and_chunking():
    aux = tiny_aux()
    aux.encoder = torch.nn.Identity()
    aux.quant_conv = torch.nn.Identity()
    aux.clamp_coeffs = False
    aux.coeff_scales.copy_(torch.tensor([2., 1.3, .7]))
    images = torch.randn(7,4,2,2, generator=torch.Generator().manual_seed(873))
    atoms, dictionary_coeffs = aux.encode_sparse_components(images)
    new_atoms, orthogonal_coeffs = encode_orthogonal_images(aux, images, chunk_size=2)
    assert torch.equal(atoms, new_atoms)
    reference = aux.physical_contributions(atoms, dictionary_coeffs).sum(-2)
    reconstructed = aux.physical_orthogonal_contributions(new_atoms, orthogonal_coeffs).sum(-2)
    torch.testing.assert_close(reconstructed, reference, atol=3e-5, rtol=3e-5)
    whole_atoms, whole_coeffs = encode_orthogonal_images(aux, images, chunk_size=7)
    assert torch.equal(new_atoms, whole_atoms)
    torch.testing.assert_close(orthogonal_coeffs, whole_coeffs, atol=3e-6, rtol=3e-6)
    assert not torch.allclose(orthogonal_coeffs, dictionary_coeffs)


def test_imagenet_batch_budget_for_both_physical_batching_strategies():
    for microbatch, accumulation in [(8,32),(128,2),(256,1)]:
        sampler = ResumableDistributedSampler(range(1281167), num_replicas=8, rank=0)
        loader = DataLoader(sampler.dataset, sampler=sampler, batch_size=microbatch, drop_last=True)
        updates = len(loader)//accumulation
        assert microbatch*8*accumulation == 2048
        assert updates == 625
        assert updates*2048 == 1280000
        assert updates*100 == 62500


def test_online_dictionary_chunks_preserve_cache_coordinates_and_prefixes():
    aux = tiny_aux()
    aux.encoder = torch.nn.Identity()
    aux.quant_conv = torch.nn.Identity()
    aux.clamp_coeffs = False
    aux.coeff_scales.copy_(torch.tensor([2., 1.3, .7]))
    images = torch.randn(7, 4, 2, 2, generator=torch.Generator().manual_seed(873))
    for prefixes in [False, True]:
        expected = aux.encode_sparse_components(images, return_prefix_coeffs=prefixes)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = encode_dictionary_images(aux, images, chunk_size=2, return_prefix_coeffs=prefixes)
        assert torch.backends.cuda.matmul.allow_tf32 == previous_tf32
        assert len(actual) == len(expected) == (3 if prefixes else 2)
        assert torch.equal(actual[0], expected[0])
        for a, b in zip(actual[1:], expected[1:]):
            assert a.dtype == torch.float32
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
