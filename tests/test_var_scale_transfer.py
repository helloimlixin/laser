from types import SimpleNamespace

import pytest
import torch

from src.models.scratch_var import build_scratch_tokenizer, load_finetune_tokenizer


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def config(patches, positions=None):
    return Config(bottleneck='laser', pretrained_vae=None, channels=4, atoms=8,
                  vae_width=32, patch_nums=patches, sparsity=2, coefficient_bins=33,
                  coefficient_range_decay=.9, residual_scale_positions=positions)


def test_scale_transfer_preserves_retained_kernels_and_ranges_and_roundtrips():
    source = build_scratch_tokenizer(config((1, 2, 3, 4, 5)), 1)
    source.quantize.coefficient_max.copy_(torch.arange(1, 6))
    target_cfg = config((1, 2, 5), (0., .25, 1.))
    target = build_scratch_tokenizer(target_cfg, 9)
    load_finetune_tokenizer(target, {'model': source.state_dict()}, (1, 2, 3, 4, 5))
    torch.testing.assert_close(target.quantize.coefficient_max, torch.tensor([1., 2., 5.]))
    for name, tensor in target.state_dict().items():
        if name != 'quantize.coefficient_max':
            torch.testing.assert_close(tensor, source.state_dict()[name])
    for new_index, old_index in enumerate((0, 1, 4)):
        pn = target.quantize.v_patch_nums[new_index]
        vectors = torch.randn(2, pn*pn, 4)
        torch.testing.assert_close(target.quantize.contribution(vectors, new_index),
                                   source.quantize.contribution(vectors, old_index))
    target.eval()
    latent = torch.randn(2, 4, 5, 5, requires_grad=True)
    codes = target.quantize.decompose(latent)
    decoded, inputs = target.quantize.from_codes(codes['atoms'], codes['coefficients'])
    torch.testing.assert_close(decoded, codes['latent'])
    torch.testing.assert_close(inputs, codes['inputs'])
    codes['loss'].backward()
    assert latent.grad.isfinite().all()
    assert target.quantize.dictionary.dictionary.grad.abs().sum() > 0
    restored = build_scratch_tokenizer(target_cfg, 2)
    restored.load_state_dict(target.state_dict(), strict=True)
    restored.eval()
    torch.testing.assert_close(restored.quantize.decompose(latent.detach())['latent'], codes['latent'])


def test_scale_transfer_rejects_wrong_kernel_mapping_and_resolution():
    source = build_scratch_tokenizer(config((1, 2, 3, 4, 5)), 1)
    target = build_scratch_tokenizer(config((1, 2, 5)), 1)
    with pytest.raises(ValueError, match='residual convolution'):
        load_finetune_tokenizer(target, {'model': source.state_dict()}, (1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match='same final resolution'):
        load_finetune_tokenizer(target, {'model': source.state_dict()}, (1, 2, 3, 4))


def test_reduced_recipes_agree_on_codec():
    from src.training.cli import load_config
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'configs/experiments'
    tokenizer = load_config(root / 'celebahq256-var341-tokenizer.yaml')
    prior = load_config(root / 'celebahq256-var341-compound.yaml')
    assert tokenizer.model.patch_nums == prior.model.patch_nums
    assert sum(x*x for x in prior.model.patch_nums) == 341
    assert tokenizer.model.residual_scale_positions == prior.model.residual_scale_positions
    assert tokenizer.model.initialization == 'finetune'
    assert prior.execution.token_cache_dir
