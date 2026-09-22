from types import SimpleNamespace

import pytest
import torch

from src.models.multiscale_laser_var import MultiScaleLaser
from src.models.compound_var import compound_decompose
from src.models.sparse_token_codec import token_temperatures, validate_tokenized_checkpoint


POLICY = dict(version='compound-hard-token-v1', atom_temperature_ratio=.001,
              coefficient_temperature_ratio=.00025)


def quantizer():
    torch.manual_seed(101)
    q = MultiScaleLaser(channels=8, atoms=16, sparsity=2, patch_nums=(1, 2, 4),
                        coefficient_bins=33, coefficient_max=3.)
    q.tokenized_sparse_policy = dict(POLICY)
    return q


def test_stage1_hard_tokens_equal_stage2_trajectory_and_receive_gradients():
    q = quantizer().train()
    latent = torch.randn(2, 8, 4, 4, requires_grad=True)
    torch.manual_seed(39)
    codes = q.decompose(latent)
    torch.manual_seed(39)
    stage2 = compound_decompose(q, latent, stochastic=True)
    for key in ('atoms', 'coefficients', 'latent', 'inputs'):
        torch.testing.assert_close(codes[key], stage2[key], rtol=0, atol=0)
    decoded, inputs = q.from_codes(codes['atoms'], codes['coefficients'])
    torch.testing.assert_close(decoded, codes['latent'], rtol=0, atol=0)
    torch.testing.assert_close(inputs, codes['inputs'], rtol=0, atol=0)
    assert codes['atoms'].dtype == codes['coefficients'].dtype == torch.int64
    assert not codes['physical_coefficients'].requires_grad
    codes['loss'].backward()
    for gradient in (latent.grad, q.dictionary.dictionary.grad):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_decoder_forward_is_exact_hard_token_latent_with_straight_through_gradient():
    q = quantizer().train()
    q.coefficient_range_decay = .9
    latent = torch.randn(2, 8, 4, 4, requires_grad=True)
    before = q.coefficient_max.clone()
    decoded, _, loss = q(latent)
    restored, _ = q.from_codes(q.last_atom_ids, q.last_coefficient_ids)
    torch.testing.assert_close(decoded, restored, rtol=0, atol=0)
    assert not torch.equal(before, q.coefficient_max)
    weight = torch.randn_like(decoded)
    torch.testing.assert_close(torch.autograd.grad((decoded * weight).sum(), latent, retain_graph=True)[0], weight)
    loss.backward()
    assert q.dictionary.dictionary.grad.abs().sum() > 0


def test_frozen_codec_and_cache_use_the_same_temperatures():
    q = quantizer().eval()
    ranges = q.coefficient_max.clone()
    z = torch.randn(2, 8, 4, 4)
    a, c = token_temperatures(q)
    implicit = compound_decompose(q, z, stochastic=True, generator=torch.Generator().manual_seed(13))
    explicit = compound_decompose(q, z, stochastic=True, atom_temperatures=a,
        coefficient_temperatures=c, generator=torch.Generator().manual_seed(13))
    for key in ('atoms', 'coefficients', 'latent', 'inputs', 'coefficient_probabilities'):
        torch.testing.assert_close(implicit[key], explicit[key], rtol=0, atol=0)
    deterministic = q.decompose(z)
    reference = compound_decompose(q, z)
    torch.testing.assert_close(deterministic['latent'], reference['latent'], rtol=0, atol=0)
    assert torch.equal(q.coefficient_max, ranges)
    assert (implicit['atoms'] != deterministic['atoms']).any()
    from src.data.var_token_cache import restore_cached_codes
    item = {key: implicit[key] for key in ('atoms', 'coefficients', 'physical_coefficients')}
    restored = restore_cached_codes(q, item, c)
    for key in ('latent', 'inputs', 'coefficient_probabilities'):
        torch.testing.assert_close(restored[key], implicit[key])


def test_loading_a_tokenizer_with_a_different_training_policy_is_rejected():
    q = quantizer()
    validate_tokenized_checkpoint(q, dict(initialization=dict(tokenized_sparse_policy=POLICY)))
    with pytest.raises(ValueError, match='different sparse-token training policies'):
        validate_tokenized_checkpoint(q, {})
    q.tokenized_sparse_policy = None
    with pytest.raises(ValueError, match='different sparse-token training policies'):
        validate_tokenized_checkpoint(q, dict(initialization=dict(tokenized_sparse_policy=POLICY)))
    validate_tokenized_checkpoint(q, {})
