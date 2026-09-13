"""Check exact RQ semantics, rather than only round-tripping the token packing."""
import torch
from torch.nn import functional as F

from src.scaled_atom_rq import ScaledAtomRQ, continuous_matching_pursuit, orthogonal_matching_pursuit


def fixture():
    torch.manual_seed(812)
    # Nonunit norms ensure the search isn't accidentally assuming normalization.
    return torch.randn(7, 11), torch.tensor([-2., -.6, .6, 2.]), torch.randn(3, 2, 2, 7)


def test_fast_search_matches_explicit_expanded_codebook_at_every_depth():
    dictionary, levels, x = fixture()
    model = ScaledAtomRQ(dictionary, levels, depth=4)
    before = x.clone()
    result = model.quantize(x, return_trajectory=True)
    residual = x.clone()
    book = model.expanded_codebook()
    for d in range(4):
        expected = (residual[..., None, :] - book).square().sum(-1).argmin(-1)
        assert torch.equal(result['codes'][..., d], expected)
        residual -= book[expected]
        torch.testing.assert_close(result['prefixes'][..., d, :], x-residual)
    torch.testing.assert_close(model.embed(result['codes']).sum(-2), result['quantized'])
    assert torch.equal(x, before)


def test_matches_released_rq_bottleneck_and_its_stochastic_soft_targets():
    from rqvae.models.rqvae.quantizations import RQBottleneck
    dictionary, levels, x = fixture()
    model = ScaledAtomRQ(dictionary, levels, depth=4)
    reference = RQBottleneck([2,2,7], [2,2,4], model.vocab_size,
                            shared_codebook=True, restart_unused_codes=False).eval()
    with torch.no_grad():
        reference.codebooks[0].weight[:-1].copy_(model.expanded_codebook())
    quantized, _, codes = reference(x)
    actual = model.quantize(x)
    assert torch.equal(actual['codes'], codes)
    torch.testing.assert_close(actual['quantized'], quantized)
    torch.manual_seed(65)
    probabilities, soft_codes = reference.get_soft_codes(x, temp=.5, stochastic=True)
    torch.manual_seed(65)
    actual_probabilities, actual_codes = model.soft_codes(x, .5, stochastic=True)
    torch.testing.assert_close(actual_probabilities, probabilities, atol=2e-6, rtol=2e-5)
    assert torch.equal(actual_codes, soft_codes)


def test_zero_and_repeated_atoms_are_valid_without_refitting():
    model = ScaledAtomRQ(torch.tensor([[1.], [0.]]), torch.tensor([-1., 1.]), depth=4)
    x = torch.tensor([[2.2, .2], [0., 0.]])
    result = model.quantize(x, return_trajectory=True)
    assert result['codes'].tolist() == [[2,2,0,0], [0,0,0,0]]
    errors = (x[..., None, :] - result['prefixes']).square().sum(-1)
    assert (errors[..., 1:] <= errors[..., :-1] + 1e-6).all()


def test_omp_control_refits_support_using_least_squares():
    torch.manual_seed(999)
    dictionary = F.normalize(torch.randn(8, 12, dtype=torch.float64), dim=0)
    x = torch.randn(9, 8, dtype=torch.float64)
    result = orthogonal_matching_pursuit(x, dictionary, dictionary.T@dictionary)
    for i in range(len(x)):
        selected = dictionary[:, result['atoms'][i]]
        expected = torch.linalg.lstsq(selected, x[i]).solution
        torch.testing.assert_close(result['coefficients'][i], expected)
    mp = continuous_matching_pursuit(x, dictionary, return_prefixes=True)
    torch.testing.assert_close(mp['quantized'], (dictionary.T[mp['atoms']] * mp['coefficients'][..., None]).sum(-2))


def test_original_transformer_uses_joint_contributions_without_future_leakage():
    from omegaconf import OmegaConf
    from rqvae.models.rqtransformer.configs import RQTransformerConfig
    from rqvae.models.rqtransformer.transformers import RQTransformer
    torch.manual_seed(32)
    quantizer = ScaledAtomRQ(torch.randn(7, 11), torch.tensor([-2., -.6, .6, 2.]))
    config = RQTransformerConfig.create(OmegaConf.create(dict(
        vocab_size=quantizer.vocab_size, block_size=[2,2,4], embed_dim=32,
        input_embed_dim=7, input_emb_vqvae=True, head_emb_vqvae=True,
        cumsum_depth_ctx=True, shared_tok_emb=True, shared_cls_emb=True,
        body=dict(n_layer=2, block=dict(n_head=4, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=4, resid_pdrop=0.)))))
    model = RQTransformer(config).eval()
    codes = torch.ones(1,2,2,4, dtype=torch.long)
    with torch.no_grad():
        logits = model(codes, model_aux=quantizer).reshape(16,-1)
        # Change only the coefficient of atom zero at depth one of pixel zero.
        changed = codes.clone()
        changed[0,0,0,1] = 2
        after_coefficient = model(changed, model_aux=quantizer).reshape(16,-1)
        torch.testing.assert_close(logits[:2], after_coefficient[:2], rtol=0, atol=0)
        assert (logits[2:]-after_coefficient[2:]).abs().max() > 1e-5
        assert (logits[4:]-after_coefficient[4:]).abs().max() > 1e-5
        # Changing just the support also reaches later depth and spatial tokens.
        changed[0,0,0,1] = 5
        after_support = model(changed, model_aux=quantizer).reshape(16,-1)
        torch.testing.assert_close(logits[:2], after_support[:2], rtol=0, atol=0)
        assert (logits[2:]-after_support[2:]).abs().max() > 1e-5
        assert (logits[4:]-after_support[4:]).abs().max() > 1e-5
