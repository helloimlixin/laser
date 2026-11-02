import torch
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf

from src.training.laser_ar_factors import coefficient_logits, configure
from src.training import laser_hard_ar
from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients, ordered_support_basis


def test_grouped_logits_are_the_exact_coarse_categorical_distribution():
    torch.manual_seed(412)
    logits = torch.randn(11, 2048, requires_grad=True)
    actual = coefficient_logits(logits, 128).softmax(-1)
    expected = logits.softmax(-1).reshape(11, 128, 16).sum(-1)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-8)
    actual.log().sum().backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad.abs().sum(0) > 0).all()


def test_orthogonal_prefixes_match_the_actual_stage1_least_squares_prefixes():
    torch.manual_seed(402)
    vectors = torch.nn.functional.normalize(torch.randn(17, 4, 16), dim=-1)
    signals = torch.randn(17, 16)
    physical = torch.linalg.lstsq(vectors.transpose(-1, -2), signals[..., None]).solution[..., 0]
    gamma, basis = dictionary_to_orthogonal_coefficients(vectors, physical)
    torch.testing.assert_close((gamma[..., None] * basis).sum(-2),
                               (physical[..., None] * vectors).sum(-2), atol=2e-6, rtol=2e-6)
    for depth in range(1, 5):
        prefix = vectors[:, :depth]
        expected_values = torch.linalg.lstsq(prefix.transpose(-1, -2), signals[..., None]).solution[..., 0]
        expected = (prefix * expected_values[..., None]).sum(-2)
        actual = (basis[:, :depth] * gamma[:, :depth, None]).sum(-2)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def factor_case():
    torch.manual_seed(99)
    aux = SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(16, 31), dim=0),
        num_atoms=31, sparsity_level=4, coeff_bins=torch.linspace(-3, 3, 2048),
        coeff_scales=torch.tensor([3., 2., 1., .5]), encoder=lambda x:x, quant_conv=lambda x:x)
    options = SimpleNamespace(encoder_batch_size=1)
    profile = SimpleNamespace(recipe=SimpleNamespace(targets=options))
    model = SimpleNamespace(_physical_pairs=lambda *a:None)
    return aux, options, profile, model, torch.randn(2, 16, 2, 2)


def test_control_factor_preserves_original_targets_loss_and_gradients_exactly():
    aux, options, profile, model, images = factor_case()
    expected = laser_hard_ar.image_tokens(aux, images, options)
    configure(profile, model, aux, coordinates='physical', bins=2048)
    actual, codes = profile.targets(aux, images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    logits = dict(atom_logits=torch.randn(2,2,2,4,31,requires_grad=True),
                  coeff_logits=torch.randn(2,2,2,4,2048,requires_grad=True))
    original = laser_hard_ar.token_cross_entropy(logits, expected)
    control = profile.loss(model, logits, actual)
    torch.testing.assert_close(control, original, rtol=0, atol=0)
    left = torch.autograd.grad(original, tuple(logits.values()), retain_graph=True)
    right = torch.autograd.grad(control, tuple(logits.values()))
    for a,b in zip(left,right):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_coarse_factor_keeps_support_and_uses_the_validated_exact_128_value_grid(monkeypatch):
    import src.models.laser_rq_adapter as adapter
    monkeypatch.setattr(adapter, 'sample_from_logits', adapter.sample_from_logits)
    aux, options, profile, model, images = factor_case()
    original = laser_hard_ar.image_tokens(aux, images, options)
    configure(profile, model, aux, coordinates='physical', bins=128)
    tokens,_ = profile.targets(aux, images)
    torch.testing.assert_close(tokens[...,0::2],original[...,0::2],rtol=0,atol=0)
    assert ((tokens[...,1::2]-aux.num_atoms)%16==8).all()
    torch.testing.assert_close(aux.coeff_bins.unique(),torch.linspace(-3,3,128),rtol=0,atol=0)


@pytest.mark.parametrize('bins',[128,2048])
@torch.no_grad()
def test_causal_coordinate_cache_matches_full_teacher_forcing_and_valid_sampling(bins,monkeypatch):
    from src.training.laser_rq_minimal import build_model
    import src.models.laser_rq_adapter as adapter
    monkeypatch.setattr(adapter,'sample_from_logits',adapter.sample_from_logits)
    torch.manual_seed(412)
    arch=OmegaConf.create(dict(type='rq-transformer',block_size=[2,2,4],
        embed_dim=12,input_embed_dim=4,shared_tok_emb=True,shared_cls_emb=True,
        input_emb_vqvae=True,head_emb_vqvae=True,cumsum_depth_ctx=True,vocab_size=7,
        vocab_size_cond=1000,block_size_cond=1,
        body=dict(n_layer=1,block=dict(n_head=3,resid_pdrop=0.)),
        head=dict(n_layer=1,block=dict(n_head=3,resid_pdrop=0.))))
    model=build_model(arch,num_atoms=7,coefficient_bins=2048,depth=4).eval()
    aux=SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(4,7),dim=0),
        coeff_bins=torch.linspace(-3,3,2048),coeff_scales=torch.tensor([3.,2.,1.,.5]),
        num_atoms=7,coeff_vocab_size=2048,sparsity_level=4)
    profile=SimpleNamespace(recipe=SimpleNamespace(targets=SimpleNamespace(encoder_batch_size=1)))
    configure(profile,model,aux,coordinates='orthogonal',bins=bins)
    truth=torch.empty(2,2,2,8,dtype=torch.long)
    truth[...,0::2]=torch.tensor([0,1,2,3]);truth[...,1::2]=7+torch.tensor([256,700,1200,1600])
    cond=torch.tensor([0,999]);expected=model(truth,aux,cond)
    generated=torch.zeros_like(truth);generated[...,1::2]=7+1024
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                actual=model.cached_forward(generated[:,:h+1],aux,cond,sample_loc=(h,w,event))
                actual=actual[:,:7] if event%2==0 else actual[:,7:]
                reference=expected['atom_logits' if event%2==0 else 'coeff_logits'][:,h,w,event//2]
                torch.testing.assert_close(actual,reference,rtol=5e-6,atol=1e-6)
                generated[:,h,w,event]=truth[:,h,w,event]
    sampled=model.sample_sparse(2,aux,cond,amp=False,atom_top_k=7)
    assert (sampled[...,0::2].sort(-1).values.diff(dim=-1)>0).all()
    if bins==128:assert ((sampled[...,1::2]-7)%16==8).all()
