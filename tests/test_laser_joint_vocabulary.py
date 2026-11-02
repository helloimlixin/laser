import copy

import pytest
import torch
from omegaconf import OmegaConf

from src import rq_upstream
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_joint_vocabulary import (
    allocate_coefficients, allocate_shared_coefficients, pack_codebook,
)


def case():
    torch.manual_seed(741)
    dictionary = torch.nn.functional.normalize(torch.randn(7, 5), dim=0)
    atoms = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 4]] * 4)
    coefficients = torch.cat((torch.zeros(4, 1), torch.randn(4, 10)), 1)
    return JointAtomCoefficientRQ(dictionary, atoms, coefficients)


def test_joint_quantizer_matches_native_separate_codebooks_and_soft_prefixes():
    from rqvae.models.rqvae.quantizations import RQBottleneck
    q = case()
    native = RQBottleneck([2, 2, 7], [2, 2, 4], q.vocab_size,
                         shared_codebook=False, restart_unused_codes=False).eval()
    with torch.no_grad():
        for d, book in enumerate(native.codebooks):
            book.weight[:-1].copy_(q.expanded_codebook(d))
    x = torch.randn(3, 2, 2, 7)
    before = x.clone()
    expected, _, codes = native(x)
    result = q.quantize(x, chunk_size=5)
    assert torch.equal(result['codes'], codes)
    torch.testing.assert_close(result['quantized'], expected, rtol=1.e-5, atol=1.e-6)
    torch.testing.assert_close(q.embed(codes).sum(-2), expected, rtol=1.e-5, atol=1.e-6)
    for stochastic in [False, True]:
        torch.manual_seed(883)
        p, c = native.get_soft_codes(x, temp=.2, stochastic=stochastic)
        torch.manual_seed(883)
        actual_p, actual_c = q.get_soft_codes(x, temp=.2, stochastic=stochastic)
        torch.testing.assert_close(actual_p, p, rtol=1.e-4, atol=5.e-6)
        assert torch.equal(actual_c, c)
    assert torch.equal(x, before)
    assert (q.embed(torch.zeros(2, 4, dtype=torch.long)) == 0).all()


def test_native_transformer_full_and_cached_logits_match_for_joint_ids():
    q = case()
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=7, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=q.vocab_size,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(arch)).eval()
    codes = torch.randint(q.vocab_size, (2, 2, 2, 4))
    labels = torch.tensor([0, 999])
    with torch.no_grad():
        expected = model(codes, model_aux=q, cond=labels)
        prefix = torch.zeros_like(codes)
        model.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    actual = model.cached_forward(prefix[:, :h + 1], q, labels, sample_loc=(h, w, d))
                    torch.testing.assert_close(actual, expected[:, h, w, d], rtol=1.e-5, atol=1.e-6)
                    prefix[:, h, w, d] = codes[:, h, w, d]
        torch.manual_seed(908)
        cached = model.sample(torch.zeros_like(codes), q, labels, top_k=7, top_p=.95)
        torch.manual_seed(908)
        full = model.sample(torch.zeros_like(codes), q, labels, top_k=7, top_p=.95, cached=False)
        assert torch.equal(cached, full)


def test_per_depth_temperatures_keep_scalar_behavior_and_exact_conditional_targets():
    q = case()
    x = torch.randn(2,2,2,7)
    torch.manual_seed(615)
    scalar_p, scalar_c = q.get_soft_codes(x,temp=.2,chunk_size=3)
    torch.manual_seed(615)
    vector_p, vector_c = q.get_soft_codes(x,temp=[.2]*4,chunk_size=3)
    assert torch.equal(scalar_p,vector_p) and torch.equal(scalar_c,vector_c)
    temperatures = [.1,.2,.3,.4]
    p,c = q.get_soft_codes(x,temp=temperatures,stochastic=False,chunk_size=3)
    residual = x.reshape(-1,7).clone()
    for depth,temperature in enumerate(temperatures):
        expected = (q.scores(residual,depth)/temperature).softmax(-1)
        torch.testing.assert_close(p[...,depth,:].reshape_as(expected),expected)
        assert torch.equal(c[...,depth].flatten(),expected.argmax(-1))
        residual -= q.embed_depth(c[...,depth].flatten(),depth)
    for invalid in [[.1]*3,[.1,.2,.3,0.],[.1,.2,.3,float('nan')]]:
        with pytest.raises(ValueError):q.get_soft_codes(x,temp=invalid)


def test_allocation_keeps_budget_and_spends_extra_entries_on_distortion():
    errors = {2: torch.tensor([100., 10., 1.]), 4: torch.tensor([30., 9., 1.]),
              6: torch.tensor([0., 8., 1.]), 8: torch.tensor([0., 7., 1.])}
    allocation = allocate_coefficients(errors, 12)
    assert allocation == [6, 4, 2] and sum(allocation) == 12
    with pytest.raises(ValueError): allocate_coefficients(errors, 11)
    books = {n: torch.arange(n).float().add(.5).expand(3, -1) for n in errors}
    spec = pack_codebook(torch.eye(3), [books] * 4, [allocation] * 4)
    q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients'])
    assert q.vocab_size == 13 and q.depth == 4
    assert torch.equal(torch.bincount(q.atom_ids[0, 1:]), torch.tensor(allocation))


def test_shared_allocation_uses_all_depths_and_preserves_integer_meaning():
    first = {2:torch.tensor([100., 1.]), 4:torch.tensor([0., 1.])}
    second = {2:torch.tensor([1., 200.]), 4:torch.tensor([1., 0.])}
    allocation = allocate_shared_coefficients([first, second], 6)
    assert allocation == [2, 4]
    assert allocate_shared_coefficients([first, second], 6, weights=[3.,1.]) == [4,2]
    books = [{2:torch.tensor([[-1.,1.]]).expand(2,-1)*scale,
              4:torch.tensor([[-2.,-1.,1.,2.]]).expand(2,-1)*scale}
             for scale in [1.,.5]]
    spec = pack_codebook(torch.eye(2), books, [allocation]*2)
    assert torch.equal(spec['atom_ids'][0],spec['atom_ids'][1])
    assert torch.equal(spec['coefficients'][0].sign(),spec['coefficients'][1].sign())
    assert spec['atom_ids'].shape == (2,7)


@pytest.mark.parametrize('weights', [[0.,0.],[-1.,1.],[float('nan'),1.],[1.]])
def test_shared_allocation_rejects_invalid_weights(weights):
    error = {2:torch.tensor([2.,2.]),4:torch.tensor([1.,1.])}
    with pytest.raises(ValueError):
        allocate_shared_coefficients([error,error],6,weights=weights)


@pytest.mark.parametrize('temperature', [.125,[.1,.125,.15,.175]])
def test_joint_recipe_shared_backward_matches_native_loss_and_parameter_gradients(tmp_path,temperature):
    from rqvae.utils.config import augment_arch_defaults
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_shared import training_backward
    from src.training.rq_imagenet_runtime import sha256
    q = case()
    inputs = tmp_path/'inputs'; inputs.mkdir()
    torch.save(dict(dictionary=q.dictionary, atom_ids=q.atom_ids, coefficients=q.coefficients),
               inputs/'joint-codebook.pt')
    config = dict(decoder='native_joint_integer', auxiliary_loss_terms=0, run_id='toy', name='toy',
        microbatch=2, epochs=6, lr_schedule_epochs=100, evaluation=dict(batch_size=2),
        tokenizer=dict(codebook_sha256=sha256(inputs/'joint-codebook.pt'), vocab_size=q.vocab_size),
        targets=dict(temperature=temperature, chunk_size=3),
        sampling=dict(temperature=1., top_k=7, top_p=.95))
    OmegaConf.save(OmegaConf.create(config), inputs/'recipe.yaml')
    profile = JointLaserRecipe(tmp_path)
    assert profile.target_temperature() == temperature
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=7, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=7,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = profile.model(augment_arch_defaults(arch)).eval()
    assert type(model) is rq_upstream.RQTransformer and model.config.vocab_size == [q.vocab_size]*4
    assert arch.vocab_size == 7
    reference = copy.deepcopy(model)
    images = torch.randn(2, 2, 2, 7); labels = torch.tensor([0, 999])
    targets, codes = q.get_soft_codes(images, temp=temperature)
    profile.targets = lambda aux, images:(targets, codes)
    actual = training_backward(profile, model, q, images, labels,
        torch.amp.GradScaler('cuda', enabled=False), .37, amp=False)
    expected = reference.compute_loss(reference(codes, q, labels), targets, use_soft_target=True)
    (expected*.37).backward()
    torch.testing.assert_close(actual, expected, rtol=1.e-6, atol=1.e-6)
    for p, r in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, r.grad, rtol=2.e-5, atol=1.e-7)
    assert profile.logging_sampling_settings() == dict(sampler_temperature=1., sampler_top_k=7,
        sampler_top_p=.95, sampler_top_k_by_depth=[7]*4)
    with (inputs/'joint-codebook.pt').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(AssertionError): JointLaserRecipe(tmp_path)
