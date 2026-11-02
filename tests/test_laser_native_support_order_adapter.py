from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models.laser_native_support_order_adapter import build_native_support_order_model
from src.rq_upstream import RQTransformer
from src.training.laser_hard_ar import verify_tokens


def case(order):
    torch.manual_seed(26101693)
    arch=OmegaConf.create(dict(type='rq-transformer',block_size=[2,2,4],embed_dim=12,
        input_embed_dim=8,shared_tok_emb=True,shared_cls_emb=True,input_emb_vqvae=True,
        head_emb_vqvae=True,cumsum_depth_ctx=True,vocab_size=11,vocab_size_cond=1000,
        block_size_cond=1,body=dict(n_layer=2,block=dict(n_head=3,resid_pdrop=0.)),
        head=dict(n_layer=2,block=dict(n_head=3,resid_pdrop=0.))))
    model=build_native_support_order_model(arch,num_atoms=11,coefficient_bins=33,order=order).eval()
    aux=SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(8,11),dim=0),
        num_atoms=11,coeff_vocab_size=33,sparsity_level=4,coeff_bins=torch.linspace(-3,3,33),
        coeff_scales=torch.tensor([3.,2.,1.,.5]))
    tokens=torch.empty(2,2,2,8,dtype=torch.long)
    tokens[...,0::2]=torch.tensor([0,1,2,3]);tokens[...,1::2]=11+torch.tensor([10,15,20,23])
    return arch,model,aux,tokens,torch.tensor([0,999])


@pytest.mark.parametrize('order',['interleaved','support_first'])
@torch.no_grad()
def test_full_and_cached_native_predictions_match_all_events(order):
    _,model,aux,truth,labels=case(order)
    expected=model(truth,aux,labels)
    prefix=torch.zeros_like(truth);prefix[...,1::2]=11+16
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event,external in enumerate(model.event_positions):
                output=model.cached_forward(prefix[:,:h+1],aux,labels,sample_loc=(h,w,event))
                atom=external%2==0
                actual=output[:,:11] if atom else output[:,11:]
                target=expected['atom_logits' if atom else 'coeff_logits'][:,h,w,external//2]
                torch.testing.assert_close(actual,target,rtol=1.e-5,atol=3.e-6)
                prefix[:,h,w,external]=truth[:,h,w,external]
    assert model.classifier.cached_atom is None


@pytest.mark.parametrize('order',['interleaved','support_first'])
@torch.no_grad()
def test_partitioned_classifier_is_exact_native_rq_without_dense_padding(order):
    _,model,aux,tokens,labels=case(order)
    reference=RQTransformer(model.config).eval()
    reference.load_state_dict(model.state_dict(),strict=True)
    reference.embed_with_model_aux=model.embed_with_model_aux
    dense=reference(model.event_tokens(tokens),aux,labels)
    atom_events=[e for e,x in enumerate(model.event_positions) if x%2==0]
    coeff_events=[e for e,x in enumerate(model.event_positions) if x%2]
    atoms=dense[...,atom_events,:11]
    for d in range(1,4):atoms[...,d,:].scatter_(-1,tokens[...,:2*d:2],-torch.inf)
    actual=model(tokens,aux,labels)
    torch.testing.assert_close(actual['atom_logits'],atoms,rtol=0,atol=0)
    torch.testing.assert_close(actual['coeff_logits'],dense[...,coeff_events,11:],rtol=0,atol=0)


@torch.no_grad()
def test_support_first_uses_entire_support_without_leaking_current_or_future_coefficients():
    _,model,aux,tokens,labels=case('support_first')
    expected=model(tokens,aux,labels)
    altered=tokens.clone();altered[:,0,0,1]=11+2
    output=model(altered,aux,labels)
    torch.testing.assert_close(output['atom_logits'][:,0,0],expected['atom_logits'][:,0,0],rtol=0,atol=0)
    torch.testing.assert_close(output['coeff_logits'][:,0,0,0],expected['coeff_logits'][:,0,0,0],rtol=0,atol=0)
    assert not torch.allclose(output['coeff_logits'][:,0,0,1],expected['coeff_logits'][:,0,0,1])
    altered=tokens.clone();altered[:,0,0,6]=10
    output=model(altered,aux,labels)
    torch.testing.assert_close(output['atom_logits'][:,0,0],expected['atom_logits'][:,0,0],rtol=0,atol=0)
    assert not torch.allclose(output['coeff_logits'][:,0,0,0],expected['coeff_logits'][:,0,0,0])
    altered=tokens.clone();altered[:,1,1,6]=10
    output=model(altered,aux,labels)
    for key in output:
        torch.testing.assert_close(output[key][:,0],expected[key][:,0],rtol=0,atol=0)


@torch.no_grad()
def test_orders_have_identical_parameters_and_preserve_exact_codec_and_physical_latent():
    _,interleaved,aux,tokens,_=case('interleaved')
    _,support_first,_,_,_=case('support_first')
    for name,value in interleaved.state_dict().items():
        torch.testing.assert_close(value,support_first.state_dict()[name],rtol=0,atol=0)
    vectors=aux.dictionary.T[tokens[...,0::2]]
    coefficients=aux.coeff_bins[tokens[...,1::2]-11]*aux.coeff_scales
    expected=(vectors*coefficients[...,None]).sum(-2)
    for model in [interleaved,support_first]:
        torch.testing.assert_close(model.codec_tokens(model.event_tokens(tokens)),tokens,rtol=0,atol=0)
        actual=model.embed_with_model_aux(model.event_tokens(tokens),aux).sum(-2)
        torch.testing.assert_close(actual,expected,rtol=1.e-6,atol=1.e-6)


@pytest.mark.parametrize('order',['interleaved','support_first'])
def test_finite_gradients_strict_reload_and_valid_external_samples(order):
    arch,model,aux,tokens,labels=case(order)
    original=aux.dictionary.clone()
    output=model(tokens,aux,labels)
    atom=torch.nn.functional.cross_entropy(output['atom_logits'].reshape(-1,11),tokens[...,0::2].reshape(-1))
    probabilities=torch.softmax(torch.randn(2,2,2,4,33),-1)
    coeff=-(probabilities*output['coeff_logits'].log_softmax(-1)).sum(-1).mean()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
    ((atom+coeff)/2).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    restored=build_native_support_order_model(arch,num_atoms=11,coefficient_bins=33,order=order).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    for key,value in model(tokens,aux,labels).items():
        torch.testing.assert_close(restored(tokens,aux,labels)[key],value,rtol=0,atol=0)
    sampled=restored.sample_sparse(2,aux,labels,amp=False)
    verify_tokens(aux,sampled)
    torch.testing.assert_close(aux.dictionary,original,rtol=0,atol=0)


def test_recipe_uses_true_joint_likelihood_without_atom_upweighting():
    from src.training.laser_native_support_order_recipe import NativeSupportOrderRecipe
    _,model,aux,tokens,labels=case('support_first')
    profile=NativeSupportOrderRecipe.__new__(NativeSupportOrderRecipe)
    profile.recipe=OmegaConf.create(dict(atom_ce_weight=1.))
    output=model(tokens,aux,labels)
    probabilities=torch.softmax(torch.randn(2,2,2,4,33),-1)
    actual=profile.loss(model,output,dict(atoms=tokens[...,0::2],coefficient_probabilities=probabilities))
    atom_logp=output['atom_logits'].log_softmax(-1).gather(-1,tokens[...,0::2,None]).squeeze(-1)
    coefficient_logp=(probabilities*output['coeff_logits'].log_softmax(-1)).sum(-1)
    expected=-(atom_logp.sum()+coefficient_logp.sum())/(tokens.numel())
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('order',['interleaved','support_first'])
def test_recipe_rejects_resuming_with_the_other_event_order(tmp_path,order):
    from src.training.laser_native_support_order_recipe import NativeSupportOrderRecipe
    recipe=OmegaConf.create(dict(run_id='order-test',name='Order test',epochs=2,microbatch=64,
        lr_schedule_epochs=100,auxiliary_loss_terms=0,decoder='native_support_order',event_order=order,
        atom_ce_weight=1.,tokenizer=dict(num_atoms=16384,coefficient_bins=512,source_rfid=4.21),
        targets=dict(temperature=.03125),evaluation=dict(batch_size=128),
        sampling=dict(atom_temperature=.9,atom_top_k=0,atom_top_p=.9,
            coefficient_temperature=1.,coefficient_top_k=0,coefficient_top_p=.85)))
    (tmp_path/'inputs').mkdir()
    OmegaConf.save(recipe,tmp_path/'inputs/recipe.yaml')
    profile=NativeSupportOrderRecipe(tmp_path)
    payload=profile.checkpoint_extra();profile.verify_resume(payload)
    assert profile.metadata()['atom_ce_weight']==profile.metadata()['coefficient_ce_weight']==1.
    payload['recipe']['event_order']='interleaved' if order=='support_first' else 'support_first'
    with pytest.raises(AssertionError):profile.verify_resume(payload)
