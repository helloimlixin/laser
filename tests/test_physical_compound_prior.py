import copy

import pytest
import torch

from src.models.physical_compound_prior import PhysicalCompoundRQTransformer
from tests.test_physical_pair_scalar_prior import case


def setup(large=False):
    model,aux,scalar,cond=case()
    original={name:parameter for name,parameter in model.named_parameters()}
    model=PhysicalCompoundRQTransformer.from_scalar(model)
    packed=scalar[...,0::2]*aux.coeff_vocab_size+scalar[...,1::2]-aux.num_atoms
    if large:
        model.block_size=torch.Size([8,8,4]);model.events=256
        model.pos_emb_hw=torch.nn.Parameter(torch.randn(1,64,12)*.02)
        packed=packed.repeat(1,4,4,1)
    return model,aux,packed,cond,original


@torch.no_grad()
def test_full_history_compound_has_no_current_coefficient_or_future_leakage():
    model,aux,packed,cond,_=setup()
    expected=model(packed,aux,cond)
    for event in range(model.events):
        changed=packed.clone().reshape(1,-1)
        changed[:,event]=changed[:,event]//5*5+(changed[:,event]%5+1)%5
        changed[:,event+1:]=torch.randint(0,35,changed[:,event+1:].shape)
        actual=model(changed.reshape_as(packed),aux,cond)
        for key in expected:
            torch.testing.assert_close(actual[key].reshape(1,model.events,-1)[:,:event+1],
                expected[key].reshape(1,model.events,-1)[:,:event+1],rtol=0,atol=0)


@torch.no_grad()
def test_both_decoders_condition_on_every_previous_compound_token():
    model,aux,packed,cond,_=setup()
    expected=model(packed,aux,cond)
    for event in range(model.events-1):
        changed=packed.clone().reshape(1,-1)
        changed[:,event]=changed[:,event]//5*5+(changed[:,event]%5+1)%5
        actual=model(changed.reshape_as(packed),aux,cond)
        for key in expected:
            original=expected[key].reshape(1,model.events,-1)[:,-1]
            updated=actual[key].reshape(1,model.events,-1)[:,-1]
            assert not torch.equal(original,updated),(event,key)


@torch.no_grad()
def test_current_atom_conditions_its_coefficient_and_class_conditions_both():
    model,aux,packed,cond,_=setup()
    expected=model(packed,aux,cond)
    changed=packed.clone();changed[:,0,0,0]=6*5+changed[:,0,0,0]%5
    actual=model(changed,aux,cond)
    torch.testing.assert_close(expected['atom_logits'][:,0,0,0],actual['atom_logits'][:,0,0,0],rtol=0,atol=0)
    assert not torch.equal(expected['coeff_logits'][:,0,0,0],actual['coeff_logits'][:,0,0,0])
    class_changed=model(packed,aux,cond-1)
    for key in expected:assert not torch.equal(expected[key][:,0,0,0],class_changed[key][:,0,0,0])


@pytest.mark.parametrize('large',[False,True])
@torch.no_grad()
def test_full_history_cache_matches_dense_without_site_resets(large):
    model,aux,packed,cond,_=setup(large)
    expected=model(packed,aux,cond)
    atoms,coefficients=model.unpack(packed,aux)
    model.init_cache()
    for event in range(model.events):
        actual_a=model.cached_atom_logits(atoms,coefficients,aux,cond,event,amp=False)
        actual_c=model.cached_coefficient_logits(atoms,coefficients,
            atoms.reshape(1,-1)[:,event],aux,event,amp=False)
        torch.testing.assert_close(actual_a,expected['atom_logits'].reshape(1,model.events,-1)[:,event],rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(actual_c,expected['coeff_logits'].reshape(1,model.events,-1)[:,event],rtol=3e-5,atol=3e-6)
    for block in model.full_history.blocks:
        assert block._cache['past_kv'].shape[-2]==model.events
    for block in model.body_transformer.blocks:
        assert block._cache['past_kv'].shape[-2]==model.events//4
    for block in model.head_transformer.blocks:
        assert block._cache['past_kv'].shape[-2]==8


@torch.no_grad()
def test_dense_and_incremental_pair_embeddings_use_signed_depth_scales_once():
    model,aux,packed,_,_=setup()
    atoms,coefficients=model.unpack(packed,aux)
    expected=model.pair_embedding(atoms,coefficients,aux).reshape(1,model.events,-1)
    for event in range(model.events):
        actual=model.pair_embedding(atoms.reshape(1,-1)[:,event],
            coefficients.reshape(1,-1)[:,event],aux,depth=event%4)
        torch.testing.assert_close(actual,expected[:,event])


def test_migration_keeps_every_old_parameter_and_adam_state_and_resumes_exactly():
    scalar,aux,tokens,cond=case()
    original_parameters=dict(scalar.named_parameters())
    optimizer=torch.optim.AdamW(scalar.parameters(),lr=1e-3)
    before=scalar(tokens,aux,cond)
    loss=before['coeff_logits'].square().mean()+before['atom_logits'][...,0,:].square().mean()
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    old_state=copy.deepcopy(optimizer.state_dict())
    old_weights={k:v.clone() for k,v in scalar.state_dict().items()}
    model=PhysicalCompoundRQTransformer.from_scalar(scalar)
    parameters=dict(model.named_parameters())
    for name,parameter in original_parameters.items():
        assert parameters[name] is parameter
        torch.testing.assert_close(parameter,old_weights[name],rtol=0,atol=0)
    added=[parameter for name,parameter in parameters.items() if name not in original_parameters]
    assert len(added)==19
    optimizer.add_param_group(dict(params=added,lr=1e-4))
    state=optimizer.state_dict()
    for index,values in old_state['state'].items():
        for key,value in values.items():torch.testing.assert_close(state['state'][index][key],value,rtol=0,atol=0)
    assert len(state['state'])==len(old_state['state'])
    packed=tokens[...,0::2]*5+tokens[...,1::2]-7
    def update():
        optimizer.zero_grad(set_to_none=True)
        output=model(packed,aux,cond)
        atoms,coefficients=model.unpack(packed,aux)
        joint=(-output['atom_logits'].log_softmax(-1).gather(-1,atoms[...,None]).mean()
            -output['coeff_logits'].log_softmax(-1).gather(-1,coefficients[...,None]).mean())/2
        joint.backward()
        for name,p in model.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),name
        optimizer.step()
    update()
    saved=copy.deepcopy((model.state_dict(),optimizer.state_dict(),torch.get_rng_state()))
    update();expected=copy.deepcopy(model.state_dict())
    model.load_state_dict(saved[0]);optimizer.load_state_dict(saved[1]);torch.set_rng_state(saved[2])
    update()
    for name,value in expected.items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)


@torch.no_grad()
def test_compound_generation_preserves_unique_atoms_and_resets_both_caches():
    model,aux,_,_,_=setup()
    atoms,coefficients=model.sample_compound(2,aux,cond=torch.tensor([19,999]),amp=False)
    assert atoms.shape==(2,2,2,4)
    assert (atoms.sort(-1).values.diff(dim=-1)>0).all()
    assert ((coefficients>=0)&(coefficients<5)).all()
    assert all(block._cache['past_kv'] is None for block in
        [*model.body_transformer.blocks,*model.head_transformer.blocks,*model.full_history.blocks])


@torch.no_grad()
def test_zero_gate_transfer_exactly_preserves_the_trained_decoder_predictions_and_rng():
    native,aux,tokens,cond=case()
    expected=native(tokens,aux,cond)
    rng=torch.get_rng_state().clone()
    model=PhysicalCompoundRQTransformer.from_scalar(native,initial_gate=0.)
    assert torch.equal(torch.get_rng_state(),rng)
    packed=tokens[...,0::2]*5+tokens[...,1::2]-7
    actual=model(packed,aux,cond)
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
