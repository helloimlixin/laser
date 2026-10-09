import copy

import pytest
import torch

from src.models.physical_compound_prior import PhysicalCompoundRQTransformer
from src.training.physical_compound_resume import migrate_checkpoint, validate_optimizer_ages, verify_live_optimizer
from src.training.full_resume_upload import recovery_metadata
from tests.test_physical_pair_scalar_prior import case


def prepared():
    model,aux,tokens,cond=case()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    result=model(tokens,aux,cond)
    (result['coeff_logits'].square().mean()+result['atom_logits'][...,0,:].square().mean()).backward()
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    original=list(dict(model.named_parameters()))
    source=dict(state_dict=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
        global_step=48202,epoch=77,batch_idx=0,scheduler={'last_epoch':48202},
        config=dict(accumulation_steps=3,total_batch_size=2048),rng_state_by_rank=[{'saved':i} for i in range(8)])
    PhysicalCompoundRQTransformer.from_scalar(model)
    return model,aux,tokens,cond,source,migrate_checkpoint(source,model,original)


def test_migration_maps_moments_by_name_when_root_gate_changes_parameter_order():
    model,aux,tokens,cond,source,migrated=prepared()
    new=set(migrated['compound_transfer']['new_parameter_names'])
    original=list(source['state_dict'])
    for index,(name,parameter) in enumerate(model.named_parameters()):
        actual=migrated['optimizer']['state'][index]
        if name in new:
            assert int(actual['step'])==0
            assert actual['exp_avg'].count_nonzero()==0
            assert actual['exp_avg_sq'].count_nonzero()==0
        else:
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(actual[key],source['optimizer']['state'][original.index(name)][key],rtol=0,atol=0)
    assert migrated['scheduler'] is source['scheduler']
    assert migrated['rng_state_by_rank'] is source['rng_state_by_rank']
    assert migrated['config']['batch_size']==64
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    optimizer.load_state_dict(migrated['optimizer'])
    verify_live_optimizer(optimizer,model,migrated['compound_transfer'],48202)
    output=model(tokens,aux,cond)
    atoms=tokens[...,::2]
    loss=(-output['atom_logits'].log_softmax(-1).gather(-1,atoms[...,None]).mean()
          +output['coeff_logits'].square().mean())
    loss.backward();optimizer.step()
    resumed=dict(migrated,global_step=48203,optimizer=optimizer.state_dict(),state_dict=model.state_dict())
    ages=validate_optimizer_ages(resumed)
    assert ages['common_adam_step']==2 and ages['new_adam_step']==1


def test_wrong_new_parameter_age_is_rejected():
    *_,migrated=prepared()
    index=migrated['compound_transfer']['parameter_names'].index('history_gates')
    migrated['optimizer']['state'][index]['step']=torch.tensor(1.)
    with pytest.raises(ValueError,match='Incorrect Adam age'):
        validate_optimizer_ages(migrated)


def test_full_recovery_accepts_only_the_explicit_migrated_parameter_ages():
    *_,migrated=prepared()
    migrated['config'].update(lr_schedule='cosine',seed=261001,training_images=1281167)
    migrated['checkpoint_world_size']=8
    migrated['rng_state_by_rank']=[dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.get_rng_state()) for _ in range(8)]
    meta=recovery_metadata(migrated)
    assert meta['adam_step']==1
    assert meta['compound_optimizer_ages']['new_adam_step']==0
    migrated.pop('compound_transfer')
    with pytest.raises(ValueError,match='Adam counters disagree'):
        recovery_metadata(migrated)
