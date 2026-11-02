import pytest
import torch
from src.models.geometric_joint_adapter import GeometricJointLinear
from src.training.rq_imagenet_shared import ensure_optimizer_covers_model


def test_late_attached_adapter_receives_an_adam_update_without_resetting_base():
    model=torch.nn.Sequential(torch.nn.Linear(5,7))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    original_ids=[id(p) for p in optimizer.param_groups[0]['params']]
    model[0]=GeometricJointLinear(model[0],torch.randn(3,2),
        torch.tensor([0,0,0,0,1,1,1]),torch.tensor([0.,-1.,1.,1.,-1.,-1.,1.]))
    proof=ensure_optimizer_covers_model(optimizer,model)
    assert proof['added_parameter_names']==['0.geometry_weight']
    assert [id(p) for p in optimizer.param_groups[0]['params'][:2]]==original_ids
    assert not optimizer.state
    model(torch.randn(4,5)).square().mean().backward()
    optimizer.step()
    assert model[0].geometry_weight.abs().sum()>0
    assert {int(v['step']) for v in optimizer.state.values()}=={1}
    again=ensure_optimizer_covers_model(optimizer,model)
    assert again['added_parameter_names']==[]
    assert len(optimizer.state)==3


def test_duplicate_and_stale_optimizer_parameters_are_rejected():
    model=torch.nn.Linear(5,7)
    optimizer=torch.optim.AdamW(model.parameters())
    optimizer.param_groups[0]['params'].append(model.weight)
    with pytest.raises(ValueError,match='duplicate'):
        ensure_optimizer_covers_model(optimizer,model)
    optimizer=torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError,match='absent'):
        ensure_optimizer_covers_model(optimizer,torch.nn.Linear(5,7))


def test_recipe_option_installs_the_projection_before_optimizer_coverage(tmp_path):
    import json
    from types import SimpleNamespace
    from src.training.laser_joint_recipe import JointLaserRecipe
    original=torch.nn.Linear(5,7)
    model=torch.nn.Module();model.classifier=torch.nn.Module();model.classifier.linear=original
    optimizer=torch.optim.AdamW(model.parameters())
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/execution.json').write_text(json.dumps(dict(align_classifier=True,geometry_shared_head=True)))
    atoms=torch.tensor([[0,0,0,0,1,1,1]]*4)
    coefficients=torch.tensor([[0.,-1.,1.,2.,-2.,-1.,1.]]*4)
    q=SimpleNamespace(dictionary=torch.randn(3,2),atom_ids=atoms,coefficients=coefficients)
    profile=JointLaserRecipe.__new__(JointLaserRecipe);profile.base=tmp_path
    profile.adapt(model,SimpleNamespace(quantizer=q))
    assert isinstance(model.classifier.linear,GeometricJointLinear)
    assert model.classifier.linear.weight is original.weight
    result=ensure_optimizer_covers_model(optimizer,model)
    assert result['added_parameter_names']==['classifier.linear.geometry_weight']
