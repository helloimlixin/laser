from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.rq_imagenet_shared import schedule_metadata
from rqvae.optimizer.scheduler import create_scheduler


def test_equal_weight_compound_objective_is_half_joint_negative_log_likelihood():
    profile = CompoundReplayRecipe.__new__(CompoundReplayRecipe)
    profile.recipe = OmegaConf.create(dict(atom_ce_weight=1.))
    atoms = torch.randn(2,1,1,4,7,requires_grad=True)
    coefficients = torch.randn(2,1,1,4,5,requires_grad=True)
    targets = dict(atoms=torch.randint(7,(2,1,1,4)),
        coefficient_probabilities=torch.randn_like(coefficients).softmax(-1))
    actual = profile.loss(None,dict(atom_logits=atoms,coeff_logits=coefficients),targets)
    atom_logp = atoms.log_softmax(-1).gather(-1,targets['atoms'][...,None]).squeeze(-1)
    coefficient_logp = (targets['coefficient_probabilities']*coefficients.log_softmax(-1)).sum(-1)
    expected = -(atom_logp+coefficient_logp).mean()/2
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,(atoms,coefficients),retain_graph=True),
        torch.autograd.grad(expected,(atoms,coefficients)))


def test_historical_noise_scale_survives_512_bin_reduction_without_rounding_centers():
    profile = CompoundReplayRecipe.__new__(CompoundReplayRecipe)
    profile.recipe = OmegaConf.create(dict(targets=dict(temperature=.01125,clip_centers=False)))
    aux = SimpleNamespace(num_atoms=7,coeff_vocab_size=512,sparsity_level=4,
        coeff_bins=torch.linspace(-3,3,512))
    support = torch.tensor([[[[0,1,2,3]]]])
    values = torch.tensor([[[[.031,-.047,3.2,-3.2]]]])
    profile.clean_components = lambda *_:(support,values)
    targets,codes = profile.targets(aux,None)
    expected = (-(values[...,None]-aux.coeff_bins).square()/.01125).softmax(-1)
    torch.testing.assert_close(targets['coefficient_probabilities'],expected)
    proof = profile.verify_targets(aux,targets,codes)
    assert proof['normalized_sigma'] == pytest.approx(.075)
    assert 'unclipped' in proof['target_centers']
    assert torch.equal(codes[...,0::2],support)


def test_two_epoch_warmup_cosine_endpoint_and_resumed_clock():
    config = OmegaConf.create(dict(optimizer=dict(init_lr=.0003,warmup=dict(
        epoch=2,multiplier=1,buffer_epoch=0,min_lr=3.e-6,mode='fix',start_from_zero=True))))
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter],lr=.0003)
    scheduler = create_scheduler(optimizer,config.optimizer.warmup,10,100)
    assert schedule_metadata(config,scheduler) == dict(initial_lr=.0003,
        initial_optimizer_lr=0.,minimum_lr=3.e-6,warmup_epochs=2.)
    for step in range(1,1001):
        optimizer.step();scheduler.step()
        if step == 10:assert optimizer.param_groups[0]['lr'] == pytest.approx(.00015)
        if step == 20:assert optimizer.param_groups[0]['lr'] == pytest.approx(.0003)
        if step == 37:
            state, opt_state = scheduler.state_dict(),optimizer.state_dict()
            other = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))],lr=.0003)
            resumed = create_scheduler(other,config.optimizer.warmup,10,100)
            other.load_state_dict(opt_state);resumed.load_state_dict(state)
        elif step > 37:
            other.step();resumed.step()
            assert other.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    assert optimizer.param_groups[0]['lr'] == pytest.approx(3.e-6)


def test_committed_checkpoint_does_not_follow_stale_public_alias(tmp_path,monkeypatch):
    from src.training import k4_checkpoint_io as io
    from runtime.rq_checkpoint_store import install
    base,out = tmp_path/'local',tmp_path/'persistent'
    base.mkdir();out.mkdir()
    monkeypatch.setenv('LASER_CHECKPOINT_STAGING_DIR',str(base/'staging'))
    monkeypatch.setenv('LASER_CHECKPOINT_UPLOAD_CACHE_DIR',str(base/'cache'))
    for name in ('_persist_serialized_checkpoint','_checkpoint_upload_source'):
        monkeypatch.setattr(io,name,getattr(io,name))
    for name in ('link_committed_checkpoint','prune_committed_checkpoint_cache'):
        monkeypatch.setattr(io,name,None,raising=False)
    install(io,base,out)
    folder = out/'persistent-train/checkpoints'
    last,best = folder/'last.pt',folder/'best.pt'
    io.atomic_torch_save(dict(global_step=1,epoch=0,evaluation=dict(fid=10.)),last)
    old = last.resolve()
    io.link_committed_checkpoint(last,best)
    original = Path.resolve
    monkeypatch.setattr(Path,'resolve',lambda path,*a,**kw:
        old if path == last else original(path,*a,**kw))
    io.atomic_torch_save(dict(global_step=2,epoch=0,evaluation=dict(fid=11.)),last)
    assert torch.load(io._checkpoint_upload_source(last),weights_only=True)['global_step']==2
    assert torch.load(io._checkpoint_upload_source(best),weights_only=True)['global_step']==1
    assert old.exists()
    install(io,base,out)
    assert torch.load(io._checkpoint_upload_source(last),weights_only=True)['global_step']==2
    assert torch.load(io._checkpoint_upload_source(best),weights_only=True)['global_step']==1
