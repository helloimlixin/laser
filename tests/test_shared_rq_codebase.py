import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src import rq_upstream
from src.models.laser_rq_adapter import LaserPairRQTransformer
from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
from src.training.laser_rq_minimal import build_model
from src.training.rq_imagenet_recipes import RQRecipe, LaserRecipe
from src.training.rq_imagenet_shared import training_backward
from rqvae.models.rqtransformer.attentions import AttentionStack, AttentionBlock


def test_fid_workspace_title_accepts_pipe_separated_training_run_names():
    import wandb_workspaces.workspaces as workspaces
    from src.training.rq_imagenet_shared import fid_workspace_title
    title = fid_workspace_title('LASER scalar-soft | restored sampler | upstream RQ | continue20')
    workspace = workspaces.Workspace(entity='test', project='test', name=title)
    assert workspace.name == 'LASER scalar-soft restored sampler upstream RQ continue20 FID'


def arch():
    return OmegaConf.create(dict(type='rq-transformer',block_size=[2,2,4],
        embed_dim=12,input_embed_dim=4,shared_tok_emb=True,shared_cls_emb=True,
        input_emb_vqvae=True,head_emb_vqvae=True,cumsum_depth_ctx=True,vocab_size=7,
        vocab_size_cond=1000,block_size_cond=1,
        body=dict(n_layer=1,block=dict(n_head=3,resid_pdrop=0.)),
        head=dict(n_layer=1,block=dict(n_head=3,resid_pdrop=0.))))


def pair_case():
    torch.manual_seed(127)
    model=build_model(arch(),num_atoms=7,coefficient_bins=5,depth=4).eval()
    aux=SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(4,7),dim=0),
        coeff_bins=torch.linspace(-3,3,5),coeff_scales=torch.tensor([3.,2.,1.,.5]),
        num_atoms=7,coeff_vocab_size=5,sparsity_level=4,vocab_size=12,
        clamp_coeffs=False,soft_target_physical=False,encoder=lambda x:x,quant_conv=lambda x:x)
    aux.sparse_targets=MethodType(FrozenLaserTokenizer.sparse_targets,aux)
    tokens=torch.empty(2,2,2,8,dtype=torch.long)
    tokens[...,0::2]=torch.tensor([0,1,2,3]);tokens[...,1::2]=7+torch.tensor([0,1,3,4])
    return model,aux,tokens,torch.tensor([0,999])


def test_both_models_use_the_same_actual_upstream_attention_classes():
    native=RQRecipe('.').model(rq_upstream.RQTransformerConfig.create(arch()))
    laser,_,_,_=pair_case()
    assert type(native) is rq_upstream.RQTransformer
    assert isinstance(laser,rq_upstream.RQTransformer)
    for model in (native,laser):
        assert type(model.body_transformer) is AttentionStack
        assert type(model.head_transformer) is AttentionStack
        assert all(type(block) is AttentionBlock for block in model.body_transformer.blocks)
    pinned=json.loads((Path(__file__).resolve().parents[1]/'configs/rq-upstream-core.json').read_text())
    assert rq_upstream.provenance()==pinned


@torch.no_grad()
def test_upstream_adapter_cache_matches_teacher_forcing_and_is_causal():
    model,aux,target,cond=pair_case()
    expected=model(target,aux,cond)
    generated=torch.zeros_like(target);generated[...,1::2]=9
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                logits=model.cached_forward(generated[:,:h+1],aux,cond,sample_loc=(h,w,event))
                actual=logits[:,:7] if event%2==0 else logits[:,7:]
                reference=expected['atom_logits' if event%2==0 else 'coeff_logits'][:,h,w,event//2]
                torch.testing.assert_close(actual,reference,rtol=3e-6,atol=8e-7)
                generated[:,h,w,event]=target[:,h,w,event]


def test_existing_laser_weights_preserve_logits_and_gradients_on_upstream_backbone():
    from src.models.physical_pair_scalar_prior import PhysicalPairScalarRQTransformer
    from src.models.rqtransformer.configs import RQTransformerConfig as LegacyConfig
    upstream,aux,tokens,cond=pair_case()
    config=arch();config.block_size[-1]=8;config.vocab_size=12
    legacy=PhysicalPairScalarRQTransformer(LegacyConfig.create(config),7).eval()
    legacy.load_state_dict(upstream.state_dict(),strict=True)
    left=legacy(tokens,aux,cond);right=upstream(tokens,aux,cond)
    for key in left:torch.testing.assert_close(right[key],left[key],rtol=3e-6,atol=8e-7)
    probabilities=torch.randn_like(right['coeff_logits']).softmax(-1)
    for model,output in ((legacy,left),(upstream,right)):
        atom=-output['atom_logits'].log_softmax(-1).gather(-1,tokens[...,0::2,None]).mean()
        coefficient=-(probabilities*output['coeff_logits'].log_softmax(-1)).sum(-1).mean()
        (atom+coefficient).backward()
    for (name,p),(other,q) in zip(legacy.named_parameters(),upstream.named_parameters()):
        assert name==other
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-6)


@pytest.mark.parametrize('kind',['rq','laser'])
def test_shared_backward_matches_direct_native_or_laser_objective(kind,tmp_path):
    torch.manual_seed(918)
    images=torch.randn(2,4,2,2);labels=torch.tensor([0,999])
    if kind=='rq':
        profile=RQRecipe(tmp_path)
        model=profile.model(rq_upstream.RQTransformerConfig.create(arch())).eval()
        codes=torch.randint(7,(2,2,2,4));targets=torch.randn(2,2,2,4,7).softmax(-1)
        embeddings=torch.randn(7,4)
        aux=SimpleNamespace(encode=lambda x:x,
            quantizer=SimpleNamespace(get_soft_codes=lambda z,**kwargs:(targets[:len(z)],codes[:len(z)])),
            get_code_emb_with_depth=lambda x:(embeddings[x],None))
    else:
        (tmp_path/'inputs').mkdir()
        recipe=dict(run_id='toy',auxiliary_loss_terms=0,microbatch=2,epochs=20,lr_schedule_epochs=100,
            evaluation=dict(batch_size=2),tokenizer=dict(num_atoms=7,coefficient_bins=5,depth=4),
            teacher=dict(atom_temperature=.4,coefficient_temperature=.5,variants=4,
                         site_chunk_size=3,encoder_batch_size=1,backend='eager'))
        OmegaConf.save(OmegaConf.create(recipe),tmp_path/'inputs/recipe.yaml')
        profile=LaserRecipe(tmp_path)
        model,aux,_,_=pair_case()
    reference=copy.deepcopy(model)
    state=torch.get_rng_state()
    actual=training_backward(profile,model,aux,images,labels,
        torch.amp.GradScaler('cuda',enabled=False),.37,amp=False)
    torch.set_rng_state(state)
    targets,codes=profile.targets(aux,images)
    expected=profile.loss(reference,reference(codes,model_aux=aux,cond=labels,amp=False),targets)
    (expected*.37).backward()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for p,q in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_shared_entry_imports_no_legacy_trainer_or_transformer():
    root=Path(__file__).resolve().parents[1]
    script="""import sys
import src.training.rq_imagenet_shared
import src.models.laser_rq_adapter
import src.models.laser_tokenizer_adapter
assert 'src.training.rqtransformer' not in sys.modules
assert 'src.models.rqtransformer.attentions' not in sys.modules
assert 'src.models.rqtransformer.transformers' not in sys.modules
"""
    subprocess.run([sys.executable,'-c',script],cwd=root,check=True,
                   env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))


def test_shared_checkpoint_preserves_best_when_last_moves(tmp_path,monkeypatch):
    from src.training.k4_checkpoint_io import atomic_torch_save
    monkeypatch.setenv('LASER_CHECKPOINT_IMMUTABLE_FILES','1')
    monkeypatch.delenv('LASER_CHECKPOINT_STAGING_DIR',raising=False)
    monkeypatch.delenv('LASER_CHECKPOINT_UPLOAD_CACHE_DIR',raising=False)
    target=tmp_path/'last.pt';best=tmp_path/'best.pt'
    atomic_torch_save(dict(step=1),target)
    best.symlink_to(os.readlink(target))
    atomic_torch_save(dict(step=2),target)
    assert torch.load(best,weights_only=True)['step']==1
    assert torch.load(target,weights_only=True)['step']==2
