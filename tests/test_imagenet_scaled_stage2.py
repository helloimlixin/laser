import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

UPSTREAM = Path(__file__).resolve().parents[1]/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path.insert(0,str(UPSTREAM))
from src.imagenet_scaled_stage2 import CachedClassLatents, enable_sdpa, load_imagenet_config
from src.scaled_atom_training import TrainingScaledAtomRQ


def test_smallest_imagenet_recipe_preserves_class_conditioning_and_schedule():
    config = load_imagenet_config(UPSTREAM)
    assert (config.arch.embed_dim,config.arch.body.n_layer,config.arch.head.n_layer)==(1536,12,4)
    assert config.arch.body.block.n_head==config.arch.head.block.n_head==24
    assert config.arch.vocab_size_cond==1000
    assert config.experiment.total_batch_size==2048 and config.experiment.epochs==100
    assert config.optimizer.init_lr==.0005 and config.optimizer.warmup.epoch==0
    assert config.arch.vocab_size==131073


def test_fused_attention_preserves_logits_gradients_and_class_causality():
    from rqvae.models import create_model
    config = load_imagenet_config(UPSTREAM,vocab_size=13)
    config.arch.embed_dim=16
    config.arch.input_embed_dim=7
    config.arch.block_size=[2,2,2]
    for stack in [config.arch.body,config.arch.head]:
        stack.n_layer=1
        stack.block.embed_dim=16
        stack.block.n_head=2
        stack.block.resid_pdrop=0.
        stack.block.attn_pdrop=0.
    torch.manual_seed(321)
    original,_=create_model(config.arch,ema=False)
    optimized=copy.deepcopy(original)
    assert enable_sdpa(optimized)==2
    aux=TrainingScaledAtomRQ(torch.randn(7,3),torch.tensor([-2.,-.5,.5,2.]),depth=2)
    codes=torch.randint(0,13,(2,2,2,2))
    labels=torch.tensor([1,999])
    a=original(codes,model_aux=aux,cond=labels)
    b=optimized(codes,model_aux=aux,cond=labels)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    for p,q in zip(original.parameters(),optimized.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
    assert optimized.cond_emb.weight.grad[1].abs().sum()>0
    assert optimized.cond_emb.weight.grad[999].abs().sum()>0
    changed=codes.clone(); changed[:,-1,-1,-1]=(changed[:,-1,-1,-1]+1)%13
    torch.testing.assert_close(b,optimized(changed,model_aux=aux,cond=labels))
    other=optimized(codes,model_aux=aux,cond=(labels+1)%1000)
    assert not torch.allclose(other[:,0,0,0],b[:,0,0,0])
    original.eval(); optimized.eval()
    original.init_cache(); optimized.init_cache()
    for position in [(0,0,0),(0,0,1),(0,1,0)]:
        a=original.cached_forward(codes,model_aux=aux,cond=labels,sample_loc=position)
        b=optimized.cached_forward(codes,model_aux=aux,cond=labels,sample_loc=position)
        torch.testing.assert_close(a,b)


def test_cache_keeps_large_integer_ids_labels_and_epoch_views(tmp_path):
    (tmp_path/'complete.json').write_text(json.dumps({'train_views':2}))
    np.save(tmp_path/'train-labels.npy',np.array([0,999],dtype=np.int16))
    for view in range(2):
        np.save(tmp_path/f'train-view{view}-latents.npy',np.full((2,8,8,256),view,dtype=np.float32))
        np.save(tmp_path/f'train-view{view}-codes.npy',np.full((2,8,8,4),131072-view,dtype=np.uint32))
    first=CachedClassLatents(tmp_path,epoch=0)
    second=CachedClassLatents(tmp_path,epoch=1)
    assert first[0][0].mean()==0 and second[0][0].mean()==1
    assert first[1][1]==999 and first[0][2].max()==131072
    assert first[0][2].dtype==torch.int64
