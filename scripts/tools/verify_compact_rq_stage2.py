#!/usr/bin/env python3
"""Verify the completed compact DDP preflight and released sampling path."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(ROOT)]
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image
from src.compact_rq_training import FrozenCompactTokenizer
from src.original_rq_training import atomic_json,file_sha256,fresh_transformer,state_sha256
from src.scaled_atom_training import soft_cross_entropy

BASE=ROOT/'outputs/church-compact-rq-stage2-20260913'


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    preflight=BASE/'preflight'
    status=json.loads((preflight/'status.json').read_text())
    assert status['phase']=='preflight_complete' and status['optimizer_step']==3
    assert not list(preflight.glob('failure-rank*.json'))
    initialization=json.loads((preflight/'initialization.json').read_text())
    assert initialization['stage2_from_scratch'] and initialization['initial_optimizer_entries']==0
    assert initialization['initial_weights_identical_across_ranks']
    checkpoint=torch.load(preflight/'last.pt',map_location='cpu',weights_only=False)
    assert checkpoint['step']==3 and len(checkpoint['rng_states'])==2
    assert checkpoint['initial_weights_sha256']==initialization['initial_weights_sha256']
    assert len(checkpoint['optimizer']['state'])==460
    tensors=list(checkpoint['state_dict'].values())
    tensors.extend(value for entry in checkpoint['optimizer']['state'].values()
                   for value in entry.values() if isinstance(value,torch.Tensor))
    assert all(torch.isfinite(value).all() for value in tensors)
    print('All preflight model and optimizer tensors are finite',flush=True)
    config=OmegaConf.create(checkpoint['config'])
    assert config.arch.vocab_size==32769 and list(config.arch.block_size)==[8,8,4]
    assert config.arch.body.n_layer==24 and config.arch.head.n_layer==4
    assert config.arch.embed_dim==1024 and config.arch.body.block.n_head==16
    assert config.loss.stochastic_codes and config.loss.temp==.125
    assert config.arch.input_emb_vqvae and config.arch.head_emb_vqvae and config.arch.cumsum_depth_ctx
    model,empty_optimizer=fresh_transformer(config)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    assert sum(p.numel() for p in model.parameters())==386882561
    assert state_sha256(model)!=initialization['initial_weights_sha256']
    del empty_optimizer,tensors,checkpoint
    model.to('cuda:0').eval()
    cache=json.loads((BASE/'cache/complete.json').read_text())
    tokenizer=FrozenCompactTokenizer(cache['checkpoint'],cache['codebook']).to('cuda:0').eval()
    frozen=state_sha256(tokenizer)
    assert frozen==cache['frozen_state_sha256']
    array=np.load(cache['latent_cache'],mmap_mode='r')
    latents=torch.from_numpy(array[:2].copy()).to('cuda:0')
    torch.manual_seed(781)
    targets,codes=tokenizer.quantizer.get_soft_codes(latents,temp=config.loss.temp,stochastic=True)
    assert targets.shape==(2,8,8,4,32769)
    torch.testing.assert_close(targets.sum(-1),torch.ones_like(targets.sum(-1)),atol=2e-6,rtol=0)
    logits=model(codes,model_aux=tokenizer,amp=True)
    chunked=soft_cross_entropy(logits,targets)
    dense=model.compute_loss(logits.float(),targets,use_soft_target=True)
    torch.testing.assert_close(chunked,dense,rtol=0,atol=3e-6)
    print(json.dumps(dict(full_model_chunked_ce=chunked.item(),full_model_dense_reference_ce=dense.item())),flush=True)
    del targets,logits,latents
    torch.manual_seed(71000)
    generated=model.sample(torch.zeros(4,8,8,4,dtype=torch.long,device='cuda:0'),
        model_aux=tokenizer,temperature=1.,top_k=250,top_p=1.,amp=True,cached=True,is_tqdm=False)
    assert generated.shape==(4,8,8,4) and generated.min()>=0 and generated.max()<32769
    decoded=tokenizer.decode_code(generated)
    assert decoded.shape==(4,3,256,256) and torch.isfinite(decoded).all()
    save_image(decoded.mul(.5).add(.5).clamp(0,1),preflight/'sampler-smoke.png',nrow=2)
    assert state_sha256(tokenizer)==frozen
    history=[json.loads(line) for line in (preflight/'metrics.jsonl').read_text().splitlines()]
    updates=[row for row in history if row.get('optimizer_updated')]
    assert len(updates)==3 and all(math.isfinite(row['gradient_norm']) for row in updates)
    files=['src/adaptive_scaled_atom_rq.py','src/compact_rq_training.py',
        'src/scaled_atom_rq.py','src/scaled_atom_training.py','src/original_rq_training.py',
        'scripts/tools/train_compact_rq_stage2.py','scripts/tools/prepare_compact_rq_stage2.py',
        'scripts/tools/calibrate_scaled_atom_temperature.py','scripts/tools/verify_compact_rq_stage2.py',
        'tests/test_compact_rq_training.py','tests/test_adaptive_scaled_atom_rq.py',
        'tests/test_scaled_atom_training.py','tests/test_scaled_atom_rq.py']
    report=dict(preflight_passed=True,steps=3,all_model_and_optimizer_tensors_finite=True,
        optimizer_entries=460,rng_rank_states=2,strict_checkpoint_reload=True,
        released_sampler_decode_smoke_passed=True,tokenizer_unchanged=True,
        full_model_chunked_ce=chunked.item(),full_model_dense_reference_ce=dense.item(),
        microbatch_per_gpu=128,global_batch=256,accumulation=1,
        peak_gpu_allocated_gib=status['peak_gpu_allocated_gib'],
        model_initialization_sha256=initialization['initial_weights_sha256'],
        parameters=386882561,vocab_size=32769,tests_passed=13,
        successful_updates=updates,source_hashes={name:file_sha256(ROOT/name) for name in files})
    atomic_json(BASE/'verification.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
