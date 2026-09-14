#!/usr/bin/env python3
"""Evaluate an immutable Church RQ checkpoint pair with auditable FID statistics."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(UPSTREAM),str(ROOT)]
os.environ.setdefault('TORCH_HOME','/workspace/tmp/official-rqvae-eval-cache')
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.utils import save_image
from rqvae.models import create_model
from rqvae.utils.config import load_config,augment_arch_defaults
from rqvae.metrics.fid import get_inception_model,mean_covar_numpy,frechet_distance
from src.original_rq_training import atomic_json,file_sha256,state_sha256,FeatureMoments,fid_from_moments


def load_frozen(checkpoint,config_path,device):
    config=load_config(config_path)
    model,_=create_model(augment_arch_defaults(config.arch),ema=False)
    # Our own stage-1 checkpoint also contains OmegaConf/optimizer metadata.
    # Permit that format only for the exact locally produced, hash-verified file.
    completed=json.loads((ROOT/'outputs/church-rq-baseline-scratch-20260912/stage1/complete.json').read_text())
    local_stage1=Path(checkpoint).resolve()==Path(completed['checkpoint']).resolve()
    if local_stage1:
        assert file_sha256(checkpoint)==completed['checkpoint_sha256']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=not local_stage1,mmap=True)
    model.load_state_dict(payload['state_dict'],strict=True)
    del payload
    model.requires_grad_(False).to(device).eval()
    return model,config


@torch.inference_mode()
def verify_paths(model,tokenizer,codes,decoded):
    # Check batched decoding against the released CLI's single-image decoding.
    sample=codes[:2]
    serial=torch.cat([tokenizer.decode_code(x[None]) for x in sample]).mul(.5).add(.5).clamp(0,1)
    decoder_error=float((serial-decoded[:2]).abs().max())
    decoder_mse=float((serial-decoded[:2]).square().mean())
    assert decoder_error<.003 and decoder_mse<1e-8,(decoder_error,decoder_mse)
    # Compare cached next-token logits against the full causal model in FP32.
    full=model(sample,model_aux=tokenizer,amp=False)
    model.init_cache()
    max_error=0.
    for index in range(256):
        h,w,d=index//32,(index//4)%8,index%4
        cached=model.cached_forward(sample,model_aux=tokenizer,amp=False,sample_loc=(h,w,d))
        error=float((cached-full[:,h,w,d]).abs().max())
        max_error=max(max_error,error)
        torch.testing.assert_close(cached,full[:,h,w,d],atol=.003,rtol=3e-4)
    model.init_cache()
    return dict(all_256_cached_positions_verified=True,cached_logit_max_abs_error=max_error,
        serial_vs_batched_decoder_max_abs_error=decoder_error,serial_vs_batched_decoder_mse=decoder_mse)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ar',type=Path,required=True)
    p.add_argument('--ar-config',type=Path,required=True)
    p.add_argument('--vqvae',type=Path,required=True)
    p.add_argument('--vqvae-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=50000)
    p.add_argument('--batch-size',type=int,default=100)
    p.add_argument('--top-k',type=int)
    p.add_argument('--temperature',type=float)
    p.add_argument('--top-p',type=float)
    p.add_argument('--seed',type=int,default=71000)
    p.add_argument('--skip-path-checks',action='store_true')
    args=p.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    assert world==2 and args.samples>world
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    out=args.output.resolve()
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    started=time.time()
    def status(phase,**kwargs):
        if rank==0:
            record=dict(phase=phase,elapsed_seconds=time.time()-started,updated_unix=time.time(),**kwargs)
            atomic_json(out/'status.json',record)
            print(json.dumps(record),flush=True)
    status('loading_verified_checkpoint_pair')
    model,config=load_frozen(args.ar,args.ar_config,device)
    tokenizer,vqconfig=load_frozen(args.vqvae,args.vqvae_config,device)
    assert list(model.get_block_size())==list(tokenizer.code_shape)==[8,8,4]
    assert config.arch.vocab_size_cond==1
    settings=OmegaConf.to_container(config.sampling,resolve=True)
    for key,value in [('top_k',args.top_k),('top_p',args.top_p),('temp',args.temperature)]:
        if value is not None:settings[key]=value
    inception=get_inception_model().eval().requires_grad_(False).to(device)
    initial_model_hash=state_sha256(model)
    initial_tokenizer_hash=state_sha256(tokenizer)
    reference=ROOT/'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz'
    if rank==0:
        active_manifest=ROOT/'outputs/church-compact-rq-stage2-20260913/source-manifest.json'
        recorded=json.loads(active_manifest.read_text())
        # Verify this evaluator's dependencies. Other experiments may extend
        # unrelated sparse-tokenizer adapters in the shared workspace.
        prefix=str(UPSTREAM.relative_to(ROOT))+'/'
        hashes={name:digest for name,digest in recorded.items()
                if name.startswith(prefix) or name=='src/original_rq_training.py'}
        for name,digest in hashes.items():
            assert file_sha256(ROOT/name)==digest,name
        spec=dict(ar=str(args.ar.resolve()),ar_sha256=file_sha256(args.ar),
            ar_config=str(args.ar_config.resolve()),ar_config_sha256=file_sha256(args.ar_config),
            vqvae=str(args.vqvae.resolve()),vqvae_sha256=file_sha256(args.vqvae),
            vqvae_config=str(args.vqvae_config.resolve()),vqvae_config_sha256=file_sha256(args.vqvae_config),
            parameters=sum(x.numel() for x in model.parameters()),settings=settings,
            samples=args.samples,world_size=world,batch_size_per_gpu=args.batch_size,
            seed=args.seed,rank_seed='seed + rank',reference=str(reference),reference_sha256=file_sha256(reference),
            precision='Released FP16 cached sampler; FP32 decoder/Inception; TF32 disabled',
            pixels='Continuous float32 RGB clamped to [0,1]; no uint8 conversion or extra resize',
            initial_model_state_sha256=initial_model_hash,initial_tokenizer_state_sha256=initial_tokenizer_hash,
            runtime_source_files_verified=len(hashes),runtime_source_hashes=hashes,
            script_sha256=file_sha256(__file__))
        atomic_json(out/'specification.json',spec)
        OmegaConf.save(config,out/'ar-config.yaml')
        OmegaConf.save(vqconfig,out/'vqvae-config.yaml')
        (out/'audit-script.py').write_text(Path(__file__).read_text())
        print(json.dumps(spec),flush=True)
    dist.barrier()
    torch.manual_seed(args.seed+rank)
    local_total=len(range(rank,args.samples,world))
    features=np.lib.format.open_memmap(out/f'features-rank{rank}.npy',mode='w+',dtype=np.float32,shape=(local_total,2048))
    token_codes=np.lib.format.open_memmap(out/f'codes-rank{rank}.npy',mode='w+',dtype=np.int32,shape=(local_total,8,8,4))
    moments=FeatureMoments(device)
    for offset in range(0,local_total,args.batch_size):
        size=min(args.batch_size,local_total-offset)
        codes=model.sample(torch.zeros(size,8,8,4,dtype=torch.long,device=device),
            model_aux=tokenizer,temperature=settings['temp'],top_k=settings['top_k'],
            top_p=settings['top_p'],amp=True,cached=True,is_tqdm=False)
        assert codes.min()>=0 and codes.max()<config.arch.vocab_size
        images=tokenizer.decode_code(codes).mul(.5).add(.5).clamp(0,1)
        assert torch.isfinite(images).all()
        if offset==0:
            if not args.skip_path_checks:
                checks=verify_paths(model,tokenizer,codes,images)
                atomic_json(out/f'path-checks-rank{rank}.json',checks)
            if rank==0:save_image(images[:64],out/'samples.png',nrow=8)
        acts=inception(images)
        assert acts.shape==(size,2048) and torch.isfinite(acts).all()
        moments.update(acts)
        features[offset:offset+size]=acts.cpu().numpy()
        token_codes[offset:offset+size]=codes.cpu().numpy()
        if offset%(10*args.batch_size)==0 or offset+size==local_total:
            status('generating',completed=min(args.samples,(offset+size)*world),samples=args.samples)
    features.flush();token_codes.flush()
    del features,token_codes
    statistics=moments.finish()
    assert statistics[0]==args.samples
    assert state_sha256(model)==initial_model_hash and state_sha256(tokenizer)==initial_tokenizer_hash
    if rank==0:
        status('computing_fid',samples=args.samples)
        score=fid_from_moments(statistics,reference)
        np.savez(out/'statistics.npz',mu=statistics[1],sigma=statistics[2],samples=args.samples)
        # Independent aggregation follows upstream's numpy mean/covariance path.
        all_features=np.concatenate([np.load(out/f'features-rank{r}.npy') for r in range(world)])
        mu,cov=mean_covar_numpy(all_features)
        real=np.load(reference)
        dense_score=float(frechet_distance(mu,cov,real['mu'],real['sigma']))
        assert abs(score-dense_score)<.001,(score,dense_score)
        result=dict(fid=score,upstream_numpy_fid=dense_score,aggregation_fid_abs_error=abs(score-dense_score),
            samples=args.samples,settings=settings,model_and_tokenizer_unchanged=True,
            elapsed_seconds=time.time()-started,specification=spec)
        atomic_json(out/'result.json',result)
        status('complete',fid=score,samples=args.samples)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1])
            if out.exists():atomic_json(out/f'failure-rank{os.environ.get("RANK","unknown")}.json',
                dict(error_type=type(error).__name__,error=str(error),updated_unix=time.time()))
        raise
