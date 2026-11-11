"""Fixed-weight native AR sampler comparisons on the original ImageNet metric."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE=Path('/tmp/laser-beam451-stage2-4h100-20261107')
sys.path[:0]=[str(BASE/'source'),str(BASE)]
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
os.environ['TORCH_HOME']=str(BASE/'torch-cache')
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from src import rq_upstream
from src import rqvae_metrics as metrics
from runtime.aligned_recipe import AlignedBeamRecipe
from runtime.preservation_guard import verify_frozen_codec


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def digest(model):
    value=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        value.update(name.encode());value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());out=Path(plan['output']);out.mkdir(parents=True,exist_ok=True)
    batch=256 if plan['samples']==50000 else 64
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world==4 and plan['samples']%1000==0
    device=torch.device('cuda',local);torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    checkpoint=Path(plan['checkpoint']);identity=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    source_step=int(state['global_step']);source_epoch=source_step/626
    with torch.device('meta'):model=rq_upstream.RQTransformer(OmegaConf.create(state['config']['arch']))
    model.load_state_dict(state['state_dict'],assign=True,strict=True);model.to(device).eval().requires_grad_(False)
    del state
    profile=AlignedBeamRecipe(BASE);aux,_=profile.tokenizer(device)
    preservation=verify_frozen_codec(profile,aux);aux_before=profile.digest(aux);model_before=digest(model)
    all_hashes=[None]*world;dist.all_gather_object(all_hashes,model_before);assert len(set(all_hashes))==1
    metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
    inception=metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    results=[]
    for policy in plan['policies']:
        name=policy['name'];sampling=policy['sampling'];destination=out/(name+'.json')
        # Safe restart after a fully written policy result.
        if destination.exists():
            result=json.loads(destination.read_text())
            assert result['checkpoint_sha256']==plan['checkpoint_sha256'] and result['sampling']==sampling
            assert result['seed']==plan['seed'] and result['generated_images']==plan['samples']
            assert result['sampling_batch_per_rank']==batch and result['model_state_sha256']==model_before
            results.append(result);continue
        seed=plan['seed'];torch.manual_seed(seed+rank);torch.cuda.manual_seed(seed+rank)
        metric=metrics.DistributedOriginalRQVAEMetrics(device,compute_inception_score=True,
            reference_stats_path=BASE/'inputs/imagenet_256_train.npz',inception=inception)
        count=plan['samples']//world;started=time.monotonic()
        for first in range(0,count,batch):
            n=min(batch,count-first)
            labels=(torch.arange(first,first+n,device=device)*world+rank)%1000
            codes=model.sample(torch.zeros(n,8,8,4,dtype=torch.long,device=device),model_aux=aux,
                cond=labels,amp=True,temperature=float(sampling['temperature']),top_k=sampling['top_k'],top_p=sampling['top_p'])
            assert codes.dtype==torch.long and codes.min()>=0 and codes.max()<16384
            for chunk in codes.split(8):
                pixels=profile.decode(aux,chunk).float().mul(.5).add(.5).clamp(0,1)
                assert pixels.isfinite().all();metric.update(pixels,real=False)
            if rank==0:
                write(out/'progress.json',dict(policy=name,generated_on_rank=first+n,samples_on_rank=count,
                    samples_global=plan['samples'],elapsed_seconds=time.monotonic()-started,time=time.time()))
        fid,is_mean,is_std=metric.compute(shuffle_for_inception=True)
        result=dict(policy=name,sampling=sampling,checkpoint=str(checkpoint),checkpoint_sha256=plan['checkpoint_sha256'],
            source_run=profile.run_id,source_step=source_step,source_epoch=source_epoch,model_state_sha256=model_before,
            generated_images=plan['samples'],reference_images=1281167,reference_split='train',
            reference_sha256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',
            fid=fid,inception_score=is_mean,inception_score_std=is_std,seed=seed,world_size=world,sampling_batch_per_rank=batch,
            balanced_all_1000_classes=True,weights_unmodified=True,training_updates_added=0,
            rejection_sampling=False,screen_only=plan['samples']!=50000,seconds=time.monotonic()-started)
        if rank==0:write(destination,result);print(json.dumps(result),flush=True)
        results.append(result);del metric;torch.cuda.empty_cache();dist.barrier()
    assert profile.digest(aux)==aux_before and digest(model)==model_before
    assert identity==(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    if rank==0:
        write(out/'results.json',dict(passed=True,weights_unmodified=True,frozen_tokenizer_unchanged=True,
            stage1_preservation=preservation,checkpoint_step=source_step,results=results))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
