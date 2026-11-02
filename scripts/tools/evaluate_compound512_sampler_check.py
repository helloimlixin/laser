#!/usr/bin/env python3
"""Matched 50k evaluation of an ancestral sampler on a fixed checkpoint."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3'))
    parser.add_argument('--epoch',type=int,default=2)
    parser.add_argument('--sampling-json',type=Path,
        help='Complete six-field atom/coefficient sampling policy; default disables atom top-k')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--baseline-json',type=Path,
        help='Optional matched fixed-checkpoint FID50k result for comparison')
    parser.add_argument('--preview',action='store_true')
    args = parser.parse_args()
    base = args.base.resolve()
    os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')
    os.environ['TORCH_HOME'] = str(base/'torch-cache')
    sys.path[:0] = [str(base/'source'),str(base/'upstream'),str(base),str(base/'runtime')]
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
    from src.training import rq_imagenet_shared as shared
    from src.training.rq_imagenet_runtime import write_json

    rank,world,local = [int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    assert world == 4
    device = torch.device('cuda',local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(13*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=60))
    profile = CompoundReplayRecipe(base)
    source = Path('/workspace/Projects/laser/outputs')/profile.run_id
    out = args.output.resolve() if args.output else source/'collapse-diagnosis'/f'epoch{args.epoch:03d}'/'full50k-no-atom-topk'
    assert not (out/'fid50k.json').exists(), 'Refusing to overwrite a completed evaluation'
    receipts = [json.loads(p.read_text()) for p in (source/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    candidates = [r for r in receipts if r['global_step']==626*args.epoch]
    assert len(candidates)==1
    receipt = candidates[0]
    checkpoint = Path(receipt['local'])
    assert checkpoint.stat().st_size==receipt['bytes']
    payload = torch.load(checkpoint,mmap=True,map_location='cpu',weights_only=False)
    assert payload['global_step']==args.epoch*626
    profile.verify_resume(payload)
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(payload['config']['arch']))
    model.load_state_dict(payload['state_dict'],strict=True,assign=True)
    model.requires_grad_(False).eval().to(device)
    del payload
    aux,_ = profile.tokenizer(device)
    original = OmegaConf.to_container(profile.recipe.sampling,resolve=True)
    assert original['atom_top_k']==256
    if args.sampling_json:
        policy = json.loads(args.sampling_json.read_text())
        assert set(policy)==set(original), 'Expected exactly the existing six sampler fields'
        for factor,maximum in [('atom',16384),('coefficient',512)]:
            temperature,p,k = [policy[f'{factor}_{suffix}'] for suffix in ('temperature','top_p','top_k')]
            assert math.isfinite(temperature) and temperature>0
            assert math.isfinite(p) and 0<p<=1
            assert isinstance(k,int) and 0<=k<=maximum
        profile.recipe.sampling = OmegaConf.create(policy)
    else:
        profile.recipe.sampling.atom_top_k = 0
    candidate = OmegaConf.to_container(profile.recipe.sampling,resolve=True)
    changes = [k for k in original if original[k]!=candidate[k]]
    assert changes
    reference = json.loads((source/'evaluations'/f'epoch{args.epoch:03d}.json').read_text())
    assert reference['generated_images']==50000 and reference['real_images']==1281167
    baseline_fid = reference['eval/fid']
    comparison_sampling = original
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text())
        assert baseline['generated_images']==50000 and baseline['reference_images']==1281167
        assert baseline['reference_split']=='train' and baseline['source_step']==args.epoch*626
        assert baseline['checkpoint']['sha256']==receipt['sha256']
        assert baseline['reference_sha256']==hashlib.sha256((base/'inputs/imagenet_256_train.npz').read_bytes()).hexdigest()
        baseline_fid,comparison_sampling = baseline['fid'],baseline['sampling']
    plan = dict(source_run=profile.run_id,checkpoint=receipt,source_epoch=args.epoch,
        source_step=626*args.epoch,source_sampling=original,sampling=candidate,
        changed_settings=changes,baseline_fid=baseline_fid,comparison_sampling=comparison_sampling,
        generated_images=50000,reference_images=1281167,reference_split='train',
        reference_sha256=hashlib.sha256((base/'inputs/imagenet_256_train.npz').read_bytes()).hexdigest(),
        generation_seed=261001,generation_batch=128,world_size=world,
        weights_unmodified=True,training_updates_added=0,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if rank==0:
        write_json(out/'plan.json',plan)
        write_json(out/'status.json',dict(phase='evaluating',pid=os.getpid(),time=time.time()))
        print(json.dumps(plan),flush=True)
    shared.BASE,shared.OUT,shared.PROFILE,shared.RUN = base,out,profile,profile.run_id
    shared.original_metrics.INCEPTION_PATH = base/'upstream/rqvae/metrics/inception.py'
    before = profile.digest(aux)
    started = time.monotonic()
    result = shared.generation(model,aux,device,rank,world,args.epoch,None,batch=128,seed=261001)
    if args.preview:
        from src.training.rq_sample_panels import designated_preview
        designated_preview(profile,model,aux,device,rank,world,args.epoch*626,None,out)
    assert profile.digest(aux)==before
    if rank==0:
        result.update(plan,weights_unmodified=True,tokenizer_unchanged=True,
            fid_delta=result['fid']-baseline_fid,
            elapsed_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            passed=True,rejection_sampling=False)
        write_json(out/'fid50k.json',result)
        write_json(out/'status.json',dict(phase='completed',fid=result['fid'],time=time.time()))
        print(json.dumps(result),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
