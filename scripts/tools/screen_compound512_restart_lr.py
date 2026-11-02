#!/usr/bin/env python3
"""Compare short LR probes at identical fresh weights and augmented targets."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(os.environ['LASER_RQ_BASE'])
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
sys.path[:0] = [str(BASE/'source'),str(BASE/'upstream'),str(BASE),str(BASE/'runtime')]
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from src.training import rq_imagenet_shared as shared
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config
from runtime.amp_replay import recoverable_amp_step


def main():
    rank,world,local = (int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    device = torch.device('cuda',local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl',device_id=device)
    profile = CompoundReplayRecipe(BASE)
    shared.BASE,shared.PROFILE = BASE,profile
    out = Path('/workspace/Projects/laser/outputs')/profile.run_id/'lr-screen'
    options = profile.recipe.optimization
    count = int(options.probe_updates)
    accumulation = 2048//(world*int(profile.microbatch))
    assert accumulation*world*profile.microbatch == 2048
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    shared.seed_all(261001)
    aux,_ = profile.tokenizer(device)
    frozen = profile.digest(aux)
    train = TarImageNet(BASE/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
    sampler = ExactGlobalBatchSampler(train,2048,world,rank,accumulation,261001)
    loader = DataLoader(train,batch_sampler=sampler,num_workers=8,pin_memory=True)
    cached = []
    for index,(images,labels) in enumerate(loader):
        with torch.no_grad():
            targets,codes = profile.targets(aux,images.to(device))
        cached.append((targets['atoms'].cpu(),targets['coefficients'].cpu(),codes.cpu(),labels))
        if rank == 0 and index%16 == 0:
            write_json(out/'progress.json',dict(phase='caching_identical_augmented_targets',
                microbatches=index+1,target_microbatches=count*accumulation,time=time.time()))
        if len(cached) == count*accumulation:
            break
    validation = TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
    indices = torch.randperm(len(validation),generator=torch.Generator().manual_seed(261016))[:int(options.probe_heldout_images)]
    heldout = []
    for images,labels in DataLoader(Subset(validation,indices[rank::world].tolist()),batch_size=8,num_workers=2):
        with torch.no_grad():
            targets,codes = profile.targets(aux,images.to(device))
        heldout.append((targets['atoms'].cpu(),targets['coefficients'].cpu(),codes.cpu(),labels))
    del loader,train,sampler,validation
    gc.collect()

    def batch(value):
        atoms,values,codes,labels = [t.to(device) for t in value]
        probabilities = (-(values[...,None]-aux.coeff_bins).square()/float(profile.recipe.targets.temperature)).softmax(-1)
        return dict(atoms=atoms,coefficient_probabilities=probabilities),codes,labels

    @torch.inference_mode()
    def measure(model):
        model.eval()
        rows = []
        for value in heldout:
            targets,codes,labels = batch(value)
            logits = model(codes,aux,labels,amp=True)
            a = F.cross_entropy(logits['atom_logits'].float().reshape(-1,16384),
                targets['atoms'].reshape(-1),reduction='none').reshape_as(targets['atoms'])
            c = -(targets['coefficient_probabilities']*logits['coeff_logits'].float().log_softmax(-1)).sum(-1)
            rows.extend(torch.stack([a.mean((1,2,3)),c.mean((1,2,3))],-1).cpu().tolist())
        gathered = [None]*world
        dist.all_gather_object(gathered,rows)
        values = torch.tensor([row for part in gathered for row in part],dtype=torch.float64)
        assert torch.isfinite(values).all() and len(values)==int(options.probe_heldout_images)
        return dict(atom_ce=float(values[:,0].mean()),coefficient_ce=float(values[:,1].mean()),
            mean_joint_ce=float(values.mean()),per_image_mean_joint_ce=values.mean(-1).tolist())

    results = []
    initial_hashes = []
    for rate in options.candidate_peak_lrs:
        rate = float(rate)
        shared.seed_all(261001)
        model,optimizer,_ = shared.fresh_model(device)
        if rank == 0:
            digest = hashlib.sha256()
            for name,value in model.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            initial_hashes.append(digest.hexdigest())
        initial = measure(model)
        ddp = DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=100)
        scaler = torch.amp.GradScaler('cuda',init_scale=128.)
        norms,retries = [],0
        started = time.monotonic()
        for update in range(count):
            optimizer.param_groups[0]['lr'] = rate*min(1.,(update+1)/int(options.probe_warmup_updates))
            optimizer.zero_grad(set_to_none=True)
            def backward():
                model.train()
                for micro in range(accumulation):
                    targets,codes,labels = batch(cached[update*accumulation+micro])
                    with shared.nullcontext() if micro==accumulation-1 else ddp.no_sync():
                        logits = ddp(codes,aux,labels,amp=True)
                        loss = profile.loss(model,logits,targets)
                        assert torch.isfinite(loss)
                        scaler.scale(loss/accumulation).backward()
            backward()
            norm,retried = recoverable_amp_step(model.parameters(),optimizer,scaler,
                max_norm=1.,replay_backward=backward)
            norms.append(float(norm));retries += retried
            if rank == 0:
                write_json(out/'progress.json',dict(phase='lr_candidate',peak_lr=rate,
                    update=update+1,total_updates=count,grad_norm=float(norm),amp_retries=retries,
                    elapsed_seconds=time.monotonic()-started,time=time.time()))
        assert {int(v['step']) for v in optimizer.state.values()} == {count}
        final = measure(model)
        row = dict(peak_lr=rate,initial=initial,final=final,finite=True,
            maximum_gradient_norm=max(norms),amp_retries=retries,updates=count,
            elapsed_seconds=time.monotonic()-started)
        results.append(row)
        if rank == 0:
            write_json(out/f'lr{rate:.0e}.json',row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('initial','final')},
                default=str),flush=True)
        del ddp,optimizer,model,scaler
        gc.collect();torch.cuda.empty_cache();dist.barrier()
    assert profile.digest(aux) == frozen
    if rank == 0:
        assert len(set(initial_hashes)) == 1
        for row in results:
            assert row['initial'] == results[0]['initial']
        best = min(row['final']['mean_joint_ce'] for row in results)
        near = [row for row in results if row['final']['mean_joint_ce'] <= best+float(options.near_best_tolerance_nats)]
        winner = min(near,key=lambda row:row['peak_lr'])
        result = dict(passed=True,selected_peak_lr=winner['peak_lr'],results=results,
            initial_weight_sha256=initial_hashes[0],same_fresh_weights_and_targets=True,
            source_tokenizer_unchanged=True,heldout_images=int(options.probe_heldout_images),
            production_updates_added=0,probe_weights_discarded=True,
            scope='Short stability and held-out CE screen; not a long-run FID optimization',time=time.time())
        write_json(out/'selection.json',result)
        print(json.dumps(dict(selected_peak_lr=winner['peak_lr'],passed=True)),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
