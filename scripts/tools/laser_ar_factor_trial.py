#!/usr/bin/env python3
"""Matched continuations of one checkpoint for sparse AR factor diagnosis."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-hard-ar-imagenet-20261015')
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(ROOT), str(BASE/'upstream')]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
from runtime.amp_replay import ReplayWindow, recoverable_amp_step
from src.training.rq_imagenet_recipes import LaserRecipe
from src.training.rq_imagenet_shared import training_backward
from src.training.rq_imagenet_runtime import TarImageNet, cpu_snapshot, write_json
from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.laser_ar_factors import configure, coefficient_logits
from src.orthogonal_sparse_codec import ordered_support_basis
from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
from rqvae.img_datasets.transforms import create_transforms
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinates', choices=('physical', 'orthogonal'), required=True)
    parser.add_argument('--bins', type=int, choices=(128, 2048), required=True)
    parser.add_argument('--updates', type=int, default=64)
    args = parser.parse_args()
    rank, world, local = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world == 4
    device = torch.device('cuda', local)
    torch.cuda.set_device(device); torch.set_num_threads(4)
    dist.init_process_group('nccl', device_id=device)
    name = f'{args.coordinates}-{args.bins}'
    local_out = BASE/'factor-trials'/name
    profile = LaserRecipe(BASE)
    durable_out = ROOT/'outputs'/profile.run_id/'diagnostics/principled/factor-trials'/name
    if rank == 0:
        local_out.mkdir(parents=True, exist_ok=True); durable_out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    audit = json.loads(
        (ROOT/'outputs'/profile.run_id/'diagnostics/paused-checkpoint-audit.json').read_text())
    parent_path = Path(audit['checkpoint_local_cache'])
    saved = torch.load(parent_path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['global_step'] == 2504 and saved['next_epoch'] == 4 and saved['next_microbatch'] == 0
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch); arch.vocab_size = config.dataset.vocab_size
    with torch.device('meta'):
        model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.to(device)
    optimizer = create_resnet_optimizer(model, augment_optimizer_defaults(config.optimizer))
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 626, 100)
    optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
    scaler = torch.amp.GradScaler('cuda'); scaler.load_state_dict(saved['scaler'])
    aux, _ = profile.tokenizer(device)
    factor = configure(profile, model, aux, coordinates=args.coordinates, bins=args.bins)
    saved_rng = saved['rng_state_by_rank'][rank]
    random.setstate(saved_rng['python']); np.random.set_state(saved_rng['numpy'])
    torch.set_rng_state(saved_rng['cpu']); torch.cuda.set_rng_state(saved_rng['cuda'], device)
    parent_runtime = saved['runtime']; parent_best = saved['best_fid']
    del saved
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False,
              gradient_as_bucket_view=True, bucket_cap_mb=100)
    train = TarImageNet(BASE/'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    accumulation = 8
    sampler = ExactGlobalBatchSampler(train, 2048, world, rank, accumulation, 261001)
    sampler.set_epoch(4)
    loader = DataLoader(train, batch_sampler=sampler, num_workers=8, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)
    validation = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'), seed=261001)
    validation_indices = np.linspace(0, len(validation)-1, 512, dtype=int).tolist()
    selected_indices = validation_indices[rank::world]
    val_loader = DataLoader(Subset(validation, selected_indices), batch_size=8, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def evaluate(step):
        model.eval(); metrics = torch.zeros(10, dtype=torch.float64, device=device)
        rows = []; cursor = 0
        for images, labels in val_loader:
            images = images.to(device); labels = labels.to(device)
            targets, codes = profile.targets(aux, images)
            output = model(codes, aux, labels, amp=True)
            atoms = -output['atom_logits'].float().log_softmax(-1).gather(-1, codes[..., 0::2, None]).squeeze(-1)
            group = 2048//args.bins
            coefficients = -coefficient_logits(output['coeff_logits'].float(), args.bins).log_softmax(-1).gather(
                -1, ((codes[..., 1::2]-aux.num_atoms)//group)[..., None]).squeeze(-1)
            per_image_a = atoms.mean((1, 2)); per_image_c = coefficients.mean((1, 2))
            correction = torch.zeros(len(images), device=device)
            if args.coordinates == 'orthogonal':
                vectors = aux.dictionary.T[codes[..., 0::2]]
                _, lower = ordered_support_basis(vectors)
                correction = -lower.diagonal(dim1=-2, dim2=-1).log().sum(-1).mean((1, 2))/4
            density = per_image_a.mean(-1)+per_image_c.mean(-1)+math.log(6/(args.bins-1))+correction
            metrics[:4] += per_image_a.sum(0).double(); metrics[4:8] += per_image_c.sum(0).double()
            metrics[8] += density.sum().double(); metrics[9] += len(images)
            for i in range(len(images)):
                rows.append(dict(index=selected_indices[cursor+i], support_nll=per_image_a[i].tolist(),
                    coefficient_nll=per_image_c[i].tolist(), density_nll_per_pair=float(density[i])))
            cursor += len(images)
        dist.all_reduce(metrics)
        gathered = [None]*world
        dist.all_gather_object(gathered, rows)
        result = dict(global_step=step, held_out_images=int(metrics[9]),
            support_nll_per_depth=(metrics[:4]/metrics[9]).tolist(),
            coefficient_nll_per_depth=(metrics[4:8]/metrics[9]).tolist(),
            density_nll_per_pair=float(metrics[8]/metrics[9]), coefficient_bins=args.bins,
            coordinates=args.coordinates, paired_rows=sorted(sum(gathered, []), key=lambda r:r['index']))
        if rank == 0:
            write_json(durable_out/f'heldout-step{step}.json', result)
            print(json.dumps({k:v for k,v in result.items() if k!='paired_rows'}), flush=True)
        model.train()
        return result

    initial = evaluate(2504)
    replay = ReplayWindow(device)
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic(); step = 2504; loss_sum = 0.; image_count = 0

    def backward(images, labels, absolute):
        images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        with nullcontext() if (absolute+1)%accumulation == 0 else ddp.no_sync():
            loss = training_backward(profile, ddp, aux, images, labels, scaler,
                world*len(images)/sampler.global_count(absolute), amp=True)
        return loss

    for absolute, (images, labels) in enumerate(loader):
        if absolute%accumulation == 0:
            replay.begin()
        replay.append(images, labels, absolute)
        loss = backward(images, labels, absolute)
        loss_sum += float(loss)*len(images); image_count += len(images)
        if (absolute+1)%accumulation:
            continue
        norm, retries = recoverable_amp_step(model.parameters(), optimizer, scaler, max_norm=1.,
            replay_backward=lambda:replay.replay(backward))
        scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
        assert int(next(iter(optimizer.state.values()))['step']) == step
        if rank == 0:
            row = dict(branch=name,global_step=step,additional_updates=step-2504,
                loss=loss_sum/image_count,lr=optimizer.param_groups[0]['lr'],
                images_per_second=2048/(time.monotonic()-start),grad_norm=float(norm),amp_retries=retries,time=time.time())
            write_json(local_out/'progress.json',row)
            if (step-2504)%8 == 0: print(json.dumps(row),flush=True)
        loss_sum = 0.; image_count = 0; start = time.monotonic()
        if step >= 2504+args.updates:
            break
    final = evaluate(step)
    rng = dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
    all_rng = [None]*world;dist.all_gather_object(all_rng,rng)
    if rank == 0:
        target = local_out/'checkpoint.pt'
        payload = cpu_snapshot(dict(run_id=profile.run_id,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),global_step=step,epoch=4,next_epoch=4,
            next_microbatch=args.updates*8,rng_state_by_rank=all_rng,recipe=OmegaConf.to_container(profile.recipe,resolve=True),
            stage1_checkpoint_sha256=profile.recipe.tokenizer.sha256,runtime=dict(parent_runtime,fresh_stage2=False,fresh_optimizer=False),
            config=OmegaConf.to_container(config,resolve=True),best_fid=parent_best,
            factor_trial=factor,parent_checkpoint=str(parent_path),parent_global_step=2504))
        torch.save(payload,target)
        receipt=dict(passed=True,branch=name,parent_step=2504,global_step=step,updates=args.updates,
            same_model_and_optimizer_initialization=True,same_data_order=True,global_batch=2048,world_size=world,
            checkpoint=str(target),checkpoint_bytes=target.stat().st_size,
            before={k:v for k,v in initial.items() if k!='paired_rows'},
            after={k:v for k,v in final.items() if k!='paired_rows'},factor=factor,time=time.time())
        write_json(durable_out/'result.json',receipt);print(json.dumps(receipt),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':
    main()
