#!/usr/bin/env python3
"""Matched checkpoint continuations over decoder and coefficient-policy factors."""
import argparse
from contextlib import nullcontext
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/workspace/Projects/laser')
PARENT = Path('/tmp/laser-rq-causal-ar-imagenet-20261015')
BASE = Path(os.environ.get('LASER_PAIR_ABLATION_BASE', '/tmp/laser-pair-ablation-20261015'))
OUT = ROOT / 'outputs/imagenet-laser-pair-ablation-20261015'
os.environ['RQ_UPSTREAM_ROOT'] = str(PARENT / 'upstream')
os.environ['TORCH_HOME'] = str(PARENT / 'torch-cache')
sys.path[:0] = [str(BASE / 'source'), str(ROOT), str(PARENT / 'upstream')]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
from runtime.amp_replay import ReplayWindow, recoverable_amp_step
from src.models.laser_compound_rq_adapter import build_compound_model
from src.training.rq_imagenet_recipes import LaserRecipe
from src.training.rq_imagenet_shared import training_backward
from src.training import rq_imagenet_shared as shared
from src.training.rq_imagenet_runtime import TarImageNet, cpu_snapshot, write_json
from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.laser_pair_ablation import configure_coefficient_policy, transplant_state, restore_named_adam
from src.training import k4_checkpoint_io as checkpoint_io
from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
from rqvae.img_datasets.transforms import create_transforms
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder', choices=('scalar', 'pair'), required=True)
    parser.add_argument('--coefficients', choices=('hard', 'soft'), required=True)
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--fid-only', action='store_true')
    parser.add_argument('--label', help='Distinct, immutable experimental variant label')
    args = parser.parse_args()
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world == 4
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', device_id=device)
    branch = args.label or f'{args.decoder}-{args.coefficients}'
    local_out, durable = BASE / branch, OUT / branch
    local_out.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    environment = json.loads((PARENT / 'environment.json').read_text())
    os.environ.update(environment)
    parent = json.loads((OUT / 'paused-parent.json').read_text())
    source_path = checkpoint_io._checkpoint_upload_source(Path(parent['checkpoint']))
    checkpoint_path = local_out / 'checkpoint.pt' if args.fid_only else source_path
    saved = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    parent_step = parent['global_step']
    assert parent_step == 3756
    if not args.fid_only:
        assert saved['global_step'] == parent_step and saved['next_epoch'] == 6
        assert saved['next_microbatch'] == 0
    config = load_config(PARENT / 'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch)
    arch.vocab_size = config.dataset.vocab_size
    profile = LaserRecipe(PARENT)
    torch.manual_seed(261015)
    with torch.device('meta'):
        source_model = profile.model(arch)
        names = list(dict(source_model.named_parameters()))
        model = (source_model if args.decoder == 'scalar' else build_compound_model(
            arch, num_atoms=16384, coefficient_bins=2048))
    if args.fid_only or args.decoder == 'scalar':
        model.load_state_dict(saved['state_dict'], strict=True, assign=True)
        transferred, mapping = {}, {name: name for name in names}
    else:
        transferred, mapping = transplant_state(model, saved['state_dict'], names, saved['optimizer'])
    del source_model
    model.to(device)
    aux, _ = profile.tokenizer(device)
    frozen_digest = profile.digest(aux)
    profile.adapt(model, aux)
    generator = configure_coefficient_policy(profile, model, aux, policy=args.coefficients, seed=261015 + rank)
    run_id = f'imagenet-laser-pair-ablation-{branch}-20261015'
    run = None
    if rank == 0:
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=run_id,
            name=f'LASER matched pair ablation | {branch}', dir=str(BASE),
            resume='must' if args.fid_only else 'allow',
            config=None if args.fid_only else dict(wandb_id=run_id,
                experiment_group='laser_pair_ablation_20261015', decoder=args.decoder,
                valid_comparison=True, pair_atom_identity='raw_dictionary',
                coefficient_policy=args.coefficients, parent_step=parent_step, additional_updates=args.updates,
                fresh_stage2=False, architecture_transfer=args.decoder == 'pair',
                shared_backbone_optimizer_moments_retained=True, auxiliary_loss_terms=0,
                sampled_omp_trajectories=0, support_encoding='deterministic_stage1',
                coordinates='orthogonal', coefficient_bins=2048, coefficient_physical_sd=.125 if args.coefficients=='soft' else 0.,
                global_batch=2048, microbatch=32, accumulation_steps=16, world_size=world,
                fid_generated_images=50000, fid_reference_images=1281167, fid_reference_split='train',
                parameter_count=sum(p.numel() for p in model.parameters())))
        run.define_metric('train/global_step')
        run.define_metric('train/epoch')
        run.define_metric('train/*', step_metric='train/global_step')
        run.define_metric('eval/fid', step_metric='train/epoch', overwrite=True)
        run.define_metric('eval/heldout_ce', step_metric='train/global_step')
        write_json(durable / 'wandb.json', dict(id=run.id, url=run.url))
    if args.fid_only:
        step = saved['global_step']
        del saved
        model.eval()
        validation = TarImageNet(PARENT / 'data', 'val', create_transforms(config.dataset, split='val'), seed=261001)
        pairs = [validation[i] for i in range(rank * 2, rank * 2 + 2)]
        images = torch.stack([p[0] for p in pairs]).to(device)
        labels = torch.tensor([p[1] for p in pairs], device=device)
        with torch.no_grad():
            _, truth = profile.targets(aux, images)
            expected = model(truth, aux, labels, amp=True)
            buffer = torch.zeros_like(truth)
            buffer[..., 1::2] = aux.num_atoms + aux.coeff_vocab_size // 2
            model.init_cache()
            divergences = []
            for w in range(4):
                for event in range(8):
                    actual = model.cached_forward(buffer[:, :1], aux, labels, amp=True, sample_loc=(0, w, event))
                    key = 'atom_logits' if event % 2 == 0 else 'coeff_logits'
                    actual = actual[:, :aux.num_atoms] if event % 2 == 0 else actual[:, aux.num_atoms:]
                    reference = expected[key][:, 0, w, event // 2]
                    assert torch.equal(torch.isfinite(actual), torch.isfinite(reference))
                    a, b = actual.float().log_softmax(-1), reference.float().log_softmax(-1)
                    divergences.append(torch.where(b.isfinite(), b.exp() * (b-a), 0.).sum(-1).mean())
                    buffer[:, 0, w, event] = truth[:, 0, w, event]
            maximum_kl = float(torch.stack(divergences).max())
            assert maximum_kl < 1.e-4, maximum_kl
            decoded = profile.decode(aux, truth)
            assert torch.isfinite(decoded).all()
            model.init_cache()
        if rank == 0:
            write_json(durable / 'sampling-verification.json', dict(passed=True,
                fp16_cached_teacher_forcing_maximum_mean_kl=maximum_kl, tested_events_per_rank=32,
                coordinates='orthogonal', code_decoder_finite=True))
        del expected, truth, buffer, images, labels, decoded, actual, reference, a, b, divergences
        torch.cuda.empty_cache()
        shared.BASE, shared.OUT, shared.PROFILE, shared.RUN = PARENT, local_out, profile, run_id
        shared.original_metrics.INCEPTION_PATH = PARENT / 'upstream/rqvae/metrics/inception.py'
        import rqvae.utils.utils as utils
        original_top_k = utils.top_k_logits
        utils.top_k_logits = lambda logits, k: logits if k == logits.shape[-1] else original_top_k(logits, k)
        start = time.time()
        result = shared.generation(model, aux, device, rank, world, step / 626, None, batch=128)
        assert profile.digest(aux) == frozen_digest
        if rank == 0:
            result.update(branch=branch, global_step=step, parent_step=parent_step,
                additional_updates=args.updates, generated_images=50000, reference_images=1281167,
                reference_split='train', sampler=dict(atom_temperature=1., atom_top_k=256, atom_top_p=.95,
                    coefficient_temperature=1., coefficient_top_p=.95), elapsed_seconds=time.time()-start)
            write_json(durable / 'fid50k.json', result)
            run.log({'eval/fid': result['fid'], 'eval/inception_score': result['inception_score'],
                     'train/global_step': step, 'train/epoch': step / 626})
            run.summary['evaluation/generated_images'] = 50000
            run.summary['evaluation/reference_images'] = 1281167
            run.save(str(durable / 'fid50k.json'), base_path=str(OUT), policy='now')
            run.finish()
        dist.barrier()
        dist.destroy_process_group()
        return

    optimizer = create_resnet_optimizer(model, augment_optimizer_defaults(config.optimizer))
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 626, 100)
    if args.decoder == 'scalar':
        optimizer.load_state_dict(saved['optimizer'])
    else:
        restore_named_adam(optimizer, model, transferred)
        for own, old in zip(optimizer.param_groups, saved['optimizer']['param_groups']):
            own.update({k: v for k, v in old.items() if k != 'params'})
    scheduler.load_state_dict(saved['scheduler'])
    scaler = torch.amp.GradScaler('cuda')
    scaler.load_state_dict(saved['scaler'])
    rng = saved['rng_state_by_rank'][rank]
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['cpu'])
    torch.cuda.set_rng_state(rng['cuda'], device)
    del saved, transferred
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False,
              gradient_as_bucket_view=True, bucket_cap_mb=100)
    train = TarImageNet(PARENT / 'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    accumulation = 16
    sampler = ExactGlobalBatchSampler(train, 2048, world, rank, accumulation, 261001)
    sampler.set_epoch(6)
    order = np.asarray(list(itertools.chain.from_iterable(itertools.islice(sampler, args.updates * accumulation))), dtype=np.int32)
    data_hash = hashlib.sha256(order.tobytes()).hexdigest()
    hashes = [None] * world
    dist.all_gather_object(hashes, data_hash)
    loader = DataLoader(train, batch_sampler=sampler, num_workers=8, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)
    validation = TarImageNet(PARENT / 'data', 'val', create_transforms(config.dataset, split='val'), seed=261001)
    indices = np.linspace(0, len(validation) - 1, 512, dtype=int).tolist()[rank::world]
    val_loader = DataLoader(Subset(validation, indices), batch_size=8, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        metrics = torch.zeros(9, device=device, dtype=torch.float64)
        rows, cursor = [], 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, codes = profile.targets(aux, images)
            output = model(codes, aux, labels, amp=True)
            a = -output['atom_logits'].float().log_softmax(-1).gather(-1, codes[..., 0::2, None]).squeeze(-1).mean((1, 2))
            c = -output['coeff_logits'].float().log_softmax(-1).gather(-1, (codes[..., 1::2] - aux.num_atoms)[..., None]).squeeze(-1).mean((1, 2))
            metrics[:4] += a.double().sum(0)
            metrics[4:8] += c.double().sum(0)
            metrics[8] += len(images)
            for i in range(len(images)):
                rows.append(dict(index=indices[cursor+i], support_nll=a[i].tolist(),
                    coefficient_nll=c[i].tolist(), ce=float((a[i].mean()+c[i].mean())/2)))
            cursor += len(images)
        dist.all_reduce(metrics)
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        report = dict(step=step, images=int(metrics[8]), support_nll=(metrics[:4]/metrics[8]).tolist(),
            coefficient_nll=(metrics[4:8]/metrics[8]).tolist(), ce=float(metrics[:8].sum()/(metrics[8]*8)),
            paired_rows=sorted(sum(gathered, []), key=lambda r: r['index']))
        if rank == 0:
            write_json(durable / f'heldout-step{step}.json', report)
            run.log({'eval/heldout_ce':report['ce'], 'train/global_step':step})
            print(json.dumps({k:v for k,v in report.items() if k!='paired_rows'}), flush=True)
        model.train()
        return report

    before = evaluate(parent_step)
    replay = ReplayWindow(device)
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    overflow_retries = 0
    step = parent_step
    window_generator = None
    update_loss = torch.zeros((), device=device)
    update_images = 0

    def backward(images, labels, absolute):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with nullcontext() if (absolute+1) % accumulation == 0 else ddp.no_sync():
            return training_backward(profile, ddp, aux, images, labels, scaler,
                                     world*len(images)/sampler.global_count(absolute), amp=True)

    def replay_backward():
        generator.set_state(window_generator)
        return replay.replay(backward)

    for absolute, (images, labels) in enumerate(loader):
        if absolute % accumulation == 0:
            replay.begin()
            window_generator = generator.get_state()
        replay.append(images, labels, absolute)
        loss = backward(images, labels, absolute)
        update_loss.add_(loss * len(images))
        update_images += len(images)
        if (absolute + 1) % accumulation:
            continue
        norm, retries = recoverable_amp_step(model.parameters(), optimizer, scaler, max_norm=1.,
                                             replay_backward=replay_backward)
        overflow_retries += retries
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        stats = torch.stack((update_loss, torch.tensor(update_images, device=device)))
        dist.all_reduce(stats)
        mean_loss = float(stats[0] / stats[1])
        if rank == 0:
            row = dict(branch=branch, global_step=step, additional_updates=step-parent_step,
                loss=mean_loss, lr=optimizer.param_groups[0]['lr'], grad_norm=float(norm),
                amp_retries=retries, images_per_second=2048/(time.monotonic()-start), time=time.time())
            write_json(local_out / 'progress.json', row)
            if (step-parent_step) % 8 == 0:
                print(json.dumps(row), flush=True)
                run.log({'train/global_step':step, 'train/epoch':step/626,
                         'train/loss':mean_loss, 'train/lr':row['lr'], 'train/images_per_second':row['images_per_second']})
        update_loss.zero_()
        update_images = 0
        start = time.monotonic()
        if step >= parent_step + args.updates:
            break
    after = evaluate(step)
    assert profile.digest(aux) == frozen_digest
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        expected_age = args.updates if mapping.get(name) is None else step
        assert int(state['step']) == expected_age, (name, state['step'], expected_age)
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))
    all_rng, private_rng = [None]*world, [None]*world
    dist.all_gather_object(all_rng, rng)
    dist.all_gather_object(private_rng, generator.get_state())
    if rank == 0:
        target = local_out / 'checkpoint.pt'
        state = cpu_snapshot(dict(run_id=run_id, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), global_step=step,
            epoch=6, next_epoch=6, next_microbatch=args.updates*accumulation,
            rng_state_by_rank=all_rng, coefficient_rng_by_rank=private_rng,
            recipe=OmegaConf.to_container(profile.recipe, resolve=True),
            stage1_checkpoint_sha256=profile.recipe.tokenizer.sha256,
            ablation=dict(decoder=args.decoder, coefficients=args.coefficients),
            parent_checkpoint=str(source_path), parent_step=parent_step))
        torch.save(state, target)
        receipt = dict(passed=True, branch=branch, parent_step=parent_step, global_step=step,
            frozen_source_base=str(BASE/'source'), decoder=args.decoder, coefficient_policy=args.coefficients,
            additional_updates=args.updates, checkpoint=str(target), checkpoint_bytes=target.stat().st_size,
            same_data_order_sha256_by_rank=hashes, same_global_batch=2048,
            shared_backbone_adam_retained=True, parameter_mapping=mapping,
            newly_initialized_parameters=[n for n,s in mapping.items() if s is None],
            auxiliary_loss_terms=0, sampled_omp_trajectories=0, amp_overflow_retries=overflow_retries,
            before={k:v for k,v in before.items() if k!='paired_rows'},
            after={k:v for k,v in after.items() if k!='paired_rows'}, time=time.time())
        write_json(durable / 'result.json', receipt)
        run.summary['trial/status'] = 'trained_awaiting_fid_selection'
        run.finish()
        print(json.dumps({k:v for k,v in receipt.items() if k not in ['parameter_mapping','newly_initialized_parameters']}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
