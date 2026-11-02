#!/usr/bin/env python3
"""Matched scalar/context continuations, including full-population FID."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path(os.environ.get('LASER_CONTEXT_TRIAL_BASE', '/tmp/laser-pair-context-ablation-20261015'))
OUT = ROOT/'outputs/imagenet-laser-pair-context-ablation-20261015'
PARENT = Path('/tmp/laser-rq-validated-ar-imagenet-20261015')
GROUP = 'laser_pair_context_matched_20261015'
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(BASE/'source'), str(BASE), str(ROOT), str(BASE/'upstream')]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
from runtime.amp_replay import ReplayWindow, recoverable_amp_step
from src.training.rq_imagenet_recipes import LaserRecipe
from src.training.rq_imagenet_shared import training_backward
from src.training import rq_imagenet_shared as shared
from src.training.rq_imagenet_runtime import TarImageNet, cpu_snapshot, write_json
from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.laser_pair_ablation import restore_named_adam
from src.training import k4_checkpoint_io as checkpoint_io
from rqvae.utils.config import load_config, augment_arch_defaults, augment_optimizer_defaults
from rqvae.img_datasets.transforms import create_transforms
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler


def restore_rng(state, device, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state(state['cuda'], device)
    generator.set_state(state['coefficient'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=['control', 'context'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, default=626)
    parser.add_argument('--fid-only', action='store_true')
    parser.add_argument('--generation-seed', type=int, default=261001)
    args = parser.parse_args()
    assert 1 <= args.stop_after <= 626
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world == 4
    device = torch.device('cuda', local)
    torch.cuda.set_device(device); torch.set_num_threads(4)
    dist.init_process_group('nccl', device_id=device)
    os.environ.update(json.loads((BASE/'environment.json').read_text()))
    local_out, durable = BASE/args.branch, OUT/args.branch
    local_out.mkdir(exist_ok=True); durable.mkdir(exist_ok=True)
    profile = LaserRecipe(BASE/('recipe-'+args.branch))
    run_id = profile.run_id
    parent = json.loads((OUT/'parent-checkpoint.json').read_text())
    parent_step, epoch, accumulation = 8764, 14, 16
    path = durable/'checkpoints/last.pt' if args.resume or args.fid_only else Path(parent['checkpoint'])
    saved = torch.load(checkpoint_io._checkpoint_upload_source(path.resolve()),
                       map_location='cpu', mmap=True, weights_only=False)
    step = int(saved['global_step'])
    assert parent_step <= step <= parent_step+626
    source_names = list(saved.get('parameter_names', saved['state_dict']))
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch); arch.vocab_size = config.dataset.vocab_size
    torch.manual_seed(261015)
    with torch.device('meta'):
        model = profile.model(arch)
    fresh_context = args.branch == 'context' and not args.resume and not args.fid_only
    if fresh_context:
        for module in (model.coefficient_identity, model.pair_context_residual):
            module.to_empty(device='cpu').apply(model._init_weights)
        model.load_scalar_state(saved['state_dict'], assign=True)
    else:
        model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.to(device)
    aux, _ = profile.tokenizer(device); profile.adapt(model, aux)
    digest = profile.digest(aux)
    generator = profile.coefficient_generator
    names = list(dict(model.named_parameters()))
    additions = sorted(set(names)-set(source_names)) if fresh_context else (
        [n for n in names if n.startswith(('coefficient_identity.', 'pair_context_residual.'))]
        if args.branch == 'context' else [])
    before_path = durable/f'heldout-step{parent_step}.json'

    @torch.no_grad()
    def evaluate(at_step):
        model.eval()
        validation = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
        indices = np.linspace(0, len(validation)-1, 512, dtype=int).tolist()[rank::world]
        loader = DataLoader(Subset(validation, indices), batch_size=8, num_workers=2, pin_memory=True)
        metrics = torch.zeros(17, device=device, dtype=torch.float64)
        rows, cursor = [], 0
        # Validation must not consume the training dropout or DataLoader RNG.
        with torch.random.fork_rng(devices=[device.index]):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                _, tokens = profile.targets(aux, images)
                output = model(tokens, aux, labels, amp=True)
                a = -output['atom_logits'].float().log_softmax(-1).gather(
                    -1, tokens[..., 0::2, None]).squeeze(-1).mean((1, 2))
                logc = output['coeff_logits'].float().log_softmax(-1)
                ids = tokens[..., 1::2]-aux.num_atoms
                c = -logc.gather(-1, ids[..., None]).squeeze(-1).mean((1, 2))
                centers = aux.coeff_bins[ids]*aux.coeff_scales
                values = aux.coeff_bins[None, :]*aux.coeff_scales[:, None]
                q = (-(centers[..., None]-values).square()/(2*.125**2)).softmax(-1)
                soft = -(q*logc).sum(-1).mean((1, 2))
                entropy = -(q*q.clamp_min(1.e-30).log()).sum(-1).mean((1, 2))
                for offset, value in [(0, a), (4, c), (8, soft), (12, entropy)]:
                    metrics[offset:offset+4] += value.double().sum(0)
                metrics[16] += len(images)
                for i in range(len(images)):
                    rows.append(dict(index=indices[cursor+i], support_nll=a[i].tolist(),
                        coefficient_nll=c[i].tolist(), ce=float((a[i].mean()+c[i].mean())/2)))
                cursor += len(images)
        dist.all_reduce(metrics)
        gathered = [None]*world; dist.all_gather_object(gathered, rows)
        report = dict(step=at_step, images=int(metrics[16]),
            support_nll=(metrics[:4]/metrics[16]).tolist(),
            coefficient_nll=(metrics[4:8]/metrics[16]).tolist(),
            coefficient_soft_ce=(metrics[8:12]/metrics[16]).tolist(),
            coefficient_target_entropy=(metrics[12:16]/metrics[16]).tolist(),
            ce=float(metrics[:8].sum()/(metrics[16]*8)),
            paired_rows=sorted(sum(gathered, []), key=lambda row: row['index']))
        if rank == 0:
            write_json(durable/f'heldout-step{at_step}.json', report)
            row = {'train/global_step': at_step, 'eval/heldout_ce': report['ce']}
            for index in range(4):
                for key in ['support_nll', 'coefficient_nll', 'coefficient_soft_ce', 'coefficient_target_entropy']:
                    row[f'eval/{key}_depth{index+1}'] = report[key][index]
            run.log(row)
        model.train()
        return report

    run = None
    if rank == 0:
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=run_id,
            name=f'LASER pair context matched - {args.branch}', group=GROUP,
            dir=str(BASE), resume='must' if args.resume or args.fid_only else 'allow',
            config=None if args.resume or args.fid_only else dict(wandb_id=run_id,
                experiment_group=GROUP, branch=args.branch, parent_step=parent_step,
                additional_updates=626, global_batch=2048, microbatch=32,
                world_size=4, accumulation_steps=16, shared_upstream=True,
                auxiliary_loss_terms=0, sampled_omp_trajectories=0,
                coefficient_physical_sd=.125, coefficient_policy='soft',
                coordinates='orthogonal', decoder=str(profile.recipe.decoder),
                sampling=OmegaConf.to_container(profile.recipe.sampling),
                fid_generated_images=50000, fid_reference_images=1281167,
                fid_reference_split='train', fid_reference_sha256=profile.recipe.evaluation.reference_sha256,
                parameter_count=sum(p.numel() for p in model.parameters())))
        run.define_metric('train/global_step'); run.define_metric('train/epoch')
        run.define_metric('train/*', step_metric='train/global_step')
        run.define_metric('eval/*', step_metric='train/global_step')
        run.define_metric('eval/fid', step_metric='train/epoch', overwrite=True)
        write_json(durable/'wandb.json', dict(id=run.id, url=run.url))

    if args.fid_only:
        assert step == parent_step+626
        del saved
        model.eval()
        data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
        pairs = [data[rank*2+i] for i in range(2)]
        images = torch.stack([p[0] for p in pairs]).to(device)
        labels = torch.tensor([p[1] for p in pairs], device=device)
        with torch.no_grad():
            _, truth = profile.targets(aux, images); expected = model(truth, aux, labels, amp=True)
            buffer = torch.zeros_like(truth); buffer[..., 1::2] = aux.num_atoms+1024
            model.init_cache(); maximum_kl = 0.
            for site in range(64):
                h, w = divmod(site, 8)
                for event in range(8):
                    actual = model.cached_forward(buffer[:, :h+1], aux, labels, amp=True,
                                                   sample_loc=(h, w, event))
                    actual = actual[:, :aux.num_atoms] if event%2 == 0 else actual[:, aux.num_atoms:]
                    reference = expected['atom_logits' if event%2 == 0 else 'coeff_logits'][:, h, w, event//2]
                    assert torch.equal(torch.isfinite(actual), torch.isfinite(reference))
                    a, b = actual.float().log_softmax(-1), reference.float().log_softmax(-1)
                    maximum_kl = max(maximum_kl, float(torch.where(b.isfinite(), b.exp()*(b-a), 0).sum(-1).max()))
                    buffer[:, h, w, event] = truth[:, h, w, event]
            assert maximum_kl < 1.e-4, maximum_kl
            model.init_cache()
        cache_kl = torch.tensor(maximum_kl, device=device); dist.all_reduce(cache_kl, op=dist.ReduceOp.MAX)
        if rank == 0:
            write_json(durable/'sampling-verification.json', dict(passed=True, events_per_rank=512,
                cached_teacher_forcing_maximum_kl=float(cache_kl), learned_context=args.branch == 'context'))
        del expected, truth, buffer, images, labels, actual, reference, a, b
        shared.BASE, shared.OUT, shared.PROFILE, shared.RUN = BASE, local_out, profile, run_id
        shared.original_metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
        import rqvae.utils.utils as utils
        original_top_k = utils.top_k_logits
        utils.top_k_logits = lambda logits, k: logits if k == logits.shape[-1] else original_top_k(logits, k)
        started = time.time()
        result = shared.generation(model, aux, device, rank, world, 15, run, batch=128,
                                   seed=args.generation_seed)
        assert profile.digest(aux) == digest
        if rank == 0:
            result.update(branch=args.branch, global_step=step, parent_step=parent_step,
                additional_updates=626, generated_images=50000, reference_images=1281167,
                reference_split='train', reference_sha256=str(profile.recipe.evaluation.reference_sha256),
                sampler=OmegaConf.to_container(profile.recipe.sampling), generation_seed=args.generation_seed,
                elapsed_seconds=time.time()-started)
            primary = args.generation_seed == 261001
            result_path = durable/('fid50k.json' if primary else f'fid50k-seed{args.generation_seed}.json')
            write_json(result_path, result)
            run.log({'train/global_step': step, 'train/epoch': 15,
                'eval/fid' if primary else 'eval/fid_repeat': result['fid'],
                'eval/fid_inherited': False, 'eval/inception_score': result['inception_score']})
            run.summary['trial/status'] = 'complete'
            run.save(str(result_path), base_path=str(OUT), policy='now'); run.finish()
        dist.barrier(); dist.destroy_process_group(); return

    optimizer = create_resnet_optimizer(model, augment_optimizer_defaults(config.optimizer))
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 626, 100)
    if fresh_context:
        indices = [i for group in saved['optimizer']['param_groups'] for i in group['params']]
        assert len(indices) == len(source_names) == 782
        states = {name: saved['optimizer']['state'][i] for name, i in zip(source_names, indices)}
        restore_named_adam(optimizer, model, states)
        for own, previous in zip(optimizer.param_groups, saved['optimizer']['param_groups']):
            own.update({k: v for k, v in previous.items() if k != 'params'})
    else:
        optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    scaler = torch.amp.GradScaler('cuda'); scaler.load_state_dict(saved['scaler'])
    rng = saved['rng_state_by_rank'][rank]
    start_micro = int(saved['next_microbatch'])
    assert saved['next_epoch'] == 14
    assert start_micro == (step-parent_step)*accumulation
    if step == parent_step:
        model.eval()
        control = LaserRecipe(PARENT)
        with torch.device('meta'):
            original = control.model(arch)
        original.load_state_dict(saved['state_dict'], strict=True, assign=True)
        original.to(device).eval(); control.adapt(original, aux)
        data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
        pairs = [data[rank*2+i] for i in range(2)]
        images = torch.stack([p[0] for p in pairs]).to(device)
        labels = torch.tensor([p[1] for p in pairs], device=device)
        with torch.no_grad():
            _, codes = control.targets(aux, images)
            reference, actual = original(codes, aux, labels, amp=True), model(codes, aux, labels, amp=True)
            for key in actual:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            if name in additions:
                assert parameter not in optimizer.state
            else:
                assert int(optimizer.state[parameter]['step']) == step
        if rank == 0:
            write_json(durable/'initialization-verification.json', dict(passed=True,
                exact_initial_logits=True, existing_adam_tensors=782, existing_adam_age=step,
                new_parameter_tensors=len(additions), new_parameters=sum(dict(model.named_parameters())[n].numel() for n in additions),
                private_coefficient_rng_restored=True, scheduler_restored=True))
            run.log({'train/global_step': step, 'train/epoch': 14, 'eval/fid': 40.36652094676475,
                     'eval/fid_inherited': True})
        del original, control, reference, actual, images, labels, codes
        gc.collect()
        torch.cuda.empty_cache()
    del saved
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False,
              gradient_as_bucket_view=True, bucket_cap_mb=100)
    train = TarImageNet(BASE/'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    sampler = ExactGlobalBatchSampler(train, 2048, world, rank, accumulation, 261001)
    sampler.set_epoch(epoch)
    order = np.asarray(list(itertools.chain.from_iterable(sampler)), dtype=np.int32)
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    hashes = [None]*world; dist.all_gather_object(hashes, order_hash)
    sampler.set_start_batch(start_micro)
    loader = DataLoader(train, batch_sampler=sampler, num_workers=8, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)
    if not before_path.exists():
        before = evaluate(parent_step)
    else:
        before = json.loads(before_path.read_text())
    restore_rng(rng, device, generator)
    replay = ReplayWindow(device, generators=[generator])
    optimizer.zero_grad(set_to_none=True); model.train()
    update_loss = torch.zeros((), device=device); update_images = 0
    started = time.monotonic(); overflow_retries = 0

    def checkpoint(at_step, completed=False):
        state = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state(device), coefficient=generator.get_state())
        all_rng = [None]*world; dist.all_gather_object(all_rng, state)
        if rank == 0:
            payload = cpu_snapshot(dict(run_id=run_id, state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                epoch=15 if completed else 14, next_epoch=15 if completed else 14,
                next_microbatch=0 if completed else (at_step-parent_step)*accumulation,
                global_step=at_step, rng_state_by_rank=all_rng, parameter_names=names,
                recipe=OmegaConf.to_container(profile.recipe, resolve=True),
                config=OmegaConf.to_container(config, resolve=True),
                stage1_checkpoint_sha256=str(profile.recipe.tokenizer.sha256),
                parent_checkpoint=parent['persistent_checkpoint'], parent_step=parent_step,
                best_fid=40.36652094676475, evaluation=None,
                runtime=dict(parameter_count=sum(p.numel() for p in model.parameters()),
                    parameter_tensors=len(names), world_size=4, microbatch=32, accumulation_steps=16)))
            checkpoint_io.atomic_torch_save(payload, durable/'checkpoints/last.pt')
            local_checkpoint = checkpoint_io._checkpoint_upload_source((durable/'checkpoints/last.pt').resolve())
            write_json(durable/'checkpoint.json', dict(global_step=at_step,
                path=str(durable/'checkpoints/last.pt'), local_checkpoint=str(local_checkpoint),
                bytes=local_checkpoint.stat().st_size, full_optimizer_and_rng=True))
            del payload
        dist.barrier()

    def backward(images, labels, absolute):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with nullcontext() if (absolute+1)%accumulation == 0 else ddp.no_sync():
            return training_backward(profile, ddp, aux, images, labels, scaler,
                world*len(images)/sampler.global_count(absolute), amp=True)

    for relative, (images, labels) in enumerate(loader):
        absolute = start_micro+relative
        if absolute%accumulation == 0:
            replay.begin()
        replay.append(images, labels, absolute)
        loss = backward(images, labels, absolute)
        update_loss.add_(loss*len(images)); update_images += len(images)
        if (absolute+1)%accumulation:
            continue
        norm, retries = recoverable_amp_step(model.parameters(), optimizer, scaler, max_norm=1.,
            replay_backward=lambda: replay.replay(backward))
        overflow_retries += retries
        scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
        stats = torch.stack((update_loss, torch.tensor(update_images, device=device))); dist.all_reduce(stats)
        if rank == 0:
            row = {'train/global_step': step, 'train/epoch': step/626,
                'train/additional_updates': step-parent_step, 'train/loss': float(stats[0]/stats[1]),
                'train/lr': optimizer.param_groups[0]['lr'], 'train/grad_norm': float(norm),
                'train/amp_overflow_retries': retries,
                'train/images_per_second': float(stats[1])/(time.monotonic()-started), 'time': time.time()}
            write_json(durable/'progress.json', row)
            if (step-parent_step)%10 == 0 or step-parent_step <= 3:
                with (local_out/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                run.log(row); print(json.dumps(row), flush=True)
        update_loss.zero_(); update_images = 0
        finished = step-parent_step >= args.stop_after
        if finished or (step-parent_step)%200 == 0:
            checkpoint(step, completed=step == parent_step+626)
        if finished:
            break
        started = time.monotonic()
    assert step == parent_step+args.stop_after
    after = evaluate(step)
    assert profile.digest(aux) == digest
    ages = {}
    for name, parameter in model.named_parameters():
        age = int(optimizer.state[parameter]['step'])
        assert age == (step-parent_step if name in additions else step), (name, age)
        ages[str(age)] = ages.get(str(age), 0)+1
    if rank == 0:
        result = dict(passed=True, branch=args.branch, parent_step=parent_step, global_step=step,
            additional_updates=step-parent_step, target_additional_updates=626,
            full_training_complete=step == parent_step+626,
            checkpoint=str(durable/'checkpoints/last.pt'), optimizer_age_groups=ages,
            same_data_order_sha256_by_rank=hashes, augmentation_epoch=14,
            coefficient_rng_inherited=True, auxiliary_loss_terms=0, sampled_omp_trajectories=0,
            amp_overflow_retries=overflow_retries,
            before={k:v for k,v in before.items() if k != 'paired_rows'},
            after={k:v for k,v in after.items() if k != 'paired_rows'})
        write_json(durable/('result.json' if result['full_training_complete'] else 'preflight.json'), result)
        run.summary['trial/status'] = 'trained_awaiting_fid' if result['full_training_complete'] else 'preflight_passed'
        run.finish(); print(json.dumps(result), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
