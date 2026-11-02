"""Frozen-feature likelihood and generation ablation for scalar mixtures.

Both heads see the same cached augmented training images and true preceding
pairs. Only the final coefficient classifiers are trained. Every generation
comparison uses the same base weights, seed, reference and sampling policy.
"""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from src.models.discretized_coefficient_mixture import install_coefficient_mixture
from src.training import rqtransformer as training
from src.training.imagenet_compound_runtime import make_aux
from src.training.rq_imagenet_runtime import TarImageNet, write_json

BASE = Path('/tmp/laser-imagenet-compound512-audit-20261026')
SAMPLING = dict(atom_temperature=.8, atom_top_k=16384, atom_top_p=.9,
                coeff_temperature=1., coeff_top_k=0, coeff_top_p=.85)


def load_base(path, device):
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    with torch.device('meta'):
        model = training.build_model(16896, 16384, coeff_vocab_size=512, sparsity_level=4,
            physical_pair_context=True, efficient_pair_backbone=True,
            compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
            coefficient_history_layers=2, coefficient_history_width=512,
            coefficient_history_heads=8, coefficient_history_normalize_fields=True,
            rq_initialization='legacy')
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    metadata = dict(source_checkpoint=str(path), source_step=int(payload['global_step']),
                    source_config=payload['config'])
    model = model.to(device).eval().requires_grad_(False)
    del payload
    return model, metadata


def saved_heads(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    container = CoefficientHeads(nn.ModuleList([
        nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 512)) for _ in range(4)
    ]))
    container.load_state_dict(payload, strict=True)
    return container.to(device)


class CoefficientHeads(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, x):
        return torch.stack([head(x[..., d, :]) for d, head in enumerate(self.heads)], -2)


@torch.no_grad()
def extract(args, device):
    world, rank = dist.get_world_size(), dist.get_rank()
    model, metadata = load_base(args.checkpoint, device)
    aux = make_aux(BASE/'inputs/stage1-tokenizer.pt', device)
    if rank == 0:
        torch.save(CoefficientHeads(model.coeff_classifier).cpu().state_dict(), args.output/'original-heads.pt')
        model.coeff_classifier.to(device)
        write_json(args.output/'source.json', metadata)
    dist.barrier()
    captured = {}
    hooks = [head.register_forward_pre_hook(
        lambda module, inputs, d=d: captured.__setitem__(d, inputs[0].detach().cpu().bfloat16()))
        for d, head in enumerate(model.coeff_classifier)]
    try:
        for split, count in [('train', args.training_images), ('val', args.validation_images)]:
            path = args.output/f'{split}-features-rank{rank}.pt'
            if path.exists():
                dist.barrier()
                continue
            transform = training.image_transform() if split == 'train' else training.val_image_transform()
            dataset = TarImageNet(BASE/'data', split, transform, seed=args.seed)
            dataset.set_epoch(10)
            generator = torch.Generator().manual_seed(args.seed + (0 if split == 'train' else 1))
            selected = torch.randperm(len(dataset), generator=generator)[:count]
            indices = selected[rank::world].tolist()
            loader = DataLoader(Subset(dataset, indices), batch_size=16, num_workers=4,
                pin_memory=True, multiprocessing_context='spawn', persistent_workers=True)
            features, targets = [], []
            started = time.monotonic()
            for i, (images, labels) in enumerate(loader):
                atoms, values = aux.encode_sparse_components(images.to(device, non_blocking=True))
                tokens, _ = aux.sparse_targets(atoms, values, hard=True, stochastic=False, compact=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(tokens, aux, labels.to(device))
                features.append(torch.stack([captured[d] for d in range(4)], -2).reshape(-1, 4, 1536))
                targets.append((tokens[..., 1::2] - 16384).reshape(-1, 4).cpu().short())
                del outputs
                if rank == 0 and (i + 1) % 16 == 0:
                    print(json.dumps(dict(phase='extract', split=split, images_per_rank=(i+1)*16,
                                          seconds=time.monotonic()-started)), flush=True)
            torch.save(dict(features=torch.cat(features), targets=torch.cat(targets),
                image_indices=indices, split=split, source_step=metadata['source_step'],
                augmentation_epoch=10, seed=args.seed), path)
            del features, targets, loader, dataset
            gc.collect()
            dist.barrier()
    finally:
        for hook in hooks:
            hook.remove()
    if rank == 0:
        write_json(args.output/'features-ready.json', dict(training_images=args.training_images,
            validation_images=args.validation_images, source_step=metadata['source_step'],
            seed=args.seed, source_checkpoint=str(args.checkpoint), world_size=world))


def raw(module):
    return module.module if isinstance(module, DDP) else module


@torch.no_grad()
def validate(heads, features, targets, device):
    heads.eval()
    bins = torch.linspace(-3, 3, 512, device=device)
    scales = torch.tensor([4.587941646575928, 2.4749884605407715,
                          1.6106663942337036, 1.0176019668579102], device=device)
    sums = torch.zeros(6, 4, dtype=torch.float64, device=device)
    for x, target in zip(features.split(256), targets.split(256), strict=True):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logp = raw(heads)(x).float().log_softmax(-1)
        p = logp.exp()
        values = bins[target]
        nll = -logp.gather(-1, target[..., None]).squeeze(-1)
        sampled_mae = (p * (bins - values[..., None]).abs()).sum(-1)
        sign_probability = (p * ((bins > 0) == (values[..., None] > 0))).sum(-1)
        expected_mae = ((p * bins).sum(-1) - values).abs()
        cdf_target = (torch.arange(512, device=device) >= target[..., None]).float()
        crps = ((p.cumsum(-1)[..., :-1] - cdf_target[..., :-1]).square()).mean(-1)
        values_to_sum = [nll, sampled_mae * scales, sign_probability, expected_mae * scales, crps]
        for d, value in enumerate(values_to_sum):
            sums[d] += value.sum(0)
        sums[-1] += len(x)
    dist.all_reduce(sums)
    result = {name: (sums[i] / sums[-1]).tolist() for i, name in enumerate(
        ['nll', 'sampled_physical_mae', 'correct_sign_probability', 'expected_physical_mae', 'crps'])}
    result['mean_nll'] = sum(result['nll']) / 4
    return result


def fit(args, device):
    rank = dist.get_rank()
    torch.manual_seed(args.seed)
    train = torch.load(args.output/f'train-features-rank{rank}.pt', map_location='cpu', weights_only=True)
    val = torch.load(args.output/f'val-features-rank{rank}.pt', map_location='cpu', weights_only=True)
    x, target = train['features'].to(device), train['targets'].to(device).long()
    vx, vy = val['features'].to(device), val['targets'].to(device).long()
    del train, val
    teacher = saved_heads(args.output/'original-heads.pt', device).eval().requires_grad_(False)
    control = copy.deepcopy(teacher).requires_grad_(True)
    # Use the same installation path as production, including the learned norms.
    from types import SimpleNamespace
    carrier = nn.Module()
    carrier.config, carrier.coeff_vocab_size = SimpleNamespace(embed_dim=1536), 512
    carrier.coeff_classifier = copy.deepcopy(teacher.heads)
    install_coefficient_mixture(carrier, components=args.components, width=args.width,
                                min_scale=args.min_scale)
    mixture = CoefficientHeads(carrier.coeff_classifier).to(device).requires_grad_(True)
    del carrier
    mixture = DDP(mixture, device_ids=[device.index], broadcast_buffers=False)
    control = DDP(control, device_ids=[device.index], broadcast_buffers=False)
    optim = torch.optim.AdamW(mixture.parameters(), lr=1.e-3, betas=(.9, .99), weight_decay=1.e-4)
    control_optim = torch.optim.AdamW(control.parameters(), lr=2.e-4, betas=(.9, .99), weight_decay=1.e-4)
    baseline = validate(teacher, vx, vy, device)
    best = dict(mixture=float('inf'), control=baseline['mean_nll'])
    best_metrics = dict(control=baseline)
    if rank == 0:
        write_json(args.output/'baseline-heldout.json', baseline)
        torch.save(dict(state_dict=raw(control).state_dict(), metrics=baseline, epoch=0), args.output/'control-best.pt')
    step = 0
    for epoch in range(args.distill_epochs + args.fit_epochs):
        distilling = epoch < args.distill_epochs
        mixture.train()
        control.train()
        rate = 1.e-3 if distilling else 2.e-5 + .5 * (2.e-4 - 2.e-5) * (
            1 + math.cos(math.pi * (epoch - args.distill_epochs) / max(1, args.fit_epochs - 1)))
        optim.param_groups[0]['lr'] = rate
        control_optim.param_groups[0]['lr'] = rate if not distilling else 2.e-4
        generator = torch.Generator(device=device).manual_seed(args.seed + 100*epoch + rank)
        order = torch.randperm(len(x), generator=generator, device=device)
        losses = torch.zeros(3, dtype=torch.float64, device=device)
        started = time.monotonic()
        for indices in order.split(args.batch_sites):
            if len(indices) < args.batch_sites:
                continue
            batch, truth = x[indices], target[indices]
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = mixture(batch)
                if distilling:
                    with torch.no_grad():
                        q = teacher(batch).float().softmax(-1)
                    loss = -(q * prediction).sum(-1).mean()
                else:
                    loss = F.cross_entropy(prediction.flatten(0, -2), truth.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mixture.parameters(), 1.)
            optim.step()
            control_loss = torch.tensor(0., device=device)
            if not distilling:
                control_optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    cp = control(batch)
                    control_loss = F.cross_entropy(cp.flatten(0, -2), truth.flatten())
                control_loss.backward()
                torch.nn.utils.clip_grad_norm_(control.parameters(), 1.)
                control_optim.step()
            losses += torch.stack((loss.detach().double(), control_loss.detach().double(),
                                   torch.ones((), device=device).double()))
            step += 1
        dist.all_reduce(losses)
        metrics = dict(mixture=validate(mixture, vx, vy, device), control=validate(control, vx, vy, device))
        for name, module in [('mixture', mixture), ('control', control)]:
            if metrics[name]['mean_nll'] < best[name]:
                best[name] = metrics[name]['mean_nll']
                best_metrics[name] = metrics[name]
                if rank == 0:
                    torch.save(dict(state_dict=raw(module).state_dict(), metrics=metrics[name],
                        epoch=epoch+1, phase='distill' if distilling else 'supervised',
                        components=args.components, width=args.width,
                        min_scale=args.min_scale, coefficient_head='discretized-logistic-mixture' if name=='mixture' else 'categorical'),
                        args.output/f'{name}-best.pt')
        if rank == 0:
            row = dict(epoch=epoch+1, step=step, phase='distill' if distilling else 'supervised',
                learning_rate=rate, mixture_train_ce=float(losses[0]/losses[2]),
                control_train_ce=float(losses[1]/losses[2]), heldout=metrics,
                seconds=time.monotonic()-started)
            with (args.output/'fit-metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            write_json(args.output/'fit-status.json', row)
            print(json.dumps(row), flush=True)
    if rank == 0:
        write_json(args.output/'fit-result.json', dict(baseline=baseline, best=best_metrics,
            training_images=args.training_images, validation_images=args.validation_images,
            distillation_epochs=args.distill_epochs, supervised_epochs=args.fit_epochs,
            components=args.components, width=args.width, backbone_frozen=True,
            trainable_parameters=dict(mixture=sum(p.numel() for p in mixture.parameters()),
                                      control=sum(p.numel() for p in control.parameters()))))


@torch.no_grad()
def verify_trained_cache(model, aux, device, output):
    """Exercise all 512 scalar events on one real image with the fitted head."""
    dataset = TarImageNet(BASE/'data', 'val', training.val_image_transform())
    image, label = dataset[1701]
    atoms, values = aux.encode_sparse_components(image[None].to(device))
    target, _ = aux.sparse_targets(atoms, values, hard=True, stochastic=False, compact=True)
    cond = torch.tensor([label], device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(target, aux, cond)
    generated = torch.zeros_like(target)
    generated[..., 1::2] = 16384 + 256
    maximum = dict(atom=0., coefficient=0.)
    model.init_cache()
    for h in range(8):
        for w in range(8):
            for event in range(8):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model.cached_forward(generated, aux, cond, amp=False, sample_loc=(h, w, event))
                name = 'coefficient' if event % 2 else 'atom'
                reference = expected['coeff_logits' if event % 2 else 'atom_logits'][:, h, w, event//2].float()
                actual = logits[:, 16384:] if event % 2 else logits[:, :16384]
                q = reference.softmax(-1)
                kl = torch.where(q > 0, q * (q.clamp_min(1.e-30).log() - actual.float().log_softmax(-1)), 0.).sum(-1)
                maximum[name] = max(maximum[name], float(kl.max()))
                generated[:, h, w, event] = target[:, h, w, event]
    model.init_cache()
    report = dict(passed=max(maximum.values()) < .005, maximum_kl=maximum, scalar_events=512)
    write_json(output/'mixture-trained-cache.json', report)
    if not report['passed']:
        raise RuntimeError(f'Fitted mixture cache parity failed: {maximum}')


@torch.no_grad()
def evaluate(args, device):
    model, metadata = load_base(args.checkpoint, device)
    if args.variant != 'original':
        payload = torch.load(args.output/f'{args.variant}-best.pt', map_location='cpu', weights_only=True)
        if args.variant == 'mixture':
            install_coefficient_mixture(model, components=payload['components'], width=payload['width'],
                                        min_scale=payload['min_scale'])
        CoefficientHeads(model.coeff_classifier).load_state_dict(payload['state_dict'], strict=True)
        model.to(device).eval()
    aux = make_aux(BASE/'inputs/stage1-tokenizer.pt', device)
    if args.variant == 'mixture':
        if any(not torch.equal(head.bin_centers, aux.coeff_bins) for head in model.coeff_classifier):
            raise ValueError('Mixture bin centers differ from the frozen tokenizer')
        if dist.get_rank() == 0 and not (args.output/'mixture-trained-cache.json').exists():
            verify_trained_cache(model, aux, device, args.output)
        dist.barrier()
        if not json.loads((args.output/'mixture-trained-cache.json').read_text())['passed']:
            raise RuntimeError('Fitted mixture cache parity failed')
    started = time.monotonic()
    fid, is_mean, is_std = training.evaluate_generation_metrics(model, aux, None, args.samples, 64,
        fid_seed=261027, metric_backend='original-rqvae',
        fid_reference_stats=BASE/'inputs/imagenet_256_train.npz', **SAMPLING)
    if dist.get_rank() == 0:
        report = dict(variant=args.variant, samples=args.samples, source_step=metadata['source_step'],
            fid=fid, inception_score=is_mean, inception_std=is_std, sampling=SAMPLING,
            fid_seed=261027, reference_images=1281167, seconds=time.monotonic()-started,
            backbone_frozen=True)
        write_json(args.output/f'{args.variant}-fid{args.samples}.json', report)
        print(json.dumps(report), flush=True)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(261026)
            training.sample_class_grid(model, aux, training.class_names_for_dataset('imagenet'),
                args.output/f'{args.variant}-preview', metadata['source_step'], num_samples=64,
                sample_batch_size=8, samples_per_class=8, setting_name=args.variant, **SAMPLING)


def canary(args, device):
    model, metadata = load_base(args.checkpoint, device)
    install_coefficient_mixture(model, components=args.components, width=args.width,
                                min_scale=args.min_scale)
    model.requires_grad_(True).train()
    aux = make_aux(BASE/'inputs/stage1-tokenizer.pt', device)
    dataset = TarImageNet(BASE/'data', 'val', training.val_image_transform())
    image, label = dataset[dist.get_rank() * 101]
    with torch.no_grad():
        atoms, values = aux.encode_sparse_components(image[None].to(device))
        tokens, _ = aux.sparse_targets(atoms, values, hard=True, stochastic=False, compact=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(tokens, aux, torch.tensor([label], device=device))
        loss = .5 * (F.cross_entropy(outputs['atom_logits'].flatten(0, -2), atoms.flatten())
            + F.cross_entropy(outputs['coeff_logits'].flatten(0, -2),
                              (tokens[..., 1::2] - 16384).flatten()))
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all())
                for p in model.parameters() if p.grad is not None)
    row = dict(rank=dist.get_rank(), loss=float(loss.detach()), finite_gradients=finite,
        missing_gradients=missing, parameters=sum(p.numel() for p in model.parameters()),
        peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30)
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, row)
    if dist.get_rank() == 0:
        write_json(args.output/'gpu-canary.json', dict(passed=all(
            r['finite_gradients'] and not r['missing_gradients'] for r in rows), ranks=rows))
        print(json.dumps(dict(phase='gpu_canary', ranks=rows)), flush=True)
    if not finite or missing:
        raise RuntimeError(f'Mixture full-model gradient canary failed on rank {dist.get_rank()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['canary', 'extract', 'fit', 'evaluate'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-images', type=int, default=32768)
    parser.add_argument('--validation-images', type=int, default=4096)
    parser.add_argument('--components', type=int, default=8)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--min-scale', type=float, default=.001)
    parser.add_argument('--distill-epochs', type=int, default=3)
    parser.add_argument('--fit-epochs', type=int, default=8)
    parser.add_argument('--batch-sites', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=261028)
    parser.add_argument('--variant', choices=['original', 'control', 'mixture'], default='mixture')
    parser.add_argument('--samples', type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    dist.init_process_group('nccl', device_id=device)
    try:
        if args.mode == 'canary':
            canary(args, device)
        elif args.mode == 'extract':
            extract(args, device)
        elif args.mode == 'fit':
            fit(args, device)
        else:
            evaluate(args, device)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
