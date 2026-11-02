#!/usr/bin/env python3
"""Controlled head-only pilot on frozen epoch-one features and exact targets."""
import copy
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-native-joint-depthtemp-imagenet-20261016')
OUT = ROOT/'outputs/imagenet-laser-joint-predictability-20261016/head-pilot'
HELDOUT_SEED = None
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
sys.path[:0] = [str(ROOT), str(BASE/'upstream'), str(ROOT/'runtime')]
import numpy as np
import torch
from omegaconf import OmegaConf
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from src import rq_upstream
from src.models.laser_joint_atom_adapter import AtomSharedJointLinear
from src.models.laser_joint_execution import AlignedJointLinear
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json


@torch.inference_mode()
def cache(device):
    sys.path.insert(0, str(ROOT/'scripts/tools'))
    from audit_laser_joint_predictability import checkpoint_at
    profile = JointLaserRecipe(BASE)
    path, saved = checkpoint_at(BASE, 626)
    profile.verify_resume(saved)
    assert saved['run_id'] == profile.run_id
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.eval().to(device)
    del saved
    aux, _ = profile.tokenizer(device)
    profile.adapt(model, aux)
    specification = load_config(BASE/'inputs/stage2-training-config.yaml')
    weight = model.classifier.linear.weight.detach().cpu().clone()
    bias = model.classifier.linear.bias.detach().cpu().clone()
    captured = []
    handle = model.classifier.linear.register_forward_pre_hook(lambda module, args: captured.append(args[0].float().cpu()))
    report = json.loads((OUT.parent/'joint_depth_temperature.json').read_text())
    populations = {}
    cache_checks = {}
    for name, split in [('calibration_train', 'train'), ('heldout_val', 'val')]:
        indices = report['splits'][name]['indices']
        if name == 'heldout_val' and HELDOUT_SEED is not None:
            forbidden = set(indices)
            candidates = np.random.default_rng(HELDOUT_SEED).choice(50000, 256, replace=False).tolist()
            indices = [i for i in candidates if i not in forbidden][:128]
            assert len(indices) == 128 and not forbidden.intersection(indices)
        imageset = TarImageNet(BASE/'data', split, create_transforms(specification.dataset, split='val'))
        features = torch.empty(len(indices)*256, weight.shape[-1])
        targets = torch.empty(len(indices)*256, weight.shape[0])
        reference_ce = 0.
        for first in range(0, len(indices), 2):
            pairs = [imageset[i] for i in indices[first:first+2]]
            images = torch.stack([p[0] for p in pairs]).to(device)
            labels = torch.tensor([p[1] for p in pairs], device=device)
            torch.manual_seed(261016001+first)
            p, codes = profile.targets(aux, images)
            logits = model(codes, aux, labels, amp=True)
            reference_ce += float(-(p*logits.float().log_softmax(-1)).sum())
            start, stop = first*256, (first+len(pairs))*256
            features[start:stop] = captured.pop().reshape(-1, weight.shape[-1])
            targets[start:stop] = p.reshape(-1, weight.shape[0]).cpu()
            del images, p, codes, logits
            if first%32 == 0:
                write_json(OUT/'progress.json', dict(phase='cache', split=name, images=first+len(pairs), time=time.time()))
        populations[name] = features, targets
        cache_checks[name] = dict(indices=indices, reference_ce=reference_ce/len(features))
    write_json(OUT/'cache-verification.json', cache_checks)
    handle.remove()
    del model, aux
    gc.collect();torch.cuda.empty_cache()
    # Convert out of inference tensors before using the weights for training.
    return weight.numpy().copy(), bias.numpy().copy(), populations


@torch.no_grad()
def evaluate(head, features, targets, device):
    errors = []
    head.eval()
    for first in range(0, len(features), 128):
        x, p = features[first:first+128].to(device), targets[first:first+128].to(device)
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            z = head(x).float()
        errors.append(-(p*z.log_softmax(-1)).sum(-1).cpu())
    return torch.cat(errors).reshape(-1, 256).mean(-1)


def main():
    global OUT, HELDOUT_SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head-lr', type=float, default=.0005)
    parser.add_argument('--output-name', default='head-pilot')
    parser.add_argument('--heldout-seed', type=int)
    args = parser.parse_args()
    OUT = OUT.parent/args.output_name
    HELDOUT_SEED = args.heldout_seed
    torch.set_num_threads(4);torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(9.5*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    def uncached_autocast(enabled=True):
        return torch.autocast('cuda', enabled=enabled, dtype=torch.float16, cache_enabled=False)
    rq_upstream.RQTransformer.forward.__globals__['autocast'] = uncached_autocast
    OUT.mkdir(parents=True, exist_ok=True)
    weight, bias, populations = cache(device)
    native = torch.nn.Linear(weight.shape[1], weight.shape[0])
    native.weight.data.copy_(torch.from_numpy(weight));native.bias.data.copy_(torch.from_numpy(bias))
    initial = AlignedJointLinear(native)
    del weight, bias
    results = {}
    for name in ['native_control', 'atom_shared_joint']:
        head = copy.deepcopy(initial).to(device)
        if name == 'atom_shared_joint':
            head = AtomSharedJointLinear(head, 16384, 4)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, betas=(.9, .95), weight_decay=.0001)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        rng = torch.Generator().manual_seed(261016)
        fit_x, fit_p = populations['calibration_train']
        val_x, val_p = populations['heldout_val']
        rows = []
        for step in range(201):
            if step in [0, 100, 200]:
                per_image = evaluate(head, val_x, val_p, device)
                row = dict(step=step, heldout_ce=float(per_image.mean()), per_image_ce=per_image.tolist())
                if step == 0:
                    baseline = json.loads((OUT/'cache-verification.json').read_text())['heldout_val']['reference_ce']
                    assert abs(row['heldout_ce']-baseline) < .001
                rows.append(row)
                write_json(OUT/(name+'.json'), dict(name=name, frozen_backbone=True, exact_original_soft_targets=True,
                    training_images=128, heldout_images=128, rows=rows, parameter_count=sum(p.numel() for p in head.parameters())))
                print(json.dumps({k:v for k,v in row.items() if k!='per_image_ce'}|dict(case=name)), flush=True)
            if step==200:break
            head.train()
            ids = torch.randint(len(fit_x), (256,), generator=rng)
            x, p = fit_x[ids].to(device), fit_p[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                logits = head(x).float()
            loss = -(p*logits.log_softmax(-1)).sum(-1).mean()
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            scaler.step(optimizer);scaler.update()
            del x, p, logits, loss
            if step%25 == 0:
                write_json(OUT/'progress.json', dict(phase='head_fit', case=name, step=step+1, time=time.time()))
        results[name] = rows
        del head, optimizer, scaler
        gc.collect();torch.cuda.empty_cache()
    control = np.array(results['native_control'][-1]['per_image_ce'])
    candidate = np.array(results['atom_shared_joint'][-1]['per_image_ce'])
    delta = candidate-control
    rng = np.random.default_rng(261016)
    bootstrap = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(-1)
    write_json(OUT/'summary.json', dict(passed=True, training_updates=200, batch_tokens=256,
        head_learning_rate=args.head_lr, heldout_seed=HELDOUT_SEED,
        native_control_heldout_ce=float(control.mean()), atom_shared_heldout_ce=float(candidate.mean()),
        candidate_minus_control=float(delta.mean()), paired_image_bootstrap_95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
        candidate_improves_control=bool(candidate.mean()<control.mean()),
        candidate_improves_frozen_baseline=bool(candidate.mean()<results['native_control'][0]['heldout_ce']),
        baseline_heldout_ce=results['native_control'][0]['heldout_ce'],
        scope='One fixed epoch-one backbone, 128 calibration training images, 128 held-out validation images; head-only updates.',
        generation_fid_improvement_measured=False, production_training_unchanged=True, time=time.time()))


if __name__ == '__main__':main()
