#!/usr/bin/env python3
"""Read-only real-batch objective audits while the fresh prior keeps training."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-pairid256k-grid512-imagenet-20261017')
RUN = 'imagenet-laser-pairid256k-grid512-scratch100-4h100-20261017'
OUT = ROOT/'outputs'/RUN
AUDIT = OUT/'objective-balance'
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(BASE/'source'), str(BASE/'upstream'), str(BASE/'runtime')]
import numpy as np
import torch
from src.training.laser_catalog_integer_recipe import CatalogIntegerLaserRecipe
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.img_datasets.transforms import create_transforms

helper = Path(__file__).with_name('joint_integer_balance.py')
if not helper.exists():
    helper = ROOT/'src/training/joint_integer_balance.py'
module_spec = importlib.util.spec_from_file_location('laser_balance_audit', helper)
diagnostics = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(diagnostics)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def publish(report):
    """Publish diagnostics with the public API; no second training SDK writer."""
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    import wandb
    run = wandb.Api(timeout=45).run('helloimlixin-rutgers/laser/'+RUN)
    root = BASE/'objective-balance-uploads'
    path = root/'objective-balance'/f'{report["checkpoint_scope"]}-step{report["checkpoint_global_step"]:06d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, report)
    run.upload_file(str(path), root=str(root))
    run.config['objective_balance_audit'] = dict(
        weights=report['weights'], additional_training_losses=0,
        checked_checkpoint_scope=report['checkpoint_scope'], checked_checkpoint_step=report['checkpoint_global_step'],
        coefficient_to_atom_context_gradient_ratio=report['coefficient_to_atom_context_gradient_ratio'],
        training_loss_or_weights_changed=False, diagnostics_file=str(path.relative_to(root)))
    run.update()
    write(AUDIT/'online-verification.json', dict(passed=True, checkpoint_scope=report['checkpoint_scope'],
        checkpoint_step=report['checkpoint_global_step'], file=str(path.relative_to(root)),
        run_url=run.url, no_additional_sdk_writer=True, time=time.time()))


@torch.inference_mode()
def audit(receipt, *, scope, device):
    checkpoint = Path(receipt['local']) if Path(receipt['local']).exists() else Path(receipt['persistent'])
    assert checkpoint.stat().st_size == receipt['bytes']
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert saved['global_step'] == receipt['global_step'] and saved['run_id'] == RUN
    profile = CatalogIntegerLaserRecipe(BASE)
    profile.verify_resume(saved)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    with torch.device('meta'):
        model = profile.model(augment_arch_defaults(config.arch))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model = model.eval().to(device)
    aux, _ = profile.tokenizer(device)
    prior = json.loads((ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015/calibration-data.json').read_text())
    fit = json.loads((ROOT/'outputs/imagenet-laser-compact-joint-vocabulary-20261016/calibration-data.json').read_text())
    excluded = set(prior['fit_indices']+prior['temperature_indices']+fit['fit_indices'])
    indices = [int(i) for i in np.random.default_rng(261017941).permutation(1281167) if int(i) not in excluded][:8]
    data = TarImageNet(BASE/'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    data.set_epoch(0)
    all_sums, all_counts, implemented_loss, total_tokens = None, None, 0., 0
    errors = dict(maximum_ce_decomposition_error=0., maximum_gradient_decomposition_error=0., maximum_target_mass_error=0.)
    metric_names = None
    for first in range(0, len(indices), 2):
        torch.manual_seed(261017941+first)
        records = [data[i] for i in indices[first:first+2]]
        images = torch.stack([row[0] for row in records]).to(device)
        labels = torch.tensor([row[1] for row in records], device=device)
        targets, codes = profile.targets(aux, images)
        logits = model(codes, aux, labels, amp=True)
        result = diagnostics.joint_integer_balance(logits, targets['atoms'], targets['coefficient_probabilities'],
            classifier_weight=model.classifier.linear.weight, row_chunk=32)
        sums, counts = np.asarray(result['sums']), np.asarray(result['counts'])
        all_sums = sums if all_sums is None else all_sums+sums
        all_counts = counts if all_counts is None else all_counts+counts
        implemented_loss += float(profile.loss(model, logits, targets))*codes.numel()
        total_tokens += codes.numel()
        metric_names = result['metric_names']
        for key in errors:
            errors[key] = max(errors[key], result[key])
        del logits, targets, images, codes
    values = all_sums/all_counts[:, None]
    per_depth = []
    for d in range(4):
        row = dict(depth=d+1, tokens=int(all_counts[d]))
        for key, val in zip(metric_names, values[d]):
            row[key.removesuffix('_sq')+'_rms' if key.endswith('_sq') else key] = math.sqrt(max(0., float(val))) if key.endswith('_sq') else float(val)
        per_depth.append(row)
    means = all_sums.sum(0)/all_counts.sum()
    lookup = dict(zip(metric_names, means))
    ce = implemented_loss/total_tokens
    loss_error = abs(ce-float(lookup['joint_ce']))
    assert loss_error < 1.e-5 and (all_counts == all_counts[0]).all()
    atom_rms = math.sqrt(float(lookup['atom_context_gradient_sq']))
    coefficient_rms = math.sqrt(float(lookup['coefficient_context_gradient_sq']))
    assert math.isfinite(atom_rms) and math.isfinite(coefficient_rms)
    result = dict(passed=True, checkpoint_scope=scope, checkpoint_global_step=receipt['global_step'],
        checkpoint_sha256=receipt['sha256'], training_images=8, training_indices=indices,
        disjoint_from_catalog_fitting=True, same_original_training_augmentation=True,
        objective='one globally normalized joint integer cross entropy',
        weights=dict(atom_marginal=1., coefficient_conditional=1., depths=[.25]*4),
        additional_training_losses=0, training_loss_or_weights_changed=False,
        global_joint_ce=ce, atom_marginal_ce=float(lookup['atom_marginal_ce']),
        coefficient_conditional_ce=float(lookup['coefficient_conditional_ce']),
        coefficient_target_entropy=float(lookup['coefficient_target_entropy']),
        atom_classifier_context_gradient_rms=atom_rms,
        coefficient_classifier_context_gradient_rms=coefficient_rms,
        coefficient_to_atom_context_gradient_ratio=coefficient_rms/max(atom_rms, 1.e-30),
        norm_scope='per-token classifier-input gradient, before common batch mean and global clipping',
        equal_objective_weights_do_not_imply_equal_gradient_norms=True,
        implemented_loss_decomposition_error=loss_error, **errors, per_depth=per_depth,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        helper_sha256=sha(helper), audit_source_sha256=sha(Path(__file__)), time=time.time())
    name = f'{scope}-step{receipt["global_step"]:06d}.json'
    write(AUDIT/name, result)
    write(AUDIT/'latest.json', result)
    print(json.dumps(result), flush=True)
    del model, aux, saved
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    smoke = BASE/'private-smoke/checkpoints/.checkpoint-data'
    receipt = max((json.loads(p.read_text()) for p in smoke.glob('*.json')), key=lambda r:r['global_step'])
    initial = AUDIT/f'preflight-step{receipt["global_step"]:06d}.json'
    report = json.loads(initial.read_text()) if initial.exists() else None
    if not (report and report['checkpoint_sha256'] == receipt['sha256'] and report['helper_sha256'] == sha(helper)):
        report = audit(receipt, scope='preflight', device=device)
    if args.publish:
        publish(report)
    if not args.watch:
        return
    last = 0
    while not (OUT/'completed.json').exists():
        receipts = [json.loads(p.read_text()) for p in (OUT/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
        available = [r for r in receipts if r['global_step'] > last]
        if available:
            receipt = max(available, key=lambda r:r['global_step'])
            report = audit(receipt, scope='training', device=device)
            if args.publish:
                try:
                    publish(report)
                except Exception as error:
                    write(AUDIT/'upload-status.json', dict(phase='publication_failed',
                        error_type=type(error).__name__, local_audit_saved=True, time=time.time()))
            last = receipt['global_step']
        time.sleep(30)


if __name__ == '__main__':
    main()
