#!/usr/bin/env python3
"""Matched saved-checkpoint diagnostics; never modify a training process."""
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
FROZEN = Path('/tmp/laser-rq-native-joint-depthtemp-imagenet-20261016')
OUT = ROOT/'outputs/imagenet-laser-joint-predictability-20261016'
os.environ['RQ_UPSTREAM_ROOT'] = str(FROZEN/'upstream')
sys.path[:0] = [str(FROZEN/'source'), str(FROZEN/'upstream'), str(FROZEN/'runtime'), str(ROOT)]
import numpy as np
import torch
from omegaconf import OmegaConf
from src import rq_upstream
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_recipes import RQRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json

TEMPERATURES = [.7, .85, 1., 1.15, 1.3]
TOP_K = [256, 512, 1024, 2048]
CASES = {
    'joint_depth_temperature': FROZEN,
    'joint_global_temperature': Path('/tmp/laser-rq-native-joint-imagenet-20261015'),
    'original_rq': Path('/tmp/laser-original-rq-imagenet-20261014'),
}


def checkpoint_at(base, step):
    for receipt in sorted((base/'checkpoint-cache/objects').glob('*.json')):
        path = receipt.with_suffix('.pt')
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if saved.get('global_step') == step:
            return path, saved
        del saved
    raise RuntimeError(f'No locally cached checkpoint at update {step} in {base}')


@torch.inference_mode()
def measure(name, base, train_indices, val_indices, device):
    profile = RQRecipe(base) if name == 'original_rq' else JointLaserRecipe(base)
    path, saved = checkpoint_at(base, 626)
    profile.verify_resume(saved)
    arch = OmegaConf.create(saved['config']['arch'])
    with torch.device('meta'):
        model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.eval().to(device)
    del saved
    aux, _ = profile.tokenizer(device)
    profile.adapt(model, aux)
    config = load_config(base/'inputs/stage2-training-config.yaml')
    report = dict(checkpoint=str(path), global_step=626, run_id=profile.run_id, splits={})
    for split, indices in [('calibration_train', train_indices), ('heldout_val', val_indices)]:
        data_split = 'train' if split == 'calibration_train' else 'val'
        data = TarImageNet(base/'data', data_split, create_transforms(config.dataset, split='val'))
        # Center crops are identical across checkpoints. The two populations
        # are disjoint, and no sampling parameter is selected on heldout_val.
        sums = torch.zeros(4, 13, dtype=torch.float64, device=device)
        ce_temperature = torch.zeros(4, len(TEMPERATURES), dtype=torch.float64, device=device)
        kept_target_mass = torch.zeros(4, len(TOP_K), dtype=torch.float64, device=device)
        for first in range(0, len(indices), 2):
            pairs = [data[i] for i in indices[first:first+2]]
            images = torch.stack([pair[0] for pair in pairs]).to(device)
            labels = torch.tensor([pair[1] for pair in pairs], device=device)
            torch.manual_seed(261016001+first)
            targets, codes = profile.targets(aux, images)
            # Identical AMP arithmetic without retaining FP16 copies of every
            # parameter alongside the saved FP32 weights during inference.
            with torch.autocast('cuda', cache_enabled=False):
                logits = model(codes, aux, labels, amp=True).float()
                wrong_logits = model(codes, aux, (labels+499)%1000, amp=True).float()
            for depth in range(4):
                p = targets[..., depth, :].reshape(-1, targets.shape[-1])
                z = logits[..., depth, :].reshape_as(p)
                log_q = z.log_softmax(-1)
                q = log_q.exp()
                p_entropy = -(p*p.clamp_min(1.e-30).log()).sum(-1)
                ce = -(p*log_q).sum(-1)
                wrong_log_q = wrong_logits[..., depth, :].reshape_as(p).log_softmax(-1)
                wrong_ce = -(p*wrong_log_q).sum(-1)
                top1 = z.argmax(-1)
                target_ids = codes[..., depth].flatten()
                atom_mass = torch.zeros(len(p), device=device)
                sign_conditional = torch.zeros_like(atom_mass)
                atom_ce, atom_h = ce, p_entropy
                sign_ce = magnitude_ce = torch.zeros_like(ce)
                sign_h = magnitude_h = torch.zeros_like(ce)
                if name != 'original_rq':
                    quantizer = aux.quantizer
                    atoms = quantizer.atom_ids[depth]
                    signs = quantizer.coefficients[depth].sign()
                    target_atoms, target_signs = atoms[target_ids], signs[target_ids]
                    same_atom = atoms[None, :] == target_atoms[:, None]
                    correct_sign = signs[None, :] == target_signs[:, None]
                    atom_mass = (q*same_atom).sum(-1)
                    sign_conditional = (q*same_atom*correct_sign).sum(-1)/atom_mass.clamp_min(1.e-30)
                    atom_index = atoms[None, :].expand(len(p), -1)
                    p_atom = p.new_zeros(len(p), quantizer.dictionary.shape[1]).scatter_add_(1, atom_index, p)
                    q_atom = p.new_zeros(p_atom.shape).scatter_add_(1, atom_index, q)
                    atom_ce = -(p_atom*q_atom.clamp_min(1.e-30).log()).sum(-1)
                    atom_h = -(p_atom*p_atom.clamp_min(1.e-30).log()).sum(-1)
                    sign_index = (atoms*3+signs.long()+1)[None, :].expand(len(p), -1)
                    p_sign = p.new_zeros(len(p), 3*quantizer.dictionary.shape[1]).scatter_add_(1, sign_index, p)
                    q_sign = p.new_zeros(p_sign.shape).scatter_add_(1, sign_index, q)
                    atom_sign_ce = -(p_sign*q_sign.clamp_min(1.e-30).log()).sum(-1)
                    atom_sign_h = -(p_sign*p_sign.clamp_min(1.e-30).log()).sum(-1)
                    sign_ce, magnitude_ce = atom_sign_ce-atom_ce, ce-atom_sign_ce
                    sign_h, magnitude_h = atom_sign_h-atom_h, p_entropy-atom_sign_h
                    torch.testing.assert_close(atom_ce+sign_ce+magnitude_ce, ce)
                    del p_atom, q_atom, p_sign, q_sign
                values = [ce.sum(), p_entropy.sum(), -(q*log_q).sum(),
                    (top1 == target_ids).float().sum(), (wrong_ce-ce).sum(),
                    atom_mass.sum(), sign_conditional.sum(), atom_ce.sum(), sign_ce.sum(),
                    magnitude_ce.sum(), atom_h.sum(), sign_h.sum(), magnitude_h.sum()]
                sums[depth] += torch.stack(values).double()
                for column, value in enumerate(TEMPERATURES):
                    ce_temperature[depth, column] -= (p*(z/value).log_softmax(-1)).sum().double()
                order = z.topk(max(TOP_K), dim=-1).indices
                cumulative = p.gather(-1, order).cumsum(-1)
                for column, value in enumerate(TOP_K):
                    kept_target_mass[depth, column] += cumulative[:, value-1].sum().double()
            del images, targets, codes, logits, wrong_logits, p, z, log_q, q, wrong_log_q
            if first % 32 == 0:
                write_json(OUT/'progress.json', dict(case=name, split=split, images=first+len(pairs), time=time.time()))
        rows = []
        for depth in range(4):
            values = (sums[depth]/(len(indices)*64)).cpu().tolist()
            rows.append(dict(depth=depth+1, ce_nats=values[0], target_entropy_nats=values[1],
                excess_ce_nats=values[0]-values[1], predicted_entropy_nats=values[2],
                hard_sample_top1_accuracy=values[3], correct_class_ce_gain_nats=values[4],
                target_atom_probability=values[5] if name != 'original_rq' else None,
                correct_sign_probability_given_target_atom=values[6] if name != 'original_rq' else None,
                ce_components=dict(atom=values[7], sign_given_atom=values[8], magnitude_given_signed_atom=values[9]),
                excess_ce_components=dict(atom=values[7]-values[10], sign_given_atom=values[8]-values[11],
                    magnitude_given_signed_atom=values[9]-values[12]),
                ce_by_temperature={str(t):float(ce_temperature[depth,k]/(len(indices)*64)) for k,t in enumerate(TEMPERATURES)},
                target_probability_mass_retained_by_top_k={str(k):float(kept_target_mass[depth,j]/(len(indices)*64)) for j,k in enumerate(TOP_K)}))
        report['splits'][split] = dict(images=len(indices), indices=indices, per_depth=rows,
            mean_ce_nats=sum(r['ce_nats'] for r in rows)/4,
            mean_excess_ce_nats=sum(r['excess_ce_nats'] for r in rows)/4)
        write_json(OUT/(name+'.json'), report)
        print(json.dumps(dict(case=name, split=split, ce=report['splits'][split]['mean_ce_nats'])), flush=True)
    fit = report['splits']['calibration_train']['per_depth']
    val = report['splits']['heldout_val']['per_depth']
    chosen = [min(row['ce_by_temperature'], key=row['ce_by_temperature'].get) for row in fit]
    report['temperature_calibration'] = dict(chosen_on='calibration_train',
        temperatures=[float(t) for t in chosen],
        heldout_original_ce=sum(row['ce_nats'] for row in val)/4,
        heldout_calibrated_ce=sum(row['ce_by_temperature'][t] for row,t in zip(val,chosen))/4,
        generation_fid_improvement_measured=False)
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device)/1024**3
    write_json(OUT/(name+'.json'), report)
    write_json(OUT/(name+'-final.json'), report)
    del model, aux, data
    gc.collect();torch.cuda.empty_cache()
    return report


def main():
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    # Share the device with training; bound the diagnostic's own allocations.
    torch.cuda.set_per_process_memory_fraction(9.5*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    # The legacy upstream autocast explicitly defaults its cache back to True.
    # Change that context only in this separate diagnostic process: arithmetic
    # and parameter dtypes stay identical, without caching the entire model.
    def uncached_autocast(enabled=True):
        return torch.autocast('cuda', enabled=enabled, dtype=torch.float16, cache_enabled=False)
    rq_upstream.RQTransformer.forward.__globals__['autocast'] = uncached_autocast
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(261016343)
    training = rng.choice(1281167, 128, replace=False).tolist()
    validation = np.random.default_rng(261015).choice(50000, 128, replace=False).tolist()
    reports = {}
    for name, base in CASES.items():
        reports[name] = measure(name, base, training, validation, device)
    write_json(OUT/'summary.json', dict(passed=True, global_step=626,
        same_images_labels_and_center_crop_transforms=True,
        original_soft_targets_and_stochastic_residual_prefixes=True,
        temperature_fitted_on_training_images_only=True,
        generation_fid_improvement_measured=False,
        limitations=['128 images per population; one checkpoint and stochastic target draw per model.',
            'Excess CE removes label entropy but differing tokenizations still have different prediction tasks.',
            'Top-k target coverage is teacher-forced; it is not a generation-quality measurement.'],
        cases={name:dict(mean_heldout_ce=x['splits']['heldout_val']['mean_ce_nats'],
            mean_heldout_excess_ce=x['splits']['heldout_val']['mean_excess_ce_nats'],
            temperature_calibration=x['temperature_calibration']) for name,x in reports.items()},time=time.time()))


if __name__ == '__main__':
    main()
