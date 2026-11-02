#!/usr/bin/env python3
"""Measure token marginals, local stability and final-refit dependence."""
import gc
import json
from pathlib import Path
import time

from audit_compound512_gap import BASE, ROOT, OUT, CompoundReplayRecipe, RQRecipe, torch
from src.training.laser_hard_ar import sparse_encoding
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config


@torch.inference_mode()
def measure(case, device):
    is_rq = case == 'original_rq'
    base = Path('/tmp/laser-original-rq-imagenet-20261014') if is_rq else BASE
    profile = RQRecipe(base) if is_rq else CompoundReplayRecipe(base)
    aux, _ = profile.tokenizer(device)
    aux.eval().requires_grad_(False)
    reference = json.loads((ROOT/'outputs/imagenet-laser-joint-predictability-20261016/original_rq-final.json').read_text())
    config = load_config(base/'inputs/stage2-training-config.yaml')
    counts = {name:torch.zeros(4, 16384, dtype=torch.float64) for name in ['fit_train96', 'select_train32', 'heldout_val128']}
    noise_levels = [.001, .01, .05]
    stability = {str(level):torch.zeros(4, dtype=torch.float64) for level in noise_levels}
    overlap = {str(level):0. for level in noise_levels}
    refit_delta = torch.zeros(3, 3, dtype=torch.float64)
    refit_signs = torch.zeros(3, 3, dtype=torch.float64)
    denominator = 128*64
    if not is_rq:
        gram = aux.dictionary.float().T@aux.dictionary.float()
    for split, label in [('train', 'calibration_train'), ('val', 'heldout_val')]:
        data = TarImageNet(base/'data', split, create_transforms(config.dataset, split='val'))
        indices = reference['splits'][label]['indices']
        for first in range(0, 128, 4):
            images = torch.stack([data[index][0] for index in indices[first:first+4]]).to(device)
            torch.manual_seed(261016001+first)
            targets, codes = profile.targets(aux, images)
            group = 'heldout_val128' if split == 'val' else 'fit_train96' if first < 96 else 'select_train32'
            if is_rq:
                counts[group] += targets.double().sum((0, 1, 2)).cpu()
            else:
                atoms = targets['atoms']
                counts[group] += torch.stack([torch.bincount(atoms[..., d].flatten(), minlength=16384) for d in range(4)]).cpu()
            if split == 'val':
                z = aux.encode(images) if is_rq else aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).contiguous()
                if is_rq:
                    _, hard = aux.quantizer.quantize(aux.quantizer.to_code_shape(z))
                else:
                    hard, final_values = sparse_encoding(z, aux.dictionary, 4, gram=gram)
                    correlation = torch.einsum('bhwc,bhwdc->bhwd', z, aux.dictionary.T[hard])
                    selected_gram = gram[hard[..., :, None], hard[..., None, :]]
                    for prefix in [1, 2, 3]:
                        values = torch.linalg.solve(selected_gram[..., :prefix, :prefix], correlation[..., :prefix, None])[..., 0]
                        delta = (final_values[..., :prefix]-values)/aux.coeff_scales[:prefix]
                        refit_delta[prefix-1, :prefix] += delta.square().double().sum((0, 1, 2)).cpu()
                        refit_signs[prefix-1, :prefix] += (values.sign() != final_values[..., :prefix].sign()).double().sum((0, 1, 2)).cpu()
                torch.manual_seed(261016488+first)
                direction = torch.randn_like(z)
                rms = z.square().mean((-1, -2, -3), keepdim=True).sqrt()
                for level in noise_levels:
                    perturbed = z+level*rms*direction
                    if is_rq:
                        _, changed = aux.quantizer.quantize(aux.quantizer.to_code_shape(perturbed))
                    else:
                        changed, _ = sparse_encoding(perturbed, aux.dictionary, 4, gram=gram)
                    stability[str(level)] += (hard == changed).double().sum((0, 1, 2)).cpu()
                    overlap[str(level)] += float((changed[..., :, None] == hard[..., None, :]).any(-1).double().sum())
                del z, direction, hard, changed
            del images, targets, codes
            write_json(OUT/'tokenizer-progress.json', dict(case=case, split=split, images=first+4, time=time.time()))
    alphas = [.001, .01, .1, 1., 10., 100.]
    def score(p, target):
        return -(target*p.log()).sum(-1)/target.sum(-1)
    candidates = [(counts['fit_train96']+alpha)/(counts['fit_train96'].sum(-1, keepdim=True)+alpha*16384) for alpha in alphas]
    selection = torch.stack([score(p, counts['select_train32']) for p in candidates])
    chosen = selection.argmin(0)
    prior = torch.stack([candidates[int(chosen[d])][d] for d in range(4)])
    heldout = score(prior, counts['heldout_val128'])
    all_counts = sum(counts.values())
    empirical = all_counts/all_counts.sum(-1, keepdim=True)
    report = dict(case=case, identical_center_crop_indices=True, images=256,
        unconditional_prior=dict(fit_images=96, selection_images=32, heldout_images=128,
            chosen_dirichlet_alpha_per_depth=[alphas[int(index)] for index in chosen],
            heldout_cross_entropy_per_depth=heldout.tolist(),
            empirical_marginal_entropy_biased_downward=(-(empirical*empirical.clamp_min(1.e-30).log()).sum(-1)).tolist()),
        latent_stability=dict(perturbation='independent Gaussian, RMS relative to each image latent RMS',
            validation_images=128, per_depth_exact_id_agreement={k:(v/denominator).tolist() for k,v in stability.items()},
            unordered_support_overlap={k:v/(4*denominator) for k,v in overlap.items()}),
        training_updates_added=0, stage1_unchanged=True)
    if not is_rq:
        report['final_refit_dependence'] = dict(
            normalized_rms_coefficient_change_from_future_support=(refit_delta/denominator).sqrt().tolist(),
            sign_change_fraction=(refit_signs/denominator).tolist(),
            rows='prefix lengths 1,2,3; columns are coefficients 1,2,3; unused entries zero',
            interpretation='A valid deterministic code, but emitted earlier coefficients depend on future selected supports.')
    write_json(OUT/f'tokenizer-{case}.json', report)
    torch.save(counts, OUT/f'token-counts-{case}.pt')
    print(json.dumps(report), flush=True)
    del aux
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(5*1024**3/total, device)
    for case in ['compound512', 'original_rq']:
        measure(case, device)
