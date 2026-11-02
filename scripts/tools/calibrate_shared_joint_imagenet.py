#!/usr/bin/env python3
"""Calibrate stochastic targets for an exactly shared physical LASER book.

Uses matched training crops for the released RQ reference and LASER. Fits on
128 images and checks a disjoint 128, excluding all shared-book fitting views.
Temperatures change sampling probabilities only, never codeword vectors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cc3m-config', type=Path,
        help='Calibrate CC3M training crops and update this CC3M recipe')
    args = parser.parse_args()
    base, out = args.base.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'report.json').exists(), 'Refuse overwriting a completed calibration'
    os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')
    sys.path[:0] = [str(base/'source'), str(base/'upstream'), str(ROOT)]
    import numpy as np
    import torch
    import yaml
    from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
    from src.training.laser_shared_joint_vocabulary import assert_shared_codewords
    from src.training.joint_noise_calibration import (
        hard_trajectory, measure_noise, public_profile,
        align_noise_budgets, fit_depth_temperatures)
    from src.training.rq_imagenet_recipes import RQRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config

    def write(name, value):
        path = out/name
        pending = path.with_name(path.name+'.pending')
        pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        pending.replace(path)

    def status(phase, **values):
        write('status.json', dict(phase=phase, pid=os.getpid(), time=time.time(), **values))
        print(json.dumps(dict(phase=phase, **values)), flush=True)

    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(device).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(26101920481)
    recipe = yaml.safe_load((base/'inputs/recipe.yaml').read_text())
    stage1_hash = digest(base/'inputs/stage1-model.pt')
    book_hash = digest(base/'inputs/joint-codebook.pt')
    assert stage1_hash == recipe['tokenizer']['sha256']
    assert book_hash == recipe['tokenizer']['codebook_sha256']
    assert digest(base/'native/inputs/stage1-model.pt') == 'e8eb312e6c0dbd8e490ca95f69e3c3b417162625ff9320bbfff791cbf832b694'
    status('loading_frozen_tokenizers')
    native, _ = RQRecipe(base/'native').tokenizer(device)
    laser = FrozenJointLaserTokenizer(base/'inputs/stage1-model.pt', base/'inputs/joint-codebook.pt').to(device)
    assert_shared_codewords(laser.quantizer)
    physical = laser.quantizer.expanded_codebook(0)
    assert all(torch.equal(physical, laser.quantizer.expanded_codebook(d)) for d in range(5))
    physical_hash = hashlib.sha256(physical.cpu().numpy().tobytes()).hexdigest()
    del physical
    initial_versions = [[v._version for v in m.state_dict().values()] for m in (native, laser)]

    class NativeBook:
        depth = 4

        def scores(self, residual, depth):
            return -native.quantizer.codebooks[depth].compute_distances(residual)

        def embed_depth(self, codes, depth):
            return native.quantizer.codebooks[depth].embed(codes)

    native_book = NativeBook()
    excluded = set()
    cc3m = yaml.safe_load(args.cc3m_config.read_text()) if args.cc3m_config else None
    exclusion_sources = [] if cc3m else [
        (ROOT/'outputs/imagenet-laser-sharedjoint20k-depth5-codebook-20261018/calibration-data.json',
         ['fit_indices', 'heldout_training_indices']),
        (ROOT/'outputs/imagenet-laser-joint20k-depth5-rq-noise-calibration-20261017/data.json',
         ['fit_indices', 'validation_indices']),
        (ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017/calibration-data.json',
         ['fit_indices', 'heldout_training_indices']),
    ]
    for path, keys in exclusion_sources:
        data = json.loads(path.read_text())
        for key in keys:
            excluded.update(data[key])
    if cc3m:
        from src.data.cc3m_joint import CC3MJointDataset, epoch_order
        assert (base/'initial-shards-ready.json').is_file()
        manifest = json.loads(Path(cc3m['data_manifest']).read_text())
        dataset = CC3MJointDataset(manifest['train'], cc3m['data_source'], cc3m['data_local'],
            train=True, seed=cc3m['seed'], epoch=0)
        shard_order = np.random.default_rng(np.random.SeedSequence([cc3m['seed'], 0])).permutation(len(manifest['train']))
        pool_size = sum(manifest['train'][int(i)]['items'] for i in shard_order[:cc3m['shuffle_group_shards']])
        pool = epoch_order(manifest['train'], cc3m['seed'], 0, cc3m['shuffle_group_shards'])[:pool_size]
        indices = np.random.default_rng(26101920481).choice(pool, 256, replace=False).tolist()
        augmentation_seed = cc3m['seed']
        fitting_dataset = 'CC3M; first randomly shuffled training shard group'
    else:
        config = load_config(base/'inputs/stage2-training-config.yaml')
        dataset = TarImageNet(base/'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
        dataset.set_epoch(0)
        indices = [int(i) for i in np.random.default_rng(26101920481).permutation(len(dataset)) if i not in excluded][:256]
        augmentation_seed, fitting_dataset = 261001, 'ImageNet'
    assert len(set(indices)) == 256 and not set(indices) & excluded
    images, native_latent, laser_latent = [], [], []
    status('encoding_matched_training_crops', fit_images=128, heldout_images=128)
    for start in range(0, len(indices), 8):
        image = torch.stack([dataset[i][0] for i in indices[start:start+8]]).to(device)
        images.append(image.cpu())
        native_latent.append(native.encode(image).cpu())
        laser_latent.append(laser.encode(image).cpu())
    images = torch.cat(images)
    latents = dict(native=torch.cat(native_latent), laser=torch.cat(laser_latent))
    torch.save(dict(indices=indices, images=images, latents=latents), base/'calibration-matched-views.pt')
    write('data.json', dict(fit_indices=indices[:128], validation_indices=indices[128:],
        fitting_split='train', heldout_split='train', fit_images=128, heldout_images=128,
        excluded_images=len(excluded), exclusion_sources=[str(p) for p,_ in exclusion_sources],
        same_augmented_images_for_both_encoders=True, image_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
        augmentation_seed=augmentation_seed, augmentation_epoch=0, fitting_dataset=fitting_dataset,
        validation_split_used_for_fitting=False))
    fit = {k:v[:128].to(device) for k,v in latents.items()}
    held = {k:v[128:].to(device) for k,v in latents.items()}
    hard = dict(native=hard_trajectory(native_book, fit['native']),
                laser=hard_trajectory(laser.quantizer, fit['laser']))
    seeds = [26101920500+i for i in range(4)]
    inherited = [0.15819901962280272, 0.13549085464477534, 0.08794634933471679,
                 0.055382978057861326, 0.0394969670343399]
    status('measuring_reference_and_inherited_noise')
    reference = measure_noise(native_book, fit['native'], [.5]*4, seeds, hard['native'])
    old = measure_noise(laser.quantizer, fit['laser'], inherited, seeds, hard['laser'])
    alignment = align_noise_budgets(hard['native']['energies'],
        [r['extra_error_over_hard_final_error'] for r in reference['profiles']], hard['laser']['energies'])
    target_ratio = reference['final_sampled_to_hard_latent_error_ratio']
    candidates = []

    def evaluate(scale):
        status('fitting_temperatures', budget_scale=scale, completed_candidates=len(candidates))
        temperatures, trace = fit_depth_temperatures(laser.quantizer, fit['laser'],
            [b*scale for b in alignment['budgets']], hard['laser']['energies'][-1], seed=26101920510)
        profile = measure_noise(laser.quantizer, fit['laser'], temperatures, seeds, hard['laser'])
        row = dict(budget_scale=scale, temperatures=temperatures, fit_trace=trace, profile=public_profile(profile),
            absolute_total_ratio_error=abs(profile['final_sampled_to_hard_latent_error_ratio']-target_ratio))
        candidates.append(row)
        write('candidates.json', candidates)

    for scale in (.5, 1., 1.5):
        evaluate(scale)
    for _ in range(3):
        ordered = sorted(candidates, key=lambda r:r['budget_scale'])
        brackets = [(a,b) for a,b in zip(ordered, ordered[1:])
            if (a['profile']['final_sampled_to_hard_latent_error_ratio']-target_ratio)*
               (b['profile']['final_sampled_to_hard_latent_error_ratio']-target_ratio) < 0]
        if brackets:
            a,b = min(brackets, key=lambda pair:pair[1]['budget_scale']-pair[0]['budget_scale'])
            scale = (a['budget_scale']+b['budget_scale'])/2
        else:
            best = min(candidates, key=lambda r:r['absolute_total_ratio_error'])
            scale = ordered[0]['budget_scale']/2 if best == ordered[0] else ordered[-1]['budget_scale']*2
        evaluate(scale)
    selected = min(candidates, key=lambda r:r['absolute_total_ratio_error'])
    temperatures = selected['temperatures']
    status('checking_independent_training_holdout', temperatures=temperatures)
    validation = {}
    held_seeds = [26101920520+i for i in range(4)]
    for key, q, x, temps in (
        ('original_rq', native_book, held['native'], [.5]*4),
        ('shared_inherited', laser.quantizer, held['laser'], inherited),
        ('shared_calibrated', laser.quantizer, held['laser'], temperatures)):
        validation[key] = measure_noise(q, x, temps, held_seeds)
    pixel = {}
    status('checking_decoded_corruption')
    for key, profile in validation.items():
        model = native if key == 'original_rq' else laser
        hard_error, sampled_error, change_error = [], [], []
        for start in range(0, 128, 8):
            real = images[128+start:128+start+8].to(device)
            hard_image = model.decode_code(profile['hard_codes'][start:start+8].to(device)).clamp(-1,1)
            hard_error.extend((hard_image-real).square().flatten(1).mean(-1).cpu().tolist())
            for codes in profile['sampled_codes']:
                sampled = model.decode_code(codes[start:start+8].to(device)).clamp(-1,1)
                sampled_error.extend((sampled-real).square().flatten(1).mean(-1).cpu().tolist())
                change_error.extend((sampled-hard_image).square().flatten(1).mean(-1).cpu().tolist())
        pixel[key] = dict(hard_mse=float(np.mean(hard_error)), stochastic_mse=float(np.mean(sampled_error)),
            sampled_to_hard_mse_ratio=float(np.mean(sampled_error)/np.mean(hard_error)),
            stochastic_vs_hard_mse=float(np.mean(change_error)))
    assert_shared_codewords(laser.quantizer)
    assert initial_versions == [[v._version for v in m.state_dict().values()] for m in (native, laser)]
    assert digest(base/'inputs/joint-codebook.pt') == book_hash
    ref_ratio = validation['original_rq']['final_sampled_to_hard_latent_error_ratio']
    old_ratio = validation['shared_inherited']['final_sampled_to_hard_latent_error_ratio']
    new_ratio = validation['shared_calibrated']['final_sampled_to_hard_latent_error_ratio']
    # Fit accuracy and a finite, modest holdout corruption budget gate launch.
    # Holdout is reported, never used to select among calibration candidates.
    passed = selected['absolute_total_ratio_error'] < .005 and abs(new_ratio-ref_ratio) < .01
    report = dict(completed=True, passed=passed, fit_images=128, heldout_images=128, monte_carlo_seeds=4,
        stage1_sha256=stage1_hash, codebook_sha256=book_hash, physical_table_sha256=physical_hash,
        physical_vectors_identical_at_all_depths=True, depth_multipliers=False,
        temperatures_only_change_target_probabilities=True, proposed_temperatures=temperatures,
        inherited_temperatures=inherited, reference=public_profile(reference), inherited=public_profile(old),
        alignment=alignment, selected=selected, selection_uses_holdout=False,
        validation={k:public_profile(v) for k,v in validation.items()}, decoded_corruption=pixel,
        original_rq_holdout_ratio=ref_ratio, inherited_holdout_ratio=old_ratio, calibrated_holdout_ratio=new_ratio,
        heldout_noise_match_improved=abs(new_ratio-ref_ratio) < abs(old_ratio-ref_ratio),
        encoder_decoder_and_codebook_unchanged=True,
        scope='Target corruption calibration; convergence and generation FID remain unmeasured.', time=time.time())
    write('report.json', report)
    assert passed, 'Noise calibration failed; inspect report before launch'
    recipe['targets']['temperature'] = temperatures
    recipe['targets']['calibration'] = dict(policy='Shared-book-specific original-RQ corruption matching',
        report=str(out/'report.json'), calibration_images=128, validation_images=128,
        selection_uses_validation=False, physical_codewords_rescaled=False,
        matching='Normalized hard-error progress and sampled-to-hard final latent error')
    if cc3m:
        cc3m['temperatures'] = temperatures
        cc3m['target_calibration'] = dict(recipe['targets']['calibration'], dataset='CC3M')
        args.cc3m_config.write_text(yaml.safe_dump(cc3m, sort_keys=False))
    else:
        (base/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
        training_path = base/'inputs/stage2-training-config.yaml'
        training = yaml.safe_load(training_path.read_text())
        training['loss']['temp'] = temperatures
        training_path.write_text(yaml.safe_dump(training, sort_keys=False))
    (base/'inputs/target-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    status('completed', passed=True, temperatures=temperatures, holdout_ratio=new_ratio, reference_ratio=ref_ratio)


if __name__ == '__main__':
    import torch
    with torch.no_grad():
        main()
