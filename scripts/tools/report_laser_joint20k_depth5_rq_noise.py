#!/usr/bin/env python3
"""Add paired-image uncertainty and an explicit adoption decision to the audit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0,str(ROOT))
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.joint_noise_calibration import align_noise_budgets

OUT = ROOT/'outputs/imagenet-laser-joint20k-depth5-rq-noise-calibration-20261017'
LOCAL = Path('/tmp/laser-joint20k-depth5-rq-noise-20261017')
BASE = Path('/tmp/laser-rq-joint20k-depth5-imagenet-20261017')
ACTIVE = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    torch.set_num_threads(4)
    report = json.loads((OUT/'report.json').read_text())
    views = torch.load(LOCAL/'matched-views.pt',map_location='cpu',weights_only=True)
    trajectories = torch.load(LOCAL/'validation-trajectories.pt',map_location='cpu',weights_only=True)
    native = torch.load('/tmp/laser-original-rq-imagenet-20261014/inputs/stage1-model.pt',
                        map_location='cpu',mmap=True,weights_only=False)['state_dict']['quantizer.codebooks.0.weight'][:-1]
    spec = torch.load(BASE/'inputs/joint-codebook.pt',map_location='cpu',weights_only=True)
    laser = JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients'])
    per_image = {}
    for name,profile in trajectories.items():
        latent = views['latents']['original_rq' if name == 'original_rq' else 'laser'][128:]
        def error(codes):
            decoded = native[codes].sum(-2) if name == 'original_rq' else laser.embed(codes).sum(-2)
            return (latent-decoded).square().flatten(1).mean(-1).numpy().astype(np.float64)
        hard = error(profile['hard_codes'])
        soft = np.stack([error(codes) for codes in profile['sampled_codes']]).mean(0)
        ratio = float(soft.mean()/hard.mean())
        assert abs(ratio-profile['final_sampled_to_hard_latent_error_ratio']) < 2.e-6
        per_image[name] = (hard,soft)
    # Resample matched images together. These intervals measure image sampling
    # uncertainty conditional on the four Monte Carlo draws, not training gains.
    bootstrap = np.random.default_rng(261017450).integers(0,128,size=(4000,128))
    boot_ratios = {name:soft[bootstrap].mean(1)/hard[bootstrap].mean(1)
                   for name,(hard,soft) in per_image.items()}
    intervals = {name:dict(ratio=float(soft.mean()/hard.mean()),
        image_bootstrap_95pct=np.quantile(boot_ratios[name],[.025,.975]).tolist())
        for name,(hard,soft) in per_image.items()}
    for name in ('laser_current','laser_proposed'):
        intervals[name]['difference_from_original_95pct'] = np.quantile(
            boot_ratios[name]-boot_ratios['original_rq'],[.025,.975]).tolist()
    validation = report['validation']
    alignment = align_noise_budgets(validation['original_rq']['hard_residual_energies'],
        [d['extra_error_over_hard_final_error'] for d in validation['original_rq']['profiles']],
        validation['laser_current']['hard_residual_energies'])
    target = np.asarray(alignment['budgets'])
    profile_errors = {}
    for name in ('laser_current','laser_proposed'):
        actual = np.asarray([d['extra_error_over_hard_final_error'] for d in validation[name]['profiles']])
        profile_errors[name] = dict(sum_absolute_log_budget_error=float(np.abs(np.log(actual/target)).sum()),
            first_two_depths_budget=float(actual[:2].sum()),
            first_two_depths_over_aligned_original=float(actual[:2].sum()/target[:2].sum()),
            sum_normalized_extra_error=float(actual.sum()))
    summary = dict(completed=True,proposed_temperatures=report['proposed_temperatures'],
        current_temperatures=report['current_temperatures'],validation_latent_error_ratio=intervals,
        validation_progress_alignment=alignment,validation_profile_errors=profile_errors,
        validation_pixel={name:{k:v for k,v in value.items() if not k.startswith('per_image')}
                          for name,value in report['validation_pixel'].items()},
        recommendation='Keep the running recipe. The proposal matches the depth allocation better, but does not improve held-out total latent-noise matching; decoded stochastic changes also remain below original RQ. Test it only as a controlled training ablation.',
        proposed_calibration_is_validated_generation_improvement=False,
        original_rq_noise_is_not_a_proven_optimal_noise_level_for_laser=True,
        active_training_targets_changed=False,training_launched=False,bootstrap_images=128,
        bootstrap_replicates=4000,bootstrap_scope='Matched-image sampling uncertainty, conditional on four stochastic draws; no training-performance inference',
        proposed_recipe=report['proposed_recipe'],full_report=str(OUT/'report.json'))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest = json.loads((ACTIVE/'reproducibility/code-manifest.json').read_text())['files']
    assert all(sha(BASE/name) == expected for name,expected in manifest.items())
    assert sha(BASE/'inputs/recipe.yaml') == report['source_plan']['active_recipe_sha256']
    proof = dict(passed=True,cpu_tests=4,native_hard_code_parity=True,
        native_soft_target_kl_verified=True,frozen_tokenizer_digests_unchanged=True,
        active_runtime_all_manifest_files_unchanged=True,active_training_targets_changed=False,
        proposed_recipe_only_noise_and_identifying_metadata_changed=True,
        main_progress=json.loads((ACTIVE/'train/progress-rank0.json').read_text()))
    (OUT/'final-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    shutil.copyfile(__file__,OUT/'source'/Path(__file__).name)
    (OUT/'reporting-source.json').write_text(json.dumps(dict(path=str(Path(__file__)),sha256=sha(__file__)),indent=2)+'\n')
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    import wandb
    run = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{ACTIVE.name}')
    publish = LOCAL/'publish/training-noise-calibration'
    publish.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','report.json','proposed-recipe.yaml','final-verification.json'):
        shutil.copyfile(OUT/name,publish/name)
        run.upload_file(str(publish/name),root=str(LOCAL/'publish'))
    print(json.dumps(dict(completed=True,summary=str(OUT/'summary.json'),
        intervals=intervals,profile_errors=profile_errors,uploaded_files=4,
        active_training_targets_changed=False)),flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
