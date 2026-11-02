"""Restart the clipped ImageNet prior with a measured smaller coefficient noise."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools import launch_imagenet_repaired_scratch as launcher
from scripts.tools.launch_imagenet_coefficient_support_scratch import copy_tree

RUN_ID = 'imagenet-rfid421-ffhq-q995-clipped-sigma42p65-scratch-4h200-20261009'
BASE = Path('/tmp/laser-imagenet-ffhq-q995-clipped-low-noise-scratch-20261009')
OUT = ROOT / 'outputs' / RUN_ID
PREVIOUS_ID = 'imagenet-rfid421-ffhq-q995-clipped-sigma170p58-scratch-4h200-20261009'
PREVIOUS = ROOT / 'outputs' / PREVIOUS_ID
PREVIOUS_BASE = Path('/tmp/laser-imagenet-ffhq-q995-clipped-scratch-20261009')
AUDIT = ROOT / 'outputs/imagenet-ffhq-q995-low-noise-policy-20261009'
SIGMA_NORMALIZED = .125
SIGMA_BINS = SIGMA_NORMALIZED / (6 / 2047)
TARGET_TEMPERATURE = 2 * SIGMA_NORMALIZED**2


def entry():
    text = (PREVIOUS_BASE / 'entry.py').read_text()
    replacements = {
        "supersedes_run='imagenet-rfid421-ffhq-support-low25-scratch-4h200-20261009'":
            f"supersedes_run='{PREVIOUS_ID}'",
        'coefficient_noise_sigma_normalized=.5, coefficient_noise_to_calibrated_radius=1/6':
            'coefficient_noise_sigma_normalized=.125, coefficient_noise_to_calibrated_radius=1/24',
        'FFHQ-matched q99.5 calibration, normalized clean clamping and Gaussian sigma .5; fresh ImageNet stage2 initialization':
            'ImageNet-calibrated lower noise: sigma .125 with retained q99.5 scales, clean clipping and native sampler; fresh model and Adam',
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def prepare():
    import yaml
    validation = json.loads((AUDIT / 'calibration-validation.json').read_text())
    tests = json.loads((AUDIT / 'test-verification.json').read_text())
    assert tests['passed'] and tests['tests_passed'] == 15
    assert validation['passed'] and validation['validation_images'] == 4096
    assert validation['target_sigma_normalized'] == SIGMA_NORMALIZED
    assert validation['target_temperature'] == TARGET_TEMPERATURE
    assert .05 < validation['expected_fixed_atom_noise_relative_rms'] < .1
    assert all(.05 < d['sigma_over_clipped_rms'] < .1 for d in validation['depths'])
    previous_policy = json.loads((PREVIOUS / 'coefficient-support-policy.json').read_text())
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(
        validation['coefficient_scales'], previous_policy['coefficient_scales']))
    BASE.mkdir(exist_ok=False)
    OUT.mkdir(exist_ok=False)
    copy_tree(PREVIOUS_BASE / 'source', BASE / 'source')
    for name in ('inputs', 'imagenet', 'torch-cache', 'inductor-cache'):
        (BASE / name).symlink_to(PREVIOUS_BASE / name, target_is_directory=True)
    for name in ('checkpoint-staging', 'checkpoint-upload-cache', 'wandb', 'wandb-cache', 'wandb-data'):
        (BASE / name).mkdir()
    policy = dict(policy='imagenet-q995-clipped-low-noise', normalized_sampling_limits=[3.] * 4,
        coefficient_scales=validation['coefficient_scales'], clean_training_coefficients_clipped=True,
        physical_sampling_limits=[3 * s for s in validation['coefficient_scales']],
        target_sigma_bins=SIGMA_BINS, target_sigma_normalized=SIGMA_NORMALIZED,
        target_temperature=TARGET_TEMPERATURE, coefficient_quantizer='uniform',
        uniform_bins=dict(count=2048, range=[-3., 3.], width=6 / 2047),
        calibration_quantile=.995, calibration_images=4096, evaluation=validation,
        enforcement='Normalize with retained per-depth q99.5/3 scales; clamp clean targets to [-3,3] before noisy contexts and clean geometry targets; sample within the same finite bins.',
        evidence='Prior same-checkpoint noise trial favored lower noise; independent current-scale validation measures 7-8% coefficient RMS perturbation.',
        limitations='One-epoch earlier trial used different scales and had encoder numerical variation; the final FID of this scratch run is unproven.')
    launcher.record(OUT / 'coefficient-support-policy.json', policy)
    launcher.record(OUT / 'support-validation.json', validation)
    launcher.record(OUT / 'original-fid-only-policy.json', dict(previous_evaluations=[],
        generation_metric_keys=['eval/fid_original_train50k', 'eval/inception_score']))
    (BASE / 'entry.py').write_text(entry())
    recipe = yaml.safe_load((PREVIOUS_BASE / 'production.yaml').read_text())
    options = dict(recipe['options'], checkpoint=str(BASE / 'inputs/stage1-tokenizer.pt'),
        data=str(BASE / 'imagenet'), output=str(OUT / 'train'), checkpoint_dir=str(OUT / 'train/checkpoints'),
        coeff_scales=validation['coefficient_scales'], coeff_target_temperature=TARGET_TEMPERATURE,
        resume=False, resume_checkpoint=None, init_stage2_checkpoint=None,
        fid_reference_stats=str(BASE / 'inputs/imagenet_256_train.npz'), wandb_id=RUN_ID,
        wandb_name='ImageNet K4 | q99.5 + clean clipping + sigma0.125 | fresh | 4 H200')
    recipe = dict(recipe, options=options)
    canonical = '# @package _global_\n' + yaml.safe_dump(recipe, sort_keys=False)
    (OUT / 'train.yaml').write_text(canonical)
    (ROOT / 'configs/stage2' / f'{RUN_ID}.yaml').write_text(canonical)
    for phase in ('preflight', 'production'):
        settings = dict(options)
        if phase == 'preflight':
            settings.update(max_optimizer_steps=20, fid_every=0, save_step_freq=0,
                            sample_grid_every=0, upload_checkpoints=False)
        else:
            settings.update(resume=True, resume_checkpoint=str(OUT / 'train/checkpoints/last.pt'))
        (BASE / (phase + '.yaml')).write_text(yaml.safe_dump(dict(recipe, options=settings), sort_keys=False))
    for name in ('validation-reference-verification.json', 'environment.txt', 'tokenizer-gpu-preflight.json'):
        shutil.copyfile(PREVIOUS / name, OUT / name)
    launcher.record(OUT / 'test-verification.json', tests)
    launcher.record(OUT / 'plan.json', dict(run=RUN_ID, supersedes_run=PREVIOUS_ID,
        stage2_initialization='scratch', fresh_model_and_adam=True, initialization_from_previous_run=False,
        coefficient_scales=validation['coefficient_scales'], coefficient_clipping=True,
        target_sigma_normalized=SIGMA_NORMALIZED, target_sigma_bins=SIGMA_BINS,
        target_temperature=TARGET_TEMPERATURE, noise_to_coefficient_rms=[d['sigma_over_clipped_rms'] for d in validation['depths']],
        physical_coefficient_limits=policy['physical_sampling_limits'], world_size=4, global_batch=2048,
        epochs=100, cosine_total_steps=62600, lr=.0005, min_lr=0., preflight_updates_retained=20,
        token_cache=None, fresh_images=True,
        generation_metric_keys=['eval/fid_original_train50k', 'eval/inception_score'],
        validation=validation, created_unix=time.time()))
    for relative in ('scripts/tools/calibrate_imagenet_ffhq_coefficient_policy.py',
                     'tests/test_ffhq_calibrated_coefficient_policy.py'):
        target = BASE / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    manifest = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (BASE / 'source').rglob('*') if p.is_file()}
    for name in ('entry.py', 'preflight.yaml', 'production.yaml'):
        manifest[name] = hashlib.sha256((BASE / name).read_bytes()).hexdigest()
    launcher.record(OUT / 'source-manifest.json', manifest)
    with tarfile.open(OUT / 'source-code.tar.gz', 'w:gz') as archive:
        archive.add(BASE / 'source', arcname='source')
        for name in ('entry.py', 'preflight.yaml', 'production.yaml'):
            archive.add(BASE / name, arcname=name)
        for name in ('coefficient-support-policy.json', 'support-validation.json', 'plan.json', 'test-verification.json'):
            archive.add(OUT / name, arcname=name)
        archive.add(Path(__file__), arcname=Path(__file__).name)
        archive.add(ROOT / 'scripts/tools/launch_imagenet_repaired_scratch.py', arcname='launch_imagenet_repaired_scratch.py')
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    print(json.dumps(dict(prepared=True, run_id=RUN_ID, target_sigma_bins=SIGMA_BINS)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'verify-preflight', 'resume-production'])
    args = parser.parse_args()
    launcher.BASE, launcher.OUT, launcher.RUN_ID = BASE, OUT, RUN_ID
    launcher.FFHQ_ADAPT, launcher.SUB_BIN = True, False
    launcher.SIGMA_BINS, launcher.REQUESTED_SIGMA_BINS = SIGMA_BINS, str(SIGMA_BINS)
    launcher.TARGET_TEMPERATURE = TARGET_TEMPERATURE
    if args.action == 'prepare':
        prepare()
    elif args.action == 'verify-preflight':
        launcher.verify_preflight()
    else:
        launcher.run(resume_only=args.action == 'resume-production')


if __name__ == '__main__':
    main()
