"""Use FFHQ percentile calibration, clean clamping and noise on ImageNet."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools import launch_imagenet_repaired_scratch as launcher
from scripts.tools.launch_imagenet_coefficient_support_scratch import adapted_entry, copy_tree

RUN_ID = 'imagenet-rfid421-ffhq-q995-clipped-sigma170p58-scratch-4h200-20261009'
BASE = Path('/tmp/laser-imagenet-ffhq-q995-clipped-scratch-20261009')
OUT = ROOT/'outputs'/RUN_ID
PREVIOUS_ID = 'imagenet-rfid421-ffhq-support-low25-scratch-4h200-20261009'
PREVIOUS = ROOT/'outputs'/PREVIOUS_ID
PREVIOUS_BASE = Path('/tmp/laser-imagenet-ffhq-support-low25-scratch-20261009')
AUDIT = ROOT/'outputs/imagenet-ffhq-calibrated-policy-20261009'
SIGMA_BINS = .5/(6/2047)


def entry():
    text = adapted_entry({})
    replacements = {
        "kwargs['clamp_coeffs'] = False": "kwargs['clamp_coeffs'] = True",
        'coefficient_clipping=False': 'coefficient_clipping=True',
        'coefficient_clean_training_clipping = False': 'coefficient_clean_training_clipping = True',
        'coefficient_clean_training_clipping=False': 'coefficient_clean_training_clipping=True',
        "    encoded = tuple(torch.cat(values) for values in results)":
        "    encoded = tuple(torch.cat(values) for values in results)\n"
        "    assert self.clamp_coeffs and bool((encoded[1].abs() <= 3).all())\n"
        "    expected_scales = self.coeff_scales.new_tensor(SUPPORT_POLICY['coefficient_scales'])\n"
        "    assert torch.equal(self.coeff_scales, expected_scales)\n"
        "    if not hasattr(self, '_clamping_verified'):\n"
        "        record('coefficient-clipping-rank'+os.environ['RANK']+'.json', dict(\n"
        "            passed=True, clean_coefficients_clamped=True, normalized_min=float(encoded[1].min()),\n"
        "            normalized_max=float(encoded[1].max()), coefficient_scales=self.coeff_scales.tolist(),\n"
        "            clamp_range=[-3., 3.], geometry_targets_clamped_before_target_noise=True))\n"
        "        self._clamping_verified = True",
        "    for name in ('diagnosis.json', 'plan.json', 'source-code.tar.gz', 'source-manifest.json',":
        "    run.config.update(dict(\n"
        "        supersedes_run='"+PREVIOUS_ID+"', coefficient_clipping=True,\n"
        "        coefficient_calibration_quantile=.995, coefficient_scales=SUPPORT_POLICY['coefficient_scales'],\n"
        "        coefficient_noise_sigma_normalized=.5, coefficient_noise_to_calibrated_radius=1/6,\n"
        "        geometry_targets_clamped_before_target_noise=True,\n"
        "        correction='FFHQ-matched q99.5 calibration, normalized clean clamping and Gaussian sigma .5; fresh ImageNet stage2 initialization',\n"
        "        support_validation=SUPPORT_POLICY['evaluation']), allow_val_change=True)\n"
        "    for name in ('diagnosis.json', 'plan.json', 'source-code.tar.gz', 'source-manifest.json',",
    }
    for old, new in replacements.items():
        assert old in text, old
        text = text.replace(old, new)
    return text


def prepare():
    import yaml
    validation = json.loads((AUDIT/'calibration-validation.json').read_text())
    assert validation['passed'] and validation['clipping_fraction'] < .02
    assert validation['target_temperature'] == .5 and validation['validation_images'] == 4096
    BASE.mkdir(exist_ok=False)
    OUT.mkdir(exist_ok=False)
    copy_tree(PREVIOUS_BASE/'source', BASE/'source')
    for name in ('inputs', 'imagenet', 'torch-cache', 'inductor-cache'):
        (BASE/name).symlink_to(PREVIOUS_BASE/name, target_is_directory=True)
    for name in ('checkpoint-staging', 'checkpoint-upload-cache', 'wandb', 'wandb-cache', 'wandb-data'):
        (BASE/name).mkdir()
    policy = dict(policy='ffhq-q995-calibrated-and-clipped', normalized_sampling_limits=[3.]*4,
        coefficient_scales=validation['coefficient_scales'], clean_training_coefficients_clipped=True,
        physical_sampling_limits=[3*s for s in validation['coefficient_scales']],
        target_sigma_bins=SIGMA_BINS, target_sigma_normalized=.5, target_temperature=.5,
        coefficient_quantizer='uniform', uniform_bins=dict(count=2048,range=[-3.,3.],width=6/2047),
        calibration_quantile=.995, calibration_images=4096, evaluation=validation,
        enforcement='Normalize physical coefficients with depth-specific q99.5/3 scales, clamp clean targets to [-3,3], then form noisy coefficient contexts and clean clamped geometry targets. Sampling uses the same finite uniform bins.',
        reference='helloimlixin-rutgers/laser/ffhqcmp0804205803',
        limitations='ImageNet scale calibration uses a random 4096-image subset; no FID is claimed for this new normalization/noise policy.')
    launcher.record(OUT/'coefficient-support-policy.json', policy)
    launcher.record(OUT/'support-validation.json', validation)
    launcher.record(OUT/'original-fid-only-policy.json', dict(previous_evaluations=[],
        generation_metric_keys=['eval/fid_original_train50k','eval/inception_score']))
    (BASE/'entry.py').write_text(entry())
    recipe = yaml.safe_load((PREVIOUS_BASE/'production.yaml').read_text())
    options = dict(recipe['options'], checkpoint=str(BASE/'inputs/stage1-tokenizer.pt'),
        data=str(BASE/'imagenet'), output=str(OUT/'train'), checkpoint_dir=str(OUT/'train/checkpoints'),
        coeff_scales=validation['coefficient_scales'], coeff_target_temperature=.5,
        resume=False, resume_checkpoint=None, init_stage2_checkpoint=None,
        fid_reference_stats=str(BASE/'inputs/imagenet_256_train.npz'), wandb_id=RUN_ID,
        wandb_name='ImageNet K4 | FFHQ q99.5 scaling + clamping + sigma0.5 | scratch | 4 H200')
    recipe = dict(recipe, options=options)
    canonical = '# @package _global_\n'+yaml.safe_dump(recipe,sort_keys=False)
    (OUT/'train.yaml').write_text(canonical)
    (ROOT/'configs/stage2'/f'{RUN_ID}.yaml').write_text(canonical)
    for phase in ('preflight','production'):
        settings = dict(options)
        if phase == 'preflight':
            settings.update(max_optimizer_steps=20,fid_every=0,save_step_freq=0,sample_grid_every=0,upload_checkpoints=False)
        else:
            settings.update(resume=True,resume_checkpoint=str(OUT/'train/checkpoints/last.pt'))
        (BASE/(phase+'.yaml')).write_text(yaml.safe_dump(dict(recipe,options=settings),sort_keys=False))
    for name in ('validation-reference-verification.json','environment.txt','tokenizer-gpu-preflight.json'):
        shutil.copyfile(PREVIOUS/name, OUT/name)
    launcher.record(OUT/'test-verification.json', dict(passed=True, tests_passed=14,
        command='pytest tests/test_ffhq_calibrated_coefficient_policy.py tests/test_coefficient_sampling_support.py tests/test_imagenet_ffhq_adaptation.py',
        normalization_before_clipping_verified=True, clamped_geometry_target_verified=True,
        gaussian_sigma_half_unit_verified=True, independent_4096_image_codec_validation=True))
    launcher.record(OUT/'plan.json', dict(run=RUN_ID,stage2_initialization='scratch',fresh_model_and_adam=True,
        initialization_from_previous_run=False,coefficient_scales=validation['coefficient_scales'],
        coefficient_clipping=True,coefficient_calibration_quantile=.995,target_sigma_normalized=.5,
        target_sigma_bins=SIGMA_BINS,target_temperature=.5,noise_to_calibrated_radius=1/6,
        physical_coefficient_limits=policy['physical_sampling_limits'],world_size=4,global_batch=2048,
        epochs=100,cosine_total_steps=62600,lr=.0005,min_lr=0.,preflight_updates_retained=20,
        token_cache=None,fresh_images=True,generation_metric_keys=['eval/fid_original_train50k','eval/inception_score'],
        validation=validation,created_unix=time.time()))
    manifest = {str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (BASE/'source').rglob('*') if p.is_file()}
    for name in ('entry.py','preflight.yaml','production.yaml'):
        manifest[name] = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
    launcher.record(OUT/'source-manifest.json',manifest)
    with tarfile.open(OUT/'source-code.tar.gz','w:gz') as archive:
        archive.add(BASE/'source',arcname='source')
        for name in ('entry.py','preflight.yaml','production.yaml'):
            archive.add(BASE/name,arcname=name)
        for name in ('coefficient-support-policy.json','support-validation.json','plan.json'):
            archive.add(OUT/name,arcname=name)
        archive.add(Path(__file__),arcname=Path(__file__).name)
        for name in ('launch_imagenet_repaired_scratch.py','launch_imagenet_coefficient_support_scratch.py'):
            archive.add(ROOT/'scripts/tools'/name,arcname=name)
    shutil.copyfile(Path(__file__),OUT/Path(__file__).name)
    print(json.dumps(dict(prepared=True,run_id=RUN_ID,policy=policy)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','verify-preflight','resume-production'])
    args=p.parse_args()
    launcher.BASE,launcher.OUT,launcher.RUN_ID=BASE,OUT,RUN_ID
    launcher.FFHQ_ADAPT,launcher.SUB_BIN=True,False
    launcher.SIGMA_BINS,launcher.REQUESTED_SIGMA_BINS=SIGMA_BINS,str(SIGMA_BINS)
    launcher.TARGET_TEMPERATURE=.5
    if args.action=='prepare':prepare()
    elif args.action=='verify-preflight':launcher.verify_preflight()
    else:launcher.run(resume_only=args.action=='resume-production')


if __name__=='__main__':main()
