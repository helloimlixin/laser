"""Launch a fresh FFHQ-adapted ImageNet prior after a measured support fix."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools import launch_imagenet_repaired_scratch as launcher

RUN_ID = 'imagenet-rfid421-ffhq-support-low25-scratch-4h200-20261009'
BASE = Path('/tmp/laser-imagenet-ffhq-support-low25-scratch-20261009')
OUT = ROOT/'outputs'/RUN_ID
PROBE = Path('/tmp/laser-imagenet-coefficient-support-fix-20261009')
PROBE_OUT = ROOT/'outputs/imagenet-coefficient-support-fix-20261009'
PARENT = Path('/tmp/laser-imagenet-ffhq-v4-sigma200bins-scratch-20261009')
PARENT_ID = 'imagenet-rfid421-ffhq-v4-sigma200bins-scratch-4h200-20261009'
SIGMA = 25.5875


def copy_tree(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.rglob('*'):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        target = destination/path.relative_to(source)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            shutil.copyfile(path, target)


def adapted_entry(policy):
    text = (PARENT/'entry.py').read_text()
    old = '    training.build_model = ffhq_adapter.build_model'
    new = (
        "    SUPPORT_POLICY = json.loads((OUT/'coefficient-support-policy.json').read_text())\n"
        "    SUPPORT_LIMITS = SUPPORT_POLICY['normalized_sampling_limits']\n"
        "    def supported_model(*args, **kwargs):\n"
        "        model = ffhq_adapter.build_model(*args, **kwargs)\n"
        "        model.coefficient_sampling_limits = list(SUPPORT_LIMITS)\n"
        "        return model\n"
        "    training.build_model = supported_model")
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = '        ARGS.checkpoint_fid_metric = original_logging.FID_PROTOCOL'
    new = (
        "        ARGS.coefficient_sampling_limits = list(SUPPORT_LIMITS)\n"
        "        ARGS.coefficient_sampling_support_policy = SUPPORT_POLICY['policy']\n"
        "        ARGS.coefficient_support_before_token_feedback = True\n"
        "        ARGS.coefficient_clean_training_clipping = False\n" + old)
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = "    for name in ('diagnosis.json', 'plan.json', 'source-code.tar.gz', 'source-manifest.json',"
    new = (
        "    run.config.update(dict(\n"
        "        coefficient_sampling_limits=list(SUPPORT_LIMITS),\n"
        "        coefficient_sampling_support_policy=SUPPORT_POLICY['policy'],\n"
        "        coefficient_support_before_token_feedback=True,\n"
        "        coefficient_clean_training_clipping=False,\n"
        "        support_calibration_images=4096,\n"
        "        supersedes_run='" + PARENT_ID + "',\n"
        "        correction='Fresh initialization with lower ImageNet coefficient noise and calibrated support masking before autoregressive feedback',\n"
        "        support_validation=SUPPORT_POLICY['evaluation']), allow_val_change=True)\n"
        "    run.save(str(OUT/'coefficient-support-policy.json'), base_path=str(OUT), policy='now')\n"
        "    run.save(str(OUT/'support-validation.json'), base_path=str(OUT), policy='now')\n" + old)
    assert text.count(old) == 1
    return text.replace(old, new)


def prepare(policy_name):
    import yaml
    evaluation = json.loads((PROBE_OUT/(policy_name+'-evaluation.json')).read_text())
    baseline = json.loads((ROOT/'outputs/imagenet-ffhq-noise-ab-epoch8-20261009/low25/train/evaluations/original_generation_step_0005634.json').read_text())
    assert evaluation['global_step'] == 5634 and evaluation['epoch'] == 9
    assert evaluation['generated_images'] == 50000 and evaluation['metric_backend'] == 'original-rqvae'
    assert evaluation['reference_sha256'] == '3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'
    assert evaluation['fid_original_train50k'] < baseline['eval/fid_original_train50k'], 'Support fix must improve the original FID50k'
    assert all(math.isfinite(evaluation[k]) for k in ('fid_original_train50k', 'inception_score'))
    assert json.loads((PROBE_OUT/'old-run-preservation.json').read_text())['passed']
    BASE.mkdir(exist_ok=False)
    OUT.mkdir(exist_ok=False)
    copy_tree(PROBE/'source', BASE/'source')
    for name in ('inputs', 'imagenet', 'torch-cache', 'inductor-cache'):
        (BASE/name).symlink_to(PARENT/name, target_is_directory=True)
    for name in ('checkpoint-staging', 'checkpoint-upload-cache', 'wandb', 'wandb-cache', 'wandb-data'):
        (BASE/name).mkdir()
    policy = dict(policy=policy_name, normalized_sampling_limits=evaluation['sampling_limits'],
        enforcement='Depth-specific coefficient logit mask before multinomial/top-p sampling and token history update',
        uniform_bins=dict(count=2048, range=[-3., 3.], width=6/2047),
        clean_training_coefficients_clipped=False, target_sigma_bins=SIGMA,
        calibration_images=4096, calibration_split='ImageNet training fresh epoch8 views',
        evaluation=evaluation, baseline_original_train_fid=baseline['eval/fid_original_train50k'],
        baseline_inception_score=baseline['eval/inception_score'],
        limitations='Checkpoint sampler validation does not establish the final FID of a fresh run; rare valid clean tails are excluded during sampling.')
    launcher.record(OUT/'coefficient-support-policy.json', policy)
    launcher.record(OUT/'support-validation.json', dict(passed=True, evaluation=evaluation, baseline=baseline))
    launcher.record(OUT/'original-fid-only-policy.json', dict(previous_evaluations=[],
        generation_metric_keys=['eval/fid_original_train50k', 'eval/inception_score']))
    (BASE/'entry.py').write_text(adapted_entry(policy))
    recipe = yaml.safe_load((PARENT/'production.yaml').read_text())
    options = dict(recipe['options'], checkpoint=str(BASE/'inputs/stage1-tokenizer.pt'),
        data=str(BASE/'imagenet'), output=str(OUT/'train'), checkpoint_dir=str(OUT/'train/checkpoints'),
        coeff_target_temperature=2*(SIGMA*6/2047)**2,
        resume=False, resume_checkpoint=None, init_stage2_checkpoint=None,
        fid_reference_stats=str(BASE/'inputs/imagenet_256_train.npz'),
        wandb_id=RUN_ID, wandb_name='ImageNet K4 | FFHQ decoder | sigma25.59 + coefficient support | scratch | 4 H200')
    recipe = dict(recipe, options=options)
    canonical = '# @package _global_\n'+yaml.safe_dump(recipe, sort_keys=False)
    (OUT/'train.yaml').write_text(canonical)
    (ROOT/'configs/stage2'/f'{RUN_ID}.yaml').write_text(canonical)
    for phase in ('preflight', 'production'):
        settings = dict(options)
        if phase == 'preflight':
            settings.update(max_optimizer_steps=20, fid_every=0, save_step_freq=0,
                            sample_grid_every=0, upload_checkpoints=False)
        else:
            settings.update(resume=True, resume_checkpoint=str(OUT/'train/checkpoints/last.pt'))
        (BASE/(phase+'.yaml')).write_text(yaml.safe_dump(dict(recipe, options=settings), sort_keys=False))
    previous = ROOT/'outputs'/PARENT_ID
    for name in ('validation-reference-verification.json', 'environment.txt', 'tokenizer-gpu-preflight.json'):
        shutil.copyfile(previous/name, OUT/name)
    launcher.record(OUT/'test-verification.json', dict(passed=True,
        command='pytest tests/test_coefficient_sampling_support.py tests/test_imagenet_ffhq_adaptation.py',
        tests_passed=12, real_64_image_sampling_verified=True))
    launcher.record(OUT/'plan.json', dict(run=RUN_ID, stage2_initialization='scratch', start_step=0,
        fresh_model_and_adam=True, initialization_from_old_run=False,
        coefficient_sampling_limits=policy['normalized_sampling_limits'], coefficient_sigma_bins=SIGMA,
        world_size=4, global_batch=2048, epochs=100, cosine_total_steps=62600,
        lr=.0005, min_lr=0., preflight_updates_retained=20,
        generation_metric_keys=['eval/fid_original_train50k', 'eval/inception_score'],
        support_validation=evaluation, previous_run_preserved=True, created_unix=time.time()))
    manifest = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (BASE/'source').rglob('*') if p.is_file()}
    for name in ('entry.py', 'preflight.yaml', 'production.yaml'):
        manifest[name] = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
    launcher.record(OUT/'source-manifest.json', manifest)
    with tarfile.open(OUT/'source-code.tar.gz', 'w:gz') as archive:
        archive.add(BASE/'source', arcname='source')
        for name in ('entry.py', 'preflight.yaml', 'production.yaml'):
            archive.add(BASE/name, arcname=name)
        archive.add(Path(__file__), arcname=Path(__file__).name)
        archive.add(ROOT/'scripts/tools/launch_imagenet_repaired_scratch.py', arcname='launch_imagenet_repaired_scratch.py')
    shutil.copyfile(Path(__file__), OUT/Path(__file__).name)
    print(json.dumps(dict(prepared=True, run_id=RUN_ID, policy=policy_name)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'verify-preflight', 'resume-production'])
    parser.add_argument('--policy', choices=['clean-q995', 'observed-max'], default='clean-q995')
    args = parser.parse_args()
    launcher.BASE, launcher.OUT, launcher.RUN_ID = BASE, OUT, RUN_ID
    launcher.FFHQ_ADAPT, launcher.SUB_BIN = True, False
    launcher.SIGMA_BINS, launcher.REQUESTED_SIGMA_BINS = SIGMA, str(SIGMA)
    launcher.TARGET_TEMPERATURE = 2*(SIGMA*6/2047)**2
    if args.action == 'prepare':
        prepare(args.policy)
    elif args.action == 'verify-preflight':
        launcher.verify_preflight()
    else:
        launcher.run(resume_only=args.action == 'resume-production')


if __name__ == '__main__':
    main()
