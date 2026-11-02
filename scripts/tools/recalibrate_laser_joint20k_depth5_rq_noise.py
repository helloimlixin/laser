#!/usr/bin/env python3
"""Matched-image noise calibration against the released rFID4.73 RQ-VAE.

Produces a proposal only. Never edits or relaunches the active training job.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-joint20k-depth5-imagenet-20261017')
ORIGINAL = Path('/tmp/laser-original-rq-imagenet-20261014')
OUT = ROOT/'outputs/imagenet-laser-joint20k-depth5-rq-noise-calibration-20261017'
LOCAL = Path('/tmp/laser-joint20k-depth5-rq-noise-20261017')
ACTIVE = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(name, value):
    path = OUT/name
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def status(phase, **values):
    write('status.json', dict(phase=phase, pid=os.getpid(), time=time.time(), **values))


def worker():
    os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
    sys.path[:0] = [str(OUT/'source'), str(BASE/'source'), str(BASE/'runtime'), str(BASE/'upstream')]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from joint_noise_calibration import (hard_trajectory, measure_noise, public_profile,
        align_noise_budgets, fit_depth_temperatures)
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_recipes import RQRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(261017420)
    plan = json.loads((OUT/'plan.json').read_text())
    assert sha(ORIGINAL/'inputs/stage1-model.pt') == plan['original_stage1_sha256']
    assert sha(BASE/'inputs/joint-codebook.pt') == plan['joint_codebook_sha256']
    native, _ = RQRecipe(ORIGINAL).tokenizer(device)
    laser, _ = JointLaserRecipe(BASE).tokenizer(device)
    models = dict(original_rq=native, laser_current=laser, laser_proposed=laser)

    class NativeBook:
        depth = 4

        def scores(self, residual, depth):
            return -native.quantizer.codebooks[depth].compute_distances(residual)

        def embed_depth(self, codes, depth):
            return native.quantizer.codebooks[depth].embed(codes)

    native_book = NativeBook()
    assert all(not p.requires_grad for model in (native, laser) for p in model.parameters())

    def model_digest(model):
        digest = hashlib.sha256()
        for key, value in model.state_dict().items():
            digest.update(key.encode())
            digest.update(value.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    initial_digests = {name:model_digest(model) for name, model in [('original_rq',native),('laser',laser)]}
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(BASE/'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    data.set_epoch(0)
    excluded = set()
    exclusion_sources = [
        (ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017/calibration-data.json',
         ['fit_indices','heldout_training_indices']),
        (ROOT/'outputs/imagenet-laser-joint-vocabulary-improvement-20261016/per-depth-temperature-validation.json',
         ['validation_indices']),
        (BASE/'inputs/target-audit.json', ['training_indices']),
    ]
    for path, keys in exclusion_sources:
        value = json.loads(path.read_text())
        for key in keys:
            excluded.update(value[key])
    indices = [int(i) for i in np.random.default_rng(261017420).permutation(len(data)) if i not in excluded][:256]
    fit_indices, validation_indices = indices[:128], indices[128:]
    assert not (set(fit_indices) & set(validation_indices)) and not (set(indices) & excluded)
    images, native_latents, laser_latents = [], [], []
    status('encoding_matched_training_views', fit_images=128, validation_images=128)
    for first in range(0, len(indices), 4):
        image = torch.stack([data[i][0] for i in indices[first:first+4]]).to(device)
        images.append(image.cpu())
        native_latents.append(native.encode(image).cpu())
        laser_latents.append(laser.encode(image).cpu())
    images = torch.cat(images)
    latent = dict(original_rq=torch.cat(native_latents), laser=torch.cat(laser_latents))
    torch.save(dict(images=images, latents=latent, indices=indices), LOCAL/'matched-views.pt')
    write('data.json', dict(fit_indices=fit_indices, validation_indices=validation_indices,
        fitting_split='train', validation_split='train', fit_images=128, validation_images=128,
        identical_augmented_images_for_both_encoders=True, excluded_indices=len(excluded),
        exclusion_sources=[str(p) for p,_ in exclusion_sources],
        augmentation='original RQ training transform; seed261001 epoch0 per-index RNG',
        image_tensor_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
        cached_views_sha256=sha(LOCAL/'matched-views.pt')))
    fit = {key:value[:128].to(device) for key,value in latent.items()}
    validation = {key:value[128:].to(device) for key,value in latent.items()}

    # Check this audit's score/embedding adapter against the exact training API.
    parity = {}
    for key, quantizer, tokenizer, x, temps in (
        ('original_rq',native_book,native,fit['original_rq'][:2],[.5]*4),
        ('laser',laser.quantizer,laser,fit['laser'][:2],plan['current_temperatures'])):
        hard = hard_trajectory(quantizer, x)
        probability, native_codes = tokenizer.quantizer.get_soft_codes(x,
            temp=.5 if key == 'original_rq' else temps, stochastic=False)
        assert torch.equal(hard['codes'], native_codes.cpu())
        residual = x.reshape(-1, x.shape[-1]).clone()
        max_kl = 0.
        for depth, temperature in enumerate(temps):
            q = (quantizer.scores(residual, depth)/temperature).softmax(-1)
            p = probability[...,depth,:].reshape_as(q)
            max_kl = max(max_kl, float((p*(p.clamp_min(1.e-30).log()-q.clamp_min(1.e-30).log())).sum(-1).max()))
            residual.sub_(quantizer.embed_depth(native_codes[...,depth].reshape(-1),depth))
        assert max_kl < 1.e-5
        parity[key] = dict(hard_codes_exact=True, maximum_soft_target_kl=max_kl)
        del probability
    write('native-parity.json', parity)
    hard_fit = {key:hard_trajectory(q,fit[key]) for key,q in
                [('original_rq',native_book),('laser',laser.quantizer)]}
    fit_seeds = [261017421+i for i in range(4)]
    status('measuring_original_and_current_noise')
    reference = measure_noise(native_book,fit['original_rq'],[.5]*4,fit_seeds,hard_fit['original_rq'])
    current = measure_noise(laser.quantizer,fit['laser'],plan['current_temperatures'],fit_seeds,hard_fit['laser'])
    alignment = align_noise_budgets(hard_fit['original_rq']['energies'],
        [r['extra_error_over_hard_final_error'] for r in reference['profiles']], hard_fit['laser']['energies'])
    write('reference-and-current.json', dict(reference=public_profile(reference),
        current=public_profile(current), alignment=alignment))
    target_ratio = reference['final_sampled_to_hard_latent_error_ratio']
    candidates = []

    def evaluate(scale):
        status('fitting_all_five_temperatures', budget_scale=scale, completed_candidates=len(candidates))
        temperatures, trace = fit_depth_temperatures(laser.quantizer, fit['laser'],
            [b*scale for b in alignment['budgets']], hard_fit['laser']['energies'][-1], seed=261017430)
        profile = measure_noise(laser.quantizer,fit['laser'],temperatures,fit_seeds,hard_fit['laser'])
        row = dict(budget_scale=scale,temperatures=temperatures,fit_trace=trace,
                   profile=public_profile(profile),
                   absolute_total_ratio_error=abs(profile['final_sampled_to_hard_latent_error_ratio']-target_ratio))
        candidates.append(row)
        write('calibration-candidates.json', candidates)
        print(json.dumps(dict(scale=scale,temperatures=temperatures,
            final_ratio=profile['final_sampled_to_hard_latent_error_ratio'], reference_ratio=target_ratio)),flush=True)

    for scale in (.5,1.,1.5):
        evaluate(scale)
    # Use measured ratios to refine a bracket, rather than assuming cumulative
    # corruption is an analytic sum (later quantization can repair earlier noise).
    for _ in range(3):
        ordered = sorted(candidates,key=lambda r:r['budget_scale'])
        brackets = [(left,right) for left,right in zip(ordered,ordered[1:])
            if (left['profile']['final_sampled_to_hard_latent_error_ratio']-target_ratio)*
               (right['profile']['final_sampled_to_hard_latent_error_ratio']-target_ratio) < 0]
        if brackets:
            left,right = min(brackets,key=lambda pair:pair[1]['budget_scale']-pair[0]['budget_scale'])
            scale = (left['budget_scale']+right['budget_scale'])/2
        else:
            closest = min(candidates,key=lambda r:r['absolute_total_ratio_error'])
            scale = closest['budget_scale']/2 if closest == ordered[0] else ordered[-1]['budget_scale']*2
        evaluate(scale)
    selected = min(candidates,key=lambda r:r['absolute_total_ratio_error'])
    temperatures = selected['temperatures']
    write('selected-calibration.json', dict(selected=selected,alignment=alignment,
        reference_ratio=target_ratio, selection_uses_validation=False))
    status('independent_validation', temperatures=temperatures)
    seeds = [261017441+i for i in range(4)]
    validation_profiles = {}
    for key,q,x,temps in (
        ('original_rq',native_book,validation['original_rq'],[.5]*4),
        ('laser_current',laser.quantizer,validation['laser'],plan['current_temperatures']),
        ('laser_proposed',laser.quantizer,validation['laser'],temperatures)):
        validation_profiles[key] = measure_noise(q,x,temps,seeds)
        write('validation-latent.json', {k:public_profile(v) for k,v in validation_profiles.items()})
    torch.save(validation_profiles, LOCAL/'validation-trajectories.pt')
    status('validating_decoded_corruption')
    pixel = {}
    target_images = images[128:]
    for key,profile in validation_profiles.items():
        model = models[key]
        hard_error, soft_error, change_error = 0., [0.]*len(seeds), [0.]*len(seeds)
        per_image_hard, per_image_soft, per_image_change = [], [[] for _ in seeds], [[] for _ in seeds]
        for first in range(0, 128, 4):
            real = target_images[first:first+4].to(device)
            hard_image = model.decode_code(profile['hard_codes'][first:first+4].to(device)).clamp(-1,1)
            e = (hard_image-real).square().flatten(1).mean(-1).cpu().tolist()
            per_image_hard.extend(e); hard_error += sum(e)
            for j,codes in enumerate(profile['sampled_codes']):
                soft_image = model.decode_code(codes[first:first+4].to(device)).clamp(-1,1)
                e = (soft_image-real).square().flatten(1).mean(-1).cpu().tolist()
                change = (soft_image-hard_image).square().flatten(1).mean(-1).cpu().tolist()
                per_image_soft[j].extend(e); per_image_change[j].extend(change)
                soft_error[j] += sum(e); change_error[j] += sum(change)
        pixel[key] = dict(hard_reconstruction_mse=hard_error/128,
            stochastic_reconstruction_mse_by_seed=[e/128 for e in soft_error],
            sampled_to_hard_reconstruction_mse_ratio=float(np.mean(soft_error)/hard_error),
            stochastic_vs_hard_mse_over_hard_reconstruction_mse=float(np.mean(change_error)/hard_error),
            per_image_hard_mse=per_image_hard, per_image_soft_mse=per_image_soft,
            per_image_soft_vs_hard_mse=per_image_change)
        write('validation-pixel.json',pixel)
    proposed_ratio = validation_profiles['laser_proposed']['final_sampled_to_hard_latent_error_ratio']
    old_ratio = validation_profiles['laser_current']['final_sampled_to_hard_latent_error_ratio']
    ref_ratio = validation_profiles['original_rq']['final_sampled_to_hard_latent_error_ratio']
    heldout_improved = abs(proposed_ratio-ref_ratio) < abs(old_ratio-ref_ratio)
    final_digests = {name:model_digest(model) for name,model in [('original_rq',native),('laser',laser)]}
    assert initial_digests == final_digests
    assert sha(BASE/'inputs/recipe.yaml') == plan['active_recipe_sha256']
    assert sha(ACTIVE/'reproducibility/code-manifest.json') == plan['active_manifest_sha256']
    recipe = OmegaConf.load(BASE/'inputs/recipe.yaml')
    recipe.version = 'laser-native-joint20k-depth5-rqnoise-proposal-v1'
    recipe.run_id = 'imagenet-laser-joint20k-depth5-rqnoise-scratch100-4h100-proposal-20261017'
    recipe.name = 'LASER joint20k depth5 | matched RQ target noise | proposed fresh100'
    recipe.targets.temperature = temperatures
    recipe.targets.calibration = dict(policy='joint five-depth original-RQ corruption matching',
        report=str(OUT/'report.json'), calibration_images=128, validation_images=128,
        matching='normalized extra distance by hard residual-error progress, then total latent-error ratio',
        first_four_temperatures_unchanged=False, training_launched=False)
    proposal = OUT/'proposed-recipe.yaml'
    OmegaConf.save(recipe,proposal)
    # Only target-noise calibration and identifying metadata may change.
    before = OmegaConf.to_container(OmegaConf.load(BASE/'inputs/recipe.yaml'),resolve=True)
    after = OmegaConf.to_container(recipe,resolve=True)
    for value in (before,after):
        for key in ('version','run_id','name'): value.pop(key)
        value['targets'].pop('temperature'); value['targets'].pop('calibration')
    assert before == after
    report = dict(completed=True,calibration_images=128,validation_images=128,
        monte_carlo_seeds_per_measurement=4, original_rq_temperature=.5,
        current_temperatures=plan['current_temperatures'], proposed_temperatures=temperatures,
        selected_budget_scale=selected['budget_scale'],alignment=alignment,
        calibration_reference=public_profile(reference),calibration_current=public_profile(current),
        calibration_selected=selected,validation={key:public_profile(value) for key,value in validation_profiles.items()},
        validation_pixel=pixel,heldout_latent_noise_match_improved=heldout_improved,
        original_rq_validation_ratio=ref_ratio,current_validation_ratio=old_ratio,proposed_validation_ratio=proposed_ratio,
        proposed_recipe=str(proposal),training_launched=False,active_training_targets_changed=False,
        sampler_changed=False,encoder_decoder_and_codebooks_unchanged=True,
        model_digests=final_digests,source_plan=plan,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        scope='Corruption calibration only; generation and convergence require a controlled training ablation. Pixel MSE is not FID.',
        time=time.time())
    write('report.json',report)
    status('completed',heldout_latent_noise_match_improved=heldout_improved,
           current_ratio=old_ratio,proposed_ratio=proposed_ratio,reference_ratio=ref_ratio)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    LOCAL.mkdir(parents=True,exist_ok=True)
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'plan.json').exists(), 'Calibration already prepared'
    (OUT/'source').mkdir(exist_ok=True)
    sources = dict(calibrate=Path(__file__),helper=ROOT/'src/training/joint_noise_calibration.py',
                   tests=ROOT/'tests/test_joint_noise_calibration.py')
    for key,source in sources.items():
        shutil.copyfile(source,OUT/'source'/source.name)
    write('source-manifest.json',{key:dict(path=str(path),sha256=sha(path)) for key,path in sources.items()})
    environment = dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    environment['PYTHONPATH'] = str(ROOT)
    with (OUT/'tests.log').open('w') as log:
        result = subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            str(sources['tests']),'--junitxml='+str(OUT/'tests.xml')],env=environment,cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode == 0, (OUT/'tests.log').read_text()
    from omegaconf import OmegaConf
    recipe = OmegaConf.load(BASE/'inputs/recipe.yaml')
    plan = dict(original_stage1_sha256='e8eb312e6c0dbd8e490ca95f69e3c3b417162625ff9320bbfff791cbf832b694',
        joint_codebook_sha256=str(recipe.tokenizer.codebook_sha256),
        current_temperatures=list(recipe.targets.temperature),
        active_recipe_sha256=sha(BASE/'inputs/recipe.yaml'),
        active_manifest_sha256=sha(ACTIVE/'reproducibility/code-manifest.json'),
        fit_images=128,validation_images=128,memory_cap_gib=8,gpu=0,
        training_launched=False,active_training_targets_changed=False,time=time.time())
    write('plan.json',plan)
    with (OUT/'calibration.log').open('a',buffering=1) as log:
        child = subprocess.Popen([sys.executable,str(OUT/'source'/Path(__file__).name),'--worker'],
            env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('worker_started',child_pid=child.pid)
        code = child.wait()
    if code:
        raise RuntimeError(f'Calibration worker failed with code {code}')
    report = json.loads((OUT/'report.json').read_text())
    assert report['completed'] and not report['active_training_targets_changed']
    print(json.dumps(dict(completed=True,report=str(OUT/'report.json'),
        temperatures=report['proposed_temperatures'],
        validation_improved=report['heldout_latent_noise_match_improved'])),flush=True)


if __name__ == '__main__':
    try:
        if '--worker' in sys.argv:
            import torch
            with torch.inference_mode():
                worker()
        else:
            main()
    except BaseException as error:
        if OUT.exists():status('failed',error_type=type(error).__name__,error=str(error))
        raise
