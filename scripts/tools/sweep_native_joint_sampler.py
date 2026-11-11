#!/usr/bin/env python3
"""Screen frozen native-joint sampling policies, then independently confirm FID50k.

All checkpoint and tokenizer weights remain frozen. The active training process,
its recipe, optimizer, and canonical sampler are never modified. Small screens
use a feature-moment MMD diagnostic, not small-sample FID. No images are logged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time


def policies():
    baseline = dict(temperature=1.0, top_k=1024, top_p=0.95)
    result = {'baseline': baseline}
    for value in [0.85, 0.90, 0.95, 1.05]:
        result[f't{round(value*100):03d}'] = dict(baseline, temperature=value)
    for value in [256, 512, 2048, 16384]:
        result[f'k{value}'] = dict(baseline, top_k=value)
    for value in [0.90, 0.98, 1.0]:
        result[f'p{round(value*100):03d}'] = dict(baseline, top_p=value)
    result['t095_k2048_p098'] = dict(temperature=.95, top_k=2048, top_p=.98)
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screen-images', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--memory-gib', type=float, default=24)
    args = parser.parse_args()
    base, local, output = [p.resolve() for p in (args.base, args.local, args.output)]
    assert args.screen_images >= 2000 and args.screen_images % 1000 == 0
    assert args.batch > 0 and args.memory_gib > 0
    local.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (local/'sweep.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (local/'report.json').exists():
        print((local/'report.json').read_text(), flush=True)
        return
    os.nice(10)
    os.environ.update(RQ_UPSTREAM_ROOT=str(base/'upstream'), TORCH_HOME=str(base/'torch-cache'),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_DISABLE_GIT='true',
        WANDB_DIR=str(local/'wandb'), WANDB_CACHE_DIR=str(local/'wandb-cache'))
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
    sys.path[:0] = [str(local/'support'), str(base/'source'), str(base), str(base/'upstream')]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from runtime.precision import install_bf16_training
    install_bf16_training()
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src import rqvae_metrics as metrics
    from screen_metrics import moment_mmd
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/properties.total_memory, device)
    for folder in ['wandb', 'screen', 'confirm']:
        (local/folder).mkdir(exist_ok=True)
    metrics.INCEPTION_PATH = base/'upstream/rqvae/metrics/inception.py'
    run = None
    phase = {'phase': 'verifying_checkpoint', 'pid': os.getpid(), 'time': time.time()}
    phase_lock = threading.Lock()
    stop_mirror = threading.Event()

    def status(**values):
        nonlocal phase
        with phase_lock:
            phase = dict(phase, **values, time=time.time(), pid=os.getpid())
            atomic_json(local/'status.json', phase)

    def persist(name, value):
        atomic_json(local/name, value)
        atomic_json(output/name, value)

    def mirror():
        while not stop_mirror.wait(20):
            try:
                with phase_lock:
                    snapshot = dict(phase)
                atomic_json(output/'status.json', snapshot)
            except Exception as error:
                print(json.dumps({'status_mirror_error': str(error)}), flush=True)

    threading.Thread(target=mirror, daemon=True).start()

    def interrupted(signum, _):
        raise InterruptedError(f'Sweep received signal {signum}; training is independent')

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, interrupted)
    try:
        status(phase='verifying_checkpoint')
        receipt = json.loads((local/'checkpoint-receipt.json').read_text())
        checkpoint = local/'frozen-checkpoint.pt'
        assert checkpoint.stat().st_size == receipt['bytes'] and receipt['state'] == 'COMMITTED'
        checkpoint_sha = digest(checkpoint)
        profile = JointLaserRecipe(base)
        saved_recipe_hash = digest(base/'inputs/recipe.yaml')
        original_sampling = OmegaConf.to_container(profile.recipe.sampling, resolve=True)
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        profile.verify_resume(payload)
        source_step = int(payload['global_step'])
        assert source_step == receipt['global_step']
        assert {int(v['step']) for v in payload['optimizer']['state'].values()} == {source_step}
        assert payload['scheduler']['after']['last_epoch'] == source_step
        assert payload['runtime']['teacher_encoding_beam_width'] == 16
        with torch.device('meta'):
            model = profile.model(OmegaConf.create(payload['config']['arch']))
        model.load_state_dict(payload['state_dict'], strict=True, assign=True)
        model.requires_grad_(False).eval().to(device)
        del payload
        import gc
        gc.collect()
        aux, _ = profile.tokenizer(device)
        assert aux.reconstruction_beam_width == 16
        assert not any(p.requires_grad for p in model.parameters())
        assert not any(p.requires_grad for p in aux.parameters())
        model_versions = [(n, p._version) for n, p in model.named_parameters()]
        frozen_tokenizer = profile.digest(aux)
        reference = base/'inputs/imagenet_256_train.npz'
        reference_sha = digest(reference)
        assert reference_sha == str(profile.recipe.evaluation.reference_sha256)
        step_epoch = source_step/626
        screen_seed, confirm_seed = 26110421, 26110422
        plan = dict(wandb_id=output.name, source_run=profile.run_id, source_step=source_step,
            source_epoch=step_epoch, checkpoint_artifact=receipt['artifact'],
            checkpoint_sha256=checkpoint_sha, source_checkpoint_bytes=receipt['bytes'],
            frozen_checkpoint=str(checkpoint), baseline_sampling=original_sampling,
            policies=policies(), screen_images=args.screen_images, screen_seed=screen_seed,
            screen_metric='reference-normalized polynomial feature-moment MMD; NOT FID',
            confirmation_images=50000, confirmation_seed=confirm_seed,
            reference_images=1281167, reference_split='train', reference_sha256=reference_sha,
            sampling_batch=args.batch, decode_batch=16, world_size=1,
            memory_cap_gib=args.memory_gib, hardware=properties.name,
            sampling_precision='float16', decoder_precision='float32',
            unchanged_tokenizer_rfid=4.5122860099749005,
            frozen_tokenizer_digest=frozen_tokenizer, active_recipe_sha256=saved_recipe_hash,
            selection='Lowest screen MMD and highest screen IS among other challengers; then compare FID50k with baseline',
            screen_and_confirmation_seeds_independent=True, balanced_class_labels=True,
            evaluator_sha256=digest(Path(__file__)), training_updates_added=0,
            active_training_sampler_changed=False, weights_frozen=True, images_logged=False)
        assert plan['policies']['baseline'] == original_sampling
        if (local/'plan.json').exists():
            assert json.loads((local/'plan.json').read_text()) == plan, 'Resume plan changed'
        persist('plan.json', plan)
        shutil.copyfile(Path(__file__), output/'evaluate.py')
        persist('verification.json', dict(passed=True, checkpoint_strictly_loaded=True,
            source_step=source_step, source_optimizer_ages_verified=True,
            model_frozen=True, tokenizer_frozen=True, reconstruction_beam_width=16,
            source_checkpoint_sha256=checkpoint_sha, active_training_recipe_unchanged=True))
        inception = metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
        mean, covariance = metrics.load_reference_statistics(reference)
        std = np.sqrt(np.maximum(np.diag(covariance), 1e-8))
        reference_second = covariance/np.outer(std, std)*(1281166/1281167)
        mean_t = torch.as_tensor(mean, dtype=torch.float64, device=device)
        std_t = torch.as_tensor(std, dtype=torch.float64, device=device)
        import wandb
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=output.name,
            name=f'ImageNet beam16 step{source_step} | frozen sampler sweep | training unchanged',
            job_type='fixed-checkpoint-sampler-sweep', resume='allow', dir=str(local/'wandb'), config=plan)
        run.use_artifact(receipt['artifact'])
        run.use_artifact(str(profile.recipe.tokenizer.paired_tokenizer_artifact))
        run.define_metric('screen/index')
        run.define_metric('screen/*', step_metric='screen/index')
        run.define_metric('confirm/index')
        run.define_metric('confirm/*', step_metric='confirm/index')
        persist('wandb.json', dict(id=run.id, url=run.url))
        run.save(str(output/'plan.json'), base_path=str(output), policy='now')
        run.save(str(output/'evaluate.py'), base_path=str(output), policy='now')

        def set_policy(setting):
            profile.recipe.sampling = OmegaConf.create(setting)

        @torch.inference_mode()
        def generate(count, seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            for first in range(0, count, args.batch):
                n = min(args.batch, count-first)
                labels = torch.arange(first, first+n, device=device) % 1000
                codes = profile.sample(model, aux, n, labels)
                assert codes.shape == (n, 8, 8, 4) and 0 <= int(codes.min()) <= int(codes.max()) < 16384
                for chunk in codes.split(16):
                    images = profile.decode(aux, chunk).float().mul(.5).add(.5).clamp(0, 1)
                    yield images
                status(generated_images=first+n, target_images=count)

        # A full generation batch checks the configured cap before any measured screen.
        status(phase='memory_preflight', policy='baseline')
        set_policy(plan['policies']['baseline'])
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for images in generate(args.batch, 26110420):
                features, logits = inception(images, return_logits=True)
                assert bool(torch.isfinite(features).all()) and bool(torch.isfinite(logits).all())
        persist('memory-preflight.json', dict(passed=True, batch=args.batch,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            memory_cap_gib=args.memory_gib, tokenizer_unchanged=profile.digest(aux)==frozen_tokenizer))
        torch.cuda.empty_cache()

        def screen(name, index, setting):
            path = local/f'screen/{name}.json'
            if path.exists():
                cached = json.loads(path.read_text())
                assert cached['checkpoint_sha256'] == checkpoint_sha and cached['sampling'] == setting
                return cached
            status(phase='screening', policy=name, policy_index=index,
                policies=len(plan['policies']), generated_images=0, target_images=args.screen_images)
            set_policy(setting)
            total = torch.zeros(2048, dtype=torch.float64, device=device)
            cross = torch.zeros(2048, 2048, dtype=torch.float64, device=device)
            fourth = torch.zeros((), dtype=torch.float64, device=device)
            probabilities = []
            started = time.monotonic()
            with torch.inference_mode():
                for images in generate(args.screen_images, screen_seed):
                    features, logits = inception(images, return_logits=True)
                    assert bool(torch.isfinite(features).all()) and bool(torch.isfinite(logits).all())
                    normalized = (features.double()-mean_t)/std_t
                    total += normalized.sum(0)
                    cross.addmm_(normalized.T, normalized)
                    fourth += normalized.square().sum(-1).square().sum()
                    probabilities.append(logits.softmax(-1).cpu())
            probs = torch.cat(probabilities)
            assert len(probs) == args.screen_images
            permutation = torch.randperm(len(probs), generator=torch.Generator().manual_seed(screen_seed))
            is_mean, is_std = metrics.inception_score(probs[permutation])
            result = dict(policy=name, sampling=setting, checkpoint_sha256=checkpoint_sha,
                generated_images=args.screen_images, seed=screen_seed, not_fid=True,
                moment_mmd=moment_mmd(total.cpu().numpy(), cross.cpu().numpy(), float(fourth),
                                      args.screen_images, reference_second),
                inception_score=is_mean, inception_score_std=is_std,
                elapsed_seconds=time.monotonic()-started)
            persist(f'screen/{name}.json', result)
            run.log({'screen/index':index, 'screen/policy':name,
                'screen/moment_mmd':result['moment_mmd'], 'screen/inception_score':is_mean})
            print(json.dumps(dict(event='screen_complete', **result)), flush=True)
            torch.cuda.empty_cache()
            return result

        screens = [screen(name, index, setting) for index, (name, setting) in enumerate(plan['policies'].items())]
        challengers = [row for row in screens if row['policy'] != 'baseline']
        best_moment = min(challengers, key=lambda row: (row['moment_mmd'], row['policy']))
        best_is = max((row for row in challengers if row['policy'] != best_moment['policy']),
                      key=lambda row: (row['inception_score'], -row['moment_mmd']))
        selected = [best_moment['policy'], best_is['policy']]
        persist('screen-summary.json', dict(results=screens, selected=selected, not_fid=True))
        confirms = []
        for index, name in enumerate(['baseline'] + selected):
            path = local/f'confirm/{name}.json'
            if path.exists():
                row = json.loads(path.read_text())
                assert row['checkpoint_sha256'] == checkpoint_sha and row['generated_images'] == 50000
                confirms.append(row)
                continue
            setting = plan['policies'][name]
            set_policy(setting)
            status(phase='confirming', policy=name, policy_index=index,
                policies=3, generated_images=0, target_images=50000)
            metric = metrics.DistributedOriginalRQVAEMetrics(device,
                compute_inception_score=True, reference_stats_path=reference, inception=inception)
            started = time.monotonic()
            with torch.inference_mode():
                for images in generate(50000, confirm_seed):
                    metric.update(images, real=False)
            assert int(metric.fake_count) == 50000
            status(phase='computing_fid', policy=name, generated_images=50000, target_images=50000)
            # Use a repeatable IS shuffle independent of policy-specific sampling RNG consumption.
            torch.manual_seed(confirm_seed+1)
            fid, is_mean, is_std = metric.compute(shuffle_for_inception=True)
            row = dict(policy=name, sampling=setting, checkpoint_sha256=checkpoint_sha,
                source_step=source_step, source_epoch=step_epoch, generated_images=50000,
                real_images=1281167, reference_split='train', reference_sha256=reference_sha,
                seed=confirm_seed, fid=fid, inception_score=is_mean, inception_score_std=is_std,
                sampling_batch=args.batch, rejection_sampling=False, weights_frozen=True,
                elapsed_seconds=time.monotonic()-started)
            assert np.isfinite(fid) and np.isfinite(is_mean)
            persist(f'confirm/{name}.json', row)
            run.log({'confirm/index':index, 'confirm/policy':name,
                'confirm/fid50k':fid, 'confirm/inception_score':is_mean})
            confirms.append(row)
            print(json.dumps(dict(event='confirmation_complete', **row)), flush=True)
            del metric
            torch.cuda.empty_cache()
        assert [(n, p._version) for n, p in model.named_parameters()] == model_versions
        assert profile.digest(aux) == frozen_tokenizer
        assert digest(base/'inputs/recipe.yaml') == saved_recipe_hash
        assert checkpoint.stat().st_size == receipt['bytes']
        baseline = confirms[0]
        winner = min(confirms, key=lambda row: row['fid'])
        report = dict(plan=plan, screen=screens, confirmation=confirms,
            selected_policy=winner['policy'], selected_sampling=winner['sampling'],
            baseline_fid50k=baseline['fid'], selected_fid50k=winner['fid'],
            fid_improvement=baseline['fid']-winner['fid'],
            baseline_inception_score=baseline['inception_score'],
            selected_inception_score=winner['inception_score'],
            weights_unmodified=True, tokenizer_unmodified=True,
            training_updates_added=0, active_training_sampler_changed=False,
            limits='One frozen checkpoint and one independent full50k confirmation seed; no guarantee at later epochs.')
        persist('report.json', report)
        run.summary.update({k:report[k] for k in ['selected_policy','baseline_fid50k',
            'selected_fid50k','fid_improvement','baseline_inception_score','selected_inception_score',
            'weights_unmodified','tokenizer_unmodified','active_training_sampler_changed']})
        run.log({'sampling/results':wandb.Table(columns=['policy','fid50k','inception_score','sampling'],
            data=[[r['policy'],r['fid'],r['inception_score'],json.dumps(r['sampling'])] for r in confirms])})
        for name in ['report.json','screen-summary.json','verification.json','memory-preflight.json']:
            run.save(str(output/name), base_path=str(output), policy='now')
        run.finish()
        run = None
        status(phase='completed', selected_policy=winner['policy'],
            baseline_fid50k=baseline['fid'], selected_fid50k=winner['fid'],
            fid_improvement=report['fid_improvement'])
        persist('status.json', dict(phase))
        print(json.dumps(dict(event='completed', **phase)), flush=True)
    except BaseException as error:
        status(phase='failed', error_type=type(error).__name__, error=str(error))
        atomic_json(output/'status.json', dict(phase))
        if run is not None:
            run.summary['sweep/error'] = str(error)
            run.finish(exit_code=1)
        raise
    finally:
        stop_mirror.set()


if __name__ == '__main__':
    main()
