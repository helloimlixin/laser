#!/usr/bin/env python3
"""Fixed epoch-six whole-integer sampler screen and independent FID50k checks."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-joint20k-depth5-imagenet-20261017')
OUT = ROOT/'outputs/imagenet-laser-joint20k-depth5-epoch6-sampler-sweep-20261017'
ACTIVE = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(OUT/'source'), str(BASE/'source'), str(BASE/'upstream'),
                str(BASE), str(BASE/'runtime'), str(ROOT)]

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_runtime import write_json
from src.training.rq_sample_panels import designated_preview, MEDIA_KEYS
from src.training import rq_imagenet_shared as shared
from src import rqvae_metrics as metrics
from joint_integer_sampling_audit import retained_probability
from sweep_compound512_samplers import moment_mmd


def main():
    from datetime import timedelta
    from PIL import Image
    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    assert world == 4
    torch.set_num_threads(4)
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(device).total_memory, device)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=60))
    plan = json.loads((OUT/'plan.json').read_text())
    profile = JointLaserRecipe(BASE)
    payload = torch.load(plan['local_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    profile.verify_resume(payload)
    assert payload['global_step'] == 3756 and payload['next_epoch'] == 6
    assert payload['next_microbatch'] == 0 and len(payload['rng_state_by_rank']) == 4
    assert all(int(state['step']) == 3756 for state in payload['optimizer']['state'].values())
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(payload['config']['arch']))
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    del payload
    model.requires_grad_(False).eval().to(device)
    aux, _ = profile.tokenizer(device)
    profile.adapt(model, aux)
    assert tuple(model.block_size) == (8, 8, 5) and model.vocab_size == [20481]*5
    parameter_versions = [(name, p._version) for name, p in model.named_parameters()]
    frozen = profile.digest(aux)
    original_recipe = OmegaConf.to_container(profile.recipe, resolve=True)
    policies = plan['policies']

    def set_policy(policy):
        profile.recipe.sampling = OmegaConf.create(policy)
        top_k = policy['top_k']
        profile.sampling_settings = lambda: dict(temperature=float(policy['temperature']),
            top_k=top_k if isinstance(top_k, list) or top_k is None else [top_k],
            top_p=[float(policy['top_p'])])

    # Verify the baseline against the existing 512-image panel, not just names
    # and checkpoint metadata, before reusing its completed FID50k measurement.
    set_policy(policies['baseline'])
    with torch.inference_mode():
        designated_preview(profile, model, aux, device, rank, world, 3756, None, OUT/'baseline-parity')
    if rank == 0:
        for key in MEDIA_KEYS:
            suffix = key.split('/')[-1]
            source = ACTIVE/f'samples/step003756-{suffix}.png'
            reproduced = OUT/f'baseline-parity/smoke/samples/step003756-{suffix}.png'
            assert np.array_equal(np.asarray(Image.open(source)), np.asarray(Image.open(reproduced))), key
        write_json(OUT/'checkpoint-verification.json', dict(passed=True,
            source_step=3756, strict_model_reload=True, native_integer_sampler=True,
            baseline_preview_pixels_exact=True, panels_verified=9, preview_images=512,
            optimizer_age=3756, baseline_fid50k=plan['baseline_fid50k'],
            checkpoint_sha256=plan['checkpoint']['sha256']))
    dist.barrier()
    run = None
    if rank == 0:
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=OUT.name,
            name='LASER joint20k depth5 epoch6 | sampler comparison', resume='allow',
            dir=os.environ['WANDB_DIR'], config=plan, job_type='fixed-checkpoint-sampler-sweep')
        run.define_metric('screen/index')
        run.define_metric('screen/*', step_metric='screen/index')
        run.define_metric('confirm/index')
        run.define_metric('confirm/fid50k', step_metric='confirm/index', summary='min,last')
        run.define_metric('samples/global_step')
        run.define_metric('samples/*', step_metric='samples/global_step')
        write_json(OUT/'wandb.json', dict(id=run.id, url=run.url))
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        workspace = ws.Workspace(entity='helloimlixin-rutgers', project='laser',
            name='LASER joint20k depth5 epoch6 sampler comparison',
            runset_settings=ws.RunsetSettings(filters=[ws.Config('wandb_id') == OUT.name]),
            sections=[ws.Section(name='Fixed checkpoint sampler comparison', is_open=True, panels=[
                wr.MarkdownPanel(markdown='Epoch-six weights held fixed. Screen metrics are not FID. '
                    'Every confirmed FID uses 50,000 generated images against all 1,281,167 training images. '
                    'The active training sampler remains unchanged.'),
                wr.LinePlot(title='Confirmed FID50k', x='confirm/index', y=['confirm/fid50k']),
                wr.LinePlot(title='Screen feature-moment distance', x='screen/index', y=['screen/moment_mmd']),
                wr.LinePlot(title='Screen Inception Score', x='screen/index', y=['screen/inception_score']),
                wr.MediaBrowser(title='Selected policy fixed classes', media_keys=['samples/fixed'], num_columns=1),
            ])])
        workspace.save()
        write_json(OUT/'workspace.json', dict(url=workspace.url))
    dist.barrier()
    metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
    mean, covariance = metrics.load_reference_statistics(BASE/'inputs/imagenet_256_train.npz')
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1e-8))
    reference_second = covariance/np.outer(standard_deviation, standard_deviation)*(1281166/1281167)
    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float64)
    std_t = torch.as_tensor(standard_deviation, device=device, dtype=torch.float64)
    inception = metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    shared.BASE, shared.PROFILE, shared.RUN = BASE, profile, profile.run_id
    shared.original_metrics.INCEPTION_PATH = metrics.INCEPTION_PATH
    retained = torch.zeros(5, 8, device=device, dtype=torch.float64)
    original_cached = model.cached_forward

    def audited_cached(*args, **kwargs):
        logits = original_cached(*args, **kwargs)
        depth = kwargs['sample_loc'][2]
        p = logits.float().softmax(-1)
        retained[depth, 0] += (-(p*p.clamp_min(1e-30).log()).sum(-1)).sum()
        for index, top_k in enumerate((1024, 2048, None)):
            mass, support = retained_probability(logits, top_k=top_k, top_p=.95)
            retained[depth, 1+2*index] += mass.sum()
            retained[depth, 2+2*index] += support.sum()
        retained[depth, 7] += len(logits)
        return logits

    @torch.inference_mode()
    def screen(name, index):
        policy = policies[name]
        result_path = OUT/f'screen/{name}/result.json'
        if result_path.exists():
            cached = json.loads(result_path.read_text())
            assert cached['sampling'] == policy and cached['generated_images'] == plan['screen_images']
            assert cached['seed'] == plan['screen_seed']
            dist.barrier()
            return cached
        set_policy(policy)
        torch.manual_seed(plan['screen_seed']+rank)
        torch.cuda.manual_seed(plan['screen_seed']+rank)
        total = torch.zeros(2048, device=device, dtype=torch.float64)
        cross = torch.zeros(2048, 2048, device=device, dtype=torch.float64)
        norm4 = torch.zeros((), device=device, dtype=torch.float64)
        probabilities = []
        started = time.monotonic()
        for first in range(0, plan['screen_images']//world, 128):
            count = min(128, plan['screen_images']//world-first)
            labels = (torch.arange(first, first+count, device=device)*world+rank)%1000
            # Audit the first 1,024 baseline images, covering every class.
            model.cached_forward = audited_cached if name == 'baseline' and first < 256 else original_cached
            codes = profile.sample(model, aux, count, labels)
            model.cached_forward = original_cached
            for part in codes.split(8):
                images = profile.decode(aux, part).float().mul(.5).add(.5).clamp(0, 1)
                features, logits = inception(images, return_logits=True)
                z = (features.double()-mean_t)/std_t
                total += z.sum(0)
                cross.addmm_(z.T, z)
                norm4 += z.square().sum(-1).square().sum()
                probabilities.append(logits.softmax(-1))
            if rank == 0:
                write_json(OUT/'progress.json', dict(phase='screening', policy=name,
                    index=index, policies=len(policies), generated=(first+count)*world,
                    target=plan['screen_images'], time=time.time()))
        for tensor in (total, cross, norm4):
            dist.all_reduce(tensor)
        probabilities = torch.cat(probabilities)
        gathered = [torch.empty_like(probabilities) for _ in range(world)]
        dist.all_gather(gathered, probabilities)
        probabilities = torch.cat(gathered)
        torch.manual_seed(plan['screen_seed'])
        is_mean, is_std = metrics.inception_score(probabilities[torch.randperm(plan['screen_images'], device=device)])
        result = dict(policy=name, sampling=policy, generated_images=plan['screen_images'],
            seed=plan['screen_seed'], not_fid=True,
            moment_mmd=moment_mmd(total.cpu().numpy(), cross.cpu().numpy(), float(norm4),
                                  plan['screen_images'], reference_second),
            inception_score=is_mean, inception_score_std=is_std,
            elapsed_seconds=time.monotonic()-started)
        if rank == 0:
            write_json(OUT/f'screen/{name}/result.json', result)
            run.log({'screen/index':index, 'screen/setting':name,
                     'screen/moment_mmd':result['moment_mmd'], 'screen/inception_score':is_mean})
            print(json.dumps(dict(event='screen_complete', **result)), flush=True)
        if name == 'baseline':
            dist.all_reduce(retained)
            if rank == 0:
                rows = []
                for depth in range(5):
                    row = retained[depth].cpu().tolist()
                    rows.append(dict(depth=depth+1, logit_rows=int(row[7]),
                        predictive_entropy=row[0]/row[7],
                        filters={str(k):dict(mean_original_mass=row[1+2*i]/row[7],
                            mean_support_size=row[2+2*i]/row[7])
                            for i, k in enumerate((1024, 2048, 'unrestricted'))}))
                write_json(OUT/'retained-probability-audit.json', dict(passed=True,
                    generated_images=1024, generated_prefixes=True,
                    temperature=1., top_p=.95, rows=rows))
        torch.cuda.empty_cache()
        dist.barrier()
        return result

    results = [screen(name, index) for index, name in enumerate(policies)]
    challengers = [r for r in results if r['policy'] != 'baseline']
    best_moment = min(challengers, key=lambda r: (r['moment_mmd'], r['policy']))
    best_is = max((r for r in challengers if r['policy'] != best_moment['policy']),
                  key=lambda r: (r['inception_score'], -r['moment_mmd']))
    selected = [best_moment['policy'], best_is['policy']]
    if rank == 0:
        write_json(OUT/'screen-summary.json', dict(results=results, selected=selected, not_fid=True))
        write_json(OUT/'status.json', dict(phase='confirming', selected=selected, time=time.time()))
    del inception
    torch.cuda.empty_cache()
    confirmations = [dict(policy='baseline', fid=plan['baseline_fid50k'],
        sampling=policies['baseline'], generated_images=50000, real_images=1281167,
        source_step=3756, seed=plan['confirmation_seed'], reused_verified_baseline=True)]
    if run:
        run.log({'confirm/index':0, 'confirm/setting':'baseline', 'confirm/fid50k':plan['baseline_fid50k']})
    for index, name in enumerate(selected, 1):
        set_policy(policies[name])
        destination = OUT/'confirm'/name
        if (destination/'fid50k.json').exists():
            cached = json.loads((destination/'fid50k.json').read_text())
            assert cached['sampling'] == policies[name] and cached['generated_images'] == 50000
            assert cached['checkpoint_sha256'] == plan['checkpoint']['sha256']
            confirmations.append(cached)
            dist.barrier()
            continue
        shared.OUT = destination
        if rank == 0:
            write_json(OUT/'progress.json', dict(phase='confirming', policy=name, index=index,
                target=50000, time=time.time()))
        started = time.monotonic()
        result = shared.generation(model, aux, device, rank, world, 6, None,
                                   batch=128, seed=plan['confirmation_seed'])
        result.update(policy=name, sampling=policies[name], source_run=ACTIVE.name,
            source_step=3756, source_epoch=6, checkpoint_sha256=plan['checkpoint']['sha256'],
            generated_images=50000, real_images=1281167, real_split='train',
            reference_sha256=plan['reference_sha256'], seed=plan['confirmation_seed'],
            world_size=4, batch=128, baseline_fid=plan['baseline_fid50k'],
            fid_delta=result['fid']-plan['baseline_fid50k'], weights_unmodified=True,
            training_updates_added=0, rejection_sampling=False,
            elapsed_seconds=time.monotonic()-started)
        with torch.inference_mode():
            designated_preview(profile, model, aux, device, rank, world, 3756, None, destination)
        if rank == 0:
            write_json(destination/'fid50k.json', result)
            run.log({'confirm/index':index, 'confirm/setting':name, 'confirm/fid50k':result['fid'],
                     'confirm/inception_score':result['inception_score']})
            print(json.dumps(dict(event='confirmation_complete', **result)), flush=True)
        confirmations.append(result)
        dist.barrier()
    assert [(name, p._version) for name, p in model.named_parameters()] == parameter_versions
    assert profile.digest(aux) == frozen
    assert OmegaConf.to_container(OmegaConf.load(BASE/'inputs/recipe.yaml'), resolve=True) == original_recipe
    winner = min(confirmations, key=lambda r: r['fid'])
    if rank == 0:
        report = dict(plan=plan, screen=results, confirmation=confirmations,
            selected_policy=winner['policy'], selected_sampling=winner['sampling'],
            selected_fid50k=winner['fid'], baseline_fid50k=plan['baseline_fid50k'],
            improvement=plan['baseline_fid50k']-winner['fid'],
            weights_unmodified=True, tokenizer_unmodified=True, training_updates_added=0,
            active_training_sampler_changed=False,
            limits='One epoch-six checkpoint and one full evaluation seed; later checkpoints need validation.')
        write_json(OUT/'report.json', report)
        run.summary.update(dict(selected_policy=winner['policy'], selected_fid50k=winner['fid'],
            baseline_fid50k=plan['baseline_fid50k'], improvement=report['improvement'],
            active_training_sampler_changed=False))
        import wandb
        run.log({'sampling/results':wandb.Table(columns=['policy', 'fid50k', 'sampling'],
            data=[[r['policy'], r['fid'], json.dumps(r['sampling'])] for r in confirmations])})
        folder = OUT/'baseline-parity' if winner['policy'] == 'baseline' else OUT/'confirm'/winner['policy']
        paths = {key:folder/f'smoke/samples/step003756-{key.split("/")[-1]}.png' for key in MEDIA_KEYS}
        run.log({'samples/global_step':3756, **{key:wandb.Image(str(path),
            caption=f"Fixed epoch-six weights; {winner['policy']}; {winner['sampling']}") for key, path in paths.items()}})
        run.save(str(OUT/'report.json'), base_path=str(OUT), policy='now')
        run.save(str(OUT/'retained-probability-audit.json'), base_path=str(OUT), policy='now')
        run.finish()
        write_json(OUT/'status.json', dict(phase='completed', selected_policy=winner['policy'],
            fid=winner['fid'], time=time.time()))
        print(json.dumps(dict(event='completed', policy=winner['policy'], fid=winner['fid'])), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
