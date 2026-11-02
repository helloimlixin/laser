#!/usr/bin/env python3
"""Screen fixed-weight ancestral samplers, then confirm two with matched FID50k."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def candidates():
    baseline = dict(atom_temperature=1., atom_top_k=0, atom_top_p=.95,
        coefficient_temperature=1., coefficient_top_k=256, coefficient_top_p=.95)
    changes = {
        'no_atom_topk': {},
        'atom_t095': dict(atom_temperature=.95),
        'atom_t105': dict(atom_temperature=1.05),
        'atom_t110': dict(atom_temperature=1.1),
        'atom_untruncated': dict(atom_top_p=1.),
        'atom_t105_p098': dict(atom_temperature=1.05, atom_top_p=.98),
        'coefficient_t080': dict(coefficient_temperature=.8),
        'coefficient_t065': dict(coefficient_temperature=.65),
        'coefficient_untruncated': dict(coefficient_top_k=0, coefficient_top_p=1.),
        'balanced': dict(atom_temperature=1.05, atom_top_p=.98,
            coefficient_temperature=.8, coefficient_top_k=0),
        'previous_laser': dict(atom_temperature=.9, atom_top_p=.9,
            coefficient_top_k=0, coefficient_top_p=.85),
    }
    return {name: dict(baseline, **change) for name, change in changes.items()}


def moment_mmd(total, cross, fourth_norm_sum, count, reference_second):
    """Unbiased generated-side polynomial MMD to fixed reference moments.

    Inputs are reference-centered, variance-normalized features. The kernel
    is (1 + x.y / dimension)**2. Reference mean is exactly zero. The reference
    empirical distribution includes its diagonal; the generated U-statistic
    excludes its diagonal. This is a screening diagnostic, never FID.
    """
    import numpy as np
    total, cross, reference_second = [np.asarray(x, dtype=np.float64)
        for x in (total, cross, reference_second)]
    dimension = total.size
    if count < 2:
        raise ValueError('At least two generated features are required')
    linear = 2 * (total @ total - np.trace(cross)) / (count*(count-1)*dimension)
    quadratic = ((np.square(cross).sum() - fourth_norm_sum) / (count*(count-1))
        - 2*(cross*reference_second).sum()/count + np.square(reference_second).sum()) / dimension**2
    return float(linear + quadratic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3'))
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--joint-module', type=Path,
        help='Use whole-pair temperature from this frozen sampler implementation')
    parser.add_argument('--factorized-controls', type=Path,
        help='Completed factorized sweep whose controls may be reused unchanged')
    parser.add_argument('--screen-count',type=int,default=4096)
    parser.add_argument('--confirm-count',type=int,choices=[1,2],default=2)
    parser.add_argument('--allow-implementation-update',action='store_true')
    parser.add_argument('--policies-json',type=Path)
    parser.add_argument('--joint-control',type=Path,
        help='Previously validated joint-policy FID50k result')
    parser.add_argument('--adaptive-search',action='store_true',
        help='Bracket a cold boundary once, then combine the strongest temperature and pair floor')
    parser.add_argument('--panels-module',type=Path,
        help='Frozen panel renderer supporting nested sampler metadata')
    args = parser.parse_args()
    base, out = args.base.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')
    os.environ['TORCH_HOME'] = str(base/'torch-cache')
    local_wandb = Path('/tmp/laser-sampler-sweeps')/out.name
    local_wandb.mkdir(parents=True,exist_ok=True)
    os.environ['WANDB_DIR'] = str(local_wandb)
    sys.path[:0] = [str(base/'source'), str(base/'upstream'), str(base), str(base/'runtime')]
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
    from src.training import rq_imagenet_shared as shared
    from src.training.rq_imagenet_runtime import write_json
    from src.training.rq_sample_panels import designated_preview
    if args.panels_module:
        import importlib.util
        spec = importlib.util.spec_from_file_location('frozen_sweep_sample_panels',args.panels_module)
        panel_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(panel_module)
        designated_preview = panel_module.designated_preview
    from src import rqvae_metrics as metrics

    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    assert world == 4
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(13*2**30/torch.cuda.get_device_properties(device).total_memory, device)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=60))
    profile = CompoundReplayRecipe(base)
    source = Path('/workspace/Projects/laser/outputs')/profile.run_id
    receipts = [json.loads(p.read_text()) for p in (source/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    receipts = [r for r in receipts if r['global_step'] == args.epoch*626]
    assert len(receipts) == 1
    receipt = receipts[0]
    checkpoint = Path(receipt['local'])
    assert checkpoint.stat().st_size == receipt['bytes']
    if rank == 0:
        with checkpoint.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['sha256']
    dist.barrier()
    payload = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=False)
    profile.verify_resume(payload)
    assert payload['global_step'] == args.epoch*626
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(payload['config']['arch']))
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).eval().to(device)
    del payload
    aux, _ = profile.tokenizer(device)
    frozen = profile.digest(aux)
    parameter_versions = [(name, p._version) for name, p in model.named_parameters()]
    policies = candidates()
    joint_module = None
    joint_control = None
    joint_statistics = dict(proposals=0,accepted=0,max_rounds=0)
    if args.joint_module:
        import importlib.util
        spec = importlib.util.spec_from_file_location('frozen_joint_pair_sampling',args.joint_module)
        joint_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(joint_module)
        original_sample = profile.sample
        def sample(model,aux,count,labels):
            temperature = profile.recipe.sampling.get('joint_temperature')
            if temperature is None:
                return original_sample(model,aux,count,labels)
            codes,stats = joint_module.sample_joint_compound(model,aux,count,labels,
                temperature=profile.recipe.sampling.get('joint_temperature_by_depth',float(temperature)),
                support_floor_ratio=float(profile.recipe.sampling.get('joint_support_floor',0.)),amp=True)
            joint_statistics['proposals'] += stats['proposals']
            joint_statistics['accepted'] += stats['accepted']
            joint_statistics['max_rounds'] = max(joint_statistics['max_rounds'],stats['max_rounds'])
            return codes
        profile.sample = sample
        controls = {name:policies[name] for name in ['no_atom_topk','previous_laser']}
        policies = dict(controls)
        for temperature in [.85,.9,.95,1.,1.05]:
            name = f'joint_t{round(temperature*100):03d}'
            policies[name] = dict(atom_temperature=temperature,atom_top_k=0,atom_top_p=1.,
                coefficient_temperature=temperature,coefficient_top_k=0,coefficient_top_p=1.,
                joint_temperature=temperature)
        if args.policies_json:
            requested = json.loads(args.policies_json.read_text())
            assert not set(requested).intersection(controls)
            for name,setting in requested.items():
                assert 'joint_temperature' in setting
                assert setting['atom_top_k']==setting['coefficient_top_k']==0
                assert setting['atom_top_p']==setting['coefficient_top_p']==1.
            policies = dict(controls,**requested)
        if args.joint_control:
            joint_control = json.loads(args.joint_control.read_text())
            assert joint_control['checkpoint']['sha256']==receipt['sha256']
            assert joint_control['source_step']==args.epoch*626
            assert joint_control['generated_images']==50000 and joint_control['reference_images']==1281167
            assert joint_control['reference_split']=='train'
            assert joint_control['policy'] not in policies
            policies[joint_control['policy']] = joint_control['sampling']
    baseline_path = source/'collapse-diagnosis'/f'epoch{args.epoch:03d}'/'full50k-no-atom-topk/fid50k.json'
    baseline = json.loads(baseline_path.read_text())
    assert baseline['sampling'] == policies['no_atom_topk']
    assert baseline['checkpoint']['sha256'] == receipt['sha256']
    assert baseline['generated_images'] == 50000 and baseline['reference_images'] == 1281167
    assert baseline['reference_split'] == 'train'
    screen_seed, confirm_seed, screen_count = 261017101, 261001, args.screen_count
    assert screen_count%world==0 and screen_count>=1000
    assert set(range(screen_seed, screen_seed+world)).isdisjoint(range(confirm_seed, confirm_seed+world))
    reference_path = base/'inputs/imagenet_256_train.npz'
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert reference_sha == baseline['reference_sha256']
    plan = dict(source_run=profile.run_id, source_epoch=args.epoch, source_step=args.epoch*626,
        checkpoint=receipt, reference_sha256=reference_sha, reference_images=1281167,
        reference_split='train', screen_generated_images=screen_count, screen_seed=screen_seed,
        confirm_generated_images=50000, confirm_seed=confirm_seed, world_size=world, batch=128,
        policies=json.loads(json.dumps(policies)), baseline_fid50k=baseline['fid'], weights_unmodified=True,
        training_updates_added=0, screen_is_not_fid=True,
        confirmation_candidates=args.confirm_count,
        selection=('Lowest feature-moment MMD among nonbaseline candidates' if args.confirm_count==1
            else 'Best feature-moment MMD and best Inception Score; two distinct nonbaseline candidates'),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.panels_module:
        plan['panel_renderer_sha256'] = hashlib.sha256(args.panels_module.read_bytes()).hexdigest()
    if joint_module is not None:
        plan.update(joint_sampler_sha256=hashlib.sha256(args.joint_module.read_bytes()).hexdigest(),
            joint_target='q(a,c) proportional to [p(a)*p(c|a)]**(1/T)',
            joint_atom_normalizer='p(a)**(1/T) * sum_c p(c|a)**(1/T)',
            atom_or_coefficient_truncation=False,image_rejection_sampling=False)
        if args.policies_json:
            plan['requested_policies_sha256'] = hashlib.sha256(args.policies_json.read_bytes()).hexdigest()
        if joint_control:
            assert joint_control['reference_sha256']==reference_sha and joint_control['seed']==confirm_seed
            plan['previous_best_joint_fid50k'] = joint_control['fid']
            plan['joint_support_filter_target'] = ('[p(a)*p(c|a)]**(1/T) times '
                'indicator[p(a)*p(c|a) >= floor_ratio*max_a(p(a))/C]')
        if args.adaptive_search:
            plan['adaptive_rule'] = ('If the best global temperature is the cold .70 boundary, screen .65 once. '
                'Then combine the best unfiltered temperature profile with the best joint-support floor.')
        if args.factorized_controls:
            assert (args.factorized_controls/'report.json').exists()
            old_plan = json.loads((args.factorized_controls/'plan.json').read_text())
            for key in ['checkpoint','reference_sha256','screen_seed','batch','world_size']:
                assert old_plan[key] == plan[key]
            plan['reused_controls'] = str(args.factorized_controls.resolve())
            if rank == 0 and old_plan['screen_generated_images']==screen_count:
                for name in controls:
                    target = out/'screen'/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():
                        target.symlink_to((args.factorized_controls/'screen'/name).resolve(),target_is_directory=True)
            dist.barrier()
        if not (out/'joint-gpu-verification.json').exists():
            labels = (torch.arange(128,device=device)*world+rank)%1000
            torch.manual_seed(261017301+rank)
            started = time.monotonic()
            native = model.sample_sparse(128,aux,cond=labels,amp=True,atom_temperature=1.,
                atom_top_k=0,atom_top_p=1.,coeff_temperature=1.,coeff_top_k=0,coeff_top_p=1.)
            torch.cuda.synchronize()
            native_seconds = time.monotonic()-started
            torch.manual_seed(261017301+rank)
            started = time.monotonic()
            joint,checks = joint_module.sample_joint_compound(model,aux,128,labels,
                temperature=1.,amp=True,verify_conditionals=True)
            torch.cuda.synchronize()
            joint_seconds = time.monotonic()-started
            agreement = float((native==joint).float().mean())
            assert agreement >= .999, f'Unfiltered T=1 control token agreement: {agreement}'
            joint,warm_checks = joint_module.sample_joint_compound(model,aux,128,labels,
                temperature=.9,amp=True,verify_conditionals=True)
            assert torch.isfinite(profile.decode(aux,joint[:8])).all()
            floor_checks = None
            if any(s.get('joint_support_floor',0.) for s in policies.values()):
                floor_codes,floor_checks = joint_module.sample_joint_compound(model,aux,32,labels[:32],
                    temperature=.85,support_floor_ratio=.03,amp=True,verify_conditionals=True)
                assert torch.isfinite(profile.decode(aux,floor_codes[:8])).all()
                del floor_codes
            if rank == 0:
                write_json(out/'joint-gpu-verification.json',dict(passed=True,
                    source_step=args.epoch*626,generated_per_policy=128,
                    native_seconds=native_seconds,joint_t1_seconds=joint_seconds,
                    t1_token_agreement=agreement,conditional_checks=checks,
                    joint_t090_checks=warm_checks,joint_floor_checks=floor_checks,training_updates_added=0))
            del native,joint
            dist.barrier()
    if rank == 0:
        if (out/'plan.json').exists():
            saved = json.loads((out/'plan.json').read_text())
            if saved != plan:
                assert args.allow_implementation_update
                for key in saved:
                    if key not in {'evaluator_sha256','joint_sampler_sha256'}:
                        assert saved[key]==plan[key], f'Evaluation setting changed: {key}'
                archive = out/'plan.before-batching.json'
                assert not archive.exists(), 'An implementation update is already recorded'
                write_json(archive,saved)
                provenance = {p.parent.name:dict(joint_sampler_sha256=saved.get('joint_sampler_sha256'),
                    evaluator_sha256=saved['evaluator_sha256']) for p in (out/'screen').glob('*/result.json')}
                write_json(out/'screen-implementation-provenance.json',provenance)
        write_json(out/'plan.json', plan)
        write_json(out/'status.json', dict(phase='screening', pid=os.getpid(), time=time.time()))
        print(json.dumps(plan), flush=True)
    run = None
    if rank == 0 and not args.offline:
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        import wandb
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=out.name,
            name=out.name, resume='allow', job_type='fixed-checkpoint-sampler-sweep',
            config=None if args.allow_implementation_update else plan,
            settings=wandb.Settings(console='off'))
        if args.allow_implementation_update:
            run.config.update(plan,allow_val_change=True)
        run.define_metric('screen/index')
        run.define_metric('screen/*', step_metric='screen/index')
        run.define_metric('confirm/index')
        run.define_metric('confirm/fid50k', step_metric='confirm/index', summary='min,last')
        run.define_metric('samples/global_step')
        run.define_metric('samples/*', step_metric='samples/global_step')
        write_json(out/'wandb.json', dict(id=run.id, url=run.url))
    metrics.INCEPTION_PATH = base/'upstream/rqvae/metrics/inception.py'
    reference_mean, reference_cov = metrics.load_reference_statistics(reference_path)
    standard_deviation = np.sqrt(np.maximum(np.diag(reference_cov), 1.e-8))
    reference_second = reference_cov / np.outer(standard_deviation, standard_deviation) * (1281166/1281167)
    mean_t = torch.as_tensor(reference_mean, dtype=torch.float64, device=device)
    std_t = torch.as_tensor(standard_deviation, dtype=torch.float64, device=device)
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    shared.BASE, shared.PROFILE, shared.RUN = base, profile, profile.run_id
    shared.original_metrics.INCEPTION_PATH = metrics.INCEPTION_PATH

    @torch.inference_mode()
    def screen(name, index):
        destination = out/'screen'/name
        result_path = destination/'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result['sampling'] == policies[name] and result['generated_images'] == screen_count
            dist.barrier()
            return result
        profile.recipe.sampling = OmegaConf.create(policies[name])
        joint_statistics.update(proposals=0,accepted=0,max_rounds=0)
        torch.manual_seed(screen_seed+rank)
        torch.cuda.manual_seed(screen_seed+rank)
        total = torch.zeros(2048, dtype=torch.float64, device=device)
        cross = torch.zeros(2048, 2048, dtype=torch.float64, device=device)
        norm4 = torch.zeros((), dtype=torch.float64, device=device)
        agreements = torch.zeros(2, dtype=torch.float64, device=device)
        probabilities = []
        started = time.monotonic()
        for first in range(0, screen_count//world, 128):
            count = min(128, screen_count//world-first)
            labels = (torch.arange(first, first+count, device=device)*world+rank)%1000
            codes = profile.sample(model, aux, count, labels)
            atoms = codes[..., 0]
            agreements[0] += (atoms[:, 1:] == atoms[:, :-1]).sum() + (atoms[:, :, 1:] == atoms[:, :, :-1]).sum()
            agreements[1] += count*2*8*7
            for part in codes.split(8):
                images = profile.decode(aux, part).float().mul(.5).add(.5).clamp(0, 1)
                features, logits = inception(images, return_logits=True)
                z = (features.double()-mean_t)/std_t
                total += z.sum(0)
                cross.addmm_(z.T, z)
                norm4 += z.square().sum(-1).square().sum()
                probabilities.append(logits.softmax(-1))
            if rank == 0:
                write_json(out/'progress.json', dict(phase='screening', policy=name,
                    index=index, policies=len(policies), generated=(first+count)*world,
                    target=screen_count, time=time.time()))
        for tensor in (total, cross, norm4, agreements):
            dist.all_reduce(tensor)
        probabilities = torch.cat(probabilities)
        gathered = [torch.empty_like(probabilities) for _ in range(world)]
        dist.all_gather(gathered, probabilities)
        probabilities = torch.cat(gathered)
        torch.manual_seed(screen_seed)
        is_mean, is_std = metrics.inception_score(probabilities[torch.randperm(screen_count, device=device)])
        result = dict(policy=name, sampling=policies[name], generated_images=screen_count,
            seed=screen_seed, moment_mmd=moment_mmd(total.cpu().numpy(), cross.cpu().numpy(),
                float(norm4), screen_count, reference_second),
            inception_score=is_mean, inception_score_std=is_std,
            first_atom_neighbor_repeat=float(agreements[0]/agreements[1]),
            not_fid=True, elapsed_seconds=time.monotonic()-started)
        if policies[name].get('joint_temperature') is not None:
            result['joint_sampler_statistics'] = dict(joint_statistics)
            result['joint_sampler_sha256'] = plan['joint_sampler_sha256']
        # Each preview uses the designated classes, ten in the overview and 64 per grid.
        designated_preview(profile, model, aux, device, rank, world, args.epoch*626,
            None, destination)
        if rank == 0:
            write_json(result_path, result)
            if run:
                run.log({'screen/index':index, 'screen/setting':name,
                    'screen/moment_mmd':result['moment_mmd'], 'screen/inception_score':is_mean,
                    'screen/first_atom_neighbor_repeat':result['first_atom_neighbor_repeat']})
            print(json.dumps(dict(event='screen_complete', **result)), flush=True)
        dist.barrier()
        torch.cuda.empty_cache()
        return result

    results = [screen(name, index) for index, name in enumerate(policies)]
    adaptive_policies = {}
    if args.adaptive_search:
        global_results = [r for r in results if r['sampling'].get('joint_temperature') is not None
            and not r['sampling'].get('joint_temperature_by_depth')
            and not r['sampling'].get('joint_support_floor',0.)]
        best_global = min(global_results,key=lambda r:r['moment_mmd'])
        if float(best_global['sampling']['joint_temperature'])==.7:
            setting = dict(best_global['sampling'])
            for key in ['joint_temperature','atom_temperature','coefficient_temperature']:
                setting[key]=.65
            name='adaptive_joint_t065'
            policies[name]=setting
            adaptive_policies[name]=setting
            results.append(screen(name,len(results)))
        unfiltered = [r for r in results if r['sampling'].get('joint_temperature') is not None
            and not r['sampling'].get('joint_support_floor',0.)]
        filtered = [r for r in results if r['sampling'].get('joint_support_floor',0.)]
        best_temperature = min(unfiltered,key=lambda r:r['moment_mmd'])
        if filtered:
            best_floor = min(filtered,key=lambda r:r['moment_mmd'])
            setting = dict(best_temperature['sampling'])
            setting['joint_support_floor'] = best_floor['sampling']['joint_support_floor']
            if setting not in policies.values():
                name='adaptive_temperature_and_floor'
                policies[name]=setting
                adaptive_policies[name]=setting
                results.append(screen(name,len(results)))
        if rank==0:
            write_json(out/'adaptive-plan.json',dict(rule=plan['adaptive_rule'],policies=adaptive_policies))
    control_names = {'no_atom_topk','previous_laser'} if joint_module is not None else {'no_atom_topk'}
    if joint_control:
        control_names.add(joint_control['policy'])
    challengers = [r for r in results if r['policy'] not in control_names]
    best_moment = min(challengers, key=lambda r: (r['moment_mmd'], r['policy']))
    best_is = max((r for r in challengers if r['policy'] != best_moment['policy']),
        key=lambda r: (r['inception_score'], -r['moment_mmd']))
    selected = [best_moment['policy'], best_is['policy']]
    selected = selected[:args.confirm_count]
    if rank == 0:
        write_json(out/'screen-summary.json', dict(results=results, selected=selected,
            not_fid=True, limits='Small-sample diagnostics select candidates; only FID50k confirms improvements.'))
        write_json(out/'status.json', dict(phase='confirming', selected=selected, time=time.time()))
        print(json.dumps(dict(event='confirmation_selected', selected=selected)), flush=True)
    del inception
    torch.cuda.empty_cache()
    confirmations = [dict(policy='no_atom_topk', fid=baseline['fid'],
        sampling=policies['no_atom_topk'], provenance=str(baseline_path),
        generated_images=50000, reference_images=1281167, reference_split='train', seed=confirm_seed)]
    if joint_module is not None and args.factorized_controls:
        prior_best = json.loads((args.factorized_controls/'confirm/previous_laser/fid50k.json').read_text())
        assert prior_best['checkpoint']['sha256'] == receipt['sha256']
        assert prior_best['seed'] == confirm_seed and prior_best['generated_images'] == 50000
        confirmations.append(prior_best)
    if joint_control:
        confirmations.append(joint_control)
    if run:
        run.log({'confirm/index':0, 'confirm/setting':'no_atom_topk', 'confirm/fid50k':baseline['fid']})
        if len(confirmations)>1:
            run.log({'confirm/index':1,'confirm/setting':'previous_laser','confirm/fid50k':confirmations[1]['fid']})
        if joint_control:
            run.log({'confirm/index':2,'confirm/setting':joint_control['policy'],'confirm/fid50k':joint_control['fid']})
    for index, name in enumerate(selected, len(confirmations)):
        destination = out/'confirm'/name
        result_path = destination/'fid50k.json'
        profile.recipe.sampling = OmegaConf.create(policies[name])
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result['sampling'] == policies[name] and result['generated_images'] == 50000
        else:
            shared.OUT = destination
            if rank == 0:
                write_json(out/'progress.json', dict(phase='confirming', policy=name, index=index,
                    target=50000, time=time.time()))
            started = time.monotonic()
            joint_statistics.update(proposals=0,accepted=0,max_rounds=0)
            result = shared.generation(model, aux, device, rank, world, args.epoch, None,
                batch=128, seed=confirm_seed)
            result.update(policy=name, sampling=policies[name], source_run=profile.run_id,
                source_step=args.epoch*626, source_epoch=args.epoch, checkpoint=receipt,
                generated_images=50000, reference_images=1281167, reference_split='train',
                reference_sha256=reference_sha, seed=confirm_seed, world_size=world, batch=128,
                baseline_fid=baseline['fid'], fid_delta=result['fid']-baseline['fid'],
                weights_unmodified=True, training_updates_added=0, rejection_sampling=False,
                elapsed_seconds=time.monotonic()-started)
            if policies[name].get('joint_temperature') is not None:
                result.update(joint_target=plan['joint_target'],
                    joint_sampler_statistics=dict(joint_statistics),image_rejection_sampling=False)
            if rank == 0:
                write_json(result_path, result)
        confirmations.append(result)
        if rank == 0:
            if run:
                run.log({'confirm/index':index, 'confirm/setting':name,
                    'confirm/fid50k':result['fid'], 'confirm/inception_score':result['inception_score']})
            print(json.dumps(dict(event='confirmation_complete', **result)), flush=True)
        dist.barrier()
    assert [(name, p._version) for name, p in model.named_parameters()] == parameter_versions
    assert profile.digest(aux) == frozen
    winner = min(confirmations, key=lambda r: r['fid'])
    if rank == 0:
        report = dict(plan=plan, screen=results, confirmation=confirmations,
            adaptive_policies=adaptive_policies,
            selected_policy=winner['policy'], selected_sampling=winner['sampling'],
            selected_fid50k=winner['fid'], baseline_fid50k=baseline['fid'],
            improvement=baseline['fid']-winner['fid'], weights_unmodified=True,
            tokenizer_unmodified=True, training_updates_added=0,
            active_training_sampler_changed=False,
            limits='One fixed early checkpoint and one full evaluation seed; no claim about later epochs.')
        if joint_control:
            report.update(previous_best_fid50k=joint_control['fid'],
                improvement_over_previous_best=joint_control['fid']-winner['fid'])
        write_json(out/'report.json', report)
        write_json(out/'selected-sampling.json', dict(sampling=winner['sampling'],
            source_run=profile.run_id, source_step=args.epoch*626, checkpoint=receipt,
            fid50k=winner['fid'], baseline_fid50k=baseline['fid'],
            generated_images=50000, reference_images=1281167, reference_split='train'))
        if run:
            import wandb
            run.summary.update(dict(selected_policy=winner['policy'], selected_fid50k=winner['fid'],
                baseline_fid50k=baseline['fid'], improvement=report['improvement'],
                active_training_sampler_changed=False))
            rows = [[r['policy'], r['fid'], r['sampling']['atom_temperature'],
                r['sampling']['atom_top_p'], r['sampling']['coefficient_temperature'],
                r['sampling']['coefficient_top_k'], r['sampling']['coefficient_top_p']]
                for r in confirmations]
            run.log({'sampling/results':wandb.Table(columns=['policy','fid50k','atom_T','atom_p',
                'coefficient_T','coefficient_k','coefficient_p'], data=rows)})
            from src.training.rq_sample_panels import MEDIA_KEYS
            folder = out/'screen'/winner['policy']/'smoke/samples'
            paths = {key:folder/f'step{args.epoch*626:06d}-{key.split("/")[-1]}.png' for key in MEDIA_KEYS}
            run.log({'samples/global_step':args.epoch*626, **{key:wandb.Image(str(path),
                caption=f"Fixed epoch-2 checkpoint; {winner['policy']}; {winner['sampling']}")
                for key, path in paths.items()}})
            run.save(str(out/'report.json'), base_path=str(out), policy='now')
            run.finish()
        write_json(out/'status.json', dict(phase='completed', selected_policy=winner['policy'],
            fid=winner['fid'], time=time.time()))
        print(json.dumps(dict(event='completed', selected_policy=winner['policy'], fid=winner['fid'])), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
