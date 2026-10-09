"""Resume-safe, fixed-epoch77 sampling sweep using official RQ-Transformer50k."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import types


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def load_helper(path):
    spec = importlib.util.spec_from_file_location('frozen_pair_sampling', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_plan(path, helper):
    plan = json.loads(path.read_text())
    if plan['generated_images'] != 50000 or plan['real_images'] != 50000:
        raise ValueError('Only full official50k evaluations are permitted')
    settings = {}
    laws = set()
    for item in plan['policies']:
        name = item['name']
        if not re.fullmatch('[a-z0-9][a-z0-9-]{0,70}', name) or name in settings:
            raise ValueError('Invalid or duplicate policy name')
        values = dict(item['policy'])
        values['coefficient_temperatures'] = tuple(values['coefficient_temperatures'])
        policy = helper.PairSamplingPolicy(**values)
        if policy.mode not in ('joint', 'ancestral') or policy.atom_proposal != 'prior' or policy.geometry_weight != 0:
            raise ValueError('Sweep requires full-prior proposals and no geometry heuristic')
        if not isinstance(policy.candidate_atoms, int) or not 1 <= policy.candidate_atoms <= 64:
            raise ValueError('Candidate count exceeds tested bounds')
        if len(policy.coefficient_temperatures) != 4:
            raise ValueError('Four coefficient depths required')
        if any(not math.isfinite(t) or t <= 0 for t in (policy.atom_temperature, *policy.coefficient_temperatures)):
            raise ValueError('Invalid temperature')
        if any(not 0 < p <= 1 for p in (policy.atom_top_p, policy.coefficient_top_p, policy.joint_top_p)):
            raise ValueError('Invalid nucleus')
        law = asdict(policy)
        if policy.mode == 'joint':
            # Joint nucleus replaces the independent coefficient nucleus.
            law.pop('coefficient_top_p')
        else:
            for key in ('candidate_atoms', 'joint_top_p', 'atom_proposal'):
                law.pop(key)
        key = json.dumps(law, sort_keys=True)
        if key in laws:
            raise ValueError('Duplicate effective sampling distributions')
        laws.add(key)
        settings[name] = policy
    if not {'anchor', 'native-control'} <= set(settings):
        raise ValueError('Both anchor and native control required')
    if plan['seed'] == plan['confirmation_seed']:
        raise ValueError('Confirmation seed must be independent')
    return plan, settings


def identity(args):
    return dict(plan_sha256=digest(args.plan), helper_sha256=digest(args.helper),
                evaluator_sha256=digest(Path(__file__)), source_epoch=77,
                source_global_step=48202, generation_batch_per_rank=args.generation_batch_size,
                decode_batch=args.decode_batch_size, world_size=8,
                metric_backend='original_rqtransformer', generated_images=50000,
                real_images=50000, real_split='val', inception_splits=10)


def completed_result(output, name, policy, seed, protocol):
    path = output / f'official-{name}.json'
    marker = output / f'completed-{name}.json'
    grid = output / f'official-{name}-samples.png'
    if not path.exists() or not marker.exists() or not grid.exists():
        return None
    result = json.loads(path.read_text())
    expected_policy = json.loads(json.dumps(asdict(policy)))
    if result.get('protocol') != protocol or result.get('policy') != expected_policy or result.get('seed') != seed:
        raise ValueError(f'Refusing incompatible completed result: {name}')
    if any(not math.isfinite(result[key]) for key in ('fid', 'inception_score', 'inception_score_std')):
        raise ValueError('Nonfinite official score')
    proof = json.loads(marker.read_text())
    if proof != dict(result_sha256=digest(path), grid_sha256=digest(grid)):
        raise ValueError(f'Completed result digest changed: {name}')
    return result


def confirmation_names(results, settings):
    ranked = sorted(settings, key=lambda name: results[name]['fid'])
    best_is = max(settings, key=lambda name: results[name]['inception_score'])
    return list(dict.fromkeys([*ranked[:2], best_is, 'anchor', 'native-control']))


def select_and_upload(args, run, results, plan, settings):
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    exploration = {k: v for k, v in results.items() if k in settings}
    best_fid = min(exploration, key=lambda k: exploration[k]['fid'])
    best_is = max(exploration, key=lambda k: exploration[k]['inception_score'])
    latest = next(reversed(results))
    selection = dict(provisional=True, source_epoch=77, source_global_step=48202,
        best_fid_policy=best_fid, best_fid=exploration[best_fid]['fid'],
        best_is_policy=best_is, best_inception_score=exploration[best_is]['inception_score'],
        latest_policy=latest, completed_evaluations=list(results),
        exploration_seed=plan['seed'], confirmation_seed=plan['confirmation_seed'])
    if len(exploration) == len(settings):
        candidates = confirmation_names(results, settings)
        selection['confirmation_policies'] = candidates
        if all('confirm-' + k in results for k in candidates):
            means = {k: (results[k]['fid'] + results['confirm-' + k]['fid']) / 2 for k in candidates}
            selection.update(provisional=False, confirmed_best_fid_policy=min(means, key=means.get),
                             two_seed_mean_fid=means)
    record(args.output / 'official-sampler-selection.json', selection)
    record(args.output / 'official-comparison-results.json', results)
    for kind, name in [('best-fid', best_fid), ('best-is', best_is), ('last', latest)]:
        shutil.copyfile(args.output / f'official-{name}-samples.png', args.output / f'{kind}-samples.png')
    run.summary['selection/provisional'] = selection['provisional']
    run.summary['selection/best_fid_policy'] = best_fid
    run.summary['selection/best_is_policy'] = best_is
    if not selection['provisional']:
        run.summary['selection/confirmed_best_fid_policy'] = selection['confirmed_best_fid_policy']
    paths = [args.output / n for n in ('official-sampler-selection.json', 'official-comparison-results.json',
                                      'best-fid-samples.png', 'best-is-samples.png', 'last-samples.png')]
    name = latest
    paths += [args.output / f'official-{name}.json', args.output / f'official-{name}-samples.png',
              args.output / f'completed-{name}.json']
    VerifiedCloudUpload(f'helloimlixin-rutgers/laser/{args.run_id}',
                        args.output / 'selected-sampler-cloud-receipt.json')(paths, epoch=77)


def load_model(args, device):
    import torch
    from src.training.rqtransformer import LaserAux, build_model
    source = args.base / 'inputs/source-epoch077-full.pt'
    payload = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    assert (payload['epoch'], payload['global_step']) == (77, 48202)
    assert payload['original_rqtransformer_metrics']['fid'] == 15.24941539209243
    config = payload['config']
    aux = LaserAux(args.base / 'inputs/resume-stage1-tokenizer.pt', config['num_atoms'],
        config['coeff_vocab_size'], config['coeff_max'], config['coeff_scale'],
        coeff_scales=config['coeff_scales'], soft_target_physical=False,
        clamp_coeffs=False, sparsity_level=4).to(device).eval()
    native_decode = aux.decode_tokens
    def bounded_decode(tokens):
        return torch.cat([native_decode(chunk) for chunk in tokens.split(args.decode_batch_size)], dim=0)
    aux.decode_tokens = bounded_decode
    with torch.device('meta'):
        model = build_model(config['num_atoms'] + config['coeff_vocab_size'], config['num_atoms'],
            physical_pair_context=True, sparsity_level=4, coeff_vocab_size=config['coeff_vocab_size'],
            model_preset=config['model_preset'])
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).to(device).eval()
    return model, aux, config


def evaluate(args):
    sys.path[:0] = [str(args.base / 'source/runtime'), str(args.base / 'support')]
    import torch
    import torch.distributed as dist
    helper = load_helper(args.helper)
    plan, settings = load_plan(args.plan, helper)
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib * 2**30 / total, device)
    if args.action != 'probe':
        assert int(os.environ['WORLD_SIZE']) == 8
        dist.init_process_group('nccl', timeout=timedelta(hours=3))
    model, aux, config = load_model(args, device)
    if args.action == 'probe':
        policy = max(settings.values(), key=lambda p: p.candidate_atoms)
        with torch.inference_mode():
            tokens = helper.sample_physical_pairs(model, args.generation_batch_size, aux,
                cond=torch.arange(args.generation_batch_size, device=device), policy=policy, amp=True)
            decoded = aux.decode_tokens(tokens)
            assert torch.isfinite(decoded).all()
        record(args.output / 'memory-probe.json', dict(finite=True, no_quality_metric_computed=True,
            batch=args.generation_batch_size, policy=asdict(policy),
            max_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            max_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            allocator_limit_gib=args.memory_limit_gib, time=time.time()))
        return
    from torch.utils.data import DataLoader, DistributedSampler
    from torchvision.datasets import ImageFolder
    from src.training.rqtransformer import val_image_transform, evaluate_generation_metrics, save_class_labeled_grid
    from src.data.imagenet_labels import class_names_for_dataset
    from official_metrics import install
    install(args.base / 'source/runtime')
    dataset = ImageFolder(Path(config['data']) / 'val', transform=val_image_transform())
    assert len(dataset) == 50000 and len(dataset.classes) == 1000
    loader = DataLoader(dataset, batch_size=64,
        sampler=DistributedSampler(dataset, num_replicas=8, rank=rank, shuffle=False, drop_last=False),
        num_workers=4, pin_memory=True)
    names = class_names_for_dataset('imagenet', dataset.classes)
    protocol = identity(args)
    run = None
    if rank == 0:
        os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
        import wandb
        wandb_dir = args.base / 'sampling-sweep-wandb'
        wandb_dir.mkdir(exist_ok=True)
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=args.run_id,
            resume='allow', mode='online', dir=str(wandb_dir),
            config=dict(plan=plan, protocol=protocol, concurrent_training=True), allow_val_change=True)
        run.summary['execution/state'] = 'evaluating_official50k'
        run.summary['evaluation/official_results_pending'] = True
        from verified_wandb_checkpoint_upload import VerifiedCloudUpload
        VerifiedCloudUpload(f'helloimlixin-rutgers/laser/{args.run_id}',
            args.output / 'online-plan-proof.json')([args.plan, args.helper, Path(__file__),
                                                    args.output / 'memory-probe.json'], epoch=77)
    results = {}
    queue = [(name, policy, plan['seed']) for name, policy in settings.items()]
    for name, policy, seed in queue:
        prior = completed_result(args.output, name, policy, seed, protocol)
        if prior is None:
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            if rank == 0:
                record(args.output / 'active-policy.json', dict(active_policy=name,
                    completed_policies=list(results), seed=seed, time=time.time()))
                print(json.dumps(dict(phase='official_sampler_begin', sampler=name,
                    generated_images=50000, seed=seed)), flush=True)
            def sample(self, batch_size, model_aux, cond=None, **kwargs):
                return helper.sample_physical_pairs(self, batch_size, model_aux, cond,
                    policy=policy, amp=kwargs.get('amp', True))
            model.sample_sparse = types.MethodType(sample, model)
            with torch.random.fork_rng(devices=[local_rank]):
                torch.manual_seed(seed + rank)
                torch.cuda.manual_seed(seed + rank)
                fid, score, std = evaluate_generation_metrics(model, aux, loader, num_samples=50000,
                    batch_size=args.generation_batch_size, num_condition_classes=1000,
                    atom_temperature=.9, atom_top_p=.9, coeff_temperature=1., coeff_top_p=.85,
                    metric_backend='original-rqvae', compute_inception_score=True)
            result = dict(fid=fid, inception_score=score, inception_score_std=std, seed=seed,
                source_epoch=77, source_global_step=48202, metric_backend='original_rqtransformer',
                generated_images=50000, real_images=50000, real_split='val', inception_splits=10,
                elapsed_seconds=time.monotonic()-started, policy=asdict(policy), protocol=protocol)
            classes = torch.tensor([269,612,265,628,689,301,932,798], device=device)
            with torch.inference_mode(), torch.random.fork_rng(devices=[local_rank]):
                torch.manual_seed(261006 + rank)
                torch.cuda.manual_seed(261006 + rank)
                tokens = model.sample_sparse(8, aux, cond=classes[rank:rank+1].repeat(8), amp=True)
                images = aux.decode_tokens(tokens).float().add(1).mul(.5).clamp(0,1)
                gathered = [torch.empty_like(images) for _ in range(8)] if rank == 0 else None
                dist.gather(images, gather_list=gathered, dst=0)
                if rank == 0:
                    path = args.output / f'official-{name}-samples.png'
                    save_class_labeled_grid(torch.cat(gathered).cpu(), classes.cpu(), names, path, samples_per_class=8)
            record(args.output / f'memory-{name}-rank{rank}.json', dict(
                max_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                max_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                allocator_limit_gib=args.memory_limit_gib, time=time.time()))
            if rank == 0:
                path = args.output / f'official-{name}.json'
                record(path, result)
                record(args.output / f'completed-{name}.json', dict(result_sha256=digest(path),
                    grid_sha256=digest(args.output / f'official-{name}-samples.png')))
                run.log({f'eval/{name}/fid_original_rqtransformer':fid,
                    f'eval/{name}/inception_score_original_rqtransformer':score,
                    f'eval/{name}/inception_score_std_original_rqtransformer':std,
                    'sampler/policy':name, 'sampler/seed':seed})
                import wandb
                run.log({f'samples/{name}':wandb.Image(str(args.output / f'official-{name}-samples.png'))})
                for key in ('fid','inception_score','inception_score_std'):
                    run.summary[f'eval/{name}/{key}_original_rqtransformer'] = result[key]
                print(json.dumps(dict(sampler=name, **result)), flush=True)
            prior = result
        results[name] = prior
        if rank == 0:
            select_and_upload(args, run, results, plan, settings)
        dist.barrier()
        if len(results) == len(settings):
            chosen = confirmation_names(results, settings)
            record(args.output / 'confirmation-plan.json', dict(policies=chosen,
                seed=plan['confirmation_seed'], selected_from_seed=plan['seed'])) if rank == 0 else None
            queue.extend(('confirm-'+key, settings[key], plan['confirmation_seed']) for key in chosen)
    if rank == 0:
        run.summary['execution/state'] = 'completed'
        run.summary['evaluation/official_results_pending'] = False
        run.summary['evaluation/official_evaluation_completed'] = True
        run.finish(exit_code=0)
    dist.destroy_process_group()


def supervise(args):
    import fcntl
    lock = (args.output / 'sweep.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    load_plan(args.plan, load_helper(args.helper))
    probe = json.loads((args.output / 'memory-probe.json').read_text())
    if not probe['finite'] or probe['batch'] != args.generation_batch_size:
        raise ValueError('Largest-candidate GPU probe must pass before launching')
    usage = subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    available = [(float(row.split(',')[0])-float(row.split(',')[1]))/1024
                 for row in usage.stdout.strip().splitlines()]
    if len(available) != 8 or min(available) < args.memory_limit_gib + 1.5:
        raise RuntimeError(f'Insufficient concurrent GPU memory: {available}')
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        str(Path(__file__).resolve()),'evaluate']
    for key in ('base','output','key_file','helper','plan','run_id','generation_batch_size','decode_batch_size','memory_limit_gib'):
        command += ['--'+key.replace('_','-'), str(getattr(args,key))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTHONUNBUFFERED='1',
        TORCH_HOME=str(args.base/'torch-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_NVLS_ENABLE='0')
    child = None
    def stop(sig, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        record(args.output/'sweep-status.json',dict(state='cancelled',time=time.time()))
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    with (args.output/'official-sweep.log').open('a') as stream:
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            record(args.output/'sweep-status.json',dict(state='running',supervisor_pid=os.getpid(),
                torchrun_pid=child.pid,run_id=args.run_id,source_epoch=77,official_images=50000,
                per_process_memory_cap_gib=args.memory_limit_gib,time=time.time()))
            time.sleep(5)
    record(args.output/'sweep-status.json',dict(state='completed' if child.returncode==0 else 'failed',
        exit_code=child.returncode,run_id=args.run_id,time=time.time()))
    if child.returncode:
        raise RuntimeError('Sweep failed; committed evaluations remain resumable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['now','evaluate','probe'])
    for name in ('base','output','key-file','helper','plan'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--generation-batch-size',type=int,default=64)
    parser.add_argument('--decode-batch-size',type=int,default=4)
    parser.add_argument('--memory-limit-gib',type=float,default=10.)
    args = parser.parse_args()
    if min(args.generation_batch_size,args.decode_batch_size,args.memory_limit_gib) <= 0:
        parser.error('Positive batches and memory cap required')
    args.output.mkdir(parents=True,exist_ok=True)
    supervise(args) if args.action == 'now' else evaluate(args)


if __name__ == '__main__':
    main()
