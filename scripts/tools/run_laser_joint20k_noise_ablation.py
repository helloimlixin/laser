#!/usr/bin/env python3
"""Compare target temperatures from scratch with a fixed eight-GPU recipe."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
CONTROL = 'imagenet-laser-joint20k-depth5-rqnoise-k2048-rngfix-scratch100-8h100-20261017'
CONTROL_OUT = ROOT/'outputs'/CONTROL
CONTROL_BASE = Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-rngfix-8h100-imagenet-20261017')
SUITE = 'imagenet-laser-joint20k-depth5-noise-ablation-20261017'
OUT = ROOT/'outputs'/SUITE
BASE = Path('/tmp/laser-joint20k-noise-ablation-20261017')
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key')
WORLD = 8
STOP_EPOCH = 4
STOP_STEP = 626 * STOP_EPOCH


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def temperature_variants(current, earlier):
    if (len(current) != 5 or len(earlier) != 5 or
            any(not math.isfinite(t) or t <= 0 for t in [*current, *earlier])):
        raise ValueError('Five finite positive temperatures are required')
    return {'earlier': list(earlier), 'restore_early': [*earlier[:2], *current[2:]]}


def behavior(recipe):
    """Strip identity, provenance, and the optional weight-hash assertion."""
    result = copy.deepcopy(recipe)
    for key in ('version', 'run_id', 'name', 'lineage'):
        result.pop(key, None)
    result['targets'].pop('calibration', None)
    optimization = result.get('optimization', {})
    optimization.pop('probe_initial_weight_sha256', None)
    if not optimization:
        result.pop('optimization', None)
    return result


def assert_temperature_only(control, candidate, temperatures):
    expected = behavior(control)
    expected['targets']['temperature'] = list(temperatures)
    if behavior(candidate) != expected:
        raise ValueError('Trial changes training or sampling beyond target temperatures')
    if (candidate['epochs'] != 100 or candidate['lr_schedule_epochs'] != 100 or
            candidate['global_batch'] != 2048 or candidate['microbatch'] != 64 or
            candidate['training_rng'] != {'policy': 'independent_rank_v1', 'base_seed': 261001}):
        raise ValueError('Trial must preserve the established batch, horizon, and RNG policy')


def comparison_rows(evaluations):
    rows = []
    for epoch in (1, 2, 4):
        available = {}
        for name, values in evaluations.items():
            row = values.get(epoch)
            if row is None:
                continue
            if (row['train/global_step'] != epoch*626 or row['generated_images'] != 50000 or
                    row['real_images'] != 1281167 or row['real_split'] != 'train' or
                    row['rejection_sampling']):
                raise ValueError('Incompatible FID evaluation')
            fid = float(row['eval/fid'])
            if not math.isfinite(fid):
                raise ValueError('FID must be finite')
            available[name] = fid
        baseline = available.get('current')
        rows.append(dict(epoch=epoch, global_step=epoch*626, fid=available,
            improvement_over_current={} if baseline is None else
                {name: baseline-value for name, value in available.items() if name != 'current'}))
    return rows


def prepare():
    import yaml
    if OUT.exists() or BASE.exists():
        raise RuntimeError('Refusing to replace an existing ablation')
    current = yaml.safe_load((CONTROL_BASE/'inputs/recipe.yaml').read_text())
    calibration = ROOT/'outputs/imagenet-laser-joint20k-depth5-rq-noise-calibration-20261017/summary.json'
    earlier = json.loads(calibration.read_text())['current_temperatures']
    variants = temperature_variants(current['targets']['temperature'], earlier)
    if current['sampling'] != {'temperature': 1., 'top_k': 2048, 'top_p': .95}:
        raise ValueError('Unexpected control sampler')
    # Reuse the frozen runtime that produced the control, including its RNG fix.
    manifest = json.loads((CONTROL_OUT/'reproducibility/code-manifest.json').read_text())
    for name, expected in manifest.items():
        if sha(CONTROL_BASE/name) != expected:
            raise RuntimeError(f'Control snapshot changed: {name}')
    BASE.mkdir(parents=True)
    OUT.mkdir(parents=True)
    trials = []
    for variant, temperatures in variants.items():
        run = f'imagenet-laser-joint20k-depth5-{variant.replace("_", "")}-k2048-rngfix-trial4-8h100-20261017'
        base, out = BASE/variant, ROOT/'outputs'/run
        if out.exists():
            raise RuntimeError(f'Trial already exists: {out}')
        base.mkdir()
        out.mkdir()
        for directory in ('source', 'runtime'):
            shutil.copytree(CONTROL_BASE/directory, base/directory,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for directory in ('upstream', 'data', 'torch-cache'):
            (base/directory).symlink_to(CONTROL_BASE/directory, target_is_directory=True)
        (base/'inputs').mkdir()
        for source in (CONTROL_BASE/'inputs').iterdir():
            target = base/'inputs'/source.name
            if source.suffix in ('.pt', '.npz'):
                os.link(source, target)
            else:
                shutil.copyfile(source, target)
        for name in ('train.py', 'verify.py'):
            shutil.copyfile(CONTROL_BASE/name, base/name)
        recipe = copy.deepcopy(current)
        recipe.update(run_id=run, version='laser-noise-temperature-ablation-v1',
            name=f'LASER depth5 | noise {variant} | paired fresh 4 epochs | 8 H100',
            lineage=dict(ablation=SUITE, control=CONTROL, stage2_initialization='scratch'))
        recipe['targets']['temperature'] = temperatures
        recipe['targets']['calibration'] = dict(policy='controlled training ablation',
            hypothesis=('Restore all earlier temperatures' if variant == 'earlier' else
                        'Restore stochastic variation at depths 1 and 2; retain current depths 3 to 5'),
            generation_improvement_established=False)
        assert_temperature_only(current, recipe, temperatures)
        (base/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
        config = yaml.safe_load((base/'inputs/stage2-training-config.yaml').read_text())
        config['loss']['temp'] = temperatures
        (base/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        environment = json.loads((CONTROL_BASE/'environment.json').read_text())
        environment = {key:value.replace(str(CONTROL_BASE), str(base)) for key,value in environment.items()}
        write(base/'environment.json', environment)
        (base/'wandb').mkdir()
        trials.append(dict(variant=variant, run_id=run, base=str(base), output=str(out),
            temperatures=temperatures, url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run}'))
    plan = dict(control=CONTROL, control_output=str(CONTROL_OUT), trials=trials,
        world_size=WORLD, global_batch=2048, microbatch=64, accumulation_steps=4,
        stop_after_epochs=STOP_EPOCH, stop_after_updates=STOP_STEP, lr_schedule_epochs=100,
        evaluation_epochs=[1,2,4], generated_images=50000, real_images=1281167,
        sampling=current['sampling'], seed=261001, independent_rank_rng=True,
        fresh_stage2=True, fresh_optimizer=True, primary_selection_epoch=4,
        auto_promote=False, automatic_baseline_resume=False,
        limits='One paired training seed and one FID seed. Early-horizon comparison; no claim of final convergence.',
        reason='User prioritized controlled tests over the current long run',
        time=time.time())
    write(OUT/'plan.json', plan)
    shutil.copyfile(ROOT/'outputs/imagenet-laser-joint20k-depth5-epochs1-2-k2048-control-20261017/report.json',
                    OUT/'prior-sampler-control.json')
    write(OUT/'status.json', dict(phase='prepared', pid=os.getpid(), time=time.time()))
    print(json.dumps(plan), flush=True)


def verify_initialization():
    import random
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf
    os.environ['RQ_UPSTREAM_ROOT'] = str(CONTROL_BASE/'upstream')
    sys.path[:0] = [str(CONTROL_BASE/'source'), str(CONTROL_BASE/'upstream'), str(CONTROL_BASE/'runtime')]
    from src.training.laser_joint_recipe import JointLaserRecipe
    from rqvae.utils.config import load_config, augment_arch_defaults
    torch.set_num_threads(8)
    plan = json.loads((OUT/'plan.json').read_text())
    hashes = {}
    for name, base in [('current', CONTROL_BASE), *[(t['variant'], Path(t['base'])) for t in plan['trials']]]:
        random.seed(261001); np.random.seed(261001); torch.manual_seed(261001)
        profile = JointLaserRecipe(base)
        aux, _ = profile.tokenizer(torch.device('cpu'))
        config = load_config(base/'inputs/stage2-training-config.yaml')
        arch = augment_arch_defaults(config.arch)
        arch.vocab_size = config.dataset.vocab_size
        model = profile.model(arch)
        profile.adapt(model, aux)
        digest = hashlib.sha256()
        for key, value in model.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().contiguous().numpy().tobytes())
        hashes[name] = digest.hexdigest()
        del model, aux
        print(json.dumps(dict(initialization=name, sha256=hashes[name])), flush=True)
    if len(set(hashes.values())) != 1:
        raise RuntimeError('Trial initial weights differ')
    current = yaml.safe_load((CONTROL_BASE/'inputs/recipe.yaml').read_text())
    for trial in plan['trials']:
        base, out = Path(trial['base']), Path(trial['output'])
        recipe = yaml.safe_load((base/'inputs/recipe.yaml').read_text())
        recipe.setdefault('optimization', {})['probe_initial_weight_sha256'] = hashes['current']
        assert_temperature_only(current, recipe, trial['temperatures'])
        (base/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
        repro = out/'reproducibility'
        repro.mkdir()
        for name in ('recipe.yaml', 'stage2-training-config.yaml'):
            shutil.copyfile(base/'inputs'/name, repro/name)
        files = [base/'train.py', base/'verify.py', base/'environment.json']
        for directory in ('source', 'runtime', 'upstream'):
            files.extend(p for p in (base/directory).rglob('*.py') if '__pycache__' not in p.parts)
        files.extend(p for p in (base/'inputs').iterdir() if p.suffix in ('.yaml', '.json'))
        write(repro/'code-manifest.json', {str(p.relative_to(base)):sha(p) for p in files})
        with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                bundle.write(path, path.relative_to(base))
            bundle.write(base/'inputs/joint-codebook.pt', 'inputs/joint-codebook.pt')
        write(repro/'trial-audit.json', dict(passed=True, temperatures_only=True,
            control=CONTROL, shared_initial_weight_sha256=hashes['current'],
            source_frozen_from_control=True, upstream_unchanged=True,
            fresh_optimizer=True, fresh_stage2=True, global_batch=2048, world_size=8,
            cosine_total_updates=62600, stop_after_updates=STOP_STEP))
    write(OUT/'initialization-verification.json', dict(passed=True, hashes=hashes,
        verified_on_cpu=True, also_asserted_at_fresh_gpu_startup=True))


def process_matches(pid, token):
    try:
        return token.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def preserve_control():
    deadline = time.monotonic()+1800
    while time.monotonic() < deadline:
        receipts = [json.loads(p.read_text()) for p in
                    (CONTROL_OUT/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
        eligible = [r for r in receipts if r['global_step'] >= STOP_STEP and r.get('fid') is not None]
        if eligible:
            receipt = min(eligible, key=lambda r:r['global_step'])
            break
        launch = json.loads((CONTROL_OUT/'launch-status.json').read_text())
        if not process_matches(launch.get('pid', 0), 'relaunch_laser_joint20k_rngfix.py'):
            raise RuntimeError('Control launcher exited before its requested checkpoint')
        write(OUT/'status.json', dict(phase='waiting_for_control_epoch4_checkpoint',
            pid=os.getpid(), minimum_step=STOP_STEP, time=time.time()))
        time.sleep(5)
    else:
        raise TimeoutError('No epoch-four checkpoint within 30 minutes; control left running')
    local, durable = Path(receipt['local']), Path(receipt['persistent'])
    if local.stat().st_size != durable.stat().st_size or local.stat().st_size != receipt['bytes']:
        raise RuntimeError('Control checkpoint size mismatch')
    if sha(local) != receipt['sha256']:
        raise RuntimeError('Control checkpoint hash mismatch')
    with local.open('rb') as a, durable.open('rb') as b:
        if a.read(64) != b.read(64):
            raise RuntimeError('Control checkpoint header mismatch')
        a.seek(-64, 2); b.seek(-64, 2)
        if a.read(64) != b.read(64):
            raise RuntimeError('Control checkpoint footer mismatch')
    pinned = BASE/f'control-epoch{receipt["epoch"]:03d}.pt'
    if not pinned.exists():
        os.link(local, pinned)
    alias = CONTROL_OUT/f'epoch{receipt["epoch"]:03d}-before-noise-ablation.pt'
    if not alias.exists():
        alias.symlink_to(os.path.relpath(durable, alias.parent))
    write(OUT/'preserved-control.json', dict(passed=True, checkpoint=receipt,
        local_pinned_checkpoint=str(pinned), persistent_alias=str(alias),
        resume_command=[sys.executable,'-m','torch.distributed.run','--standalone',
            '--nproc-per-node=8',str(CONTROL_BASE/'train.py'),'--epochs','100','--resume',str(pinned)],
        environment_file=str(CONTROL_BASE/'environment.json'), time=time.time()))
    launch = json.loads((CONTROL_OUT/'launch-status.json').read_text())
    ranks = [json.loads((CONTROL_OUT/f'train/progress-rank{rank}.json').read_text())['pid'] for rank in range(8)]
    if not process_matches(launch['pid'], 'relaunch_laser_joint20k_rngfix.py'):
        raise RuntimeError('Refusing to signal an unrecognized control launcher')
    os.kill(launch['pid'], signal.SIGTERM)
    for second in range(60):
        live = [pid for pid in ranks if process_matches(pid, str(CONTROL_BASE/'train.py'))]
        if (not live and not process_matches(launch['child_pid'], str(CONTROL_BASE/'train.py'))
                and not process_matches(launch['pid'], 'relaunch_laser_joint20k_rngfix.py')):
            break
        if second == 30:
            for pid in live:
                os.kill(pid, signal.SIGTERM)
        time.sleep(1)
    else:
        raise RuntimeError('Control workers did not exit; ablation not launched')
    record = dict(phase='stopped_for_noise_ablation', suite=SUITE,
        preserved_global_step=receipt['global_step'], preserved_epoch=receipt['epoch'], time=time.time())
    write(CONTROL_OUT/'launch-status.json', record)
    write(OUT/'control-stopped.json', record)
    os.environ['WANDB_API_KEY'] = KEY.read_text().strip()
    import wandb
    run = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{CONTROL}')
    run.summary.update({'training/status':'stopped_for_noise_ablation',
        'training/preserved_checkpoint_step':receipt['global_step'],'training/noise_ablation':SUITE})
    run.update()


def report():
    plan = json.loads((OUT/'plan.json').read_text())
    paths = {'current': CONTROL_OUT, **{t['variant']:Path(t['output']) for t in plan['trials']}}
    evaluations = {name:{int(p.stem.removeprefix('epoch')):json.loads(p.read_text())
        for p in (path/'evaluations').glob('epoch*.json')} for name,path in paths.items()}
    rows = comparison_rows(evaluations)
    complete = all(name in rows[-1]['fid'] for name in paths)
    winner = min(rows[-1]['fid'], key=rows[-1]['fid'].get) if complete else None
    result = dict(completed=complete, results=rows, provisional_best_at_epoch4=winner,
        limitations=plan['limits'], observed_epoch4_improvement=bool(
            complete and winner != 'current'), final_convergence_improvement_established=False,
        auto_promoted=False, trial_urls=[t['url'] for t in plan['trials']])
    write(OUT/'report.json', result)
    print(json.dumps(result), flush=True)
    return result


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    return {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}


def run_suite():
    lock = (BASE/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not json.loads((OUT/'initialization-verification.json').read_text())['passed']:
        raise RuntimeError('Initialization verification is required')
    plan = json.loads((OUT/'plan.json').read_text())
    for trial in plan['trials']:
        base, out = Path(trial['base']), Path(trial['output'])
        if (out/'train').exists():
            raise RuntimeError('Suite refuses to overwrite an existing production trial')
        for name, expected in json.loads((out/'reproducibility/code-manifest.json').read_text()).items():
            if sha(base/name) != expected:
                raise RuntimeError(f'Trial snapshot changed: {name}')
    if any(not process_matches(pid, str(CONTROL_BASE/'train.py')) for pid in gpu_processes()):
        raise RuntimeError('GPU has a process outside the owned control run')
    preserve_control()
    if gpu_processes():
        raise RuntimeError('GPUs are still occupied after stopping the control')
    child = None

    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
        raise KeyboardInterrupt('Noise ablation interrupted')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for trial in plan['trials']:
        base, out = Path(trial['base']), Path(trial['output'])
        env = dict(os.environ, **json.loads((base/'environment.json').read_text()),
                   WANDB_API_KEY=KEY.read_text().strip())
        distributed = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8']
        phases = [
            ('gpu_preflight', [*distributed,str(base/'train.py'),'--smoke-updates','2']),
            ('checkpoint_verification', [sys.executable,str(base/'verify.py')]),
            ('training', [*distributed,str(base/'train.py'),'--epochs',str(STOP_EPOCH)]),
        ]
        for phase, command in phases:
            if phase == 'training':
                proof = json.loads((out/'smoke-reload-verification.json').read_text())
                if not proof['passed'] or proof['targets']['temperature'] != trial['temperatures']:
                    raise RuntimeError('Trial preflight failed')
            with (out/f'{phase}.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(command, cwd=base,
                    env=dict(env,LASER_PRIVATE_SMOKE_DIR=str(base/'private-smoke')),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state = dict(phase=phase,variant=trial['variant'],run_id=trial['run_id'],
                    pid=os.getpid(),child_pid=child.pid,command=command,time=time.time())
                write(OUT/'status.json', state)
                write(out/'launch-status.json', state)
                result = child.wait()
            if result:
                raise RuntimeError(f'{trial["variant"]} {phase} exited {result}; see {out/phase}.log')
        completed = json.loads((out/'completed.json').read_text())
        if completed['epochs'] != STOP_EPOCH or completed['global_step'] != STOP_STEP:
            raise RuntimeError('Trial ended at an unexpected training budget')
        write(out/'launch-status.json', dict(phase='completed_ablation',epochs=STOP_EPOCH,
            global_step=STOP_STEP,time=time.time()))
        report()
    result = report()
    if not result['completed']:
        raise RuntimeError('Missing final paired FIDs')
    write(OUT/'status.json', dict(phase='completed',results=str(OUT/'report.json'),time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','verify-initialization','run','report'])
    args = parser.parse_args()
    {'prepare':prepare,'verify-initialization':verify_initialization,
     'run':run_suite,'report':report}[args.phase]()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            write(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),time=time.time()))
        raise
