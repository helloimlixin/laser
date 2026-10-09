"""Prepare and supervise matched one-epoch compound-energy/control forks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def environment(base, output, key, phase):
    return dict(os.environ, LASER_RUN_BASE=str(base), LASER_PERSISTENT_BASE=str(output),
        LASER_ACCUMULATION='4', LASER_PHASE=phase, LASER_CHECKPOINT_STAGING_DIR=str(base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base/'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1', LASER_COMPILE_BLOCKS='1', LASER_PREVIEW_KEEP_OPTIMIZER='1',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', WANDB_API_KEY=key.read_text().strip(),
        WANDB_DIR=str(base/'wandb'), WANDB_CACHE_DIR=str(base/'wandb-cache'),
        WANDB_DATA_DIR=str(base/'wandb-data'), WANDB_CONFIG_DIR=str(base/'wandb-config'),
        TORCHINDUCTOR_CACHE_DIR=str(base/'inductor-cache'), TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCH_HOME=str(base/'torch-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        NCCL_NVLS_ENABLE='0', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')


def patch_trial_runtime(runtime):
    """Commit a full recovery snapshot before entering the costly evaluator."""
    path = runtime/'src/training/rqtransformer.py'
    source = path.read_text()
    marker = '            save_before_official_evaluation(\n'
    if marker in source:
        return
    needle = ('        if run_fid:\n'
              '            with optimizer_state_offloaded_for_generation(model, optimizer, device):\n')
    replacement = ('        if run_fid:\n'
        '            save_before_official_evaluation(\n'
        '                model, optimizer, parameter_names, device, epoch, global_step, scheduler,\n'
        '                runtime_config, best_fid, best_inception, last_checkpoint)\n'
        '            with optimizer_state_offloaded_for_generation(model, optimizer, device):\n')
    if source.count(needle) != 1:
        raise RuntimeError('Frozen trainer evaluation boundary changed')
    path.write_text(source.replace(needle, replacement))


def prepare(args):
    import yaml
    import torch
    sys.path[:0] = [str(args.repo), str(args.parent/'support')]
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    if (args.output/'preparation-complete.json').exists():
        raise RuntimeError('A prepared trial already exists; use run, not prepare')
    args.base.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(args.parent/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    source = _checkpoint_upload_source(args.parent_output/'train/checkpoints/best_fid_15.0819_epoch_079.pt')
    payload = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(payload)
    if (metadata['global_step'], metadata['epoch'], metadata['rng_ranks'], payload['scheduler']['last_epoch']) != (49454,79,8,49454):
        raise RuntimeError('The requested full-state best checkpoint is inconsistent')
    anchor = args.base/'source-epoch079-full.pt'
    if not anchor.exists():
        os.link(source, anchor)
    elif not os.path.samefile(source, anchor):
        raise RuntimeError('Partially prepared trial has a different source anchor')
    plan = dict(version='compound-energy-distance-ab-v1', source_checkpoint=str(anchor),
        source_epoch=79, source_step=49454, source_fid=15.081930549156368,
        source_run='helloimlixin-rutgers/laser/imagenet-rfid421-epoch77-full-compound-history-lr1e6-8h100-20261006',
        common_adam_step=metadata['compound_optimizer_ages']['common_adam_step'],
        history_adam_step=metadata['compound_optimizer_ages']['new_adam_step'],
        initial_lr=metadata['saved_learning_rates'][0], target_epoch=80, updates_per_branch=626,
        optimizer_and_lr_policy='saved Adam state and absolute100epoch zero-floor cosine unchanged',
        candidate_top_k=4, sites_per_image=2, coefficient_groups=64, target_gradient_ratio=.1,
        weight_cap=.2, ramp_updates=100, evaluation_seeds=[261001,261101],
        calibration_file=str(args.output/'shared-gradient-calibration.json'),
        geometry='energy distance between joint signed contribution distribution and teacher soft distribution',
        geometry_approximation='top4 atoms plus teacher once; 64 adjacent coefficient groups; depth-scale normalized',
        sampler=dict(mode='native ancestral', atom_temperature=.9, atom_top_p=.9,
                     coefficient_temperature=1., coefficient_top_p=.85),
        metrics=dict(backend='original_rqtransformer', generated_images=50000, real_images=50000,
                     real_split='val', inception_splits=10),
        branches=['control','geometry'], parent_base=str(args.parent), parent_output=str(args.parent_output),
        auto_resume_preserved_continuation=True, prepared_unix=time.time())
    record(args.base/'plan.json', plan)
    record(args.output/'plan.json', plan)
    record(args.output/'source-integrity.json', metadata)
    runtime = args.base/'runtime'
    shutil.copytree(args.parent/'source/runtime', runtime, ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    changed = ['src/models/physical_compound_prior.py', 'src/training/physical_compound_geometry.py',
               'src/training/compound_geometry_trial_hook.py', 'src/training/training_loss_tracker.py']
    for name in changed:
        shutil.copy2(args.repo/name, runtime/name)
    patch_trial_runtime(runtime)
    changed.append('src/training/rqtransformer.py')
    template = (args.parent/'entry.py').read_text()
    insertion = ("from src.training.compound_geometry_trial_hook import install as install_energy_trial\n"
                 "install_energy_trial(globals(), json.loads((BASE.parent/'plan.json').read_text()))\n\n")
    if template.count('from src.training.cli import main') != 1:
        raise RuntimeError('Unexpected original trainer entrypoint')
    template = template.replace('from src.training.cli import main', insertion+'from src.training.cli import main')
    recipe = yaml.safe_load((args.parent/'recipe.yaml').read_text())
    for branch in plan['branches']:
        base, output = args.base/branch, args.output/branch
        (base/'source').mkdir(parents=True, exist_ok=True)
        if not (base/'source/runtime').exists():
            (base/'source/runtime').symlink_to(runtime)
        shutil.copytree(args.parent/'support', base/'support', ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
        shutil.copy2(args.parent/'compound-transfer.json', base/'compound-transfer.json')
        shutil.copy2(args.parent/'fixed-teacher-batch.pt', base/'fixed-teacher-batch.pt')
        (base/'inputs').mkdir(exist_ok=True)
        if not (base/'inputs/resume-stage1-tokenizer.pt').exists():
            (base/'inputs/resume-stage1-tokenizer.pt').symlink_to(args.parent/'inputs/resume-stage1-tokenizer.pt')
        (base/'entry.py').write_text(template)
        (output/'train/checkpoints').mkdir(parents=True, exist_ok=True)
        options = dict(recipe['options'], epochs=80, output=str(output/'train'),
            checkpoint=str(base/'inputs/resume-stage1-tokenizer.pt'),
            resume_checkpoint=str(anchor), checkpoint_dir=str(output/'train/checkpoints'),
            wandb_id=f'imagenet-rfid421-epoch79-energy-{branch}-official50k-ab-20261007',
            wandb_name=f'ImageNet K4 | epoch79 | {branch} | matched energy trial',
            save_step_freq=250, sample_grid_every=0, upload_checkpoints=False,
            sample_grid_on_start=False, max_optimizer_steps=0)
        (base/'recipe.yaml').write_text(yaml.safe_dump(dict(recipe, options=options), sort_keys=False))
        shutil.copyfile(base/'recipe.yaml', output/'recipe.yaml')
        if branch == 'geometry':
            preflight = dict(options, max_optimizer_steps=20, fid_every=0)
            (base/'preflight.yaml').write_text(yaml.safe_dump(dict(recipe, options=preflight),sort_keys=False))
    code = args.output/'compound-energy-code.zip'
    with zipfile.ZipFile(code, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in changed:
            archive.write(runtime/name, name)
        for name in ['scripts/tools/run_compound_energy_trial.py', 'tests/test_physical_compound_geometry.py']:
            archive.write(args.repo/name, name)
        archive.write(args.base/'control/entry.py','entry.py')
        archive.write(args.base/'plan.json','plan.json')
    manifest = {name:hashlib.sha256((runtime/name).read_bytes()).hexdigest() for name in changed}
    record(args.output/'code-manifest.json', manifest)
    record(args.output/'preparation-complete.json',dict(plan=plan, code_manifest=manifest, time=time.time()))
    print(json.dumps(dict(prepared=True, plan=str(args.output/'plan.json'), source_lr=plan['initial_lr'],
                          source_step=49454, branches=plan['branches'])), flush=True)


def stop_and_preserve(args):
    status = json.loads((args.parent_output/'training-status.json').read_text())
    supervisor, torchrun = status['supervisor_pid'], status['torchrun_pid']
    if str(args.parent) not in command(supervisor) or 'restart_compound_history_schedule' not in command(supervisor):
        raise RuntimeError('Current continuation supervisor identity changed')
    ranks = [json.loads(p.read_text())['pid'] for p in
             (args.parent_output/'verification'/status['phase']).glob('process-rank*.json')]
    if len(ranks) != 8 or any(str(args.parent) not in command(pid) for pid in ranks):
        raise RuntimeError('Current continuation rank ownership changed')
    ready = args.parent_output/'trial-checkpoint-ready.json'
    if ready.exists():
        raise RuntimeError('Unexpected stale current-continuation stop marker')
    os.kill(supervisor, signal.SIGTERM)
    deadline = time.monotonic()+900
    while True:
        if ready.exists():
            state = json.loads(ready.read_text())
            saved = json.loads((args.parent_output/'last-local-save.json').read_text())
            if state['no_further_updates'] and state['global_step'] == saved['step']:
                break
        if time.monotonic() > deadline:
            raise RuntimeError('Current continuation did not commit its full stop state')
        time.sleep(5)
    sys.path[:0] = [str(args.parent/'source/runtime'), str(args.parent/'support')]
    import torch
    from src.training.full_resume_upload import recovery_metadata
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(args.parent/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    payload = torch.load(_checkpoint_upload_source(Path(saved['target'])), map_location='cpu',mmap=True,weights_only=False)
    metadata = recovery_metadata(payload)
    if metadata['global_step'] != state['global_step'] or metadata['rng_ranks'] != 8:
        raise RuntimeError('Current continuation stop checkpoint is incomplete')
    anchor = args.parent/f'preserved-before-energy-trial-step{metadata["global_step"]}.pt'
    os.link(_checkpoint_upload_source(Path(saved['target'])), anchor)
    record(args.output/'preserved-continuation.json', dict(metadata=metadata, anchor=str(anchor), ready=state))
    del payload
    for pid in [*ranks, torchrun]:
        cmd = command(pid)
        if cmd and str(args.parent) not in cmd:
            raise RuntimeError('Frozen continuation process ownership changed')
        if cmd:
            os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic()+90
    while command(supervisor) or any(command(pid) for pid in ranks):
        if time.monotonic() > deadline:
            raise RuntimeError('Frozen continuation did not release its GPUs')
        time.sleep(1)
    ready.rename(args.output/'preserved-continuation-ready.json')
    record(args.parent_output/'training-status.json', dict(state='suspended_for_matched_energy_trial',
        preserved_global_step=metadata['global_step'], trial_status=str(args.output/'status.json'),
        automatic_resume_after_trial=True, time=time.time()))


def publish(args, branch):
    base, output = args.base/branch, args.output/branch
    sys.path[:0] = [str(args.base/'runtime'), str(base/'support')]
    os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(base/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    import yaml
    config = yaml.safe_load((base/'recipe.yaml').read_text())['options']
    checkpoints = output/'train/checkpoints'
    slots = base/'upload-slots'
    slots.mkdir(exist_ok=True)
    sources = [('last.pt', checkpoints/'last.pt'),
               ('best-fid-resume.pt', next(checkpoints.glob('best_fid_*.pt'))),
               ('best-is-resume.pt', next(checkpoints.glob('best_is_*.pt')))]
    paths = []
    for name, source in sources:
        target = slots/name
        if not target.exists():
            os.link(_checkpoint_upload_source(source), target)
        paths.append(target)
    run_path = 'helloimlixin-rutgers/laser/'+config['wandb_id']
    VerifiedCloudUpload(run_path, output/'cloud-checkpoint-receipt.json')(paths, 80)
    extras = [args.output/'plan.json', args.output/'code-manifest.json', args.output/'compound-energy-code.zip',
              output/'official-two-seed-summary.json', *output.glob('official-seed*.json'),
              output/'latest-training-diagnostics.json', args.output/'shared-gradient-calibration.json']
    if (args.output/'comparison.json').exists():
        extras.append(args.output/'comparison.json')
    for name in ('continuation-policy.json','continuation-policy-code.zip','geometry-finite20-proof.json',
                 'geometry-preflight-verification.json','evaluator-repair-verification.json',
                 'evaluator-repair-code.zip','evaluator-repair-tests.txt'):
        if (args.output/name).exists():
            extras.append(args.output/name)
    VerifiedCloudUpload(run_path, output/'cloud-evidence-receipt.json')(extras,80)
    import wandb
    run = wandb.Api().run(run_path)
    run.summary['trial/full_checkpoint_uploads_verified'] = True
    run.summary.update()
    all_uploaded = all((args.output/b/name).exists() and json.loads((args.output/b/name).read_text()).get('complete')
                       for b in ('control','geometry')
                       for name in ('cloud-checkpoint-receipt.json','cloud-evidence-receipt.json'))
    if all_uploaded:
        record(args.output/'status.json',dict(state='completed',official_evaluations_completed=4,
            full_best_and_last_checkpoints_verified_online=True,
            automatic_promotion=False,source_epoch=79,source_step=49454,
            comparison=str(args.output/'comparison.json'),time=time.time()))


def run(args):
    import yaml
    lock = (args.base/'trial-supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = json.loads((args.base/'plan.json').read_text())
    manifest = json.loads((args.output/'code-manifest.json').read_text())
    if not (args.output/'preparation-complete.json').exists() or any(
            hashlib.sha256((args.base/'runtime'/name).read_bytes()).hexdigest() != digest
            for name,digest in manifest.items()):
        raise RuntimeError('Trial code is incomplete or changed after preparation')
    status = args.output/'status.json'
    parent_lock = None
    suspended = False
    recovering = args.action == 'resume-trial'
    try:
        if recovering:
            policy = json.loads((args.output/'continuation-policy.json').read_text())
            preflight = json.loads((args.output/'geometry-preflight-verification.json').read_text())
            if policy['resume_preserved_later_checkpoint'] or not preflight['passed']:
                raise RuntimeError('Recovery requires the selected epoch79 source and passed geometry preflight')
        else:
            record(status, dict(state='checkpointing_continuation', supervisor_pid=os.getpid(), time=time.time()))
            stop_and_preserve(args)
            suspended = True
        parent_lock = (args.parent/'training-supervisor.lock').open('w')
        fcntl.flock(parent_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        phases = [('geometry',False),('control',False)] if recovering else [('geometry',True),('control',False),('geometry',False)]
        for branch, preflight in phases:
            base, output = args.base/branch, args.output/branch
            phase = 'energy_'+branch+('_preflight' if preflight else '')+('_recovery' if recovering else '')
            env = environment(base, output, args.key_file, phase)
            env['LASER_GEOMETRY_BRANCH'] = branch
            config = base/'preflight.yaml' if preflight else base/'recipe.yaml'
            if preflight:
                env['LASER_GEOMETRY_PREFLIGHT'] = '1'
            elif branch == 'geometry' or recovering:
                recipe = yaml.safe_load(config.read_text())
                recipe['options']['resume_checkpoint'] = str(output/'train/checkpoints/last.pt')
                recipe['options']['save_step_freq'] = 250
                config = base/('resume-after-evaluator-fix.yaml' if recovering else 'resume-after-preflight.yaml')
                config.write_text(yaml.safe_dump(recipe,sort_keys=False))
            ready_path = output/'trial-checkpoint-ready.json'
            if ready_path.exists():
                ready_path.rename(output/f'previous-checkpoint-ready-{int(time.time())}.json')
            cmd = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                   str(base/'entry.py'),'--config',str(config)]
            with (output/'training.log').open('a') as stream:
                child = subprocess.Popen(cmd,cwd=base,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    record(status,dict(state='running',branch=branch,supervisor_pid=os.getpid(),
                        torchrun_pid=child.pid,phase=phase,source_step=49454,
                        target_step=49474 if preflight else 50080,
                        official_evaluations_per_branch=2,time=time.time()))
                    time.sleep(5)
            if child.returncode:
                raise RuntimeError(f'Matched {branch} branch failed with code {child.returncode}')
            ready = json.loads((output/'trial-checkpoint-ready.json').read_text())
            saved = json.loads((output/'last-local-save.json').read_text())
            target_step = 49474 if preflight else 50080
            if not ready['no_further_updates'] or saved['step'] != target_step or ready['global_step'] != target_step:
                raise RuntimeError(f'Matched {branch} final checkpoint is incomplete')
            if preflight:
                verify = output/'verification'/phase
                checks = [json.loads((verify/f'step20-rank{rank}.json').read_text()) for rank in range(8)]
                energy = [json.loads((verify/f'energy-step20-rank{rank}.json').read_text()) for rank in range(8)]
                if not all(x['finite'] for x in checks) or not all(x['applied_weight']>0 for x in energy):
                    raise RuntimeError('Nonzero geometry objective failed its eight-rank finite-state preflight')
                from src.training.k4_checkpoint_io import _checkpoint_upload_source
                from src.training.full_resume_upload import recovery_metadata
                import torch
                os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(base/'checkpoint-upload-cache')
                payload = torch.load(_checkpoint_upload_source(Path(saved['target'])),map_location='cpu',mmap=True,weights_only=False)
                metadata = recovery_metadata(payload)
                if metadata['global_step'] != 49474 or metadata['rng_ranks'] != 8 or payload['scheduler']['last_epoch'] != 49474:
                    raise RuntimeError('Geometry preflight recovery state is incomplete')
                record(args.output/'geometry-preflight-verification.json',dict(passed=True,
                    full_recovery_metadata=metadata, finite_state_ranks=8, updates=20,
                    adam_reset=False, energy=energy[0], time=time.time()))
                del payload
                (output/'trial-checkpoint-ready.json').rename(output/'preflight-checkpoint-ready.json')
                continue
            if not (output/'official-two-seed-summary.json').exists():
                raise RuntimeError(f'Matched {branch} official evaluations are missing')
        summaries = {b:json.loads((args.output/b/'official-two-seed-summary.json').read_text()) for b in plan['branches']}
        paired = [dict(seed=s, control_fid=summaries['control']['evaluations'][i]['fid'],
            geometry_fid=summaries['geometry']['evaluations'][i]['fid'],
            fid_change=summaries['geometry']['evaluations'][i]['fid']-summaries['control']['evaluations'][i]['fid'])
            for i,s in enumerate(plan['evaluation_seeds'])]
        comparison = dict(source_epoch=79,epochs_per_branch=1,official_protocol=plan['metrics'],
            summaries=summaries,paired_results=paired,
            fid_mean_change=summaries['geometry']['fid_mean']-summaries['control']['fid_mean'],
            geometry_better_on_both_seeds=all(x['fid_change']<0 for x in paired),
            automatic_promotion=False,time=time.time())
        record(args.output/'comparison.json',comparison)
        record(status,dict(state='training_and_evaluation_completed_uploads_pending',comparison=comparison,time=time.time()))
        for branch in plan['branches']:
            cmd = [sys.executable,str(Path(__file__).resolve()),'publish','--base',str(args.base),
                '--output',str(args.output),'--parent',str(args.parent),'--parent-output',str(args.parent_output),
                '--key-file',str(args.key_file),'--repo',str(args.repo),'--branch',branch]
            with (args.output/branch/'publisher.log').open('a') as stream:
                publisher = subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                             env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            record(args.output/branch/'publisher-launch.json',dict(pid=publisher.pid,time=time.time()))
    except Exception as error:
        record(status,dict(state='failed_checkpoints_preserved',error=repr(error),
            resume_preserved_later_checkpoint=False,source_epoch=79,source_step=49454,time=time.time()))
        raise
    finally:
        if parent_lock is not None:
            parent_lock.close()
        if suspended:
            cmd = [sys.executable,str(Path(__file__).resolve()),'resume-parent','--base',str(args.base),
                '--output',str(args.output),'--parent',str(args.parent),'--parent-output',str(args.parent_output),
                '--key-file',str(args.key_file),'--repo',str(args.repo)]
            with (args.output/'parent-resume.log').open('a') as stream:
                resumed = subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record(args.output/'parent-resume-launch.json',dict(pid=resumed.pid,time=time.time()))


def resume_parent(args):
    policy_path = args.output/'continuation-policy.json'
    if policy_path.exists() and not json.loads(policy_path.read_text()).get('resume_preserved_later_checkpoint', True):
        record(args.output/'parent-resume-skipped.json',dict(
            reason='User explicitly selected the epoch79 FID15.08193 checkpoint after fixes',
            preserved_continuation_is_backup_only=True, source_epoch=79, source_step=49454,time=time.time()))
        record(args.parent_output/'training-status.json',dict(state='preserved_backup_superseded_by_epoch79_trial',
            active_trial_status=str(args.output/'status.json'),source_epoch=79,source_step=49454,time=time.time()))
        return
    lock = (args.parent/'training-supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    phase = 'compound_after_energy_trial'
    cmd = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
           str(args.parent/'entry-objective-schedule.py'),'--config',str(args.parent/'recipe.yaml')]
    env = environment(args.parent,args.parent_output,args.key_file,phase)
    with (args.parent_output/'compound_objective_schedule.log').open('a') as stream:
        child = subprocess.Popen(cmd,cwd=args.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        def stop(sig, frame):
            for path in (args.parent_output/'verification'/phase).glob('process-rank*.json'):
                pid = json.loads(path.read_text())['pid']
                if str(args.parent) in command(pid):
                    os.kill(pid,signal.SIGTERM)
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        while child.poll() is None:
            record(args.parent_output/'training-status.json',dict(state='training',supervisor_pid=os.getpid(),
                torchrun_pid=child.pid,phase=phase,
                run_id='imagenet-rfid421-epoch77-full-compound-history-lr1e6-8h100-20261006',
                target_epoch=100,official_evaluations_only=True,time=time.time()))
            time.sleep(5)
    record(args.parent_output/'training-status.json',dict(state='completed' if child.returncode==0 else 'failed',
        exit_code=child.returncode,phase=phase,time=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','resume-trial','publish','resume-parent'])
    for name in ('base','output','parent','parent-output','key-file','repo'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--branch',choices=['control','geometry'])
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args)
    elif args.action in ('run','resume-trial'): run(args)
    elif args.action == 'publish': publish(args,args.branch)
    else: resume_parent(args)


if __name__=='__main__':
    main()
