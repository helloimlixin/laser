"""Run a bounded higher-LR trial using the complete protected best checkpoint."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import continue_imagenet_fid16379 as recovery

RUN_ID = 'imagenet-rfid421-lossdecay-warm200-lr1e5-3epoch-8h100-20261006'
RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID
record = recovery.record


def patch_entry(base, original, parent_id, target_epoch):
    p = base / 'entry.py'
    code = p.read_text()
    def replace(old, new):
        nonlocal code
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    replace('ARGS.lr==0.0001', 'ARGS.lr==0.00001')
    replace('from imagenet_zero_floor_lr import create_scheduler, observe_official_fid',
            'from imagenet_loss_decay_lr import create_scheduler, observe_official_fid')
    start = code.index(' from imagenet_fid_rewind_guard import regression_request\n')
    end = code.index(' return result\ntraining.evaluate_generation_metrics=evaluate', start)
    code = code[:start] + code[end:]
    replace(" config.update(automatic_fid_rewind=True,fid_rewind_policy='strictly worse than the saved best; no further updates; full-state rewind at reduced LR')\n",
            " config.update(automatic_fid_rewind=False,fid_rewind_policy='bounded three-epoch sustained trial; temporary regressions retained; protected full best kept',\n"
            f"  trial_target_epoch={target_epoch},trial_epochs=3,trial_lr_warmup_updates=200,trial_peak_lr=1e-5)\n")
    replace(
        "  training_loss_components='cross entropy and CRPS separately; resumable EMA decay0.95',manual_lr_increase_factor=1000.)\n",
        "  training_loss_components='cross entropy and CRPS separately; resumable EMA decay0.95; sample-weighted epoch means')\n")
    start = code.index(' config.update(resume_source_epoch=')
    end = code.index(' WB=original_init(*args,**kwargs)', start)
    metrics = original['original_rqtransformer_metrics']
    code = code[:start] + (
        f" config.update(resume_source_epoch={original['epoch']},resume_source_global_step={original['global_step']},\n"
        f"  source_run={'helloimlixin-rutgers/laser/'+parent_id!r},source_checkpoint_epoch={original['epoch']},\n"
        f"  source_checkpoint_step={original['global_step']},source_checkpoint_fid={metrics['fid']!r},\n"
        "  source_optimizer_restored=True,trained_source_optimizer_available=True,\n"
        f"  optimizer_initialization='trained AdamW moments and all782 counters preserved at{original['adam_step']}',\n"
        "  rng_initialization='all8 saved CPU/CUDA RNG streams restored',\n"
        "  evaluation_protocol='Official RQ-Transformer FID/10-split IS only; validation50k/generated50k',\n"
        "  learning_rate_schedule='200-update warmup3e-7 to1e-5; cosine to zero at original epoch100 endpoint; no FID-triggered LR reductions during three-epoch trial',\n"
        "  lr_continuation='intentional policy migration with original schedule clock and trained Adam preserved',\n"
        f"  source_scheduler_step={original['learning_rate_schedule']['state']['last_epoch']},\n"
        "  restart_initial_lr=3e-7,lr_floor_after=0.,cosine_decay_end_epoch=100)\n"
    ) + code[end:]
    code, count = re.subn(r'(?m)^CHECKPOINT_EPOCH=\d+$',
                         f"CHECKPOINT_EPOCH={original['epoch']}", code)
    assert count == 1
    code, count = re.subn(r"WB.log\(\{'train/epoch':\d+,'train/global_step':\d+,",
        f"WB.log({{'train/epoch':{original['epoch']},'train/global_step':{original['global_step']},", code)
    assert count == 1
    replace('TRAIN_LOSS_TRACKER=None\n', 'TRAIN_LOSS_TRACKER=None\nEPOCH_LOSS_TRACKER=None\nLOSS_EPOCH=None\n')
    replace('global TRAIN_LOSS_TRACKER\n', 'global TRAIN_LOSS_TRACKER,EPOCH_LOSS_TRACKER,LOSS_EPOCH\n')
    replace("  INITIAL_RECOVERY=recovery_metadata(payload)\n",
            "  epoch_stats=payload.get('train_epoch_loss_tracking')\n"
            "  if epoch_stats is not None:\n"
            "   LOSS_EPOCH=epoch_stats['epoch']\n"
            "   EPOCH_LOSS_TRACKER=TrainingLossTracker(payload['global_step'],epoch_stats['state'])\n"
            "  INITIAL_RECOVERY=recovery_metadata(payload)\n")
    replace(' global UPDATES,INITIAL_ADAM_STEP,LAST_STEP_END,PROFILER,TRAIN_OBJECTIVE_SUM,TRAIN_LOSS_TRACKER,LATEST_TRAIN_LOSS\n',
            ' global UPDATES,INITIAL_ADAM_STEP,LAST_STEP_END,PROFILER,TRAIN_OBJECTIVE_SUM,TRAIN_LOSS_TRACKER,LATEST_TRAIN_LOSS,EPOCH_LOSS_TRACKER,LOSS_EPOCH\n')
    replace(' TRAIN_OBJECTIVE_SUM=None\n if UPDATES in (1,20):\n',
            " global_step=INITIAL_RECOVERY['global_step']+UPDATES\n"
            " epoch=(global_step-1)//626\n"
            " if EPOCH_LOSS_TRACKER is None or LOSS_EPOCH!=epoch:\n"
            "  from src.training.training_loss_tracker import TrainingLossTracker\n"
            "  EPOCH_LOSS_TRACKER=TrainingLossTracker(global_step-1);LOSS_EPOCH=epoch\n"
            " epoch_metrics=EPOCH_LOSS_TRACKER.update(*values,samples,global_step)\n"
            " LATEST_TRAIN_LOSS.update({'train/epoch_cross_entropy_mean':epoch_metrics['train/cross_entropy_running_mean'],\n"
            "  'train/epoch_loss_mean':epoch_metrics['train/loss_running_mean']})\n"
            ' TRAIN_OBJECTIVE_SUM=None\n if UPDATES in (1,20):\n')
    replace("  payload=dict(payload,train_loss_tracking=TRAIN_LOSS_TRACKER.state_dict())\n",
            "  payload=dict(payload,train_loss_tracking=TRAIN_LOSS_TRACKER.state_dict())\n"
            "  if EPOCH_LOSS_TRACKER is not None:\n"
            "   payload=dict(payload,train_epoch_loss_tracking=dict(epoch=LOSS_EPOCH,state=EPOCH_LOSS_TRACKER.state_dict()))\n")
    start = code.index('  if FID_REWIND_REQUEST is not None:\n')
    end = code.index('  if UPLOADER is not None:UPLOADER.close()', start)
    code = code[:start] + (
        "  record(EVIDENCE/'trial-checkpoint-ready.json',dict(global_step=INITIAL_RECOVERY['global_step']+UPDATES,\n"
        "   epoch=CHECKPOINT_EPOCH,best_fid=BEST_FID,best_is=BEST_IS,stopping=STOPPING,\n"
        "   no_further_updates=True,loss=LATEST_TRAIN_LOSS,time=time.time()))\n"
    ) + code[end:]
    compile(code, str(p), 'exec')
    p.write_text(code)


def prepare(base, evidence, parent_base):
    import torch
    import yaml
    torch.set_num_threads(4)
    audit = evidence / 'continuation-20261005'
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    assert not (checkpoints / 'last.pt').exists(), 'Refuse to overwrite a continuation'
    for name in ('source/runtime', 'support'):
        shutil.copytree(parent_base / name, base / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('continue_imagenet_loss_decay.py', 'imagenet_loss_decay_lr.py', 'verify_imagenet_loss_decay.py'):
        shutil.copyfile(Path(__file__).with_name(name), base / 'support' / name)
    sys.path[:0] = [str(base / 'source/runtime'), str(base / 'support')]
    from imagenet_loss_decay_lr import migrate_continuation, create_scheduler
    from src.training.k4_checkpoint_io import (
        atomic_torch_save, _checkpoint_upload_source, _persist_serialized_checkpoint)
    from src.training.full_resume_upload import recovery_metadata
    from continue_imagenet_lr1000 import compare_tensors
    inputs = base / 'inputs'; inputs.mkdir(exist_ok=True)
    source = inputs / 'source-best-full.pt'
    os.link(parent_base / 'final-cpu-upload/best-fid-resume.pt', source)
    raw = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    original = recovery_metadata(raw)
    metrics = original['original_rqtransformer_metrics']
    assert metrics['metric_backend'] == 'original_rqtransformer'
    assert metrics['real_images'] == metrics['generated_images'] == 50000
    assert original['rng_ranks'] == original['world_size'] == 8 and original['adam_parameters'] == 782
    assert original['next_microbatch'] == 0
    assert original['adam_step'] == original['global_step'] - 34430
    assert raw['scheduler']['last_epoch'] == original['global_step'] - 35056
    optimizer, state = migrate_continuation(raw['optimizer'], raw['scheduler'])
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-7)
    assert create_scheduler(dummy, initial_lr=1e-5, min_lr=0.,
        total_steps=state['policy']['total_steps'], completed_steps=state['last_epoch'],
        state_dict=state).state_dict() == state
    recipe = yaml.safe_load((parent_base / 'recipe.yaml').read_text())
    parent_id = recipe['options']['wandb_id']
    target_epoch = original['epoch'] + 3
    recipe['options'].update(checkpoint=str(inputs / 'resume-stage1-tokenizer.pt'),
        output=str(base / 'production/train'), checkpoint_dir=str(checkpoints),
        wandb_id=RUN_ID, lr_schedule_restart_id=RUN_ID, lr=1e-5, min_lr=0.,
        epochs=target_epoch, max_optimizer_steps=0, fid_every=1,
        wandb_name=f"ImageNet K4 | bestFID{metrics['fid']:.4f} epoch{original['epoch']} | warmup200 to1e-5 | sustained3epoch | 8H100")
    recipe['options'].pop('resume_checkpoint', None)
    for name in ('resume-stage1-tokenizer.pt', 'resume-weights-inception-2015-12-05-6726825d.pth'):
        os.link(parent_base / 'inputs' / name, inputs / name)
    for name in ('torch-cache', 'inductor-cache'):
        (base / name).symlink_to((parent_base / name).resolve(), target_is_directory=True)
    shutil.copyfile(parent_base / 'entry.py', base / 'entry.py')
    patch_entry(base, original, parent_id, target_epoch)
    record(base / 'official-baseline.json', metrics)
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    fid_alias = checkpoints / f"best_fid_{metrics['fid']:.4f}_epoch_{original['epoch']:03d}.pt"
    is_source = parent_base / 'final-cpu-upload/best-is-resume.pt'
    is_payload = torch.load(is_source, map_location='cpu', mmap=True, weights_only=False)
    is_metadata = recovery_metadata(is_payload)
    is_score = is_metadata['original_rqtransformer_metrics']['inception_score']
    is_alias = checkpoints / f"best_is_{is_score:.4f}_epoch_{is_metadata['epoch']:03d}.pt"
    revision = dict(source_run=parent_id, source_epoch=original['epoch'],
        source_global_step=original['global_step'], original_lr=original['saved_learning_rates'][0],
        initial_lr=3e-7, peak_lr=1e-5, warmup_steps=200, trial_epochs=3,
        target_epoch=target_epoch, old_scheduler=raw['scheduler'], new_scheduler=state,
        schedule_clock_preserved=True, optimizer_moments_reset=False, official_evaluations_only=True,
        temporary_fid_regressions_permitted=True)
    payload = dict(raw, optimizer=optimizer, scheduler=state,
        config=dict(raw['config'], **recipe['options']), lr_revision=revision,
        best_fid=[(metrics['fid'], str(fid_alias))], best_inception=[(is_score, str(is_alias))])
    # Start a new reporting interval; keep the old statistics as provenance.
    payload['source_train_loss_tracking'] = payload.pop('train_loss_tracking', None)
    payload.pop('train_epoch_loss_tracking', None)
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    latest = checkpoints / 'last.pt'
    atomic_torch_save(payload, latest)
    for alias in (fid_alias, checkpoints / 'resume-anchor.pt'):
        alias.symlink_to(latest.resolve().relative_to(checkpoints))
    is_pin = base / 'checkpoint-staging/best-is-persist.pt'
    os.link(is_source, is_pin)
    _persist_serialized_checkpoint(is_pin, is_alias)
    assert os.path.samefile(is_source, _checkpoint_upload_source(is_alias))
    restored = torch.load(_checkpoint_upload_source(latest), map_location='cpu', mmap=True, weights_only=False)
    compared = compare_tensors(raw, restored)
    assert restored['scheduler'] == state
    metadata = recovery_metadata(restored)
    with source.open('rb') as f:digest = hashlib.file_digest(f, 'md5').hexdigest()
    record(audit / 'source-parent.json', dict(base=str(parent_base), run_id=parent_id))
    record(audit / 'source-state-verification.json', original)
    record(audit / 'restart-state-verification.json', dict(metadata=metadata,
        source_md5=digest, tensor_values_compared=compared, model_and_adam_tensors_identical=True,
        rank_rng_identical=True, schedule_clock_preserved=True, revision=revision))
    record(audit / 'source-checkpoint-selection.json', dict(file=str(source), epoch=original['epoch'],
        global_step=original['global_step'], adam_step=original['adam_step'], fid=metrics['fid'],
        md5=digest, bytes=source.stat().st_size, full_optimizer_state=True))
    for name in ('entry.py', 'recipe.yaml', 'official-baseline.json'):
        shutil.copyfile(base / name, audit / name)
    (evidence / 'status.json').symlink_to('continuation-20261005/status.json')
    print(json.dumps(dict(prepared=True, run=RUN_ID, source_fid=metrics['fid'],
        source_step=original['global_step'], initial_lr=3e-7, peak_lr=1e-5,
        target_epoch=target_epoch, tensor_values_compared=compared)), flush=True)


def finalize_trial(base, evidence, child, key_file):
    import torch
    torch.set_num_threads(4)
    sys.path[:0] = [str(base / 'source/runtime'), str(base / 'support')]
    os.environ.update(LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
                      LASER_CHECKPOINT_IMMUTABLE_FILES='1', CUDA_VISIBLE_DEVICES='')
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    extension = base / 'continuation-extension.json'
    extended = extension.exists()
    ready_name = 'continuation-checkpoint-ready.json' if extended else 'trial-checkpoint-ready.json'
    ready = json.loads((evidence / ready_name).read_text())
    receipt = json.loads((evidence / 'last-local-save.json').read_text())
    assert ready['no_further_updates'] and receipt['step'] == ready['global_step']
    native = evidence / 'continuation-20261005/train/checkpoints/last.pt'
    raw = torch.load(_checkpoint_upload_source(native.resolve()), map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(raw)
    assert metadata['global_step'] == ready['global_step']
    assert metadata['adam_step'] == metadata['global_step'] - 34430
    assert raw['scheduler']['last_epoch'] == metadata['global_step'] - 35056
    assert metadata['rng_ranks'] == 8 and metadata['adam_parameters'] == 782
    slots = base / ('final-cpu-upload-epoch100' if extended else 'final-cpu-upload');slots.mkdir(exist_ok=True)
    for name, source in [('last.pt',native), ('best-fid-resume.pt',Path(raw['best_fid'][0][1])),
                         ('best-is-resume.pt',Path(raw['best_inception'][0][1]))]:
        cached = Path(_checkpoint_upload_source(source.resolve()))
        os.link(cached, slots / name)
        pinned = torch.load(slots / name, map_location='cpu', mmap=True, weights_only=False)
        record((slots / name).with_suffix('.json'), recovery_metadata(pinned))
    status = json.loads((evidence / 'continuation-20261005/status.json').read_text())
    phase = status.get('phase','continuation_20261005_'+str(status['attempt']))
    rank_paths = list((evidence / 'verification' / phase).glob('process-rank*.json'))
    assert len(rank_paths) == 8
    pids = [json.loads(p.read_text())['pid'] for p in rank_paths] + [status['torchrun_pid'],child.pid]
    for pid in pids:
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists():assert str(base).encode() in command.read_bytes()
    final_name = 'continuation-final-state-verification.json' if extended else 'trial-final-state-verification.json'
    record(evidence / final_name, dict(metadata=metadata,ready=ready,
        full_best_and_last_pinned=True,no_further_updates=True,verified_at=time.time()))
    for pid in pids:
        try:os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:pass
    child.wait(timeout=30)
    command = [sys.executable,str(Path(__file__).with_name('drain_imagenet_full_checkpoints.py')),
        '--base',str(base),'--evidence',str(evidence),'--run',RUN_PATH,
        '--key-file',str(key_file),'--stop-reason','bounded sustained higher-LR trial finished; full best protected']
    if extended:
        command[-1] = 'user-requested continuation endpoint; full best and latest protected'
        command.extend(['--directory',str(slots)])
    with (evidence / 'continuation-20261005/final-cpu-upload.log').open('a') as stream:
        upload = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                  env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    record(evidence / 'continuation-20261005/final-cpu-upload-launch.json',dict(pid=upload.pid,time=time.time()))
    status.update(state='stopped_resumable',final_step=metadata['global_step'],
                  full_final_checkpoint_verified=True,
                  stop_reason='continuation endpoint reached' if extended else 'bounded loss-decay trial finished')
    record(evidence / 'continuation-20261005/status.json',status)
    if extended:
        os.environ['WANDB_API_KEY'] = key_file.read_text().strip()
        import wandb
        run = wandb.init(entity='helloimlixin-rutgers',project='laser',id=RUN_ID,
                         resume='must',mode='online',dir=str(base))
        run.summary.update({'execution/state':'completed' if metadata['epoch']>=100 else 'stopped_resumable',
            'execution/final_epoch':metadata['epoch'],'execution/final_step':metadata['global_step'],
            'verification/full_best_and_last_saved':True})
        run.finish(exit_code=0)
    return metadata


def supervise(base, evidence, key_file):
    lock = (base / 'loss-decay-trial.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child = None
    stopping = False
    def stop(_sig,_frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    audit = evidence / 'continuation-20261005'
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')
    extension = base / 'continuation-extension.json'
    extended = extension.exists()
    if extended:env['LASER_CONTINUATION_TAG'] = json.loads(extension.read_text())['phase_prefix']
    ready_name = 'continuation-checkpoint-ready.json' if extended else 'trial-checkpoint-ready.json'
    command = [sys.executable,str(Path(__file__).resolve()),'upload-recovery',
        '--base',str(base),'--evidence',str(evidence),'--key-file',str(key_file)]
    with (audit / 'recovery-upload.log').open('a') as stream:
        upload = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    record(audit / 'recovery-upload-launch.json',dict(pid=upload.pid,time=time.time()))
    deadline = time.monotonic()+300
    while not (audit / 'online-run-created.json').exists():
        if stopping:return
        if upload.poll() is not None or time.monotonic()>deadline:
            raise RuntimeError('Online run creation failed; training not launched')
        time.sleep(2)
    command[2] = 'train-once'
    with (audit / 'train-supervisor.log').open('a') as stream:
        child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    while child.poll() is None:
        progress = dict(state='running',supervisor_pid=os.getpid(),
            trainer_supervisor_pid=child.pid,active_base=str(base),active_evidence=str(evidence),
            active_run=RUN_ID,temporary_fid_regressions_permitted=True,trial_epochs=3,
            warmup_updates=200,peak_lr=1e-5,official_evaluations_only=True,heartbeat_unix=time.time())
        if extended:progress.update(trial_epochs=None,bounded_trial=False,target_epoch=100,warmup_restarted=False)
        record(evidence / 'loss-decay-trial-status.json',progress)
        if extended:record(evidence / 'continuation-status.json',progress)
        if (evidence / ready_name).exists():
            metadata = finalize_trial(base,evidence,child,key_file)
            final_progress = dict(state='stopped_resumable',
                reason='continuation endpoint reached' if extended else 'bounded trial finished',active_run=RUN_ID,final_step=metadata['global_step'],
                full_best_and_last_pinned=True,official_evaluations_only=True,time=time.time())
            record(evidence / 'loss-decay-trial-status.json',final_progress)
            if extended:record(evidence / 'continuation-status.json',final_progress)
            return
        time.sleep(5)
    child.wait()
    record(evidence / 'loss-decay-trial-status.json',dict(state='failed' if child.returncode else 'completed',
        exit_code=child.returncode,active_run=RUN_ID,time=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','upload-recovery','supervise','train-once'])
    for name in ('base','evidence','key-file'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-base',type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        assert args.parent_base is not None
        prepare(args.base,args.evidence,args.parent_base);return
    recovery.RUN_ID = recovery.runner.RUN_ID = RUN_ID
    recovery.RUN_PATH = recovery.runner.RUN_PATH = RUN_PATH
    parent = json.loads((args.evidence / 'continuation-20261005/source-parent.json').read_text())
    recovery.PARENT,recovery.SOURCE_RUN_ID = Path(parent['base']),parent['run_id']
    if args.action == 'upload-recovery':recovery.upload_recovery(args.base,args.evidence,args.key_file)
    elif args.action == 'train-once':
        import yaml
        target_epoch = yaml.safe_load((args.base / 'recipe.yaml').read_text())['options']['epochs']
        original_record = recovery.runner.record
        def progress(path, value):
            if Path(path).name == 'status.json':value['target_epoch'] = target_epoch
            original_record(path,value)
        recovery.runner.record = progress
        recovery.runner.supervise(args.base,args.evidence,args.key_file)
    else:supervise(args.base,args.evidence,args.key_file)


if __name__ == '__main__':main()
