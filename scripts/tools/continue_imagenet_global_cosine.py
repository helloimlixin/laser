"""Continue the protected epoch77 prior on a scaled 100-epoch cosine to zero."""
import argparse
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

import continue_imagenet_fid16379 as recovery

RUN_ID = 'imagenet-rfid421-epoch77-rqcos0-lr1p249e6-1epoch-8h100-20261006'
RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID
record = recovery.record


def prepare(base, evidence, parent, *, target_epoch=78, resume_lr=None):
    import torch
    import yaml
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[2]
    audit = evidence / 'continuation-20261005'
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    assert not (checkpoints / 'last.pt').exists(), 'Refuse to overwrite a prepared trial'
    for folder in ('source/runtime', 'support'):
        shutil.copytree(parent / folder, base / folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('imagenet_global_cosine_lr.py', 'continue_imagenet_global_cosine.py',
                 'verify_imagenet_global_cosine.py', 'continue_imagenet_fid16379.py',
                 'continue_imagenet_pairfix_repair.py', 'drain_imagenet_full_checkpoints.py'):
        shutil.copyfile(repo / 'scripts/tools' / name, base / 'support' / name)
    sys.path[:0] = [str(base / 'source/runtime'), str(base / 'support')]
    from src.training import k4_checkpoint_io as checkpoint_io
    from src.training.full_resume_upload import recovery_metadata
    from imagenet_global_cosine_lr import migrate_global_cosine, create_scheduler, peak_for_resume_lr
    from continue_imagenet_lr1000 import compare_tensors
    inputs = base / 'inputs'; inputs.mkdir(exist_ok=True)
    source = inputs / 'source-epoch077-full.pt'
    os.link(parent / 'fluctuation-audit-20261006/best-fid-resume.pt', source)
    raw = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    original = recovery_metadata(raw)
    assert (original['epoch'], original['global_step'], original['adam_step'], original['next_microbatch']) == (77, 48202, 13772, 0)
    assert original['adam_parameters'] == 782 and original['rng_ranks'] == original['world_size'] == 8
    metrics = original['original_rqtransformer_metrics']
    assert metrics['fid'] == 15.24941539209243 and metrics['metric_backend'] == 'original_rqtransformer'
    if not 77 < target_epoch <= 100:
        raise ValueError('Continuation target must be between epochs78 and100')
    peak_lr = (1e-5 if resume_lr is None else
               peak_for_resume_lr(resume_lr, global_step=48202, total_steps=62600))
    optimizer, state = migrate_global_cosine(raw['optimizer'], raw['scheduler'],
        global_step=48202, steps_per_epoch=626, peak_lr=peak_lr, baseline_fid=metrics['fid'])
    lr = optimizer['param_groups'][0]['lr']
    if resume_lr is not None:assert math.isclose(lr, resume_lr, rel_tol=1e-12)
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    assert create_scheduler(dummy, initial_lr=peak_lr, min_lr=0., total_steps=62600,
        completed_steps=48202, state_dict=state).state_dict() == state
    recipe = yaml.safe_load((parent / 'recipe.yaml').read_text())
    parent_id = recipe['options']['wandb_id']
    recipe['options'].update(checkpoint=str(inputs / 'resume-stage1-tokenizer.pt'),
        output=str(base / 'production/train'), checkpoint_dir=str(checkpoints),
        wandb_id=RUN_ID, lr=peak_lr, min_lr=0., lr_schedule_epochs=100, epochs=target_epoch,
        lr_schedule_restart_id=raw['config']['lr_schedule_restart_id'], max_optimizer_steps=0,
        wandb_name=f'ImageNet K4 | epoch77 FID15.2494 | LR{lr:.6g} cosine to0 at100 | continue to{target_epoch} | 8H100')
    recipe['options'].pop('resume_checkpoint', None)
    for name in ('resume-stage1-tokenizer.pt', 'resume-weights-inception-2015-12-05-6726825d.pth'):
        os.link(parent / 'inputs' / name, inputs / name)
    for name in ('torch-cache', 'inductor-cache'):
        (base / name).symlink_to((parent / name).resolve(), target_is_directory=True)
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    record(base / 'official-baseline.json', metrics)
    fid_alias = checkpoints / 'best_fid_15.2494_epoch_077.pt'
    is_source = parent / 'fluctuation-audit-20261006/best-is-resume.pt'
    is_payload = torch.load(is_source, map_location='cpu', mmap=True, weights_only=False)
    is_meta = recovery_metadata(is_payload)
    is_score = is_meta['original_rqtransformer_metrics']['inception_score']
    is_alias = checkpoints / f"best_is_{is_score:.4f}_epoch_{is_meta['epoch']:03d}.pt"
    revision = dict(run_id=RUN_ID, run_path=RUN_PATH,
        source_run=parent_id, source_epoch=77, source_global_step=48202,
        source_adam_step=13772, original_lr=original['saved_learning_rates'][0],
        initial_lr=lr, peak_lr=peak_lr, requested_resume_lr=resume_lr,
        floor=0., schedule_epochs=100, schedule_steps=62600,
        target_epoch=target_epoch, trial_epochs=target_epoch-77,
        automatic_fid_rewind=False, old_scheduler=raw['scheduler'], new_scheduler=state,
        global_progress_preserved=True, scheduler_clock_rebased_to_global=True,
        source_scheduler_step=13146, new_scheduler_step=48202,
        optimizer_moments_reset=False, warmup_restarted=False,
        objective_unchanged=True, sampling_unchanged=True, parameter_ema_enabled=False,
        official_evaluations_only=True,
        lr_peak_interpretation=f'Original RQ100epoch cosine shape scaled to peak{peak_lr!r} for epoch77 LR{lr!r}; published5e-4 peak would give6.24722325923851e-5 at77.')
    payload = dict(raw, optimizer=optimizer, scheduler=state,
        config=dict(raw['config'], **recipe['options']), lr_revision=revision,
        best_fid=[(metrics['fid'], str(fid_alias))], best_inception=[(is_score, str(is_alias))])
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    checkpoint_io.atomic_torch_save(payload, checkpoints / 'last.pt')
    native = (checkpoints / 'last.pt').resolve()
    for alias in (fid_alias, checkpoints / 'resume-epoch077-anchor.pt'):
        alias.symlink_to(native.relative_to(checkpoints))
    disposable = inputs / '.initial-is-persist.pt'; os.link(is_source, disposable)
    checkpoint_io._persist_serialized_checkpoint(disposable, is_alias)
    restored = torch.load(checkpoint_io._checkpoint_upload_source(native), map_location='cpu', mmap=True, weights_only=False)
    compared = compare_tensors(raw, restored)
    metadata = recovery_metadata(restored)
    assert metadata['adam_step'] == 13772 and metadata['saved_learning_rates'] == [lr]
    assert restored['scheduler']['last_epoch'] == restored['global_step'] == 48202
    with source.open('rb') as stream:digest = hashlib.file_digest(stream, 'md5').hexdigest()
    record(audit / 'source-parent.json', dict(base=str(parent), run_id=parent_id))
    record(audit / 'source-state-verification.json', original)
    record(audit / 'restart-state-verification.json', dict(metadata=metadata, revision=revision,
        source_md5=digest, model_and_adam_tensors_identical=True, rank_rng_identical=True,
        tensor_values_compared=compared, source_train_loss_tracking=raw['train_loss_tracking']))
    record(audit / 'source-checkpoint-selection.json', dict(file=str(source), epoch=77,
        global_step=48202, adam_step=13772, fid=metrics['fid'], md5=digest,
        bytes=source.stat().st_size, full_optimizer_state=True))
    record(base / 'global-cosine-trial.json', revision)
    code = (parent / 'entry.py').read_text()
    assert code.count('ARGS.lr_schedule_epochs==44') == 1
    code = code.replace('ARGS.lr_schedule_epochs==44', 'ARGS.lr_schedule_epochs==100')
    assert code.count('ARGS.lr==0.00001') == 1
    code = code.replace('ARGS.lr==0.00001', f'ARGS.lr=={peak_lr!r}')
    code = code.replace('CHECKPOINT_EPOCH=68', 'CHECKPOINT_EPOCH=77')
    code = code.replace("WB.log({'train/epoch':68,'train/global_step':42568,", "WB.log({'train/epoch':77,'train/global_step':48202,")
    code = code.replace("record(EVIDENCE/'continuation-checkpoint-ready.json'", "record(EVIDENCE/'trial-checkpoint-ready.json'")
    code = code.replace('from imagenet_loss_decay_lr import create_scheduler, observe_official_fid',
        'from imagenet_global_cosine_lr import create_scheduler, observe_official_fid')
    start, end = code.index(' config.update(automatic_fid_rewind='), code.index(' config.update(objective_revision=')
    code = code[:start] + (
        f" config.update(automatic_fid_rewind=False,bounded_trial={target_epoch<100!r},trial_epochs={target_epoch-77},trial_target_epoch={target_epoch},continuation_target_epoch={target_epoch},\n"
        "  fid_rewind_policy='continue to requested endpoint; retain full best checkpoint',\n"
        "  parameter_ema_enabled=False,objective_changed=False,warmup_restarted=False,\n"
        f"  cosine_clock='absolute global data progress',cosine_peak_lr={peak_lr!r},cosine_total_epochs=100)\n"
    ) + code[end:]
    start, end = code.index(' config.update(resume_source_epoch='), code.index(' WB=original_init(*args,**kwargs)')
    code = code[:start] + (
        f" config.update(resume_source_epoch=77,resume_source_global_step=48202,source_run={'helloimlixin-rutgers/laser/'+parent_id!r},\n"
        "  source_checkpoint_epoch=77,source_checkpoint_step=48202,source_checkpoint_fid=15.24941539209243,\n"
        "  source_optimizer_restored=True,source_adam_step=13772,source_scheduler_step=13146,\n"
        f"  restart_initial_lr={lr!r},new_scheduler_step=48202,lr_floor_after=0.,cosine_decay_end_epoch=100,\n"
        "  optimizer_initialization='trained AdamW moments and all782 counters preserved',\n"
        "  rng_initialization='all8 saved CPU/CUDA RNG streams restored',\n"
        "  evaluation_protocol='Official RQ-Transformer FID/10-split IS only; validation50k/generated50k',\n"
        f"  learning_rate_schedule='original100epoch cosine shape scaled to epoch77 LR{lr:.12g}; zero floor; source global progress77 preserved',\n"
        "  lr_continuation='intentional clock mapping from custom continuation to absolute epoch77; no warmup or Adam reset')\n"
    ) + code[end:]
    compile(code, str(base / 'entry.py'), 'exec'); (base / 'entry.py').write_text(code)
    for name in ('entry.py', 'recipe.yaml', 'official-baseline.json', 'global-cosine-trial.json'):
        shutil.copyfile(base / name, audit / name)
    (evidence / 'status.json').symlink_to('continuation-20261005/status.json')
    print(json.dumps(dict(prepared=True, run=RUN_PATH, source_epoch=77, source_fid=metrics['fid'],
        initial_lr=lr, peak_lr=peak_lr, scheduler_step=48202, adam_step=13772,
        target_epoch=target_epoch, tensor_values_compared=compared)), flush=True)


def finalize(base, evidence, child, key_file):
    import torch
    torch.set_num_threads(4)
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    ready = json.loads((evidence / 'trial-checkpoint-ready.json').read_text())
    receipt = json.loads((evidence / 'last-local-save.json').read_text())
    assert ready['no_further_updates'] and receipt['step'] == ready['global_step']
    checkpoints = evidence / 'continuation-20261005/train/checkpoints'
    native = (checkpoints / 'last.pt').resolve(strict=True)
    raw = torch.load(_checkpoint_upload_source(native), map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(raw)
    control = json.loads((base / 'global-cosine-trial.json').read_text())
    target_epoch = control['target_epoch']
    assert metadata['global_step'] == ready['global_step'] == raw['scheduler']['last_epoch']
    assert metadata['adam_step'] == metadata['global_step'] - 34430
    if metadata['epoch'] >= 100:
        assert metadata['global_step'] == 62600 and metadata['saved_learning_rates'] == [0.]
    slots = base / 'final-cpu-upload'; slots.mkdir(exist_ok=True)
    sources = [('last.pt', native), ('best-fid-resume.pt', Path(raw['best_fid'][0][1])),
               ('best-is-resume.pt', Path(raw['best_inception'][0][1]))]
    for name, source in sources:
        target = slots / name
        if not target.exists():os.link(_checkpoint_upload_source(source.resolve()), target)
        record(target.with_suffix('.json'), recovery_metadata(torch.load(target, map_location='cpu', mmap=True, weights_only=False)))
    record(evidence / 'trial-final-state-verification.json', dict(metadata=metadata,
        ready=ready, full_best_and_last_pinned=True, no_further_updates=True, verified_at=time.time()))
    status = json.loads((evidence / 'continuation-20261005/status.json').read_text())
    ranks = list((evidence / 'verification' / status['phase']).glob('process-rank*.json'))
    assert len(ranks) == 8
    pids = [json.loads(p.read_text())['pid'] for p in ranks] + [status['torchrun_pid'], child.pid]
    for pid in pids:
        proc = Path(f'/proc/{pid}/cmdline')
        if proc.exists():assert str(base).encode() in proc.read_bytes()
    for pid in pids:
        try:os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:pass
    child.wait(timeout=30)
    command = [sys.executable, str(base / 'support/drain_imagenet_full_checkpoints.py'),
        '--base', str(base), '--evidence', str(evidence), '--run', RUN_PATH,
        '--key-file', str(key_file), '--stop-reason', f'epoch77 lower-rate cosine continuation stopped at epoch{metadata["epoch"]}']
    if metadata['epoch'] >= target_epoch:command.append('--completed')
    with (evidence / 'continuation-20261005/final-cpu-upload.log').open('a') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    record(evidence / 'continuation-20261005/final-cpu-upload-launch.json', dict(pid=process.pid, time=time.time()))
    os.environ['WANDB_API_KEY'] = key_file.read_text().strip()
    import wandb
    run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN_ID,
                     resume='must', mode='online', dir=str(base))
    run.summary.update({'execution/state':'completed' if metadata['epoch'] >= target_epoch else 'stopped_resumable',
        'execution/final_epoch':metadata['epoch'], 'execution/final_step':metadata['global_step'],
        'verification/full_best_and_last_saved':True})
    run.finish(exit_code=0)
    return metadata


def supervise(base, evidence, key_file):
    lock = (base / 'global-cosine-trial.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    audit = evidence / 'continuation-20261005'
    control = json.loads((base / 'global-cosine-trial.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1',
               LASER_CONTINUATION_TAG='rqcos0_20261006_epoch77')
    child = None
    stopping = False
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    command = [sys.executable, str(Path(__file__).resolve()), 'upload-recovery',
        '--base', str(base), '--evidence', str(evidence), '--key-file', str(key_file)]
    with (audit / 'recovery-upload.log').open('a') as stream:
        upload = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True, env=env)
    deadline = time.monotonic() + 300
    while not (audit / 'online-run-created.json').exists():
        if stopping:return
        if upload.poll() is not None or time.monotonic() > deadline:raise RuntimeError('Online run creation failed')
        time.sleep(2)
    command[2] = 'train-once'
    with (audit / 'train-supervisor.log').open('a') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
    while True:
        if (evidence / 'trial-checkpoint-ready.json').exists():
            metadata = finalize(base, evidence, child, key_file)
            record(evidence / 'global-cosine-trial-status.json', dict(
                state='completed' if metadata['epoch'] >= control['target_epoch'] else 'stopped_resumable',
                active_run=RUN_ID, final_step=metadata['global_step'], target_epoch=control['target_epoch'],
                full_best_and_last_pinned=True, official_evaluations_only=True, time=time.time()))
            return
        if child.poll() is not None:break
        record(evidence / 'global-cosine-trial-status.json', dict(state='running',
            supervisor_pid=os.getpid(), trainer_supervisor_pid=child.pid,
            active_base=str(base), active_run=RUN_ID, source_epoch=77, target_epoch=control['target_epoch'],
            initial_lr=control['initial_lr'], peak_lr=control['peak_lr'], cosine_zero_epoch=100,
            objective_unchanged=True, parameter_ema_enabled=False,
            official_evaluations_only=True, heartbeat_unix=time.time()))
        time.sleep(5)
    record(evidence / 'global-cosine-trial-status.json', dict(state='failed',
        exit_code=child.returncode, active_run=RUN_ID, time=time.time()))


def main():
    global RUN_ID, RUN_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload-recovery', 'train-once', 'supervise'])
    for name in ('base', 'evidence', 'key-file'):parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parent-base', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--target-epoch', type=int, default=78)
    parser.add_argument('--initial-lr-at-epoch77', type=float)
    args = parser.parse_args()
    if args.action == 'prepare':
        RUN_ID = args.run_id or RUN_ID
        RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID
        prepare(args.base, args.evidence, args.parent_base,
                target_epoch=args.target_epoch, resume_lr=args.initial_lr_at_epoch77)
        return
    control = json.loads((args.base / 'global-cosine-trial.json').read_text())
    RUN_ID = control.get('run_id', RUN_ID)
    RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID
    if args.run_id is not None and args.run_id != RUN_ID:
        raise ValueError('Run identity differs from the saved continuation')
    sys.path[:0] = [str(args.base / 'source/runtime'), str(args.base / 'support')]
    os.environ.update(LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base / 'checkpoint-upload-cache'),
                      LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    recovery.RUN_ID = recovery.runner.RUN_ID = RUN_ID
    recovery.RUN_PATH = recovery.runner.RUN_PATH = RUN_PATH
    parent = json.loads((args.evidence / 'continuation-20261005/source-parent.json').read_text())
    recovery.PARENT, recovery.SOURCE_RUN_ID = Path(parent['base']), parent['run_id']
    if args.action == 'upload-recovery':recovery.upload_recovery(args.base, args.evidence, args.key_file)
    elif args.action == 'train-once':
        original_record = recovery.runner.record
        def progress(path, value):
            if Path(path).name == 'status.json':value['target_epoch'] = control['target_epoch']
            original_record(path, value)
        recovery.runner.record = progress
        recovery.runner.supervise(args.base, args.evidence, args.key_file)
    else:supervise(args.base, args.evidence, args.key_file)


if __name__ == '__main__':main()
