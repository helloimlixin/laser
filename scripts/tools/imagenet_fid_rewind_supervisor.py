"""Supervise official-FID regression stops, full-state rewinds, and online forks."""
import argparse
import fcntl
import hashlib
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

try:
    from imagenet_fid_rewind_guard import install_guard
except ModuleNotFoundError:
    from scripts.tools.imagenet_fid_rewind_guard import install_guard


def record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def cpu_runtime(base):
    os.environ.update(CUDA_VISIBLE_DEVICES='',
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    for path in (str(base / 'support'), str(base / 'source/runtime')):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    import torch
    torch.set_num_threads(4)


def release_stopped_run(base, evidence, child, key_file, *, request):
    """Release GPUs only after the exact stopped state and both winners are pinned."""
    cpu_runtime(base)
    import torch
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from src.training.full_resume_upload import recovery_metadata
    from src.training.fid_adaptive_schedule import FidAdaptiveSchedule
    ready = json.loads((evidence / 'fid-rewind-checkpoint-ready.json').read_text())
    receipt = json.loads((evidence / 'last-local-save.json').read_text())
    step = int(request['global_step'])
    assert ready['no_further_updates'] and ready['request'] == request
    assert int(receipt['step']) == step
    native = evidence / 'continuation-20261005/train/checkpoints/last.pt'
    local = Path(_checkpoint_upload_source(native.resolve(strict=True)))
    raw = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(raw)
    state = metadata['learning_rate_schedule']['state']
    assert metadata['global_step'] == step and metadata['adam_step'] == step - 34430
    assert state['last_epoch'] == step - 35056
    assert state['last_observation_step'] == state['last_epoch']
    assert metadata['original_rqtransformer_metrics']['fid'] == request['fid']
    assert metadata['world_size'] == metadata['rng_ranks'] == 8
    assert metadata['adam_parameters'] == 782 and metadata['next_microbatch'] == 0
    assert all(math.isclose(lr, FidAdaptiveSchedule.lr_at_step(
        state['policy'], state['last_epoch'], state['multiplier']),
        rel_tol=1e-12, abs_tol=1e-20) for lr in metadata['saved_learning_rates'])
    assert math.isclose(metadata['saved_learning_rates'][0], request['requested_lr'],
                        rel_tol=1e-12, abs_tol=1e-20)
    status_path = evidence / 'continuation-20261005/status.json'
    status = json.loads(status_path.read_text())
    assert status['supervisor_pid'] == child.pid
    phase = evidence / 'verification' / ('continuation_20261005_' + str(status['attempt']))
    guards = sorted(phase.glob('fid-rewind-rank*.json'))
    assert len(guards) == 8
    for path in guards:
        guard = json.loads(path.read_text())
        assert guard['request'] == request and guard['no_further_updates']
        assert guard['adam_step'] == metadata['adam_step']
        assert guard['updates'] == ready['updates']
    directory = base / 'final-cpu-upload'
    directory.mkdir(exist_ok=True)
    pins = {}
    sources = [('last.pt', native), ('best-fid-resume.pt', Path(raw['best_fid'][0][1])),
               ('best-is-resume.pt', Path(raw['best_inception'][0][1]))]
    for name, source in sources:
        cached = Path(_checkpoint_upload_source(source.resolve(strict=True)))
        target = directory / name
        if not target.exists():
            os.link(cached, target)
        assert os.path.samefile(cached, target)
        pin = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
        pin_metadata = recovery_metadata(pin)
        assert pin_metadata['rng_ranks'] == 8 and pin_metadata['adam_parameters'] == 782
        record(target.with_suffix('.json'), pin_metadata)
        pins[name] = dict(path=str(target), bytes=target.stat().st_size,
                          epoch=pin_metadata['epoch'], global_step=pin_metadata['global_step'],
                          adam_step=pin_metadata['adam_step'])
    assert os.path.samefile(directory / 'best-fid-resume.pt',
                            _checkpoint_upload_source(Path(request['source_checkpoint']).resolve()))
    record(evidence / 'automatic-rewind-stop-verification.json',
           dict(metadata=metadata, request=request, pins=pins,
                all8_ranks_stopped_before_next_update=True, verified_at=time.time()))
    pids = [json.loads(path.read_text())['pid'] for path in sorted(phase.glob('process-rank*.json'))]
    assert len(pids) == len(set(pids)) == 8
    pids += [status['torchrun_pid'], status['supervisor_pid']]
    for pid in pids:
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists():
            assert str(base).encode() in command.read_bytes()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=30)
    status.update(state='stopped_resumable', stop_reason=request['reason'],
                  final_step=step, full_final_checkpoint_verified=True,
                  heartbeat_unix=time.time(), automatic_rewind=True)
    record(status_path, status)
    command = [sys.executable, str(Path(__file__).with_name('drain_imagenet_full_checkpoints.py')),
               '--base', str(base), '--evidence', str(evidence), '--run', status['run_path'],
               '--key-file', str(key_file), '--stop-reason', request['reason']]
    log = evidence / 'continuation-20261005/final-cpu-upload.log'
    with log.open('a') as stream:
        upload = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    record(evidence / 'continuation-20261005/final-cpu-upload-launch.json',
           dict(pid=upload.pid, log=str(log), launched_at=time.time(), automatic_rewind=True))
    return metadata


def prepare_rewind(base, evidence, destination, persistent, *, request, run_id):
    """Restore the saved best Adam/RNG state and reduce only its LR amplitude."""
    cpu_runtime(base)
    import torch
    import yaml
    from src.training.k4_checkpoint_io import _checkpoint_upload_source, atomic_torch_save, _persist_serialized_checkpoint
    from src.training.full_resume_upload import recovery_metadata
    from imagenet_epoch64_lower_lr import lower_rewind_lr
    from imagenet_zero_floor_lr import create_scheduler
    source = base / 'final-cpu-upload/best-fid-resume.pt'
    raw = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    original = recovery_metadata(raw)
    metrics = original['original_rqtransformer_metrics']
    assert metrics is not None and metrics['fid'] == request['best_fid']
    assert metrics['global_step'] == original['global_step']
    assert original['epoch'] >= 64 and original['next_microbatch'] == 0
    assert original['adam_step'] == original['global_step'] - 34430
    assert original['learning_rate_schedule']['state']['last_epoch'] == original['global_step'] - 35056
    assert original['rng_ranks'] == 8 and original['adam_parameters'] == 782
    # A regression's existing controller already halves the live LR. Never
    # raise the saved-best LR when returning to its earlier cosine clock.
    target_lr = min(float(request['requested_lr']), original['saved_learning_rates'][0] * .5)
    optimizer, state = lower_rewind_lr(raw['optimizer'], raw['scheduler'],
        initial_lr=target_lr, scored_epoch64=False, allow_zero=True)
    initial_lr = optimizer['param_groups'][0]['lr']
    fake = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial_lr)
    schedule = create_scheduler(fake, initial_lr=state['policy']['initial_lr'], min_lr=0.,
        total_steps=state['policy']['total_steps'], completed_steps=state['last_epoch'], state_dict=state)
    assert schedule.state_dict() == state
    assert {k:v for k,v in state.items() if k != 'multiplier'} == {
        k:v for k,v in raw['scheduler'].items() if k != 'multiplier'}
    audit = persistent / 'continuation-20261005'
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    assert not (checkpoints / 'last.pt').exists()
    for directory in ('source/runtime', 'support'):
        shutil.copytree(base / directory, destination / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    inputs = destination / 'inputs'
    inputs.mkdir(exist_ok=True)
    pinned_source = inputs / f"source-epoch{original['epoch']:03d}.pt"
    os.link(source, pinned_source)
    for name in ('resume-stage1-tokenizer.pt', 'resume-weights-inception-2015-12-05-6726825d.pth'):
        os.link(base / 'inputs' / name, inputs / name)
    for name in ('torch-cache', 'inductor-cache'):
        (destination / name).symlink_to((base / name).resolve(), target_is_directory=True)
    recipe = yaml.safe_load((base / 'recipe.yaml').read_text())
    if 'objective_revision' in raw:
        objective = raw['objective_revision']
        options = recipe['options']
        from src.training.physical_pair_crps import crps_weight_at_step
        assert objective['version'] == 'physical-pair-range-normalized-crps-v1'
        assert (objective['max_weight'], objective['start_step'], objective['ramp_steps']) == (
            options['coeff_crps_weight'], options['coeff_crps_start_step'], options['coeff_crps_ramp_steps'])
        assert objective['completed_steps'] == max(0, original['global_step'] - objective['start_step'])
        assert objective['next_weight'] == crps_weight_at_step(
            objective['max_weight'], original['global_step'], objective['start_step'], objective['ramp_steps'])
    parent_id = recipe['options']['wandb_id']
    recipe['options'].update(checkpoint=str(inputs / 'resume-stage1-tokenizer.pt'),
        output=str(destination / 'production/train'), checkpoint_dir=str(checkpoints),
        lr_schedule_restart_id=run_id, wandb_id=run_id, fid_every=1, min_lr=0.,
        wandb_name=f"ImageNet rFID4.21 K4 | auto rewind epoch{original['epoch']} FID{metrics['fid']:.4f} | LR{initial_lr:.3e}, floor0 | 8 H100")
    recipe['options'].pop('resume_checkpoint', None)
    record(destination / 'official-baseline.json', metrics)
    entry = (base / 'entry.py').read_text()
    start = entry.index(' config.update(resume_source_epoch=')
    end = entry.index(' WB=original_init(*args,**kwargs)', start)
    entry = entry[:start] + (
        f" config.update(resume_source_epoch={original['epoch']},resume_source_global_step={original['global_step']},\n"
        f"  source_run={'helloimlixin-rutgers/laser/'+parent_id!r},\n"
        f"  source_checkpoint_epoch={original['epoch']},source_checkpoint_step={original['global_step']},source_checkpoint_fid={metrics['fid']!r},\n"
        f"  source_inception_score={metrics['inception_score']!r},source_optimizer_restored=True,trained_source_optimizer_available=True,\n"
        f"  optimizer_initialization='restored trained best checkpoint AdamW moments and 782 counters at{original['adam_step']}',\n"
        "  rng_initialization='restored all 8 best checkpoint RNG streams',\n"
        "  evaluation_protocol='Official RQ-Transformer Inception/FID/10-split IS only; validation50k/generated50k',\n"
        "  baseline_evaluation_provenance='same saved best model; model and Adam tensors verified unchanged',\n"
        "  learning_rate_schedule='saved zero-floor cosine through epoch100; reduced LR amplitude; strict automatic rewind on FID regression',\n"
        "  lr_continuation='best checkpoint clock and FID controller history preserved; trained Adam preserved; lower LR amplitude',\n"
        f"  source_scheduler_step={state['last_epoch']},continuation_scheduler_origin_adam_step=626,continuation_scheduler_origin_global_step=35056,\n"
        f"  restart_initial_lr={initial_lr!r},lr_floor_before=0.,lr_floor_after=0.,automatic_fid_rewind=True,cosine_decay_end_epoch=100)\n"
    ) + entry[end:]
    entry, count = re.subn(r'(?m)^CHECKPOINT_EPOCH=\d+$', f"CHECKPOINT_EPOCH={original['epoch']}", entry)
    assert count == 1
    entry, count = re.subn(r"WB.log\(\{'train/epoch':\d+,'train/global_step':\d+,",
        f"WB.log({{'train/epoch':{original['epoch']},'train/global_step':{original['global_step']},", entry)
    assert count == 1
    (destination / 'entry.py').write_text(entry)
    (destination / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    install_guard(destination)
    fid_alias = checkpoints / f"best_fid_{metrics['fid']:.4f}_epoch_{original['epoch']:03d}.pt"
    best_is_source = base / 'final-cpu-upload/best-is-resume.pt'
    best_is_payload = torch.load(best_is_source, map_location='cpu', mmap=True, weights_only=False)
    best_is = recovery_metadata(best_is_payload)
    best_is_metric = best_is['original_rqtransformer_metrics']
    assert best_is_metric is not None
    is_alias = checkpoints / f"best_is_{best_is_metric['inception_score']:.4f}_epoch_{best_is['epoch']:03d}.pt"
    same_is_source = os.path.samefile(best_is_source, source)
    revision = dict(source_run=parent_id, source_epoch=original['epoch'],
        source_global_step=original['global_step'], source_adam_step=original['adam_step'],
        original_lr=original['saved_learning_rates'][0], initial_lr=initial_lr,
        old_scheduler=raw['scheduler'], new_scheduler=state, optimizer_moments_reset=False,
        schedule_clock_preserved=True, new_floor=0., decay_end_epoch=100,
        automatic_rewind_request=request, terminal_zero_lr=initial_lr == 0.)
    payload = dict(raw, optimizer=optimizer, scheduler=state,
        config=dict(raw['config'], **recipe['options']), lr_revision=revision,
        best_fid=[(metrics['fid'], str(fid_alias))],
        best_inception=[(best_is_metric['inception_score'], str(is_alias))])
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(destination / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(destination / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    latest = checkpoints / 'last.pt'
    atomic_torch_save(payload, latest)
    for alias in (fid_alias, checkpoints / 'resume-anchor.pt'):
        alias.symlink_to(latest.resolve().relative_to(checkpoints))
    if same_is_source:
        is_alias.symlink_to(latest.resolve().relative_to(checkpoints))
    else:
        # Persistence consumes its temporary input; retain the old upload pin.
        is_pin = destination / 'checkpoint-staging/best-is-persist.pt'
        is_pin.parent.mkdir(exist_ok=True)
        os.link(best_is_source, is_pin)
        _persist_serialized_checkpoint(is_pin, is_alias)
        # This independent IS winner retains its own original full optimizer.
        copied_is = torch.load(_checkpoint_upload_source(is_alias), map_location='cpu', mmap=True, weights_only=False)
        assert recovery_metadata(copied_is)['original_rqtransformer_metrics'] == best_is_metric
        assert os.path.samefile(best_is_source, _checkpoint_upload_source(is_alias))
    restored = torch.load(_checkpoint_upload_source(latest), map_location='cpu', mmap=True, weights_only=False)
    compared = 0
    for key, tensor in raw['state_dict'].items():
        assert torch.equal(tensor, restored['state_dict'][key]), key
        compared += tensor.numel()
    for key, fields in raw['optimizer']['state'].items():
        for field, tensor in fields.items():
            if torch.is_tensor(tensor):
                assert torch.equal(tensor, restored['optimizer']['state'][key][field]), (key,field)
                compared += tensor.numel()
    for rank, streams in enumerate(raw['rng_state_by_rank']):
        for name, tensor in streams.items():
            assert torch.equal(tensor, restored['rng_state_by_rank'][rank][name])
    assert restored['scheduler'] == state
    if 'objective_revision' in raw:
        assert restored['objective_revision'] == raw['objective_revision']
    metadata = recovery_metadata(restored)
    assert metadata['adam_step'] == original['adam_step']
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'md5').hexdigest()
    record(audit / 'source-parent.json', dict(base=str(base), run_id=parent_id))
    record(audit / 'source-state-verification.json', original)
    record(audit / 'restart-state-verification.json', dict(metadata=metadata, revision=revision,
        source_md5=digest, tensor_values_compared=compared, model_and_adam_tensors_identical=True,
        rank_rng_identical=True, schedule_clock_preserved=True,
        objective_revision=restored.get('objective_revision')))
    record(audit / 'source-checkpoint-selection.json', dict(file=str(pinned_source),
        epoch=original['epoch'], global_step=original['global_step'], adam_step=original['adam_step'],
        fid=metrics['fid'], md5=digest, bytes=source.stat().st_size, full_optimizer_state=True))
    for name in ('entry.py','recipe.yaml','official-baseline.json'):
        shutil.copyfile(destination / name, audit / name)
    (persistent / 'status.json').symlink_to('continuation-20261005/status.json')
    return initial_lr


def supervise_with_rewinds(base, evidence, key_file, *, driver):
    root_base, root_evidence = base, evidence
    lock = (root_base / 'automatic-rewind.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = False
    child = None
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    import yaml
    root_run = yaml.safe_load((base / 'recipe.yaml').read_text())['options']['wandb_id']
    policy = root_evidence / 'automatic-rewind-status.json'
    rewinds = 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
    terminal = False
    while not stopping:
        run_id = yaml.safe_load((base / 'recipe.yaml').read_text())['options']['wandb_id']
        install_guard(base)
        audit = evidence / 'continuation-20261005'
        shutil.copyfile(base / 'entry.py', audit / 'entry.py')
        marker = audit / 'online-run-created.json'
        if not marker.exists():
            log = audit / 'recovery-upload.log'
            command = [sys.executable,str(driver),'upload-recovery','--base',str(base),
                       '--evidence',str(evidence),'--key-file',str(key_file)]
            with log.open('a') as stream:
                upload = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True, env=env)
            record(audit / 'recovery-upload-launch.json',dict(pid=upload.pid,log=str(log)))
            deadline = time.monotonic()+300
            while not marker.exists():
                if stopping:
                    return
                if upload.poll() is not None or time.monotonic()>=deadline:
                    record(policy,dict(state='failed',reason='online run creation failed',log=str(log)))
                    raise RuntimeError('Online run creation failed; training was not launched')
                time.sleep(2)
        if terminal:
            cpu_runtime(base)
            from src.training.k4_checkpoint_io import _checkpoint_upload_source
            slots = base / 'final-cpu-upload';slots.mkdir(exist_ok=True)
            checkpoints = audit / 'train/checkpoints'
            for name, source in [('last.pt',checkpoints/'last.pt'),
                                 ('best-fid-resume.pt',next(checkpoints.glob('best_fid_*.pt'))),
                                 ('best-is-resume.pt',next(checkpoints.glob('best_is_*.pt')))]:
                os.link(_checkpoint_upload_source(source.resolve()),slots/name)
            command = [sys.executable,str(driver.with_name('drain_imagenet_full_checkpoints.py')),
                       '--base',str(base),'--evidence',str(evidence),'--run','helloimlixin-rutgers/laser/'+run_id,
                       '--key-file',str(key_file),'--stop-reason','rewound to best after zero LR endpoint']
            with (audit/'final-cpu-upload.log').open('a') as stream:
                subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            record(policy,dict(state='stopped_after_zero_lr_rewind',active_base=str(base),
                               active_evidence=str(evidence),active_run=run_id,rewinds=rewinds))
            return
        log = audit / 'train-supervisor.log'
        command = [sys.executable,str(driver),'train-once','--base',str(base),
                   '--evidence',str(evidence),'--key-file',str(key_file)]
        with log.open('a') as stream:
            child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                     start_new_session=True,env=env)
        request = None
        regression_started = None
        while child.poll() is None:
            record(policy,dict(state='running',supervisor_pid=os.getpid(),
                trainer_supervisor_pid=child.pid,active_base=str(base),active_evidence=str(evidence),
                active_run=run_id,rewinds=rewinds,heartbeat_unix=time.time(),
                strict_fid_regression=True,restore_trained_adam=True,restore_all8_rng=True))
            requested = evidence / 'fid-rewind-request.json'
            ready = evidence / 'fid-rewind-checkpoint-ready.json'
            if requested.exists() and not stopping:
                if regression_started is None:
                    regression_started = time.monotonic()
                if ready.exists():
                    request = json.loads(requested.read_text())
                    break
                if time.monotonic()-regression_started > 1200:
                    record(policy,dict(state='failed',reason='regression checkpoint did not commit',active_run=run_id))
                    raise RuntimeError('Retain stopped GPU workers: regression checkpoint did not commit')
            time.sleep(5)
        if stopping:
            child.wait()
            record(policy,dict(state='stopped',active_run=run_id,rewinds=rewinds))
            return
        if request is None:
            child.wait()
            status = json.loads((audit / 'status.json').read_text())
            record(policy,dict(state=status['state'],active_run=run_id,
                               active_base=str(base),active_evidence=str(evidence),rewinds=rewinds))
            return
        release_stopped_run(base,evidence,child,key_file,request=request)
        rewinds += 1
        new_id = root_run + f'-rewind{rewinds:03d}'
        new_base = Path(str(root_base)+f'-rewind{rewinds:03d}')
        new_evidence = root_evidence.parent / new_id
        record(policy,dict(state='preparing_rewind',active_run=run_id,next_run=new_id,
                           source_fid=request['best_fid'],rejected_fid=request['fid'],rewinds=rewinds))
        lr = prepare_rewind(base,evidence,new_base,new_evidence,request=request,run_id=new_id)
        print(json.dumps(dict(rewound=True,run=new_id,best_fid=request['best_fid'],lr=lr)),flush=True)
        base,evidence = new_base,new_evidence
        terminal = lr == 0.
