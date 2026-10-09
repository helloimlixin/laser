"""Fork epoch77 for five epochs of paired official raw/EMA evaluation."""
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

import continue_imagenet_fid16379 as recovery

RUN_ID = 'imagenet-rfid421-epoch77-ema999-lr10x-5epoch-8h100-20261006'
RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID
record = recovery.record


def prepare(base, evidence, parent):
    import torch
    import yaml
    torch.set_num_threads(4)
    audit = evidence / 'continuation-20261005'
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    assert not (checkpoints / 'last.pt').exists(), 'Refuse to replace an existing trial'
    repo = Path(__file__).resolve().parents[2]
    for folder in ('source/runtime', 'support'):
        shutil.copytree(parent / folder, base / folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('parameter_ema.py', 'ema_recovery.py', 'full_resume_upload.py'):
        shutil.copyfile(repo / 'src/training' / name, base / 'source/runtime/src/training' / name)
    for name in ('imagenet_loss_decay_lr.py', 'imagenet_ema_hooks.py', 'ema_checkpoint_cloud_upload.py',
                 'continue_imagenet_ema_trial.py', 'verify_imagenet_ema_trial.py',
                 'verified_wandb_checkpoint_upload.py', 'continue_imagenet_pairfix_repair.py',
                 'continue_imagenet_fid16379.py'):
        shutil.copyfile(repo / 'scripts/tools' / name, base / 'support' / name)
    sys.path[:0] = [str(base / 'source/runtime'), str(base / 'support')]
    from src.training import k4_checkpoint_io as checkpoint_io
    from src.training.full_resume_upload import recovery_metadata
    from imagenet_loss_decay_lr import scale_saved_continuation, create_scheduler
    from continue_imagenet_lr1000 import compare_tensors
    inputs = base / 'inputs'; inputs.mkdir(exist_ok=True)
    source = inputs / 'source-epoch077-full.pt'
    os.link(parent / 'fluctuation-audit-20261006/best-fid-resume.pt', source)
    raw = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    original = recovery_metadata(raw)
    assert (original['epoch'], original['global_step'], original['adam_step'], original['next_microbatch']) == (77, 48202, 13772, 0)
    assert original['rng_ranks'] == original['world_size'] == 8 and original['adam_parameters'] == 782
    metrics = original['original_rqtransformer_metrics']
    assert metrics['fid'] == 15.24941539209243 and metrics['metric_backend'] == 'original_rqtransformer'
    optimizer, scheduler = scale_saved_continuation(raw['optimizer'], raw['scheduler'], 10.)
    lr = optimizer['param_groups'][0]['lr']
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    assert create_scheduler(dummy, initial_lr=1e-4, min_lr=0., total_steps=27544,
        completed_steps=scheduler['last_epoch'], state_dict=scheduler).state_dict() == scheduler
    recipe = yaml.safe_load((parent / 'recipe.yaml').read_text())
    parent_id = recipe['options']['wandb_id']
    recipe['options'].update(checkpoint=str(inputs / 'resume-stage1-tokenizer.pt'),
        output=str(base / 'production/train'), checkpoint_dir=str(checkpoints),
        wandb_id=RUN_ID, lr=1e-4, epochs=82, max_optimizer_steps=0,
        wandb_name='ImageNet K4 | epoch77 FID15.2494 | trained Adam LR x10 | weight EMA0.999 | official raw/EMA50k | 5epoch 8H100')
    # A new W&B ID does not restart the saved LR controller.
    recipe['options']['lr_schedule_restart_id'] = raw['config']['lr_schedule_restart_id']
    recipe['options'].pop('resume_checkpoint', None)
    for name in ('resume-stage1-tokenizer.pt', 'resume-weights-inception-2015-12-05-6726825d.pth'):
        os.link(parent / 'inputs' / name, inputs / name)
    for name in ('torch-cache', 'inductor-cache'):
        (base / name).symlink_to((parent / name).resolve(), target_is_directory=True)
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    record(base / 'official-baseline.json', metrics)
    raw_fid = checkpoints / 'best_fid_15.2494_epoch_077.pt'
    is_source = parent / 'fluctuation-audit-20261006/best-is-resume.pt'
    is_payload = torch.load(is_source, map_location='cpu', mmap=True, weights_only=False)
    is_meta = recovery_metadata(is_payload)
    raw_is = checkpoints / f"best_is_{is_meta['original_rqtransformer_metrics']['inception_score']:.4f}_epoch_{is_meta['epoch']:03d}.pt"
    ema_best = {kind: dict(score=metrics[field], path=str(checkpoints / f'best_ema_{kind}_{metrics[field]:.4f}_epoch_077.pt'),
        global_step=48202, epoch=77, metrics=dict(metrics, weight_state='ema', ema_updates=0,
        initialized_from_raw_exactly=True)) for kind, field in (('fid', 'fid'), ('is', 'inception_score'))}
    control = dict(run_id=RUN_ID, source_run=parent_id, source_step=48202, source_epoch=77,
        source_adam_step=13772, target_epoch=82, trial_epochs=5, decay=.999, lr_factor=10.,
        initial_lr=lr, peak_lr=1e-4, source_learning_rate=original['saved_learning_rates'][0],
        initial_ema_best=ema_best, phase_prefix='ema_20261006_epoch77', official_evaluations_only=True)
    record(base / 'ema-trial.json', control)
    revision = dict(source_run=parent_id, source_epoch=77, source_global_step=48202,
        source_adam_step=13772, original_lr=original['saved_learning_rates'][0], initial_lr=lr,
        peak_lr=1e-4, factor=10., old_scheduler=raw['scheduler'], new_scheduler=scheduler,
        optimizer_moments_reset=False, schedule_clock_preserved=True, warmup_restarted=False,
        target_epoch=82, objective_unchanged=True, sampling_unchanged=True, ema_decay=.999)
    payload = dict(raw, optimizer=optimizer, scheduler=scheduler, config=dict(raw['config'], **recipe['options']),
                   best_fid=[(metrics['fid'], str(raw_fid))],
                   best_inception=[(is_meta['original_rqtransformer_metrics']['inception_score'], str(raw_is))],
                   ema_best=ema_best, ema_trial_revision=revision)
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    checkpoint_io.atomic_torch_save(payload, checkpoints / 'last.pt')
    native = (checkpoints / 'last.pt').resolve()
    for alias in (raw_fid, checkpoints / 'resume-epoch077-anchor.pt',
                  *(Path(w['path']) for w in ema_best.values())):
        alias.symlink_to(native.relative_to(checkpoints))
    disposable = inputs / '.initial-is-copy.pt'; os.link(is_source, disposable)
    checkpoint_io._persist_serialized_checkpoint(disposable, raw_is)
    restored = torch.load(checkpoint_io._checkpoint_upload_source(native), map_location='cpu', mmap=True, weights_only=False)
    compared = compare_tensors(raw, restored)
    metadata = recovery_metadata(restored)
    assert metadata['adam_step'] == 13772 and metadata['saved_learning_rates'] == [lr]
    assert restored['scheduler']['last_epoch'] == raw['scheduler']['last_epoch'] == 13146
    with source.open('rb') as stream: digest = hashlib.file_digest(stream, 'md5').hexdigest()
    record(audit / 'source-state-verification.json', original)
    record(audit / 'restart-state-verification.json', dict(metadata=metadata, revision=revision,
        model_and_adam_tensors_identical=True, rank_rng_identical=True, tensor_values_compared=compared,
        source_md5=digest, source_train_loss_tracking=raw['train_loss_tracking']))
    record(audit / 'source-checkpoint-selection.json', dict(file=str(source), epoch=77, global_step=48202,
        fid=metrics['fid'], adam_step=13772, md5=digest, bytes=source.stat().st_size, full_optimizer_state=True))
    record(audit / 'source-parent.json', dict(base=str(parent), run_id=parent_id))
    code = (parent / 'entry.py').read_text()
    assert code.count('ARGS.lr==0.00001') == 1
    code = code.replace('ARGS.lr==0.00001', 'ARGS.lr==0.0001')
    code = code.replace('CHECKPOINT_EPOCH=68', 'CHECKPOINT_EPOCH=77')
    code = code.replace("WB.log({'train/epoch':68,'train/global_step':42568,", "WB.log({'train/epoch':77,'train/global_step':48202,")
    code = code.replace("record(EVIDENCE/'continuation-checkpoint-ready.json'", "record(EVIDENCE/'trial-checkpoint-ready.json'")
    code = code.replace('VerifiedCloudUpload(\'helloimlixin-rutgers/laser/\'+WB.id,EVIDENCE/\'cloud-checkpoint-receipt.json\')',
                        'EMACheckpointCloudUpload(\'helloimlixin-rutgers/laser/\'+WB.id,EVIDENCE/\'cloud-checkpoint-receipt.json\')')
    code = code.replace('from verified_wandb_checkpoint_upload import VerifiedCloudUpload',
                        'from ema_checkpoint_cloud_upload import EMACheckpointCloudUpload')
    start, end = code.index(' config.update(automatic_fid_rewind='), code.index(' config.update(objective_revision=')
    code = code[:start] + (" config.update(automatic_fid_rewind=False,bounded_trial=True,trial_epochs=5,trial_target_epoch=82,\n"
        "  parameter_ema_decay=.999,paired_raw_and_ema_official_evaluation=True,learning_rate_factor=10.,warmup_restarted=False)\n") + code[end:]
    start, end = code.index(' config.update(resume_source_epoch='), code.index(' WB=original_init(*args,**kwargs)')
    code = code[:start] + (
        f" config.update(resume_source_epoch=77,resume_source_global_step=48202,source_run={'helloimlixin-rutgers/laser/'+parent_id!r},\n"
        "  source_checkpoint_epoch=77,source_checkpoint_step=48202,source_checkpoint_fid=15.24941539209243,\n"
        "  source_optimizer_restored=True,source_scheduler_step=13146,source_adam_step=13772,\n"
        f"  restart_initial_lr={lr!r},lr_floor_after=0.,cosine_decay_end_epoch=100,\n"
        "  optimizer_initialization='trained AdamW moments and all782 counters preserved; LR amplitude x10',\n"
        "  rng_initialization='all8 saved CPU/CUDA RNG streams restored',\n"
        "  evaluation_protocol='Official RQ-Transformer FID/10-split IS; raw and EMA; validation50k/generated50k',\n"
        "  lr_continuation='saved cosine clock and zero floor retained; amplitude x10; no new warmup',\n"
        "  checkpoint_training_weights='raw',checkpoint_ema_weights='parameter_ema.values')\n"
    ) + code[end:]
    # Distinguish raw official metrics from the paired EMA metrics.
    code = code.replace('eval/fid_original_rqtransformer', 'eval/raw/fid_original_rqtransformer')
    code = code.replace('eval/inception_score_original_rqtransformer', 'eval/raw/inception_score_original_rqtransformer')
    code = code.replace('eval/inception_score_std_original_rqtransformer', 'eval/raw/inception_score_std_original_rqtransformer')
    baseline_end = " WB.summary['evaluation/backend']='official RQ-Transformer only'"
    code = code.replace(baseline_end,
        " for prefix in ('raw','ema'):\n"
        "  for field in ('fid','inception_score','inception_score_std'):\n"
        "   WB.define_metric('eval/'+prefix+'/'+field+'_original_rqtransformer',step_metric='train/global_step')\n"
        " WB.summary['evaluation/ema_baseline_fid']=15.24941539209243\n"
        + baseline_end)
    assert code.count('from src.training.cli import main') == 1
    code = code.replace('from src.training.cli import main',
        'from imagenet_ema_hooks import install as install_ema_trial\ninstall_ema_trial(globals())\n\nfrom src.training.cli import main')
    compile(code, str(base / 'entry.py'), 'exec'); (base / 'entry.py').write_text(code)
    for name in ('entry.py', 'recipe.yaml', 'official-baseline.json', 'ema-trial.json'):
        shutil.copyfile(base / name, audit / name)
    (evidence / 'status.json').symlink_to('continuation-20261005/status.json')
    print(json.dumps(dict(prepared=True, run=RUN_PATH, source_epoch=77, source_fid=metrics['fid'],
        initial_lr=lr, adam_step=13772, scheduler_step=13146, target_epoch=82,
        tensor_values_compared=compared, ema_decay=.999)), flush=True)


def configure_recovery(base, evidence):
    recovery.RUN_ID = recovery.runner.RUN_ID = RUN_ID
    recovery.RUN_PATH = recovery.runner.RUN_PATH = RUN_PATH
    parent = json.loads((evidence / 'continuation-20261005/source-parent.json').read_text())
    recovery.PARENT, recovery.SOURCE_RUN_ID = Path(parent['base']), parent['run_id']


def drain(base, evidence, key_file):
    import torch
    import wandb
    from ema_checkpoint_cloud_upload import EMACheckpointCloudUpload
    os.environ.update(WANDB_API_KEY=key_file.read_text().strip(), CUDA_VISIBLE_DEVICES='')
    paths = sorted((base / 'final-cpu-upload').glob('*.pt'))
    assert len(paths) == 5
    ready = json.loads((evidence / 'trial-final-state-verification.json').read_text())
    EMACheckpointCloudUpload(RUN_PATH, evidence / 'final-cloud-checkpoint-receipt.json')(paths, ready['metadata']['epoch'])
    run = wandb.Api(timeout=120).run(RUN_PATH)
    run.summary.update({'checkpoints/final_cloud_verified':True,
                        'checkpoints/raw_and_ema_full_recovery_saved':True})
    print('Verified all final raw/EMA full checkpoints and metadata online', flush=True)


def finalize(base, evidence, child, key_file):
    import torch
    torch.set_num_threads(4)
    from src.training.full_resume_upload import recovery_metadata
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    ready = json.loads((evidence / 'trial-checkpoint-ready.json').read_text())
    receipt = json.loads((evidence / 'last-local-save.json').read_text())
    assert ready['no_further_updates'] and receipt['step'] == ready['global_step']
    checkpoints = evidence / 'continuation-20261005/train/checkpoints'
    native = (checkpoints / 'last.pt').resolve(strict=True)
    raw = torch.load(_checkpoint_upload_source(native), map_location='cpu', mmap=True, weights_only=False)
    metadata = recovery_metadata(raw)
    assert metadata['global_step'] == ready['global_step']
    assert metadata['parameter_ema']['updates'] == metadata['global_step'] - 48202
    assert metadata['adam_step'] == metadata['global_step'] - 34430
    slots = base / 'final-cpu-upload'; slots.mkdir(exist_ok=True)
    sources = [('last.pt', native)]
    sources += [(f'best-raw-{kind}-resume.pt', Path(ranking[0][1]))
                for kind, ranking in (('fid', raw['best_fid']), ('is', raw['best_inception']))]
    sources += [(f'best-ema-{kind}-resume.pt', Path(w['path'])) for kind, w in raw['ema_best'].items()]
    for name, source in sources:
        cached = Path(_checkpoint_upload_source(source.resolve()))
        destination = slots / name
        if not destination.exists():os.link(cached, destination)
        saved = torch.load(destination, map_location='cpu', mmap=True, weights_only=False)
        record(destination.with_suffix('.json'), recovery_metadata(saved))
    record(evidence / 'trial-final-state-verification.json', dict(metadata=metadata, ready=ready,
        all_five_full_slots_pinned=True, no_further_updates=True, verified_at=time.time()))
    status = json.loads((evidence / 'continuation-20261005/status.json').read_text())
    ranks = list((evidence / 'verification' / status['phase']).glob('process-rank*.json'))
    assert len(ranks) == 8
    pids = [json.loads(p.read_text())['pid'] for p in ranks] + [status['torchrun_pid'], child.pid]
    for pid in pids:
        path = Path(f'/proc/{pid}/cmdline')
        if path.exists():assert str(base).encode() in path.read_bytes()
    for pid in pids:
        try:os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:pass
    child.wait(timeout=30)
    command = [sys.executable, str(Path(__file__).resolve()), 'drain', '--base', str(base),
               '--evidence', str(evidence), '--key-file', str(key_file)]
    with (evidence / 'continuation-20261005/final-cpu-upload.log').open('a') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    record(evidence / 'continuation-20261005/final-cpu-upload-launch.json', dict(pid=process.pid, time=time.time()))
    os.environ['WANDB_API_KEY'] = key_file.read_text().strip()
    import wandb
    run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN_ID,
                    resume='must', mode='online', dir=str(base))
    run.summary.update({'execution/state':'completed' if metadata['epoch'] >= 82 else 'stopped_resumable',
        'execution/final_epoch':metadata['epoch'], 'execution/final_step':metadata['global_step'],
        'verification/full_best_and_last_saved':True})
    run.finish(exit_code=0)
    status.update(state='stopped_resumable', final_step=metadata['global_step'],
                  full_final_checkpoint_verified=True)
    record(evidence / 'continuation-20261005/status.json', status)
    return metadata


def supervise(base, evidence, key_file):
    lock = (base / 'ema-trial.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    audit = evidence / 'continuation-20261005'
    control = json.loads((base / 'ema-trial.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1',
               LASER_CONTINUATION_TAG=control['phase_prefix'])
    child = None
    stopping = False
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    command = [sys.executable, str(Path(__file__).resolve()), 'upload-recovery', '--base', str(base),
               '--evidence', str(evidence), '--key-file', str(key_file)]
    with (audit / 'recovery-upload.log').open('a') as stream:
        upload = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    deadline = time.monotonic() + 300
    while not (audit / 'online-run-created.json').exists():
        if stopping:return
        if upload.poll() is not None or time.monotonic() > deadline:raise RuntimeError('Online run creation failed')
        time.sleep(2)
    command[2] = 'train-once'
    with (audit / 'train-supervisor.log').open('a') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    marker = evidence / 'trial-checkpoint-ready.json'
    while True:
        if marker.exists():
            metadata = finalize(base, evidence, child, key_file)
            record(evidence / 'ema-trial-status.json', dict(state='stopped_resumable', active_run=RUN_ID,
                final_step=metadata['global_step'], full_best_and_last_pinned=True,
                target_epoch=82, official_evaluations_only=True, time=time.time()))
            return
        if child.poll() is not None:break
        record(evidence / 'ema-trial-status.json', dict(state='running', supervisor_pid=os.getpid(),
            trainer_supervisor_pid=child.pid, active_base=str(base), active_run=RUN_ID,
            source_epoch=77, target_epoch=82, trial_epochs=5, parameter_ema_decay=.999,
            initial_lr=control['initial_lr'], lr_factor=10., warmup_restarted=False,
            official_evaluations_only=True, heartbeat_unix=time.time()))
        time.sleep(5)
    record(evidence / 'ema-trial-status.json', dict(state='failed', exit_code=child.returncode, time=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'upload-recovery', 'train-once', 'supervise', 'drain'])
    for name in ('base', 'evidence', 'key-file'):p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--parent-base', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':prepare(args.base, args.evidence, args.parent_base); return
    sys.path[:0] = [str(args.base / 'source/runtime'), str(args.base / 'support')]
    os.environ.update(LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base / 'checkpoint-upload-cache'),
                      LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    configure_recovery(args.base, args.evidence)
    if args.action == 'upload-recovery':recovery.upload_recovery(args.base, args.evidence, args.key_file)
    elif args.action == 'train-once':
        original_record = recovery.runner.record
        def progress(path, value):
            if Path(path).name == 'status.json':value['target_epoch'] = 82
            original_record(path, value)
        recovery.runner.record = progress
        recovery.runner.supervise(args.base, args.evidence, args.key_file)
    elif args.action == 'drain':drain(args.base, args.evidence, args.key_file)
    else:supervise(args.base, args.evidence, args.key_file)


if __name__ == '__main__':main()
