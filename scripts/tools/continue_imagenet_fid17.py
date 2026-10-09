"""Restart the FID17 epoch55 winner with verified fresh Adam and a warmup/cosine LR."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import continue_imagenet_pairfix_repair as runner

SOURCE_RUN_ID = 'imagenet-rfid421-fid1745-cosine-8h100-20261004'
RUN_ID = 'imagenet-rfid421-fid17459-epoch55-warmcosine-8h100-20261005'
RUN_PATH = 'helloimlixin-rutgers/laser/' + RUN_ID
runner.RUN_ID, runner.RUN_PATH = RUN_ID, RUN_PATH
record, replace_once = runner.record, runner.replace_once


def verify_progress(metadata):
    assert metadata['epoch'] == 55 and metadata['next_microbatch'] == 0
    assert metadata['global_step'] == 34430 and metadata['adam_step'] == 0
    assert metadata['adam_parameters'] == 782 and metadata['rng_ranks'] == 8
    assert metadata['world_size'] == 8 and metadata['gradient_accumulation_steps'] == 3
    state = metadata['learning_rate_schedule']['state']
    assert state['last_epoch'] == 0 and state['total_steps'] == 28170
    assert state['schedule_version'] == 'laser-epoch55-warmup-cosine-v1'
    assert state['peak_lr'] == .0003 and state['min_lr'] == 3e-7 and state['warmup_steps'] == 626
    expected = .0003 * .1
    assert math.isclose(metadata['saved_learning_rates'][0], expected, rel_tol=1e-10)
    return expected


def prepare(base, evidence):
    import torch
    import yaml
    sys.path.insert(0, str(base / 'source/runtime'))
    from src.training.full_resume_upload import recovery_metadata
    audit = evidence / 'continuation-20261005'
    receipt = json.loads((audit / 'source-checkpoint-selection.json').read_text())
    source = Path(receipt['file'])
    assert receipt['phase'] == 'verified' and source.stat().st_size == receipt['bytes']
    payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    metadata = recovery_metadata(payload)
    assert metadata['epoch'] == 55 and metadata['global_step'] == 34430
    assert metadata['adam_step'] == 0 and metadata['rng_ranks'] == 8
    assert metadata['next_microbatch'] == 0 and metadata['adam_parameters'] == 782
    assert math.isclose(metadata['fid'], 17.45907974243164, abs_tol=1e-7)
    torch.set_num_threads(4)
    for state in payload['optimizer']['state'].values():
        assert not bool(state['exp_avg'].count_nonzero())
        assert not bool(state['exp_avg_sq'].count_nonzero())
    record(audit / 'source-state-verification.json', metadata)
    checkpoints = audit / 'train/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / 'last.pt'
    entry = (base / 'source/entry.py').read_text()
    entry = replace_once(entry, "kwargs['resume']='allow'", "kwargs['resume']='must'")
    entry = replace_once(entry, 'def stop(_sig,_frame):',
        "record(VERIFY/('process-rank'+os.environ['RANK']+'.json'),\n"
        " dict(pid=os.getpid(),rank=int(os.environ['RANK']),time=time.time()))\n\n"
        'def stop(_sig,_frame):')
    entry = replace_once(entry, "ARGS.lr==0.00045", "ARGS.lr==0.0003")
    entry = replace_once(entry,
        "  assert not ARGS.model_only_best_checkpoints and ARGS.lr_schedule_epochs==45",
        "  assert not ARGS.model_only_best_checkpoints and ARGS.lr_schedule_epochs==45\n"
        "  assert ARGS.metric_backend=='original-rqvae' and ARGS.token_cache is None")
    entry = replace_once(entry, "  saved=payload.get('config',{})\n",
        "  saved=payload.get('config',{})\n"
        "  if saved.get('metric_backend')!='original-rqvae':\n"
        "   payload['best_fid']=[];payload['best_inception']=[]\n")
    entry = replace_once(entry, " step=int(payload['global_step']);epoch=int(payload['epoch'])",
        " step=int(payload['global_step']);epoch=int(payload['epoch'])\n"
        " if OFFICIAL_METRICS is not None and OFFICIAL_METRICS['global_step']==step:\n"
        "  payload=dict(payload,original_rqtransformer_metrics=OFFICIAL_METRICS)")
    entry = replace_once(entry,
        " if not WB.summary.get('restart/baseline_logged',False):\n"
        "  WB.log({'train/epoch':55,'train/global_step':34430,'val/fid':17.45907974243164,'val/inception_score':62.45886993408203})\n"
        "  WB.summary['restart/baseline_logged']=True\n",
        " baseline=BASE/'official-baseline.json'\n"
        " if baseline.exists():\n"
        "  metrics=json.loads(baseline.read_text())\n"
        "  WB.log({'train/epoch':55,'train/global_step':34430,\n"
        "   'val/fid':metrics['fid'],'val/inception_score':metrics['inception_score'],\n"
        "   'val/inception_score_std':metrics['inception_score_std'],\n"
        "   'eval/fid_original_rqtransformer':metrics['fid'],\n"
        "   'eval/inception_score_original_rqtransformer':metrics['inception_score'],\n"
        "   'eval/inception_score_std_original_rqtransformer':metrics['inception_score_std']})\n"
        " WB.summary['evaluation/backend']='official RQ-Transformer only'\n"
        " WB.summary['evaluation/reference']='ImageNet validation50k, generated50k'\n")
    entry = replace_once(entry, " WB=original_init(*args,**kwargs)",
        " config.update(resume_source_epoch=55,resume_source_global_step=34430,\n"
        "  source_run='helloimlixin-rutgers/laser/imagenet-rfid421-fid1745-cosine-8h100-20261004',\n"
        "  optimizer_initialization='verified zero AdamW moments and counters at epoch55',\n"
        "  source_optimizer_restored=True,trained_source_optimizer_available=False,\n"
        "  rng_initialization='restored all 8 epoch55 checkpoint RNG streams',\n"
        "  evaluation_protocol='Official RQ-Transformer Inception/FID/10-split IS only; validation50k/generated50k',\n"
        "  evaluation_backend_changed_at_step=34430,previous_metric_backend='torchmetrics',\n"
        "  learning_rate_schedule='epoch55-to56 linear warmup 3e-5 to3e-4; cosine to3e-7 at epoch100',\n"
        "  lr_continuation='new conservative warmup/cosine for verified fresh Adam at epoch55')\n"
        " WB=original_init(*args,**kwargs)")
    entry = replace_once(entry, " WB.summary['execution/state']='training'",
        " submit_upload()\n WB.summary['execution/state']='training'")
    entry = replace_once(entry, " result=original_step(optimizer,*args,**kwargs);UPDATES+=1",
        " result=original_step(optimizer,*args,**kwargs);UPDATES+=1\n"
        " if UPDATES==20:ARGS.save_step_freq=1\n"
        " elif UPDATES==21:ARGS.save_step_freq=250")
    injected = '''
from official_metrics import install as install_official_metrics
install_official_metrics(ROOT)
from imagenet_fid17_lr import create_scheduler
training.create_cosine_lr_scheduler=create_scheduler
OFFICIAL_METRICS=None
original_evaluate=training.evaluate_generation_metrics
def evaluate(*args,**kwargs):
 global OFFICIAL_METRICS
 assert kwargs['metric_backend']=='original-rqvae'
 device=next(args[0].parameters()).device
 with torch.random.fork_rng(devices=[device.index]):
  torch.random.default_generator.manual_seed(261001+dist.get_rank())
  torch.cuda.manual_seed(261001+dist.get_rank())
  result=original_evaluate(*args,**kwargs)
 # This run restarted Adam at epoch55: global step and Adam step differ by 34430.
 step=INITIAL_RECOVERY['global_step']+UPDATES
 OFFICIAL_METRICS=dict(global_step=step,fid=result[0],inception_score=result[1],
  inception_score_std=result[2],metric_backend='original_rqtransformer',real_images=50000,
  generated_images=50000,real_split='val',inception_splits=10,seed=261001)
 if dist.get_rank()==0:
  record(EVIDENCE/'continuation-20261005'/f'official-metrics-step{step}.json',OFFICIAL_METRICS)
  if WB is not None:
   WB.log({'eval/fid_original_rqtransformer':result[0],
    'eval/inception_score_original_rqtransformer':result[1],
    'eval/inception_score_std_original_rqtransformer':result[2],
    'train/global_step':step})
 return result
training.evaluate_generation_metrics=evaluate
'''
    entry = replace_once(entry, 'from src.training.cli import main\n', injected + '\nfrom src.training.cli import main\n')
    (base / 'entry.py').write_text(entry)
    shutil.copyfile('/tmp/laser-imagenet-pairfix-repair-20261005-resume/support/official_metrics.py',
                    base / 'support/official_metrics.py')
    shutil.copyfile(Path(__file__).with_name('imagenet_fid17_lr.py'), base / 'support/imagenet_fid17_lr.py')
    recipe = yaml.safe_load((base / 'inputs/resume-active-config.yaml').read_text())
    recipe['options'].update(checkpoint=str(base / 'inputs/resume-stage1-tokenizer.pt'),
        token_cache=None, data='/tmp/laser-imagenet-stage2/imagenet',
        output=str(base / 'production/train'), checkpoint_dir=str(checkpoints), resume=True,
        wandb_mode='online', metric_backend='original-rqvae', fid_reference_stats=None,
        batch_size=86, fid_every=2, save_step_freq=250, save_ckpt_freq=1,
        lr=.0003, warmup_epochs=1.0, warmup_start_ratio=.1, lr_schedule_restart_id=RUN_ID,
        wandb_id=RUN_ID, wandb_name='ImageNet rFID4.21 K4 | epoch55 FID17.459 restart | Adam warmup/cosine | 8 H100')
    recipe['options'].pop('resume_checkpoint', None)
    from imagenet_fid17_lr import create_scheduler
    fake = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=.0003)
    schedule = create_scheduler(fake, initial_lr=.0003, min_lr=3e-7, total_steps=28170)
    optimizer = dict(payload['optimizer'], param_groups=[dict(group, lr=.0003*.1, initial_lr=.0003)
                                                        for group in payload['optimizer']['param_groups']])
    bootstrap = dict(payload, optimizer=optimizer, scheduler=schedule.state_dict(),
        config=dict(payload['config'], **recipe['options'], accumulation_steps=3, world_size=8),
        best_fid=[], best_inception=[],
        fid=None, inception_score=None)
    expected = verify_progress(recovery_metadata(bootstrap))
    prepared = base / 'inputs/tuned-epoch055.pt'
    torch.save(bootstrap, prepared)
    assert not latest.exists()
    latest.symlink_to(prepared)
    record(audit / 'optimizer-bootstrap-verification.json', dict(
        source_epoch=55, source_global_step=34430, adam_step=0, adam_moments_all_zero=True,
        weights_and_adam_tensors_unchanged=True, rank_rng_unchanged=True,
        initial_lr=expected, peak_lr=.0003, warmup_steps=626, total_schedule_steps=28170,
        betas=optimizer['param_groups'][0]['betas'], weight_decay=optimizer['param_groups'][0]['weight_decay']))
    (base / 'recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    baseline = json.loads(json.dumps(recipe))
    baseline['options'].update(fid_only=True, wandb_mode='disabled', upload_checkpoints=False,
                              output=str(base / 'baseline-evaluation/train'))
    (base / 'baseline.yaml').write_text(yaml.safe_dump(baseline, sort_keys=False))
    weights = base / 'torch-cache/hub/checkpoints'
    weights.mkdir(parents=True, exist_ok=True)
    for name in ('weights-inception-2015-12-05-6726825d.pth',
                 'pt_inception-2015-12-05-6726825d.pth'):
        shutil.copyfile(base / 'inputs/resume-weights-inception-2015-12-05-6726825d.pth',
                        weights / name)
    for file in ('entry.py', 'recipe.yaml', 'baseline.yaml'):
        shutil.copyfile(base / file, audit / file)
    shutil.copyfile(base / 'support/official_metrics.py', audit / 'official_metrics.py')
    record(audit / 'lr-continuation.json', dict(initial_resume_lr=expected,
        adam_step=0, global_step=34430, scheduler_step=0, scheduler_total_steps=28170,
        trained_optimizer_available=False, schedule='one-epoch warmup then44-epoch cosine', remaining_epochs=45,
        epoch_lrs={epoch:schedule.lr_at((epoch-55)*626) for epoch in (55,56,60,65,70,80,90,100)}))
    print(json.dumps(dict(resume_epoch=55, resume_step=34430, adam_step=0,
                          lr=expected, metric_backend='official RQ-Transformer only')), flush=True)


def seed_baseline(base, evidence):
    import torch
    import yaml
    torch.set_num_threads(4)
    sys.path.insert(0, str(base / 'source/runtime'))
    from src.training.full_resume_upload import recovery_metadata
    from src.training import k4_checkpoint_io as checkpoint_io
    audit = evidence / 'continuation-20261005'
    metrics = json.loads((audit / 'official-metrics-step34430.json').read_text())
    assert metrics['metric_backend'] == 'original_rqtransformer' and metrics['generated_images'] == 50000
    assert math.isfinite(metrics['fid']) and metrics['fid'] < 22, 'Restored model quality needs investigation'
    raw = torch.load(base / 'inputs/tuned-epoch055.pt', map_location='cpu', weights_only=False, mmap=True)
    verify_progress(recovery_metadata(raw))
    options = yaml.safe_load((base / 'recipe.yaml').read_text())['options']
    checkpoints = Path(options['checkpoint_dir'])
    fid_path = checkpoints / f"best_fid_{metrics['fid']:.4f}_epoch_055.pt"
    is_path = checkpoints / f"best_is_{metrics['inception_score']:.4f}_epoch_055.pt"
    payload = dict(raw, config=dict(raw['config'], **options),
        best_fid=[(metrics['fid'], str(fid_path))],
        best_inception=[(metrics['inception_score'], str(is_path))],
        fid=metrics['fid'], inception_score=metrics['inception_score'],
        inception_score_std=metrics['inception_score_std'], original_rqtransformer_metrics=metrics)
    os.environ.update(LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    checkpoint_io.atomic_torch_save(payload, checkpoints / 'last.pt')
    source = (checkpoints / 'last.pt').resolve()
    for alias in (fid_path, is_path, checkpoints / 'resume-anchor.pt'):
        assert not alias.exists()
        alias.symlink_to(source.relative_to(checkpoints))
    reloaded = torch.load(checkpoint_io._checkpoint_upload_source(checkpoints / 'last.pt'),
                          map_location='cpu', weights_only=False, mmap=True)
    verify_progress(recovery_metadata(reloaded))
    assert raw['scheduler'] == reloaded['scheduler']
    assert raw['optimizer']['param_groups'] == reloaded['optimizer']['param_groups']
    compared = 0
    for key, tensor in raw['state_dict'].items():
        assert torch.equal(tensor, reloaded['state_dict'][key]), key
        compared += tensor.numel()
    for key, state in raw['optimizer']['state'].items():
        for name, tensor in state.items():
            if torch.is_tensor(tensor):
                assert torch.equal(tensor, reloaded['optimizer']['state'][key][name]), (key, name)
                compared += tensor.numel()
    for rank, streams in enumerate(raw['rng_state_by_rank']):
        for key, state in streams.items():
            assert torch.equal(state, reloaded['rng_state_by_rank'][rank][key])
    shutil.copyfile(audit / 'official-metrics-step34430.json', base / 'official-baseline.json')
    record(audit / 'official-anchor-verification.json', dict(
        metadata=recovery_metadata(reloaded), tensor_values_compared=compared,
        model_and_adam_tensors_identical=True, scheduler_identical=True, rank_rng_identical=True,
        checkpoint=str(source), evaluation_only_no_optimizer_updates=True))
    print(json.dumps(dict(official_fid=metrics['fid'], official_is=metrics['inception_score'],
                          tensor_values_compared=compared, restored_state_unchanged=True)), flush=True)


def baseline(base, evidence, key_file):
    env = dict(os.environ, LASER_RUN_BASE=str(base), LASER_PERSISTENT_BASE=str(evidence),
        LASER_PHASE='official_baseline_20261005', LASER_PREFLIGHT='1', LASER_ACCUMULATION='3',
        LASER_COMPILE_BLOCKS='0', LASER_PREVIEW_KEEP_OPTIMIZER='1',
        TORCH_HOME=str(base / 'torch-cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        NCCL_NVLS_ENABLE='0', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
               str(base / 'entry.py'), '--config', str(base / 'baseline.yaml')]
    log = evidence / 'continuation-20261005/official-baseline.log'
    with log.open('a') as stream:
        child = subprocess.Popen(command, env=env, cwd=base, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        record(evidence / 'continuation-20261005/baseline-launch.json',
               dict(pid=child.pid, command=command, log=str(log)))
        code = child.wait()
    if code:
        raise RuntimeError('Official baseline evaluation failed; inspect ' + str(log))
    seed_baseline(base, evidence)


def upload_recovery(base, evidence, key_file):
    import base64
    import importlib.metadata
    import platform
    import tarfile
    import wandb
    ready = json.loads(Path('/tmp/laser-imagenet-stage2/imagenet/training-ready.json').read_text())
    assert ready['training_images'] == 1281167 and ready['md5'] == '1d675b47d978889d74fa0da5fadfb00e'
    os.environ['WANDB_API_KEY'] = key_file.read_text().strip()
    directory = base / 'recovery-upload'
    directory.mkdir(exist_ok=True)
    shutil.copyfile(base / 'recipe.yaml', directory / 'resume-active-config.yaml')
    with tarfile.open(directory / 'resume-frozen-code.tar.gz', 'w:gz', compresslevel=4) as archive:
        for name in ('entry.py', 'support', 'official-baseline.json'):
            archive.add(base / name, arcname=name,
                        filter=lambda item:None if '__pycache__' in item.name else item)
        archive.add(base / 'source/runtime', arcname='runtime',
                    filter=lambda item:None if '__pycache__' in item.name else item)
    record(directory / 'resume-dataset-identity.json', dict(
        training=ready, validation=json.loads(Path('/tmp/laser-imagenet-stage2/imagenet/validation-ready.json').read_text()),
        training_views='Resize(short edge256), RandomCrop256, RandomHorizontalFlip0.5; fresh each epoch',
        online_encoder_precision='BF16; OMP FP32', token_cache_used=False,
        persistent_archive_expected='/workspace/Projects/data/imagenet/archives/ILSVRC2012_img_train.verified-20261005.tar'))
    record(directory / 'resume-environment.json', dict(python=platform.python_version(),
        packages={item.metadata['Name']:item.version for item in importlib.metadata.distributions()
                  if item.metadata.get('Name')}, required_scipy='>=1.16,<1.18'))
    for name in ('resume-stage1-tokenizer.pt','resume-weights-inception-2015-12-05-6726825d.pth'):
        destination = directory / name
        if not destination.exists():
            os.link(base / 'inputs' / name, destination)
    dependencies = []
    for path in sorted(directory.iterdir()):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        dependencies.append(dict(file=path.name, sha256=digest, bytes=path.stat().st_size))
    bundle = dict(schema='laser-full-recovery-v1', run=RUN_ID, dependencies=dependencies,
        slots=dict(latest='last.pt',fid='best-fid-resume.pt',is_score='best-is-resume.pt'),
        source_run='helloimlixin-rutgers/laser/'+SOURCE_RUN_ID, source_checkpoint_epoch=55,
        source_checkpoint_step=34430, source_trained_optimizer_available=False,
        optimizer_initialization='verified zero AdamW moments and counters',
        learning_rate_policy='626-update warmup3e-5 to3e-4, then cosine to3e-7 at epoch100',
        active_accumulation_steps=3, original_rqtransformer_evaluations_only=True)
    record(directory / 'resume-bundle.json', bundle)
    record(directory / 'resume-instructions.json', dict(run=RUN_PATH,
        restore_command='python scripts/tools/restore_imagenet_fid1745_cosine.py --run '+RUN_PATH+
            ' --destination /tmp/laser-recovered --checkpoint-dir /workspace/recovered-checkpoints'+
            ' --data /path/to/verified/imagenet --key-file /path/to/private-wandb-key --launch',
        optimizer_steps_preserved=True, warmup_cosine_state_preserved=True, rng_ranks=8,
        global_batch=2048, physical_microbatch=86, accumulation=3,
        dataset_requires_verified_original_training_images=True, metrics_backend='official RQ-Transformer only'))
    shutil.copyfile(base / 'official-baseline.json', directory / 'official-baseline.json')
    run = wandb.Api(timeout=120).run(RUN_PATH)
    receipts = []
    for path in sorted(directory.iterdir(),key=lambda item:(item.name=='resume-bundle.json', item.name)):
        run.upload_file(str(path),root=str(directory))
        with path.open('rb') as stream:
            digest = base64.b64encode(hashlib.file_digest(stream,'md5').digest()).decode()
        remote = run.file(path.name)
        assert remote.size == path.stat().st_size and remote.md5 == digest, path.name
        receipts.append(dict(file=path.name, bytes=remote.size, md5=remote.md5))
    audit = evidence / 'continuation-20261005'
    for path in directory.iterdir():
        if path.stat().st_size < 10_000_000:
            shutil.copyfile(path, audit / path.name)
    record(audit / 'cloud-recovery-bundle-receipt.json', dict(
        run=RUN_PATH, complete=True, files=receipts, verified_unix=time.time()))
    print(json.dumps(dict(run=RUN_PATH,recovery_dependencies_verified_online=True)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--action', choices=['prepare','baseline','seed','upload-recovery','supervise'], default='prepare')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.base, args.evidence)
    elif args.action == 'baseline':
        baseline(args.base, args.evidence, args.key_file)
    elif args.action == 'seed':
        seed_baseline(args.base, args.evidence)
    elif args.action == 'upload-recovery':
        upload_recovery(args.base, args.evidence, args.key_file)
    else:
        assert 'LASER_PREFLIGHT' not in os.environ
        ready = json.loads(Path('/tmp/laser-imagenet-stage2/imagenet/training-ready.json').read_text())
        assert ready['training_images'] == 1281167 and ready['md5'] == '1d675b47d978889d74fa0da5fadfb00e'
        assert (args.base / 'official-baseline.json').exists()
        runner.supervise(args.base, args.evidence, args.key_file)


if __name__ == '__main__':
    main()
