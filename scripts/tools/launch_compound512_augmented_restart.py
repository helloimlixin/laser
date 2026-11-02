#!/usr/bin/env python3
"""Prepare, validate, screen LR, and launch the fresh augmented compound run."""
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
import zipfile

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-compound512-augwarm-imagenet-20261016')
RUN = 'imagenet-laser-compound512-augwarm-scratch100-4h100-20261016'
OUT = ROOT/'outputs'/RUN
SOURCE = Path('/tmp/laser-compound512-native-imagenet-20261016')
RECIPE = ROOT/'configs/stage2/imagenet-laser-compound512-augmented-scratch100-20261016.yaml'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def archive():
    reproduction=OUT/'reproducibility'
    for p in (BASE/'inputs').iterdir():
        if p.suffix in ('.yaml','.json'):
            shutil.copyfile(p,reproduction/p.name)
    files=[BASE/'train.py',BASE/'verify.py',BASE/'screen_lr.py',BASE/'environment.json']
    if (BASE/'configure_samples.py').exists():
        files.append(BASE/'configure_samples.py')
    for name in ('source','runtime','upstream'):
        files.extend((BASE/name).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml'))
    files.extend((BASE/'inputs').glob('*.json'))
    write(reproduction/'code-manifest.json',{str(p.relative_to(BASE)):digest(p) for p in files})
    with zipfile.ZipFile(reproduction/'source.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for p in files:
            bundle.write(p,p.relative_to(BASE))
        bundle.write(Path(__file__),'controller.py')


def prepare():
    import yaml
    sys.path[:0]=[str(ROOT),str(ROOT/'runtime')]
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    assert not (OUT/'prepared.json').exists(), 'Inspect the existing preparation instead of replacing it'
    recipe=yaml.safe_load(RECIPE.read_text())
    assert recipe['run_id']==RUN and recipe['epochs']==100
    assert recipe['tokenizer']['coefficient_bins']==512
    assert recipe['atom_ce_weight']==1. and recipe['targets']['temperature']==.01125
    (BASE/'inputs').mkdir(parents=True,exist_ok=True)
    (OUT/'reproducibility').mkdir(parents=True,exist_ok=True)
    freeze_shared_runtime(BASE,ROOT,include_stochastic_teacher=False)
    (BASE/'upstream').symlink_to(SOURCE/'upstream',target_is_directory=True)
    for name in ('data','torch-cache'):
        (BASE/name).symlink_to(SOURCE/name,target_is_directory=True)
    for name in ('stage2-training-config.yaml','stage2-archive-config.yaml','imagenet_256_train.npz','rq-upstream-core.json'):
        shutil.copyfile(SOURCE/'inputs'/name,BASE/'inputs'/name)
    os.link(SOURCE/'inputs/stage1-model.pt',BASE/'inputs/stage1-model.pt')
    assert digest(BASE/'inputs/stage1-model.pt')==recipe['tokenizer']['sha256']
    assert digest(BASE/'inputs/imagenet_256_train.npz')==recipe['evaluation']['reference_sha256']
    config=yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    published=recipe['optimization'].get('selection')=='published_original_rq'
    if published:
        original=Path('/tmp/laser-original-rq-imagenet-20261014/inputs/stage2-training-config.yaml')
        reference=yaml.safe_load(original.read_text())
        shutil.copyfile(original,BASE/'inputs/original-rq-training-config.yaml')
        config['optimizer']=reference['optimizer']
        assert config['optimizer']['init_lr']==.0005
        assert config['optimizer']['warmup']['epoch']==0 and config['optimizer']['warmup']['min_lr']==0
    else:
        config['optimizer']['init_lr']=.0003
        config['optimizer']['warmup'].update(epoch=recipe['optimization'].get('warmup_epochs',2),
            min_lr=.0003*recipe['optimization'].get('minimum_lr_ratio',.01),multiplier=1,
            buffer_epoch=0,mode='fix',start_from_zero=True)
    config['experiment'].update(epochs=100,save_ckpt_freq=1,test_freq=2,total_batch_size=2048)
    config['loss']=dict(type=recipe['objective'],atom_weight=1.,
        coefficient_temperature=.01125,coefficient_bins=512,auxiliary_loss_terms=0)
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    shutil.copyfile(RECIPE,BASE/'inputs/recipe.yaml')
    (BASE/'wandb').mkdir(exist_ok=True)
    environment=dict(LASER_RQ_BASE=str(BASE),LASER_RQ_SAFE_CHECKPOINTS='1',
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1',TORCH_HOME=str(BASE/'torch-cache'),
        WANDB_DIR=str(BASE/'wandb'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NCCL_NVLS_ENABLE='0',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    write(BASE/'environment.json',environment)
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__=='__main__':\n    launch('laser_compound_replay')\n")
    shutil.copyfile(ROOT/'scripts/tools/screen_compound512_restart_lr.py',BASE/'screen_lr.py')
    if recipe.get('preview',{}).get('designated_classes'):
        shutil.copyfile(ROOT/'scripts/tools/configure_rq_sample_panels.py',BASE/'configure_samples.py')
    verification=(ROOT/'scripts/tools/verify_laser_compound512.py').read_text()
    verification=verification.replace('normalized_noise_sigma=.125,sigma_bins=.125/(6/511)',
        'normalized_noise_sigma=.075,sigma_bins=.075/(6/511)')
    warmup=config['optimizer']['warmup']['epoch']
    if warmup:
        verification=verification.replace('assert scheduler.after_scheduler.last_epoch == 2 and scheduler.after_scheduler.T_max == 62600',
            'assert scheduler.warmup_scheduler.last_epoch == 2 and scheduler.after_scheduler.last_epoch == 0\n'
            f'    assert scheduler.after_scheduler.T_max == {62600-int(warmup*626)}')
    verification=verification.replace("    write_json(out/'smoke-reload-verification.json',dict", """    import hashlib
    receipts=[json.loads(p.read_text()) for p in (out/'smoke/checkpoints/.checkpoint-data').glob('*.json')]
    receipt=max(receipts,key=lambda row:row['global_step'])
    with Path(receipt['persistent']).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['sha256']
    write_json(out/'smoke-persistence-verification.json',dict(passed=True,
        persistent_sha256_verified=True,bytes=receipt['bytes'],sha256=receipt['sha256']))
    write_json(out/'smoke-reload-verification.json',dict""")
    (BASE/'verify.py').write_text(verification)
    archive()
    write(OUT/'prepared.json',dict(prepared=True,run_id=RUN,fresh_stage2=True,
        fresh_optimizer=True,source_stage2_checkpoint=None,target_epoch=100,
        target_updates=62600,updates_per_epoch=626,global_batch=2048,
        coefficient_bins=512,atoms=16384,compound_integer_vocabulary=16384*512,
        classifier='factorized exact p(atom) p(coefficient|atom), mean joint NLL',
        auxiliary_loss_terms=0,coefficient_sigma=.075,augmentation_from_step_zero=True,
        warmup_epochs=warmup,lr_screen_pending=not published,tokenizer_gate_pending=True,time=time.time()))
    print(json.dumps(dict(prepared=True,run_id=RUN)),flush=True)


def run():
    import yaml
    lock=(BASE/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT/'cpu-verification.json').read_text())['passed']
    cancelled=False
    child=None
    panel_process=None
    def status(phase,**values):
        row=dict(phase=phase,pid=os.getpid(),run_id=RUN,
            target_epoch=100,target_updates=62600,time=time.time())
        row.update(values)
        write(OUT/'launch-status.json',row)
    def stop(*_):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
        if panel_process is not None and panel_process.poll() is None:
            panel_process.terminate()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    status('waiting_for_previous_run_to_release_gpus')
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        if cancelled:
            status('cancelled');return
        time.sleep(5)
    sizes=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.total',
        '--format=csv,noheader,nounits'],text=True).splitlines()]
    world=len(sizes)
    micro=min(64,64 if min(sizes)>=75000 else 32 if min(sizes)>=45000 else 16)
    assert world==4 and micro==64, 'Recompute the LR screen batch plan if hardware changes'
    environment=dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    environment['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    def launch(command,phase):
        nonlocal child
        with (OUT/f'{phase}.log').open('a') as stream:
            child=subprocess.Popen(command,cwd=BASE,env=environment,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            status(phase,child_pid=child.pid,world_size=world,microbatch=micro,
                accumulation_steps=math.ceil(2048/(world*micro)))
            result=child.wait()
        if cancelled:
            status('cancelled',returncode=result);raise SystemExit(130)
        if result:
            status('failed',failed_phase=phase,returncode=result)
            raise RuntimeError(f'{phase} failed; inspect its log')
    distributed=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}']
    def passed(path):
        return path.exists() and json.loads(path.read_text()).get('passed',False)
    if not passed(OUT/'tokenizer-verification.json'):
        launch([*distributed,str(BASE/'verify.py'),'--reconstruction'],'tokenizer_reconstruction')
    assert json.loads((OUT/'tokenizer-verification.json').read_text())['passed']
    recipe=yaml.safe_load((BASE/'inputs/recipe.yaml').read_text())
    published=recipe['optimization'].get('selection')=='published_original_rq'
    if published:
        original=yaml.safe_load((BASE/'inputs/original-rq-training-config.yaml').read_text())
        selection=dict(passed=True,production_updates_added=0,selection='published_original_rq',
            selected_peak_lr=float(original['optimizer']['init_lr']),
            initial_weight_sha256=recipe['optimization']['seeded_initial_weight_sha256'],
            lr_screen_used=False,warmup_epochs=0,minimum_lr=0.)
        write(OUT/'optimization-selection.json',selection)
    else:
        if not passed(OUT/'lr-screen/selection.json'):
            launch([*distributed,str(BASE/'screen_lr.py')],'lr_screen')
        selection=json.loads((OUT/'lr-screen/selection.json').read_text())
    assert selection['passed'] and selection['production_updates_added']==0
    peak=selection['selected_peak_lr']
    config=yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['optimizer']['init_lr']=peak
    minimum=peak*float(recipe['optimization'].get('minimum_lr_ratio',.01))
    config['optimizer']['warmup']['min_lr']=minimum
    if published:
        assert config['optimizer']==original['optimizer'], 'Optimizer must match the original RQ recipe exactly'
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    recipe['optimization'].update(selected_peak_lr=peak,minimum_lr=minimum,
        probe_initial_weight_sha256=selection['initial_weight_sha256'])
    (BASE/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    archive()
    if not passed(OUT/'smoke/verification.json'):
        launch([*distributed,str(BASE/'train.py'),'--microbatch',str(micro),'--smoke-updates','2'],'gpu_preflight')
    if not passed(OUT/'smoke-reload-verification.json'):
        launch([sys.executable,str(BASE/'verify.py')],'checkpoint_verification')
    proof=json.loads((OUT/'smoke-reload-verification.json').read_text())
    assert proof['passed'] and proof['production_starts_from_zero']
    ready=json.loads((OUT/'prepared.json').read_text())
    ready.update(lr_screen_pending=False,tokenizer_gate_pending=False,selected_peak_lr=peak,
        minimum_lr=minimum,world_size=world,microbatch=micro,
        accumulation_steps=math.ceil(2048/(world*micro)),production_starts_from_zero=True)
    write(OUT/'prepared.json',ready)
    for name,expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
        assert digest(BASE/name)==expected,name
    if recipe.get('preview',{}).get('designated_classes'):
        with (OUT/'sample-workspace.log').open('a') as stream:
            panel_process=subprocess.Popen([sys.executable,str(BASE/'configure_samples.py'),
                '--run',RUN,'--watch'],cwd=BASE,env=environment,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'sample-workspace-process.json',dict(pid=panel_process.pid,time=time.time()))
    launch([*distributed,str(BASE/'train.py'),'--microbatch',str(micro),'--epochs','100'],'train')
    complete=json.loads((OUT/'completed.json').read_text())
    assert complete['global_step']==62600 and complete['epochs']==100
    status('completed',**complete)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--recipe',type=Path,default=RECIPE)
    args=parser.parse_args()
    BASE=args.base.resolve()
    RECIPE=args.recipe.resolve()
    import yaml
    RUN=yaml.safe_load(RECIPE.read_text())['run_id']
    OUT=ROOT/'outputs'/RUN
    prepare() if args.action=='prepare' else run()
