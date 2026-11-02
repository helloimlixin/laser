#!/usr/bin/env python3
"""Select, validate and launch a fresh compact joint-integer ImageNet pilot."""
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

ROOT=Path('/workspace/Projects/laser')
RUN='imagenet-laser-joint20k-scratch2-4h100-20261017'
OUT=ROOT/'outputs'/RUN
BASE=Path('/tmp/laser-rq-joint20k-imagenet-20261017')
SOURCE=Path('/tmp/laser-rq-native-joint32k-imagenet-20261016')
STUDY=ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017'
RECIPE=ROOT/'configs/stage2/imagenet-laser-joint20k-scratch2-20261017.yaml'
OLD_RUN='imagenet-laser-pairid256k-grid512-scratch100-4h100-20261017'
OLD=ROOT/'outputs'/OLD_RUN
OLD_BASE=Path('/tmp/laser-pairid256k-grid512-imagenet-20261017')
sys.path.insert(0,str(ROOT))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name,**values):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(dict(values,time=time.time()),indent=2)+'\n');tmp.replace(path)


def matches(pid,token):
    try:return token.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False


def preserve_and_stop():
    if (OUT/'preserved-source-checkpoint.json').exists():
        assert json.loads((OUT/'preserved-source-checkpoint.json').read_text())['passed']
        return
    os.environ['RQ_UPSTREAM_ROOT']=str(OLD_BASE/'upstream')
    sys.path.insert(0,str(OLD_BASE/'upstream'))
    from src import rq_upstream
    import torch
    receipts=[json.loads(p.read_text()) for p in (OLD/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    receipt=max(receipts,key=lambda r:r['global_step'])
    write('launch-status.json',phase='verifying_current_checkpoint',pid=os.getpid(),preserved_step=receipt['global_step'])
    for key in ['local','persistent']:
        path=Path(receipt[key]);assert path.stat().st_size==receipt['bytes']
        assert sha(path)==receipt['sha256'],key
    saved=torch.load(receipt['local'],map_location='cpu',mmap=True,weights_only=False)
    step=receipt['global_step']
    assert saved['run_id']==OLD_RUN and saved['global_step']==step
    assert saved['next_microbatch']==0 and len(saved['rng_state_by_rank'])==4
    assert all(int(s['step'])==step for s in saved['optimizer']['state'].values())
    del saved
    progress=[json.loads((OLD/'train'/f'progress-rank{r}.json').read_text()) for r in range(4)]
    pids=[r['pid'] for r in progress]
    status=json.loads((OLD/'launch-status.json').read_text())
    controller=status['pid']
    assert matches(controller,'launch_laser_catalog_integer.py')
    assert all(matches(pid,str(OLD_BASE/'train.py')) for pid in pids)
    assert not list((OLD_BASE/'checkpoint-staging').glob('*.staging'))
    observed=max(r['train/global_step'] for r in progress)
    write('preserved-source-checkpoint.json',passed=True,source_run=OLD_RUN,receipt=receipt,
        local_and_persistent_sha256_verified=True,all_adam_steps=step,all_rank_rng_preserved=True,
        latest_observed_step=observed,unfinished_updates_not_in_checkpoint=observed-step)
    os.kill(controller,signal.SIGTERM)
    for second in range(40):
        live=[pid for pid in pids if matches(pid,str(OLD_BASE/'train.py'))]
        if not live:break
        if second in [5,20,35]:
            for pid in live:os.kill(pid,signal.SIGINT if second==5 else signal.SIGTERM if second==20 else signal.SIGKILL)
        time.sleep(1)
    assert not any(matches(pid,str(OLD_BASE/'train.py')) for pid in pids)
    for item in Path('/proc').iterdir():
        if item.name.isdigit() and matches(int(item.name),str(OLD/'objective-balance/source/audit.py')):
            os.kill(int(item.name),signal.SIGTERM)
    old_status=dict(phase='paused_for_joint20k_trial',preserved_step=step,latest_observed_step=observed,
        replacement_run=RUN,time=time.time())
    (OLD/'launch-status.json').write_text(json.dumps(old_status,indent=2)+'\n')
    write('source-handoff-status.json',phase='source_stopped_with_verified_checkpoint',preserved_step=step)


def prepare():
    import yaml
    import numpy as np
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    from src.training.rq_sample_panels import CLASSES
    if (OUT/'prepared.json').exists():return
    report=json.loads((STUDY/'full-rfid.json').read_text())
    assert report['completed'] and report['passed'] and report['vocabulary']==20481
    assert report['real_images']==report['reconstructed_images']==50000
    recipe=yaml.safe_load(RECIPE.read_text())
    book=Path(report['codebook_path'])
    assert sha(book)==report['codebook_sha256']
    recipe['tokenizer'].update(codebook_sha256=sha(book),measured_rfid=report['measured_rfid'],
        candidate=report['candidate'],atoms=8192 if report['candidate'].startswith('adaptive') else 5120,
        adaptive_coefficient_allocation=report['candidate'].startswith('adaptive'))
    RECIPE.write_text(yaml.safe_dump(recipe,sort_keys=False))
    assert sha(SOURCE/'inputs/stage1-model.pt')==recipe['tokenizer']['sha256']
    assert sha(SOURCE/'inputs/imagenet_256_train.npz')==recipe['evaluation']['reference_sha256']
    (BASE/'inputs').mkdir(parents=True)
    (OUT/'reproducibility').mkdir(exist_ok=True)
    freeze_shared_runtime(BASE,ROOT,include_stochastic_teacher=False)
    for name in ['upstream','data','torch-cache']:(BASE/name).symlink_to((SOURCE/name).resolve(),target_is_directory=True)
    for name in ['stage1-model.pt','imagenet_256_train.npz']:(BASE/'inputs'/name).symlink_to((SOURCE/'inputs'/name).resolve())
    for name in ['stage2-training-config.yaml','stage2-archive-config.yaml','rq-upstream-core.json']:
        shutil.copyfile(SOURCE/'inputs'/name,BASE/'inputs'/name)
    shutil.copyfile(book,BASE/'inputs/joint-codebook.pt')
    shutil.copyfile(RECIPE,BASE/'inputs/recipe.yaml')
    fit=json.loads((STUDY/'calibration-data.json').read_text())
    excluded=set(fit['fit_indices']+fit['heldout_training_indices'])
    indices=[int(i) for i in np.random.default_rng(26101720481).permutation(1281167) if int(i) not in excluded][:32]
    (BASE/'inputs/target-audit.json').write_text(json.dumps(dict(training_indices=indices,disjoint_from_fit=True),indent=2)+'\n')
    config=yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss']=dict(type='soft_target_cross_entropy',stochastic_codes=True,temp=recipe['targets']['temperature'])
    config['experiment'].update(epochs=2,batch_size=64,total_batch_size=2048,save_ckpt_freq=1,test_freq=2)
    assert config['optimizer']['init_lr']==.0005 and config['optimizer']['warmup']['epoch']==0
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__=='__main__':\n    launch('laser_joint')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_joint20k_smoke.py',BASE/'verify.py')
    shutil.copyfile(ROOT/'scripts/tools/configure_rq_sample_panels.py',BASE/'configure_samples.py')
    environment=dict(LASER_RQ_BASE=str(BASE),LASER_RQ_SAFE_CHECKPOINTS='1',
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        TORCH_HOME=str(BASE/'torch-cache'),WANDB_DIR=str(BASE/'wandb'),OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',NCCL_NVLS_ENABLE='0',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    (BASE/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    files=[BASE/name for name in ['train.py','verify.py','configure_samples.py','environment.json']]
    for directory in ['source','runtime']:files.extend((BASE/directory).rglob('*.py'))
    files+=list((BASE/'inputs').glob('*.yaml'))+list((BASE/'inputs').glob('*.json'))+[BASE/'inputs/joint-codebook.pt']
    manifest={str(p.relative_to(BASE)):sha(p) for p in files}
    write('reproducibility/code-manifest.json',files=manifest)
    for name in ['full-rfid.json','selected.json','screen-results.json','calibration-data.json','source-manifest.json']:
        shutil.copyfile(STUDY/name,OUT/'reproducibility'/name)
    for name in ['recipe.yaml','stage2-training-config.yaml']:
        shutil.copyfile(BASE/'inputs'/name,OUT/'reproducibility'/name)
    with zipfile.ZipFile(OUT/'reproducibility/source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,p.relative_to(BASE))
        archive.write(Path(__file__),Path('scripts/tools')/Path(__file__).name)
    write('prepared.json',prepared=True,training_launched=False,run_id=RUN,recipe=recipe,
        fresh_stage2=True,fresh_optimizer=True,stage2_checkpoint_loaded=False,initial_global_step=0,
        world_size=4,global_batch=2048,optimizer_updates_per_epoch=626,cosine_total_updates=62600,
        stop_after_updates=1252,epochs=2,measured_reconstruction_rfid=report['measured_rfid'],
        vocabulary=20481,candidate=report['candidate'],sample_classes=[dict(name=n,imagenet_id=i) for n,i in CLASSES],
        paired_ablation_reference='imagenet-laser-native-joint32k-scratch2-4h100-20261016',
        code_manifest_sha256=sha(OUT/'reproducibility/code-manifest.json'))


def compare():
    results={}
    for epoch in [1,2]:
        current=json.loads((OUT/f'evaluations/epoch{epoch:03d}.json').read_text())
        row=dict(joint20k_fid=current['eval/fid'])
        for label,run in [('original_rq','imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014'),
                          ('joint32k','imagenet-laser-native-joint32k-scratch2-4h100-20261016'),
                          ('joint256k',OLD_RUN)]:
            path=ROOT/'outputs'/run/f'evaluations/epoch{epoch:03d}.json'
            if path.exists():
                ref=json.loads(path.read_text())
                assert ref['train/global_step']==current['train/global_step']==626*epoch
                assert ref['generated_images']==current['generated_images']==50000
                assert ref['real_images']==current['real_images']==1281167
                row[label]=dict(fid=ref['eval/fid'],joint20k_minus_reference=current['eval/fid']-ref['eval/fid'])
        results[str(epoch)]=row
    write('comparison.json',epochs=results,matched_population_and_updates=True,
        joint32k_has_same_backbone_schedule_targets_and_sampler=True,
        original_rq_and_joint256k_have_different_tokenizers_and_samplers=True)


def run():
    lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    child,panel=None,None
    def stop(*_):
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
        if panel is not None and panel.poll() is None:panel.terminate()
        raise KeyboardInterrupt('Joint20k trial interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(command,name,execution_env=env):
        nonlocal child
        write('launch-status.json',phase=name,pid=os.getpid())
        with (OUT/(name+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=ROOT,env=execution_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write('launch-status.json',phase=name,pid=os.getpid(),child_pid=child.pid)
            result=child.wait()
        if result:raise RuntimeError(f'{name} failed with exit code{result}')
    deadline=time.monotonic()+3600
    write('launch-status.json',phase='waiting_for_pair_compression_fit',pid=os.getpid())
    while not (STUDY/'fit-completed.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Pair compression fit has not completed')
        time.sleep(5)
    study_script=str(ROOT/'scripts/tools/fit_laser_adaptive_joint20k.py')
    if not (STUDY/'selected.json').exists():launch([sys.executable,study_script,'screen'],'reconstruction_screen')
    preserve_and_stop()
    for _ in range(30):
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not active:break
        time.sleep(2)
    assert not active,'Unrelated GPU work remains; refusing to launch over it'
    distributed=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4']
    if not (STUDY/'full-rfid.json').exists():
        launch([*distributed,study_script,'full','--memory-fraction','.9'],'full_reconstruction_validation')
    prepare()
    launch([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
        'tests/test_laser_adaptive_joint_vocabulary.py','tests/test_laser_compact_joint_vocabulary.py',
        'tests/test_laser_joint_vocabulary.py','tests/test_laser_joint_compact_runtime.py',
        '--junitxml='+str(OUT/'reproducibility/tests.xml')],'cpu_verification',
        dict(env,PYTHONPATH='.:runtime:scripts/tools'))
    for rel,expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text())['files'].items():assert sha(BASE/rel)==expected
    production_env=dict(env,**json.loads((BASE/'environment.json').read_text()))
    production_env['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    smoke_env=dict(production_env,LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke'))
    launch([*distributed,str(BASE/'train.py'),'--microbatch','64','--smoke-updates','2'],'gpu_preflight',smoke_env)
    launch([sys.executable,str(BASE/'verify.py')],'checkpoint_verification',smoke_env)
    assert json.loads((OUT/'smoke-reload-verification.json').read_text())['passed']
    with (OUT/'sample-workspace.log').open('ab',buffering=0) as log:
        panel=subprocess.Popen([sys.executable,str(BASE/'configure_samples.py'),'--run',RUN,'--watch'],
            env=production_env,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    plan=json.loads((OUT/'prepared.json').read_text())
    plan.update(training_launched=True,preflight_passed=True,production_starts_from_zero=True,
        actual_microbatch=64,accumulation_steps=8,launched_at=time.time())
    write('prepared.json',**plan)
    launch([*distributed,str(BASE/'train.py'),'--microbatch','64','--epochs','2'],'training',production_env)
    compare();write('launch-status.json',phase='completed',epochs=2,updates=1252)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    try:prepare() if args.action=='prepare' else run()
    except BaseException as error:
        write('launch-status.json',phase='failed',error_type=type(error).__name__,error=str(error));raise
