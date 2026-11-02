#!/usr/bin/env python3
"""Fresh compound512 trial, queued behind the current six-epoch joint trial."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path('/workspace/Projects/laser')
sys.path[:0]=[str(ROOT),str(ROOT/'runtime'),str(ROOT/'scripts/tools')]
import launch_laser_hard_ar_imagenet as launcher
from runtime.rq_imagenet_entry import freeze_shared_runtime

RUN='imagenet-laser-compound512-native-scratch2-4h100-20261016'
BASE=Path('/tmp/laser-compound512-native-imagenet-20261016')
OUT=ROOT/'outputs'/RUN
SOURCE=Path('/tmp/laser-rq-native-joint32k-imagenet-20261016')
WAIT_FOR=ROOT/'outputs/imagenet-laser-native-joint32k-scratch2-4h100-20261016'
RECIPE=ROOT/'configs/stage2/imagenet-laser-compound512-native-scratch2-20261016.yaml'
launcher.BASE,launcher.OUT,launcher.RUN=BASE,OUT,RUN


def prepare():
    import torch
    import yaml
    from omegaconf import OmegaConf
    from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
    from rqvae.utils.config import load_config,augment_arch_defaults
    assert not (OUT/'prepared.json').exists(),'Already prepared'
    recipe=yaml.safe_load(RECIPE.read_text())
    assert recipe['run_id']==RUN and recipe['epochs']==2 and recipe['auxiliary_loss_terms']==0
    assert recipe['tokenizer']['num_atoms']==16384 and recipe['tokenizer']['coefficient_bins']==512
    assert recipe['targets']['temperature']==.03125
    assert abs(recipe['targets']['sigma_bins']-.125/(6/511))<1.e-12
    assert recipe['initialization']==dict(paired_fresh_backbone=True,reference_vocab_size=65537)
    (BASE/'inputs').mkdir(parents=True,exist_ok=True)
    (OUT/'reproducibility').mkdir(parents=True,exist_ok=True)
    freeze_shared_runtime(BASE,ROOT,include_stochastic_teacher=False)
    shutil.copytree(SOURCE/'upstream',BASE/'upstream',ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc'))
    for name in ['stage2-training-config.yaml','stage2-archive-config.yaml','imagenet_256_train.npz','rq-upstream-core.json']:
        shutil.copyfile(SOURCE/'inputs'/name,BASE/'inputs'/name)
    os.link(SOURCE/'inputs/stage1-model.pt',BASE/'inputs/stage1-model.pt')
    assert launcher.digest(BASE/'inputs/stage1-model.pt')==recipe['tokenizer']['sha256']
    assert launcher.digest(BASE/'inputs/imagenet_256_train.npz')==recipe['evaluation']['reference_sha256']
    config=yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss']=dict(type=recipe['objective'],atom_weight=1.5,coefficient_temperature=.03125,
        stochastic_coefficient_contexts=True,coefficient_bins=512,geometry_weight=0.)
    assert config['optimizer']['init_lr']==.0005 and config['optimizer']['warmup']['epoch']==0
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    shutil.copyfile(RECIPE,BASE/'inputs/recipe.yaml')
    for name in ['data','torch-cache']:(BASE/name).symlink_to(SOURCE/name,target_is_directory=True)
    (BASE/'wandb').mkdir(exist_ok=True)
    environment=dict(LASER_RQ_BASE=str(BASE),TORCH_HOME=str(BASE/'torch-cache'),WANDB_DIR=str(BASE/'wandb'),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    launcher.write(BASE/'environment.json',environment)
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__ == '__main__':\n    launch('laser_compound_replay')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_compound512.py',BASE/'verify.py')
    profile=CompoundReplayRecipe(BASE)
    training=load_config(BASE/'inputs/stage2-training-config.yaml')
    arch=augment_arch_defaults(training.arch);arch.vocab_size=training.dataset.vocab_size
    with torch.device('meta'):model=profile.model(arch)
    assert sum(p.numel() for p in model.parameters())==1448144384
    assert len(list(model.parameters()))==837 and model.coeff_classifier[0][1].out_features==512
    for p in (BASE/'inputs').iterdir():
        if p.suffix in ['.json','.yaml']:shutil.copyfile(p,OUT/'reproducibility'/p.name)
    replay=ROOT/'outputs/imagenet-compound-q995-early-sign-replay2-4h100-20261016'
    shutil.copyfile(replay/'comparison.json',OUT/'reproducibility/compound2048-replay-comparison.json')
    files=[BASE/'train.py',BASE/'verify.py',BASE/'environment.json']
    for folder in ['source','runtime','upstream']:files.extend((BASE/folder).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml'));files.extend((BASE/'inputs').glob('*.json'))
    launcher.write(OUT/'reproducibility/code-manifest.json',
                   {str(p.relative_to(BASE)):launcher.digest(p) for p in files})
    tests=['tests/test_laser_native_compound_adapter.py','tests/test_laser_hard_ar.py',
           'tests/test_shared_rq_codebase.py','tests/test_amp_replay.py']
    with zipfile.ZipFile(OUT/'reproducibility/source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(BASE))
        for p in [Path(__file__),ROOT/'scripts/tools/launch_laser_hard_ar_imagenet.py',
                  *[ROOT/name for name in tests]]:z.write(p,p.relative_to(ROOT))
    test_env=dict(os.environ,PYTHONPATH='.:runtime:scripts/tools',OMP_NUM_THREADS='4',
                  MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (OUT/'cpu-tests.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q',*tests,'-p','no:cacheprovider',
            '--junitxml='+str(OUT/'reproducibility/cpu-tests.xml')],cwd=ROOT,env=test_env,
            stdout=log,stderr=subprocess.STDOUT)
    suites=ET.parse(OUT/'reproducibility/cpu-tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','errors','failures','skipped']}
    passed=result.returncode==0 and counts['tests']>=26 and not counts['errors'] and not counts['failures']
    launcher.write(OUT/'cpu-verification.json',dict(counts,passed=passed))
    assert passed,'CPU verification failed; see cpu-tests.log'
    launcher.write(OUT/'prepared.json',dict(prepared=True,run_id=RUN,recipe=recipe,
        fresh_stage2=True,fresh_optimizer=True,source_stage2_checkpoint=None,
        initial_global_step=0,initial_optimizer_state_count=0,train_images=1281167,
        global_batch=2048,updates_per_epoch=626,stop_updates=1252,cosine_total_updates=62600,
        evaluation_epochs=[1,2],original_stage1_checkpoint=True,atoms=16384,coefficient_bins=512,
        coefficient_sigma_normalized=.125,coefficient_sigma_bins=.125/(6/511),
        auxiliary_loss_terms=0,sampled_trajectories=0,paired_initial_backbone=True,
        wait_for_run=WAIT_FOR.name,wait_for_epoch=6,
        tokenizer_reconstruction_gate_pending=True,
        source_manifest_sha256=launcher.digest(OUT/'reproducibility/code-manifest.json'),time=time.time()))
    print('Prepared fresh compound512 trial; all CPU checks passed',RUN,flush=True)


def run(start_now=False):
    lock=(BASE/'compound-controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT/'prepared.json').read_text())['prepared']
    assert json.loads((OUT/'cpu-verification.json').read_text())['passed']
    child=None;cancelled=False
    def status(phase,**values):
        launcher.write(OUT/'launch-status.json',dict(phase=phase,pid=os.getpid(),run_id=RUN,
            coefficient_bins=512,atoms=16384,world_size=4,time=time.time(),**values))
    def stop(*_):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    status('waiting_for_free_gpus' if start_now else 'queued',waiting_for_run=None if start_now else WAIT_FOR.name,
           waiting_for_epoch=None if start_now else 6,start_now=start_now,
           reconstruction_gate_pending=True,fresh_stage2=True)
    while not cancelled:
        if start_now:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if not active.strip():break
            time.sleep(2)
            continue
        previous=json.loads((WAIT_FOR/'launch-status.json').read_text())
        if previous.get('phase')=='completed' and previous.get('global_step')==3756:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if not active.strip():break
        if previous.get('phase') in ['failed','cancelled','stopped_or_failed']:
            status('prerequisite_failed',previous=previous)
            raise RuntimeError('Current 32k trial did not reach the requested stopping point')
        time.sleep(5)
    if cancelled:status('cancelled');return 130
    for name,expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
        assert launcher.digest(BASE/name)==expected,name
    environment=dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
             str(BASE/'verify.py'),'--reconstruction']
    with (OUT/'reconstruction.log').open('a') as log:
        child=subprocess.Popen(command,cwd=BASE,env=environment,stdout=log,stderr=subprocess.STDOUT)
        status('reconstruction_gate',child_pid=child.pid)
        code=child.wait()
    if code or cancelled:
        status('cancelled' if cancelled else 'failed',failed_phase='reconstruction_gate',returncode=code)
        raise RuntimeError('512-bin reconstruction gate failed; no AR run started')
    assert json.loads((OUT/'tokenizer-verification.json').read_text())['passed']
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    code=launcher.run(resume=False)
    if code==0:
        comparisons={}
        old=json.loads((ROOT/'outputs/imagenet-compound-q995-early-sign-replay2-4h100-20261016/comparison.json').read_text())
        for epoch in [1,2]:
            current=json.loads((OUT/'evaluations'/f'epoch{epoch:03d}.json').read_text())
            joint=json.loads((WAIT_FOR/'evaluations'/f'epoch{epoch:03d}.json').read_text())
            rq=json.loads((ROOT/'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014/evaluations'/f'epoch{epoch:03d}.json').read_text())
            assert current['train/global_step']==joint['train/global_step']==rq['train/global_step']==epoch*626
            assert current['generated_images']==joint['generated_images']==rq['generated_images']==50000
            assert current['real_images']==joint['real_images']==rq['real_images']==1281167
            comparisons[str(epoch)]=dict(compound512=current['eval/fid'],joint32k=joint['eval/fid'],
                original_rq=rq['eval/fid'],compound2048_replay=old['curves'][str(epoch)]['evaluation']['eval/fid_original_train50k'])
        launcher.write(OUT/'comparison.json',dict(completed_epochs=2,global_step=1252,curves=comparisons,
            scope='Matched updates and FID populations; architectures, precision and tokenization differ',
            automatic_extension=False))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--start-now',action='store_true',help='Skip the epoch-six queue after an explicitly requested switch')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:raise SystemExit(run(start_now=args.start_now))
