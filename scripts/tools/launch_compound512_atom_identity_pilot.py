#!/usr/bin/env python3
"""Prepare and execute the gated, matched 128-update atom-identity pilot."""
import argparse
import concurrent.futures
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
SOURCE=Path('/tmp/laser-compound512-native-imagenet-20261016')
BASE=Path('/tmp/laser-compound512-atom-identity-pilot-20261016')
OUT=ROOT/'outputs/imagenet-compound512-atom-identity-pilot-20261016'
CONFIG=ROOT/'configs/stage2/imagenet-laser-compound512-atom-identity-pilot-20261016.yaml'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    import torch,yaml
    assert not (OUT/'plan.json').exists(),'Already prepared'
    os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
    sys.path[:0]=[str(BASE/'source'),str(BASE/'runtime'),str(BASE/'upstream')]
    proposal=yaml.safe_load(CONFIG.read_text())
    BASE.mkdir(exist_ok=True)
    for folder in ['source','runtime','upstream']:
        shutil.copytree(SOURCE/folder,BASE/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    additions=['src/models/laser_atom_identity_adapter.py','src/training/laser_identity_pilot_analysis.py']
    for name in additions:shutil.copyfile(ROOT/name,BASE/'source'/name)
    (BASE/'inputs').mkdir(exist_ok=True)
    for p in (SOURCE/'inputs').iterdir():
        if p.suffix in ['.yaml','.json','.npz']:shutil.copyfile(p,BASE/'inputs'/p.name)
    if not (BASE/'inputs/stage1-model.pt').exists():
        os.link(SOURCE/'inputs/stage1-model.pt',BASE/'inputs/stage1-model.pt')
    shutil.copyfile(SOURCE/'tokenizer-verification.json',BASE/'tokenizer-verification.json')
    for name in ['data','torch-cache']:
        if not (BASE/name).exists():(BASE/name).symlink_to(SOURCE/name,target_is_directory=True)
    (BASE/'wandb').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'scripts/tools/run_compound512_atom_identity_pilot.py',BASE/'trial.py')
    shutil.copyfile(CONFIG,BASE/'inputs/pilot-plan.yaml')
    checkpoint=SOURCE/'checkpoint-cache/objects/fd443074bdaa225e59db2f9beca34fd23bc528eb6fa3bff94f0e5d71564fd77a.pt'
    assert digest(checkpoint)==proposal['source']['checkpoint_sha256']
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    assert payload['global_step']==1252 and payload['next_epoch']==2 and payload['next_microbatch']==0
    assert len(payload['state_dict'])==len(payload['optimizer']['state'])==837
    assert all(int(state['step'])==1252 for state in payload['optimizer']['state'].values())
    assert len(payload['rng_state_by_rank'])==4
    del payload
    old=json.loads((ROOT/'outputs/imagenet-compound512-gap-20261016/compound512-epoch001.json').read_text())
    write(OUT/'prior-validation-indices.json',old['splits']['heldout_val']['indices'])
    env=dict(LASER_RQ_BASE=str(BASE),TORCH_HOME=str(BASE/'torch-cache'),WANDB_DIR=str(BASE/'wandb'),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    write(BASE/'environment.json',env)
    repro=OUT/'reproducibility';repro.mkdir(exist_ok=True)
    files=[BASE/'trial.py',BASE/'environment.json',BASE/'tokenizer-verification.json']
    for folder in ['source','runtime','upstream']:files.extend((BASE/folder).rglob('*.py'))
    files.extend(p for p in (BASE/'inputs').iterdir() if p.suffix in ['.json','.yaml'])
    manifest={str(p.relative_to(BASE)):digest(p) for p in files}
    write(repro/'code-manifest.json',manifest)
    with zipfile.ZipFile(repro/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,p.relative_to(BASE))
        for name in ['tests/test_laser_atom_identity_adapter.py','tests/test_laser_identity_pilot_analysis.py',
                     'scripts/tools/launch_compound512_atom_identity_pilot.py']:
            archive.write(ROOT/name,name)
    tests=['tests/test_laser_atom_identity_adapter.py','tests/test_laser_identity_pilot_analysis.py','tests/test_amp_replay.py']
    test_env=dict(os.environ,PYTHONPATH='.:runtime:tests',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (OUT/'cpu-tests.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q',*tests,'-p','no:cacheprovider',
            '--junitxml='+str(repro/'cpu-tests.xml')],cwd=ROOT,env=test_env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'CPU tests failed'
    write(OUT/'cpu-verification.json',dict(passed=True,suites=tests))
    plan=dict(prepared=True,training_runner_prepared=True,source_checkpoint_sha256=proposal['source']['checkpoint_sha256'],
        local_source_checkpoint=str(checkpoint),source_step=1252,stop_step=1380,additional_updates_per_branch=128,
        global_batch=2048,world_size=4,microbatch=64,accumulation_steps=8,auxiliary_loss_terms=0,
        sampled_trajectory_banks=0,gate_required_for_fid=True,automatic_training_extension=False,
        automatic_adoption=False,source_manifest_sha256=digest(repro/'code-manifest.json'),time=time.time())
    write(OUT/'plan.json',plan);print(json.dumps(plan),flush=True)


def workspace():
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    space=ws.Workspace(entity='helloimlixin-rutgers',project='laser',name='Compound512 atom identity matched pilot FID',
        runset_settings=ws.RunsetSettings(filters=[ws.Config('experiment_group')==OUT.name]),
        sections=[ws.Section(name='FID',is_open=True,pinned=True,panels=[
            wr.MarkdownPanel(markdown='Matched continuations from update 1252. Each branch trains 128 additional updates on identical batches. '
                'The only candidate change is a zero-initialized learned atom-ID embedding. '
                'A paired 1024-image prediction gate precedes FID. All FID evaluations use 50,000 generated images versus all 1,281,167 training images.'),
            wr.LinePlot(title='FID50k vs full training set - after gate',x='train/global_step',y=['eval/fid']),
            wr.LinePlot(title='Held-out atom cross entropy',x='train/global_step',y=['eval/atom_ce']),
            wr.LinePlot(title='Held-out joint pair cross entropy',x='train/global_step',y=['eval/joint_pair_ce']),
            wr.LinePlot(title='Training loss',x='train/global_step',y=['train/loss']),
        ])])
    space.save();write(OUT/'workspace.json',dict(url=space.url))


def run():
    lock=(BASE/'pilot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT/'cpu-verification.json').read_text())['passed']
    for name,expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
        assert digest(BASE/name)==expected,name
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not active.strip(),'GPUs are occupied'
    workspace()
    environment=dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    child=None;cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(action,branch='control'):
        nonlocal child
        if cancelled:raise KeyboardInterrupt
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(BASE/'trial.py'),action,'--branch',branch]
        with (OUT/f'{action}-{branch}.log').open('a') as log:
            child=subprocess.Popen(command,cwd=BASE,env=environment,stdout=log,stderr=subprocess.STDOUT)
            write(OUT/'status.json',dict(phase=action,branch=branch,pid=os.getpid(),child_pid=child.pid,time=time.time()))
            result=child.wait()
        if result:raise RuntimeError(f'{action} {branch} returned {result}')
    # Persist immutable checkpoints on the CPU while the next GPU phase runs.
    sys.path[:0]=[str(BASE/'source'),str(BASE/'runtime'),str(BASE/'upstream')]
    os.environ.update(json.loads((BASE/'environment.json').read_text()))
    from src.training import k4_checkpoint_io
    def persist(branch):
        folder=OUT/branch;receipt=json.loads((folder/'checkpoint.json').read_text())
        original=Path(receipt['local_path']);staged=BASE/(branch+'-durable-staging.pt');os.link(original,staged)
        target=folder/'checkpoints/last.pt';target.parent.mkdir(parents=True,exist_ok=True)
        k4_checkpoint_io._persist_immutable_checkpoint(staged,target)
        receipt.update(durable_path=str(target),sha256=digest(original),durable=True)
        write(folder/'durable-checkpoint.json',receipt)
    writer=concurrent.futures.ThreadPoolExecutor(max_workers=1);copies=[]
    try:
        launch('probe')
        for branch in ['control','atom_identity']:
            launch('train',branch);copies.append(writer.submit(persist,branch))
        from src.training.laser_identity_pilot_analysis import analyze
        def read(branch,name):return json.loads((OUT/branch/name).read_text())
        gate=analyze(read('control','heldout-step1252.json'),read('atom_identity','heldout-step1252.json'),
            read('control','heldout-step1380.json'),read('atom_identity','heldout-step1380.json'),
            read('control','result.json'),read('atom_identity','result.json'))
        write(OUT/'gate.json',gate);print(json.dumps(gate),flush=True)
        if gate['qualifies_for_fid']:
            for branch in ['control','atom_identity']:launch('fid',branch)
        for copy in copies:copy.result()
        result=dict(completed=True,gate=gate,fid_evaluated=gate['qualifies_for_fid'],
            automatic_adoption=False,automatic_training_extension=False,time=time.time())
        if gate['qualifies_for_fid']:
            control=read('control','fid50k.json');candidate=read('atom_identity','fid50k.json')
            result['fid']=dict(control=control['fid'],candidate=candidate['fid'],delta=candidate['fid']-control['fid'])
        write(OUT/'comparison.json',result)
        write(OUT/'status.json',dict(phase='completed',passed=True,fid_evaluated=gate['qualifies_for_fid'],time=time.time()))
    except BaseException as error:
        write(OUT/'status.json',dict(phase='cancelled' if cancelled else 'failed',error=str(error),time=time.time()))
        raise
    finally:writer.shutdown(wait=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run'])
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:run()
