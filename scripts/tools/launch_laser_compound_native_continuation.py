#!/usr/bin/env python3
"""Queue a verified full-dictionary compound continuation after the 32k trial."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-compound-native-continue6-20261016')
SOURCE = Path('/tmp/laser-rq-native-joint32k-imagenet-20261016')
REPLAY = ROOT/'outputs/imagenet-compound-q995-early-sign-replay2-4h100-20261016'
RUN = 'imagenet-laser-compound-q995-native-continue6-4h100-20261016'
OUT = ROOT/'outputs'/RUN
WAIT_FOR = ROOT/'outputs/imagenet-laser-native-joint32k-scratch2-4h100-20261016'
RECIPE = ROOT/'configs/stage2/imagenet-laser-compound-q995-native-continue6-20261016.yaml'
sys.path[:0] = [str(ROOT), str(ROOT/'runtime'), str(ROOT/'scripts/tools')]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare():
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
    from rqvae.utils.config import load_config, augment_arch_defaults
    torch.set_num_threads(4)
    assert not (OUT/'prepared.json').exists(), 'Already prepared'
    recipe = yaml.safe_load(RECIPE.read_text())
    assert recipe['run_id'] == RUN and recipe['epochs'] == 6 and recipe['auxiliary_loss_terms'] == 0
    manifest_path = REPLAY/'train/checkpoints/last.pt'
    original = json.loads(manifest_path.read_text())
    cached = Path('/tmp/laser-compound-early-replay-20261016/chunk-cache')/(original['sha256']+'.pt')
    assert cached.stat().st_size == original['bytes'] and digest(cached) == original['sha256']
    (BASE/'inputs').mkdir(parents=True,exist_ok=True)
    (OUT/'reproducibility').mkdir(parents=True,exist_ok=True)
    os.link(cached,BASE/'historical-epoch002.pt')
    shutil.copyfile(manifest_path,OUT/'reproducibility/source-checkpoint-manifest.json')
    freeze_shared_runtime(BASE,ROOT,include_stochastic_teacher=False)
    shutil.copytree(SOURCE/'upstream',BASE/'upstream',ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc'))
    for name in ['stage2-training-config.yaml','stage2-archive-config.yaml','imagenet_256_train.npz','rq-upstream-core.json']:
        shutil.copyfile(SOURCE/'inputs'/name,BASE/'inputs'/name)
    os.link(SOURCE/'inputs/stage1-model.pt',BASE/'inputs/stage1-model.pt')
    assert digest(BASE/'inputs/stage1-model.pt') == recipe['tokenizer']['sha256']
    assert digest(BASE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss'] = dict(type=recipe['objective'],atom_weight=1.5,coefficient_temperature=.03125,
                          stochastic_coefficient_contexts=True,geometry_weight=0.)
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    shutil.copyfile(RECIPE,BASE/'inputs/recipe.yaml')
    for name in ['data','torch-cache']:
        (BASE/name).symlink_to(SOURCE/name,target_is_directory=True)
    (BASE/'wandb').mkdir(exist_ok=True)
    environment = dict(LASER_RQ_BASE=str(BASE),TORCH_HOME=str(BASE/'torch-cache'),WANDB_DIR=str(BASE/'wandb'),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    write(BASE/'environment.json',environment)
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__ == '__main__':\n    launch('laser_compound_replay')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_compound_replay.py',BASE/'verify.py')
    payload = torch.load(cached,map_location='cpu',mmap=True,weights_only=False)
    assert payload['global_step'] == 1252 and payload['epoch'] == 2 and payload['checkpoint_world_size'] == 4
    assert payload['config']['compound_pair_autoregressive'] and payload['config']['compound_micro_transformer_layers'] == 2
    assert payload['config']['atom_loss_weight'] == recipe['atom_ce_weight']
    assert payload['config']['coeff_scales'] == recipe['tokenizer']['coefficient_scales']
    profile = CompoundReplayRecipe(BASE)
    training = load_config(BASE/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(training.arch)
    arch.vocab_size = training.dataset.vocab_size
    with torch.device('meta'):
        model = profile.model(arch)
    model.load_state_dict(payload['state_dict'],strict=True,assign=True)
    names = list(dict(model.named_parameters()))
    assert names == list(payload['state_dict']) and len(names) == 837
    assert sum(p.numel() for p in model.parameters()) == 1457980928
    optimizer = payload['optimizer']
    indices = [i for group in optimizer['param_groups'] for i in group['params']]
    assert len(indices) == len(optimizer['state']) == len(names)
    for name,index in zip(names,indices):
        assert torch.isfinite(payload['state_dict'][name]).all(), name
        state = optimizer['state'][index]
        assert int(state['step']) == 1252
        for field in ['exp_avg','exp_avg_sq']:
            assert state[field].shape == payload['state_dict'][name].shape and torch.isfinite(state[field]).all(), name
    assert payload['scheduler']['T_max'] == 62600 and payload['scheduler']['last_epoch'] == 1252
    rng = [dict(cpu=value['torch_cpu'],cuda=value['torch_cuda'],
        python=random.Random(261001+rank).getstate(),numpy=np.random.RandomState(261001+rank).get_state())
        for rank,value in enumerate(payload['rng_state_by_rank'])]
    source_eval = dict(fid=payload['fid'],inception_score=payload['inception_score'],
                       inception_score_std=payload['inception_score_std'])
    assert source_eval['fid'] == recipe['continuation']['source_fid']
    training.arch = model.config
    converted = dict(run_id=RUN,state_dict=model.state_dict(),optimizer=optimizer,
        scheduler=dict(warmup=None,after=payload['scheduler']),
        scaler=torch.amp.GradScaler('cuda',enabled=True).state_dict(),
        epoch=2,global_step=1252,next_epoch=2,next_microbatch=0,rng_state_by_rank=rng,
        config=OmegaConf.to_container(training,resolve=True),
        runtime=dict(fresh_stage2=False,fresh_optimizer=False,global_step=1252,world_size=4,
            microbatch=64,accumulation_steps=8,global_batch=2048,parameter_tensors=837,
            parameter_count=1457980928,**profile.metadata()),
        stage1_checkpoint_sha256=recipe['tokenizer']['sha256'],**profile.checkpoint_extra(),
        evaluation=source_eval,best_fid=source_eval['fid'],
        continuation_origin=dict(source_run=REPLAY.name,source_checkpoint_sha256=original['sha256'],
            global_step=1252,weights_and_all_adam_states_unchanged=True,
            tensor_order_identical=True,original_torch_cpu_cuda_rng_restored=True,
            python_numpy_rng_not_present_in_source=True,geometry_never_active_in_source=True,
            geometry_remains_disabled=True,bitwise_trajectory_reproduction=False))
    profile.verify_resume(converted)
    torch.save(converted,BASE/'converted-epoch002.pt')
    converted_hash = digest(BASE/'converted-epoch002.pt')
    proof = dict(passed=True,source_run=REPLAY.name,source_step=1252,source_epoch=2,
        source_checkpoint_sha256=original['sha256'],converted_checkpoint_sha256=converted_hash,
        model_parameters=1457980928,parameter_tensors=837,optimizer_states=837,all_adam_ages=1252,
        model_and_adam_tensor_order_identical=True,strict_native_model_reload=True,
        original_torch_rng_streams=4,scheduler_step=1252,cosine_total_updates=62600,
        source_encoder_precision='BF16',encoder_precision='FP32',source_training_precision='BF16',
        training_precision='native FP16 AMP with newly initialized loss scaler',
        original_stage1_source_rfid=4.215039034200402,actual_quantized_rfid_pending=True,
        geometry_loss_disabled=True,atom_ce_weight=1.5,coefficient_ce_weight=1.,auxiliary_loss_terms=0)
    write(OUT/'checkpoint-transfer-verification.json',proof)
    for p in (BASE/'inputs').iterdir():
        if p.suffix in ['.yaml','.json']:
            shutil.copyfile(p,OUT/'reproducibility'/p.name)
    shutil.copyfile(REPLAY/'comparison.json',OUT/'reproducibility/compound-early-replay-comparison.json')
    files=[BASE/'train.py',BASE/'verify.py',BASE/'environment.json']
    for folder in ['source','runtime','upstream']:
        files.extend((BASE/folder).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml')); files.extend((BASE/'inputs').glob('*.json'))
    write(OUT/'reproducibility/code-manifest.json',{str(p.relative_to(BASE)):digest(p) for p in files})
    tests=['tests/test_laser_native_compound_adapter.py','tests/test_laser_hard_ar.py',
           'tests/test_shared_rq_codebase.py','tests/test_amp_replay.py']
    with zipfile.ZipFile(OUT/'reproducibility/source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(BASE))
        for p in [Path(__file__),*[ROOT/name for name in tests]]:z.write(p,p.relative_to(ROOT))
    environment_tests=dict(os.environ,PYTHONPATH='.:runtime:scripts/tools',OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (OUT/'cpu-tests.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q',*tests,'-p','no:cacheprovider',
            '--junitxml='+str(OUT/'reproducibility/cpu-tests.xml')],cwd=ROOT,env=environment_tests,
            stdout=log,stderr=subprocess.STDOUT)
    suites=ET.parse(OUT/'reproducibility/cpu-tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','errors','failures','skipped']}
    passed=result.returncode == 0 and counts['tests'] >= 4 and not counts['errors'] and not counts['failures']
    write(OUT/'cpu-verification.json',dict(counts,passed=passed))
    assert passed,'CPU verification failed; see cpu-tests.log'
    write(OUT/'prepared.json',dict(prepared=True,run_id=RUN,recipe=recipe,start_step=1252,
        stop_step=3756,start_epoch=2,stop_epoch=6,fresh_stage2=False,fresh_optimizer=False,
        source_run=REPLAY.name,converted_checkpoint_sha256=converted_hash,
        source_manifest_sha256=digest(OUT/'reproducibility/code-manifest.json'),
        wait_for_run=WAIT_FOR.name,wait_for_epoch=6,queued_gpu_checks=['cache parity',
            '50000-image native generation baseline','50000-image finite-code reconstruction',
            'two disposable optimizer updates and checkpoint reload']))
    print('Prepared and CPU-verified compound continuation',RUN,flush=True)


def run():
    lock=(BASE/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'prepared.json').read_text())
    assert plan['prepared'] and json.loads((OUT/'cpu-verification.json').read_text())['passed']
    for name,expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
        assert digest(BASE/name) == expected,name
    child=None
    cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(phase,**values):
        write(OUT/'launch-status.json',dict(phase=phase,pid=os.getpid(),run_id=RUN,
            source_step=1252,stop_step=3756,world_size=4,time=time.time(),**values))
    status('queued',waiting_for=WAIT_FOR.name,waiting_for_epoch=6)
    while not cancelled:
        previous=json.loads((WAIT_FOR/'launch-status.json').read_text())
        if previous.get('phase') == 'completed' and previous.get('global_step') == 3756:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if not active.strip():break
        if previous.get('phase') in ['failed','cancelled','stopped_or_failed']:
            status('prerequisite_failed',previous=previous)
            raise RuntimeError('Current 32k trial did not reach the requested stopping point')
        time.sleep(5)
    if cancelled:
        status('cancelled');return 130
    environment=dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    environment['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    def launch(command,phase):
        nonlocal child
        with (OUT/(phase+'.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=BASE,env=environment,stdout=log,stderr=subprocess.STDOUT)
            status(phase,child_pid=child.pid)
            code=child.wait()
        if code or cancelled:
            status('cancelled' if cancelled else 'failed',failed_phase=phase,returncode=code)
            raise RuntimeError(f'{phase} returned {code}')
    distributed=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4']
    launch([*distributed,str(BASE/'verify.py')],'baseline_verification')
    proof=json.loads((OUT/'baseline/runtime-verification.json').read_text())
    assert proof['passed']
    os.environ.update(json.loads((BASE/'environment.json').read_text()))
    import torch
    from src.training.k4_checkpoint_io import atomic_torch_save,_checkpoint_upload_source
    torch.set_num_threads(4)
    assert digest(BASE/'converted-epoch002.pt') == plan['converted_checkpoint_sha256']
    payload=torch.load(BASE/'converted-epoch002.pt',map_location='cpu',mmap=True,weights_only=False)
    measured=json.loads((OUT/'baseline/fid50k.json').read_text())
    payload['evaluation']=measured
    payload['best_fid']=measured['fid']
    (OUT/'persistent-train/checkpoints').mkdir(parents=True,exist_ok=True)
    (OUT/'checkpoints').symlink_to('persistent-train/checkpoints',target_is_directory=True)
    atomic_torch_save(payload,OUT/'checkpoints/last.pt')
    best=OUT/'checkpoints/best.pt';best.symlink_to(os.readlink(OUT/'checkpoints/last.pt'))
    del payload
    resume=str(OUT/'checkpoints/last.pt')
    launch([*distributed,str(BASE/'train.py'),'--epochs','6','--microbatch','64',
            '--resume',resume,'--smoke-updates','1254'],'gpu_smoke')
    smoke=(OUT/'smoke/checkpoints/last.pt').resolve()
    saved=torch.load(_checkpoint_upload_source(smoke),map_location='cpu',mmap=True,weights_only=False)
    assert saved['global_step'] == 1254 and saved['scheduler']['after']['last_epoch'] == 1254
    assert len(saved['optimizer']['state']) == 837
    assert all(int(v['step']) == 1254 and torch.isfinite(v['exp_avg']).all()
               and torch.isfinite(v['exp_avg_sq']).all() for v in saved['optimizer']['state'].values())
    write(OUT/'continuation-verification.json',dict(passed=True,source_step=1252,
        disposable_smoke_step=1254,all_smoke_adam_ages=1254,production_resumes_source_step=1252,
        original_torch_rng_and_adam_restored=True,cosine_updates=62600,
        native_baseline_fid=proof['full_generation_fid'],actual_tokenizer_rfid=proof['actual_compound_tokenizer_rfid']))
    del saved
    (OUT/'evaluations').mkdir(exist_ok=True)
    write(OUT/'evaluations/epoch002.json',dict(measured,**{'train/epoch':2,'train/global_step':1252,
        'eval/fid':measured['fid'],'eval/inception_score':measured['inception_score']},
        origin='same trained weights re-evaluated using the shared native runtime'))
    launch([*distributed,str(BASE/'train.py'),'--epochs','6','--microbatch','64',
            '--resume',resume,'--new-wandb-run'],'train')
    status('completed',completed_epochs=6,global_step=3756,returncode=0)
    return 0


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    arguments=parser.parse_args()
    if arguments.action == 'prepare':prepare()
    else:raise SystemExit(run())
