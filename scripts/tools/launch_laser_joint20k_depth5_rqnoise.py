#!/usr/bin/env python3
"""Fresh five-depth run with the approved RQ-noise proposal and K2048 sampler."""
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

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-laser-joint20k-depth5-rqnoise-k2048-scratch100-4h100-20261017'
OUT = ROOT/'outputs'/RUN
BASE = Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-imagenet-20261017')
SOURCE = Path('/tmp/laser-rq-joint20k-depth5-imagenet-20261017')
OLD = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
CALIBRATION = ROOT/'outputs/imagenet-laser-joint20k-depth5-rq-noise-calibration-20261017'
SAMPLER = ROOT/'outputs/imagenet-laser-joint20k-depth5-epoch6-sampler-sweep-20261017'
RECIPE = ROOT/'configs/stage2/imagenet-laser-joint20k-depth5-rqnoise-k2048-scratch100-20261017.yaml'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def write(name, **values):
    path = OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(values,time=time.time()),indent=2)+'\n')
    temporary.replace(path)


def matches(pid, token):
    try:
        return token.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def prepare():
    import numpy as np
    import yaml
    report = json.loads((CALIBRATION/'report.json').read_text())
    selected = json.loads((SAMPLER/'selected-sampling.json').read_text())
    assert report['completed'] and not report['active_training_targets_changed']
    assert json.loads((CALIBRATION/'final-verification.json').read_text())['passed']
    assert json.loads((SAMPLER/'final-verification.json').read_text())['passed']
    assert selected['generated_images'] == 50000 and selected['reference_images'] == 1281167
    recipe = yaml.safe_load((CALIBRATION/'proposed-recipe.yaml').read_text())
    recipe.update(version='laser-native-joint20k-depth5-rqnoise-k2048-v1',run_id=RUN,
        name='LASER joint20k depth5 | RQ target noise + K2048 | fresh100')
    recipe['sampling'] = selected['sampling']
    recipe['targets']['calibration']['training_launched'] = True
    assert recipe['targets']['temperature'] == report['proposed_temperatures']
    assert recipe['sampling'] == dict(temperature=1.,top_k=2048,top_p=.95)
    assert recipe['epochs'] == recipe['lr_schedule_epochs'] == 100
    assert recipe['microbatch'] == 64 and recipe['global_batch'] == 2048
    assert recipe['tokenizer']['depth'] == 5 and recipe['tokenizer']['vocab_size'] == 20481
    assert recipe['decoder'] == 'native_joint_integer' and recipe['auxiliary_loss_terms'] == 0
    # A matched restart changes targets and sampler, plus identifying metadata.
    previous = yaml.safe_load((SOURCE/'inputs/recipe.yaml').read_text())
    import copy
    before, after = copy.deepcopy(previous),copy.deepcopy(recipe)
    for value in (before,after):
        for key in ('version','run_id','name','sampling'):value.pop(key)
        value['targets'].pop('temperature'); value['targets'].pop('calibration')
    assert before == after
    manifest = json.loads((OLD/'reproducibility/code-manifest.json').read_text())['files']
    assert all(sha(SOURCE/name) == expected for name,expected in manifest.items())
    assert sha(SOURCE/'inputs/stage1-model.pt') == recipe['tokenizer']['sha256']
    assert sha(SOURCE/'inputs/joint-codebook.pt') == recipe['tokenizer']['codebook_sha256']
    assert sha(SOURCE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    assert not BASE.exists() and not RECIPE.exists()
    RECIPE.write_text(yaml.safe_dump(recipe,sort_keys=False))
    (BASE/'inputs').mkdir(parents=True)
    identical_runtime = {}
    for directory in ('source','runtime'):
        for path in (SOURCE/directory).rglob('*.py'):
            destination = BASE/path.relative_to(SOURCE)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination)
            assert sha(path) == sha(destination)
            identical_runtime[str(destination.relative_to(BASE))] = sha(destination)
    for name in ('upstream','data','torch-cache'):
        (BASE/name).symlink_to((SOURCE/name).resolve(),target_is_directory=True)
    for name in ('stage1-model.pt','imagenet_256_train.npz'):
        (BASE/'inputs'/name).symlink_to((SOURCE/'inputs'/name).resolve())
    for name in ('stage2-training-config.yaml','stage2-archive-config.yaml','rq-upstream-core.json'):
        shutil.copyfile(SOURCE/'inputs'/name,BASE/'inputs'/name)
    shutil.copyfile(SOURCE/'inputs/joint-codebook.pt',BASE/'inputs/joint-codebook.pt')
    shutil.copyfile(RECIPE,BASE/'inputs/recipe.yaml')
    fit = json.loads((ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017/calibration-data.json').read_text())
    calibration_data = json.loads((CALIBRATION/'data.json').read_text())
    excluded = set(fit['fit_indices']+fit['heldout_training_indices']+
        calibration_data['fit_indices']+calibration_data['validation_indices']+
        json.loads((SOURCE/'inputs/target-audit.json').read_text())['training_indices'])
    indices = [int(i) for i in np.random.default_rng(261017204820).permutation(1281167)
               if int(i) not in excluded][:32]
    (BASE/'inputs/target-audit.json').write_text(json.dumps(dict(training_indices=indices,
        disjoint_from_fit=True,disjoint_from_temperature_calibration_and_validation=True),indent=2)+'\n')
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss']['temp'] = recipe['targets']['temperature']
    config['experiment']['sample'] = dict(temp=1.,top_k=2048,top_p=.95)
    assert config['optimizer']['init_lr'] == .0005 and config['optimizer']['warmup']['epoch'] == 0
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    archive = yaml.safe_load((BASE/'inputs/stage2-archive-config.yaml').read_text())
    archive['sampling'] = dict(temp=1.,top_k=2048,top_p=.95)
    (BASE/'inputs/stage2-archive-config.yaml').write_text(yaml.safe_dump(archive,sort_keys=False))
    for name in ('train.py','verify.py','configure_samples.py'):
        shutil.copyfile(SOURCE/name,BASE/name)
    environment = json.loads((SOURCE/'environment.json').read_text())
    environment = {key:value.replace(str(SOURCE),str(BASE)) for key,value in environment.items()}
    (BASE/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    files = [BASE/name for name in ('train.py','verify.py','configure_samples.py','environment.json')]
    for directory in ('source','runtime'):files.extend((BASE/directory).rglob('*.py'))
    files += list((BASE/'inputs').glob('*.yaml'))+list((BASE/'inputs').glob('*.json'))
    files.append(BASE/'inputs/joint-codebook.pt')
    write('reproducibility/code-manifest.json',files={str(p.relative_to(BASE)):sha(p) for p in files})
    write('reproducibility/adaptation-verification.json',passed=True,
        copied_runtime_files=identical_runtime,all_runtime_python_identical_to_previous_run=True,
        native_upstream_unchanged=True,only_target_temperatures_and_sampling_behavior_changed=True,
        tokenizer_unchanged=True,stage1_sha256=recipe['tokenizer']['sha256'],
        codebook_sha256=recipe['tokenizer']['codebook_sha256'])
    for name,source in [('recipe.yaml',RECIPE),('noise-calibration-report.json',CALIBRATION/'report.json'),
                        ('sampler-selection.json',SAMPLER/'selected-sampling.json'),
                        ('stage2-training-config.yaml',BASE/'inputs/stage2-training-config.yaml')]:
        shutil.copyfile(source,OUT/'reproducibility'/name)
    with zipfile.ZipFile(OUT/'reproducibility/source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(BASE))
        archive.write(Path(__file__),Path(__file__).relative_to(ROOT))
    write('prepared.json',prepared=True,training_launched=False,run_id=RUN,recipe=recipe,
        fresh_stage2=True,fresh_optimizer=True,stage2_checkpoint_loaded=False,initial_global_step=0,
        world_size=4,global_batch=2048,microbatch=64,accumulation_steps=8,
        optimizer_updates_per_epoch=626,epochs=100,stop_after_updates=62600,cosine_total_updates=62600,
        tokens_per_image=320,vocabulary=20481,previous_training_run=OLD.name,
        sampler_selection_fid50k=selected['fid50k'],sampler_selection_epoch=6,
        noise_calibration_generation_improvement_proven=False,
        code_manifest_sha256=sha(OUT/'reproducibility/code-manifest.json'))


def preserve_and_stop():
    import torch
    torch.set_num_threads(4)
    records = [json.loads(p.read_text()) for p in
               (OLD/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    receipt = max(records,key=lambda r:r['global_step'])
    write('launch-status.json',phase='preserving_previous_checkpoint',pid=os.getpid(),
          previous_saved_step=receipt['global_step'])
    for key in ('local','persistent'):
        assert Path(receipt[key]).stat().st_size == receipt['bytes']
        assert sha(receipt[key]) == receipt['sha256']
    saved = torch.load(receipt['local'],map_location='cpu',mmap=True,weights_only=False)
    step = receipt['global_step']
    assert saved['run_id'] == OLD.name and saved['global_step'] == step
    assert saved['next_microbatch'] == 0 and len(saved['rng_state_by_rank']) == 4
    assert all(int(s['step']) == step for s in saved['optimizer']['state'].values())
    del saved
    alias = OLD/f'epoch{receipt["epoch"]:03d}-before-rqnoise-k2048.pt'
    alias.symlink_to(Path(receipt['persistent']).relative_to(OLD))
    local = SOURCE/f'epoch{receipt["epoch"]:03d}-before-rqnoise-k2048.pt'
    os.link(receipt['local'],local)
    controller = json.loads((OLD/'launch-status.json').read_text())['pid']
    progress = [json.loads((OLD/f'train/progress-rank{rank}.json').read_text()) for rank in range(4)]
    pids = [p['pid'] for p in progress]
    assert matches(controller,'launch_laser_joint20k_depth5.py')
    assert all(matches(pid,str(SOURCE/'train.py')) for pid in pids)
    observed = max(p['train/global_step'] for p in progress)
    write('preserved-source-checkpoint.json',passed=True,receipt=receipt,preserved_alias=str(alias),
        local_alias=str(local),local_and_persistent_sha256_verified=True,
        latest_observed_step=observed,unfinished_updates_not_in_checkpoint=observed-step)
    os.kill(controller,signal.SIGTERM)
    for second in range(45):
        live = [pid for pid in pids if matches(pid,str(SOURCE/'train.py'))]
        if not live and not matches(controller,'launch_laser_joint20k_depth5.py'):break
        if second in (10,25,40):
            for pid in live:os.kill(pid,signal.SIGINT if second == 10 else signal.SIGTERM if second == 25 else signal.SIGKILL)
        time.sleep(1)
    assert not any(matches(pid,str(SOURCE/'train.py')) for pid in pids)
    assert not matches(controller,'launch_laser_joint20k_depth5.py')
    stopped = dict(phase='stopped_for_fresh_rqnoise_k2048',preserved_step=step,
        latest_observed_step=observed,replacement_run=RUN,time=time.time())
    (OLD/'launch-status.json').write_text(json.dumps(stopped,indent=2)+'\n')
    import wandb
    previous = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{OLD.name}')
    previous.summary.update({'training/status':stopped['phase'],
        'training/preserved_checkpoint_step':step,'training/replacement_run':RUN})
    previous.update()


def main():
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'prepared.json').exists(), 'Run already prepared'
    environment = dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                       PYTHONUNBUFFERED='1')
    child = panel = None

    def stop(*_):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT)
        if panel is not None and panel.poll() is None:panel.terminate()
        raise KeyboardInterrupt('RQ-noise K2048 run interrupted')

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)

    def launch(command,phase,env):
        nonlocal child
        with (OUT/(phase+'.log')).open('ab',buffering=0) as log:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write('launch-status.json',phase=phase,pid=os.getpid(),child_pid=child.pid,
                  run_id=RUN,epochs=100,stop_step=62600)
            result = child.wait()
        if result:raise RuntimeError(f'{phase} returned {result}')

    prepare()
    launch([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
        'tests/test_laser_joint_refinement.py','tests/test_laser_joint_compact_runtime.py',
        'tests/test_laser_joint_vocabulary.py','tests/test_joint_noise_calibration.py',
        'tests/test_joint_integer_sampling_audit.py','--junitxml='+str(OUT/'reproducibility/tests.xml')],
        'cpu_verification',dict(environment,PYTHONPATH='.:runtime:scripts/tools'))
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    preserve_and_stop()
    for _ in range(30):
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not active:break
        time.sleep(1)
    assert not active,'GPU processes remain after stopping the prior run'
    production = dict(environment,**json.loads((BASE/'environment.json').read_text()),
                      WANDB_API_KEY=os.environ['WANDB_API_KEY'])
    smoke = dict(production,LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke'))
    distributed = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4']
    for name,expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text())['files'].items():
        assert sha(BASE/name) == expected
    launch([*distributed,str(BASE/'train.py'),'--microbatch','64','--smoke-updates','2'],'gpu_preflight',smoke)
    launch([sys.executable,str(BASE/'verify.py')],'checkpoint_verification',smoke)
    proof = json.loads((OUT/'smoke-reload-verification.json').read_text())
    assert proof['passed'] and proof['depth'] == 5 and proof['cached_events_verified'] == 320
    assert proof['targets']['temperature'] == json.loads((OUT/'prepared.json').read_text())['recipe']['targets']['temperature']
    with (OUT/'sample-workspace.log').open('ab',buffering=0) as log:
        panel = subprocess.Popen([sys.executable,str(BASE/'configure_samples.py'),'--run',RUN,'--watch'],
            cwd=BASE,env=production,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    plan = json.loads((OUT/'prepared.json').read_text())
    plan.update(training_launched=True,preflight_passed=True,production_starts_from_zero=True)
    write('prepared.json',**plan)
    launch([*distributed,str(BASE/'train.py'),'--microbatch','64','--epochs','100'],'training',production)
    completed = json.loads((OUT/'completed.json').read_text())
    assert completed['epochs'] == 100 and completed['global_step'] == 62600
    write('launch-status.json',phase='completed',epochs=100,updates=62600)


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    try:main()
    except BaseException as error:
        write('launch-status.json',phase='failed',error_type=type(error).__name__,error=str(error))
        raise
