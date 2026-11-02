#!/usr/bin/env python3
"""One paired four-epoch shared-codebook trial on all four B200s."""
import argparse
import ast
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
CONTROL = 'imagenet-laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018'
SOURCE = Path('/tmp/laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018')
CONTROL_OUT = ROOT / 'outputs' / CONTROL
CALIBRATION = ROOT / 'outputs/imagenet-laser-sharedjoint20k-depth5-codebook-20261018'
RUN = 'imagenet-laser-sharedjoint20k-depth5-uniformce-k2048-scratch4-4b200-20261018'
BASE = Path('/tmp/laser-sharedjoint20k-depth5-uniformce-k2048-scratch4-4b200-20261018')
OUT = ROOT / 'outputs' / RUN
RECIPE = ROOT / 'configs/stage2' / (RUN + '.yaml')
CHILD = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f'.{os.getpid()}.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def status(phase, **values):
    write(OUT / 'status.json', dict(phase=phase, run_id=RUN, supervisor_pid=os.getpid(),
        epochs=4, lr_schedule_epochs=100, world_size=4, hardware='4 NVIDIA B200',
        global_batch=2048, microbatch=256, accumulation_steps=2,
        objective='native_soft_target_cross_entropy', shared_codebook_across_depths=True,
        sampling=dict(temperature=1., top_k=2048, top_p=.95), time=time.time(), **values))


def initial_hashes():
    import gc
    import random
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
    sys.path[:0] = [str(BASE / 'source'), str(BASE / 'upstream'), str(BASE / 'runtime')]
    from src.training.laser_joint_recipe import JointLaserRecipe
    from rqvae.utils.config import load_config, augment_arch_defaults
    torch.set_num_threads(4)
    results = {}
    for name, folder in [('control', SOURCE), ('shared', BASE)]:
        random.seed(261001); np.random.seed(261001); torch.manual_seed(261001)
        profile = JointLaserRecipe(folder)
        aux, _ = profile.tokenizer(torch.device('cpu'))
        config = load_config(folder / 'inputs/stage2-training-config.yaml')
        arch = augment_arch_defaults(config.arch); arch.vocab_size = config.dataset.vocab_size
        model = profile.model(arch); profile.adapt(model, aux)
        h = hashlib.sha256()
        for key,value in model.state_dict().items():
            h.update(key.encode()); h.update(value.detach().contiguous().numpy().tobytes())
        results[name] = h.hexdigest()
        del model, aux, profile
        gc.collect()
    assert results['control'] == results['shared'], 'Paired initial model weights differ'
    write(OUT / 'initialization-verification.json', dict(passed=True,
        initial_weight_sha256=results['shared'], control_initial_weight_sha256=results['control'],
        stage2_checkpoint_loaded=False, fresh_optimizer=True))
    return results['shared']


def prepare():
    import yaml
    validation = json.loads((CALIBRATION / 'full-rfid.json').read_text())
    assert validation['completed'] and validation['passed'] and validation['physical_vectors_exactly_shared']
    assert json.loads((CALIBRATION / 'sharing-proof.json').read_text())['passed']
    assert not BASE.exists() and not RECIPE.exists()
    (BASE / 'inputs').mkdir(parents=True); OUT.mkdir(parents=True, exist_ok=True)
    for directory in ('source', 'runtime'):
        shutil.copytree(SOURCE / directory, BASE / directory,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for directory in ('upstream', 'data', 'torch-cache'):
        (BASE / directory).symlink_to(SOURCE / directory, target_is_directory=True)
    for path in (SOURCE / 'inputs').iterdir():
        destination = BASE / 'inputs' / path.name
        if path.name == 'joint-codebook.pt':
            shutil.copyfile(validation['codebook_path'], destination)
        elif path.suffix in ('.pt', '.npz'):
            os.link(path, destination)
        else:
            shutil.copyfile(path, destination)
    shutil.copyfile(SOURCE / 'train.py', BASE / 'train.py')
    # Add sharing validation and metadata; preserve the actual model and loss.
    for name in ('src/models/laser_joint_tokenizer.py', 'src/training/laser_shared_joint_vocabulary.py',
                 'src/training/laser_compact_joint_vocabulary.py'):
        shutil.copyfile(ROOT / name, BASE / 'source' / name)
    profile_path = BASE / 'source/src/training/laser_joint_recipe.py'
    source = profile_path.read_text()
    old = 'integer_joint_ids=True, scalar_events_per_site='
    assert source.count(old) == 1
    source = source.replace(old, "codebook_shared_across_depths=bool(self.recipe.tokenizer.get('shared_codebook_across_depths',False)),\n            " + old)
    profile_path.write_text(source)
    shared_path = BASE / 'source/src/training/rq_imagenet_shared.py'
    original_shared = shared_path.read_text()
    old_tree = ast.parse(original_shared)
    new_tree = ast.parse((ROOT / 'src/training/rq_imagenet_shared.py').read_text())
    old_view = next(node for node in old_tree.body if isinstance(node,ast.FunctionDef) and node.name=='fid_workspace')
    new_view = next(node for node in new_tree.body if isinstance(node,ast.FunctionDef) and node.name=='fid_workspace')
    old_segment = ast.get_source_segment(original_shared,old_view)
    new_segment = ast.get_source_segment((ROOT / 'src/training/rq_imagenet_shared.py').read_text(),new_view)
    shared_path.write_text(original_shared.replace(old_segment,new_segment))
    after_tree = ast.parse(shared_path.read_text())
    for name in ('fresh_model','training_backward','main','generation','preview'):
        before = next(node for node in old_tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
        after = next(node for node in after_tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
        assert ast.dump(before)==ast.dump(after), name
    recipe = yaml.safe_load((SOURCE / 'inputs/recipe.yaml').read_text())
    recipe.update(run_id=RUN, name='LASER shared joint20k depth5 | uniform CE K2048 | paired fresh4 | 4 B200',
        version='laser-shared-joint20k-v1', epochs=100, lr_schedule_epochs=100)
    recipe['tokenizer'].update(codebook_sha256=validation['codebook_sha256'],
        shared_codebook_across_depths=True, measured_rfid=validation['measured_rfid'],
        first_four_depths_unchanged=False, coefficient_values_shared=True)
    recipe['targets']['calibration'] = dict(policy='Inherited numerical target temperatures; codebook-only training comparison',
        temperatures_unchanged=True, new_shared_book_noise_verified_before_production=True)
    recipe['lineage'] = dict(control_run=CONTROL, stage2_initialization='paired_fresh_random',
        stage2_checkpoint_loaded=False, change='One identical physical sparse codebook at every depth',
        bounded_trial_epochs=4, auto_promote=False)
    assert recipe['loss'] == dict(type='soft_target_cross_entropy', reduction='mean')
    assert recipe['sampling'] == dict(temperature=1., top_k=2048, top_p=.95)
    (BASE / 'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    for name in ('stage2-training-config.yaml', 'stage2-archive-config.yaml'):
        path = BASE / 'inputs' / name; config = yaml.safe_load(path.read_text())
        if name == 'stage2-training-config.yaml':
            config['experiment']['sample'] = dict(temp=1., top_k=2048, top_p=.95)
        else:
            config['sampling'] = dict(temp=1., top_k=2048, top_p=.95)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = {key:value.replace(str(SOURCE),str(BASE)) for key,value in
        json.loads((SOURCE / 'environment.json').read_text()).items()}
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', LASER_WANDB_CHECKPOINT_UPLOADS='0')
    write(BASE / 'environment.json', env)
    for directory in ('wandb','wandb-cache','wandb-data','checkpoint-staging','checkpoint-cache'):
        (BASE / directory).mkdir(exist_ok=True)
    assert digest(BASE / 'inputs/joint-codebook.pt') == validation['codebook_sha256']
    status('verifying_paired_initialization')
    recipe.setdefault('optimization', {})['probe_initial_weight_sha256'] = initial_hashes()
    for path in (BASE / 'inputs/recipe.yaml', RECIPE):
        path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    repro = OUT / 'reproducibility'; repro.mkdir(exist_ok=True)
    for path in (BASE / 'inputs').iterdir():
        if path.suffix in ('.yaml','.json') or path.name == 'joint-codebook.pt':
            shutil.copyfile(path, repro / path.name)
    shutil.copyfile(__file__, repro / Path(__file__).name)
    shutil.copyfile(CALIBRATION / 'full-rfid.json', repro / 'shared-codebook-rfid.json')
    files = [BASE / 'train.py', BASE / 'environment.json']
    for directory in ('source','runtime','upstream'):
        files.extend(path for path in (BASE / directory).rglob('*.py') if '__pycache__' not in path.parts)
    files.extend(path for path in (BASE / 'inputs').iterdir() if path.suffix in ('.yaml','.json'))
    write(repro / 'code-manifest.json', {str(path.relative_to(BASE)):digest(path) for path in files})
    with zipfile.ZipFile(repro / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, path.relative_to(BASE))
    plan = dict(run_id=RUN, control_run=CONTROL, shared_codebook_sha256=validation['codebook_sha256'],
        shared_reconstruction_fid=validation['measured_rfid'], global_batch=2048, microbatch=256,
        epochs=4, recipe_horizon=100, lr_schedule_epochs=100, target_update=2504,
        vocabulary=20481, depth=5, tokens_per_image=320, world_size=4,
        sampling=recipe['sampling'], target_temperatures_unchanged=True,
        initial_weights_identical_to_control=True, new_model_and_optimizer=True,
        training_code_math_unchanged=True, representation='Exactly shared sparse codeword table',
        baseline_fid_by_epoch={'1':88.15302424275671,'2':69.58891028355418,'4':57.35028955878977},
        baseline_sampling='K2048, T1, top-p0.95; same world, batch, and generation seed',
        auto_promote=False, maximum_concurrent_prior_training_runs=1,
        resume_control_after_trial='Preserved handoff checkpoint; user can select which trial to continue')
    write(OUT / 'prepared.json', plan); status('prepared')


def stop_previous():
    previous = json.loads((CONTROL_OUT / 'status.json').read_text())
    assert previous['phase'] == 'training'
    target = math.ceil(previous['latest_progress']['train/global_step'] / 626) * 626
    directory = CONTROL_OUT / 'persistent-train/checkpoints/.checkpoint-data'
    while True:
        receipts = [json.loads(p.read_text()) for p in directory.glob('*.json')]
        latest = max(receipts,key=lambda row:row['global_step'])
        if latest['global_step'] >= target: break
        progress = json.loads((CONTROL_OUT / 'train/progress-rank0.json').read_text())
        status('waiting_for_control_checkpoint', target_saved_step=target,
            latest_saved_step=latest['global_step'], latest_control_step=progress['train/global_step'])
        time.sleep(10)
    assert digest(Path(latest['local'])) == latest['sha256']
    pin = BASE / 'preserved-control.pt'
    os.link(latest['local'],pin)
    controller = previous['supervisor_pid']
    assert b'adopt_joint20k_k2048_epoch2.py' in Path(f'/proc/{controller}/cmdline').read_bytes()
    write(OUT / 'control-handoff.json', dict(preserved_checkpoint=latest,
        preserved_local_pin=str(pin), previous_supervisor_pid=controller, reason='One shared-codebook training experiment'))
    os.kill(controller,signal.SIGTERM)
    for _ in range(55):
        occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not occupied: break
        time.sleep(1)
    assert not occupied, occupied
    write(CONTROL_OUT / 'status.json', dict(phase='preserved_for_shared_codebook_trial',
        preserved_checkpoint=latest, replacement_run=RUN, optimizer_state_preserved=True, time=time.time()))


def stop(signum, _frame):
    if CHILD is not None and CHILD.poll() is None: os.killpg(CHILD.pid, signal.SIGINT)
    raise SystemExit(128+signum)


def child(command, env, phase):
    global CHILD
    with (OUT / (phase + '.log')).open('ab',buffering=0) as log:
        CHILD = subprocess.Popen(command,cwd=BASE,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            progress_path = BASE / 'live-training/train/progress-rank0.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            status(phase, child_pid=CHILD.pid, latest_progress=progress, command=command)
            try: code=CHILD.wait(timeout=20); break
            except subprocess.TimeoutExpired: pass
    if code: raise RuntimeError(f'{phase} exited {code}; inspect {OUT/(phase+".log")}')


def run():
    lock=(BASE/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'train/progress-rank0.json').exists()
    stop_previous()
    env=dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    for key in ('WANDB_SERVICE','_WANDB_SERVICE'):env.pop(key,None)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(BASE/'train.py'),'--epochs','4','--microbatch','256']
    smoke=BASE/'private-preflight'
    child([*command,'--smoke-updates','1'],dict(env,LASER_PRIVATE_SMOKE_DIR=str(smoke)),'preflight')
    for rank in range(4):
        startup=json.loads((smoke/f'startup-rank{rank}.json').read_text())
        assert startup['fresh_stage2'] and startup['fresh_optimizer'] and startup['global_step']==0
        assert startup['training_epochs']==4 and startup['lr_schedule_epochs']==100
        assert startup['global_batch']==2048 and startup['microbatch']==256
        assert startup['objective']=='native_soft_target_cross_entropy' and startup['depth_loss_weights'] is None
        assert startup['codebook_shared_across_depths'] and startup['integer_joint_ids']
        assert startup['native_sampling']==dict(temperature=1.,top_k=[2048],top_p=[.95])
        assert json.loads((smoke/f'preview-verification-rank{rank}.json').read_text())['passed']
    proof=json.loads((smoke/'verification.json').read_text())
    assert proof['passed'] and proof['global_step']==proof['all_adam_steps']==1
    assert json.loads((smoke/'initial-weight-verification.json').read_text())['passed']
    write(OUT/'preflight.json',dict(passed=True,proof=proof,preflight_weights_discarded=True,
        production_starts_from_zero=True,shared_codebook_verified=True))
    write(ROOT/'outputs/imagenet-laser-joint20k-active.json',dict(run_id=RUN,base=str(BASE),
        url=f'https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/{RUN}',
        shared_codebook_across_depths=True,epochs=4,lr_schedule_epochs=100,
        sampling=dict(temperature=1.,top_k=2048,top_p=.95),world_size=4,
        hardware='4 NVIDIA B200',preserved_control=CONTROL,time=time.time()))
    child(command,env,'training')
    completed=json.loads((OUT/'completed.json').read_text())
    assert completed['epochs']==4 and completed['global_step']==2504
    plan=json.loads((OUT/'prepared.json').read_text()); rows=[]
    for epoch in (1,2,4):
        value=json.loads((OUT/f'evaluations/epoch{epoch:03d}.json').read_text())
        assert value['generated_images']==50000 and value['real_images']==1281167
        baseline=plan['baseline_fid_by_epoch'][str(epoch)]
        rows.append(dict(epoch=epoch,control_fid=baseline,shared_fid=value['eval/fid'],
            improvement=baseline-value['eval/fid']))
    write(OUT/'comparison.json',dict(completed=True,results=rows,auto_promoted=False,
        interpretation='One paired training seed and fixed K2048 sampler; long-horizon behavior untested'))
    status('completed',results=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','run'))
    args=parser.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try: {'prepare':prepare,'run':run}[args.phase]()
    except BaseException as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise
