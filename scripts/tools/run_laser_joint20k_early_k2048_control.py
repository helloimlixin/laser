#!/usr/bin/env python3
"""Verify immutable epoch-one/two checkpoints and supervise their FID50k control."""
import copy
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

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-rq-joint20k-depth5-imagenet-20261017')
OLD = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
NEW = ROOT/'outputs/imagenet-laser-joint20k-depth5-rqnoise-k2048-scratch100-4h100-20261017'
CURRENT = Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-imagenet-20261017')
OUT = ROOT/'outputs/imagenet-laser-joint20k-depth5-epochs1-2-k2048-control-20261017'
LOCAL = Path('/tmp/laser-joint20k-depth5-early-k2048-control-20261017')


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def write(name,**values):
    path = OUT/name
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(values,time=time.time()),indent=2)+'\n')
    temporary.replace(path)


def main():
    import torch
    import yaml
    torch.set_num_threads(4)
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT/'plan.json').exists():
        assert '--resume' in sys.argv,'Control already prepared; use --resume'
        return resume()
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'wandb').mkdir(exist_ok=True)
    write('status.json',phase='verifying_checkpoints',pid=os.getpid())
    old_recipe = yaml.safe_load((BASE/'inputs/recipe.yaml').read_text())
    new_recipe = yaml.safe_load((CURRENT/'inputs/recipe.yaml').read_text())
    before,after = copy.deepcopy(old_recipe),copy.deepcopy(new_recipe)
    for value in (before,after):
        for key in ('version','run_id','name','sampling'):value.pop(key)
        value['targets'].pop('temperature');value['targets'].pop('calibration')
    assert before == after
    assert old_recipe['sampling'] == dict(temperature=1.,top_k=1024,top_p=.95)
    assert new_recipe['sampling'] == dict(temperature=1.,top_k=2048,top_p=.95)
    for folder,output in [(BASE,OLD),(CURRENT,NEW)]:
        manifest = json.loads((output/'reproducibility/code-manifest.json').read_text())['files']
        assert all(sha(folder/name) == expected for name,expected in manifest.items())
    records = [json.loads(p.read_text()) for p in (OLD/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    checkpoints = []
    for epoch in (1,2):
        receipt = next(r for r in records if r['global_step'] == epoch*626)
        source = Path(receipt['local'])
        assert source.stat().st_size == Path(receipt['persistent']).stat().st_size == receipt['bytes']
        assert sha(source) == receipt['sha256']
        saved = torch.load(source,map_location='cpu',mmap=True,weights_only=False)
        assert saved['run_id'] == OLD.name and saved['global_step'] == epoch*626
        assert saved['next_epoch'] == epoch and saved['next_microbatch'] == 0
        assert saved['recipe'] == old_recipe and len(saved['rng_state_by_rank']) == 4
        assert saved['scheduler']['after']['last_epoch'] == epoch*626
        assert all(int(s['step']) == epoch*626 for s in saved['optimizer']['state'].values())
        evaluations = [json.loads((output/f'evaluations/epoch{epoch:03d}.json').read_text())
                       for output in (OLD,NEW,ROOT/'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014')]
        assert saved['evaluation']['fid'] == receipt['fid'] == evaluations[0]['eval/fid']
        for evaluation in evaluations:
            assert evaluation['generated_images'] == 50000 and evaluation['real_images'] == 1281167
            assert evaluation['real_split'] == 'train' and evaluation['train/global_step'] == epoch*626
        del saved
        local = LOCAL/f'epoch{epoch:03d}.pt'
        assert not local.exists()
        os.link(source,local)
        checkpoints.append(dict(epoch=epoch,global_step=epoch*626,receipt=receipt,local_checkpoint=str(local),
            old_evaluation=evaluations[0],new_evaluation=evaluations[1],original_rq_evaluation=evaluations[2]))
    (OUT/'source').mkdir(exist_ok=True)
    sources = {'evaluate.py':ROOT/'scripts/tools/evaluate_laser_joint20k_early_k2048_control.py',
        'controller.py':Path(__file__),
        'joint_integer_sampling_audit.py':ROOT/'src/training/joint_integer_sampling_audit.py'}
    for name,path in sources.items():shutil.copyfile(path,OUT/'source'/name)
    write('source-manifest.json',files={name:sha(OUT/'source'/name) for name in sources})
    write('plan.json',wandb_id=OUT.name,checkpoints=checkpoints,old_training_run=OLD.name,
        new_training_run=NEW.name,sampling=dict(temperature=1.,top_k=2048,top_p=.95),
        generated_images_per_checkpoint=50000,reference_images=1281167,reference_split='train',
        reference_sha256=sha(BASE/'inputs/imagenet_256_train.npz'),seed=261001,world_size=4,
        sampling_batch_per_gpu=128,memory_cap_gib_per_gpu=16,
        training_updates_added=0,active_training_settings_changed=False,
        active_recipe_sha256=sha(CURRENT/'inputs/recipe.yaml'),
        active_manifest_sha256=sha(NEW/'reproducibility/code-manifest.json'),
        source_recipe_sha256=sha(BASE/'inputs/recipe.yaml'),
        old_noise_temperatures=old_recipe['targets']['temperature'],new_noise_temperatures=new_recipe['targets']['temperature'],
        all_other_training_behavior_identical=True,tokenizer_codebook_sha256=old_recipe['tokenizer']['codebook_sha256'])
    environment = dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    environment.update(WANDB_DIR=str(LOCAL/'wandb'),PYTHONPATH='.:runtime:scripts/tools',
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip())
    child = None

    def stop(*_):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT)
        raise KeyboardInterrupt('Control interrupted; current training stays running')

    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (OUT/'evaluation.log').open('ab',buffering=0) as log:
        child = subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            str(OUT/'source/evaluate.py')],env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('status.json',phase='evaluation_starting',pid=os.getpid(),child_pid=child.pid)
        code = child.wait()
    if code:raise RuntimeError(f'Control evaluation returned {code}')
    report = json.loads((OUT/'report.json').read_text());assert report['completed']
    plan = report['plan']
    assert sha(CURRENT/'inputs/recipe.yaml') == plan['active_recipe_sha256']
    assert sha(NEW/'reproducibility/code-manifest.json') == plan['active_manifest_sha256']
    assert sha(BASE/'inputs/recipe.yaml') == plan['source_recipe_sha256']
    write('final-verification.json',passed=True,both_checkpoints_sha256_verified=True,
        both_baseline_panels_reproduced_pixel_exact=True,generated_images_per_checkpoint=50000,
        reference_images=1281167,active_training_settings_unchanged=True,
        active_progress=json.loads((NEW/'train/progress-rank0.json').read_text()))
    print(json.dumps(dict(completed=True,results=report['results'],gain_shrink=report['gain_shrink'])),flush=True)


def resume():
    plan = json.loads((OUT/'plan.json').read_text())
    assert sha(CURRENT/'inputs/recipe.yaml') == plan['active_recipe_sha256']
    assert sha(NEW/'reproducibility/code-manifest.json') == plan['active_manifest_sha256']
    for item in plan['checkpoints']:
        assert sha(item['local_checkpoint']) == item['receipt']['sha256']
    sources = {'evaluate.py':ROOT/'scripts/tools/evaluate_laser_joint20k_early_k2048_control.py',
        'controller.py':Path(__file__)}
    for name,path in sources.items():shutil.copyfile(path,OUT/'source'/name)
    write('resume-source-manifest.json',files={name:sha(OUT/'source'/name) for name in sources},
        reason='Dashboard name validation; completed FID50k is reused, remaining checkpoint evaluated')
    # Validate the dashboard name locally before spending more GPU time.
    import wandb_workspaces.workspaces as ws
    ws.Workspace(entity='helloimlixin-rutgers',project='laser',
                 name='LASER depth5 early sampler and noise comparison')
    environment = dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    environment.update(WANDB_DIR=str(LOCAL/'wandb'),PYTHONPATH='.:runtime:scripts/tools',
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip())
    child = None

    def stop(*_):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT)
        raise KeyboardInterrupt('Control interrupted; current training stays running')

    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (OUT/'evaluation-resume.log').open('ab',buffering=0) as log:
        child = subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            str(OUT/'source/evaluate.py')],env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('status.json',phase='evaluation_resuming',pid=os.getpid(),child_pid=child.pid)
        code = child.wait()
    if code:raise RuntimeError(f'Control evaluation returned {code}')
    report = json.loads((OUT/'report.json').read_text());assert report['completed']
    assert sha(CURRENT/'inputs/recipe.yaml') == plan['active_recipe_sha256']
    assert sha(NEW/'reproducibility/code-manifest.json') == plan['active_manifest_sha256']
    assert sha(BASE/'inputs/recipe.yaml') == plan['source_recipe_sha256']
    write('final-verification.json',passed=True,both_checkpoints_sha256_verified=True,
        both_baseline_panels_reproduced_pixel_exact=True,generated_images_per_checkpoint=50000,
        reference_images=1281167,active_training_settings_unchanged=True,
        active_progress=json.loads((NEW/'train/progress-rank0.json').read_text()))
    print(json.dumps(dict(completed=True,results=report['results'],gain_shrink=report['gain_shrink'])),flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    try:main()
    except BaseException as error:
        write('status.json',phase='failed',error_type=type(error).__name__,error=str(error))
        raise
