#!/usr/bin/env python3
"""Checkpoint the coarse trial, audit joint512, and launch a fresh bounded trial."""
import argparse
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
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/tools')]
import launch_laser_hard_ar_imagenet as launcher
from runtime.rq_imagenet_entry import freeze_shared_runtime

RUN = 'imagenet-laser-fine-joint512-scratch6-4h100-20261015'
BASE = Path('/tmp/laser-rq-fine-joint512-imagenet-20261015')
OUT = ROOT/'outputs'/RUN
SOURCE = Path('/tmp/laser-rq-native-joint-imagenet-20261015')
OLD = ROOT/'outputs/imagenet-laser-native-joint-scratch6-4h100-20261015'
RECIPE = ROOT/'configs/stage2/imagenet-laser-fine-joint512-scratch6-20261015.yaml'
launcher.BASE, launcher.OUT, launcher.RUN = BASE, OUT, RUN


def archive_sources():
    repro = OUT/'reproducibility'; repro.mkdir(exist_ok=True)
    for p in (BASE/'inputs').iterdir():
        if p.suffix in ('.yaml', '.json'): shutil.copyfile(p, repro/p.name)
    files = [BASE/'train.py', BASE/'verify.py', BASE/'audit.py', BASE/'environment.json']
    for directory in ['source', 'runtime', 'upstream']: files.extend((BASE/directory).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml')); files.extend((BASE/'inputs').glob('*.json'))
    manifest = {str(p.relative_to(BASE)): launcher.digest(p) for p in files}
    launcher.write(repro/'code-manifest.json', manifest)
    with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p, p.relative_to(BASE))
        z.write(Path(__file__), 'scripts/tools/'+Path(__file__).name)
        z.write(ROOT/'tests/test_laser_fine_joint.py', 'tests/test_laser_fine_joint.py')
    prepared = json.loads((OUT/'prepared.json').read_text())
    prepared['source_manifest_sha256'] = launcher.digest(repro/'code-manifest.json')
    launcher.write(OUT/'prepared.json', prepared)


def prepare():
    import yaml
    if (OUT/'prepared.json').exists(): raise RuntimeError('Already prepared')
    recipe = yaml.safe_load(RECIPE.read_text())
    assert recipe['run_id'] == RUN and recipe['epochs'] == 6
    assert recipe['global_batch'] == 2048 and recipe['lr_schedule_epochs'] == 100
    assert recipe['auxiliary_loss_terms'] == 0 and recipe['tokenizer']['coefficient_bins'] == 512
    (BASE/'inputs').mkdir(parents=True); OUT.mkdir(parents=True, exist_ok=True)
    freeze_shared_runtime(BASE, ROOT, include_stochastic_teacher=False)
    shutil.copytree(SOURCE/'upstream', BASE/'upstream',
        ignore=shutil.ignore_patterns('__pycache__', '.git', '*.pyc'))
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__ == '__main__':\n    launch('laser_fine_joint')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_fine_joint_checkpoint.py', BASE/'verify.py')
    shutil.copyfile(ROOT/'scripts/tools/audit_laser_fine_joint.py', BASE/'audit.py')
    for name in ['stage2-training-config.yaml', 'stage2-archive-config.yaml', 'imagenet_256_train.npz',
                 'rq-upstream-core.json']:
        shutil.copyfile(SOURCE/'inputs'/name, BASE/'inputs'/name)
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    assert config['optimizer']['init_lr'] == .0005
    config['loss'] = dict(type='joint_categorical_cross_entropy', stochastic_codes=True)
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    os.link(SOURCE/'inputs/stage1-model.pt', BASE/'inputs/stage1-model.pt')
    assert launcher.digest(BASE/'inputs/stage1-model.pt') == recipe['tokenizer']['sha256']
    shutil.copyfile(RECIPE, BASE/'inputs/recipe.yaml')
    assert launcher.digest(BASE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    for name in ['data', 'torch-cache']:
        (BASE/name).symlink_to(SOURCE/name, target_is_directory=True)
    launcher.write(BASE/'environment.json', dict(LASER_RQ_BASE=str(BASE),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        TORCH_HOME=str(BASE/'torch-cache'), WANDB_DIR=str(BASE/'wandb'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    launcher.write(OUT/'prepared.json', dict(prepared=True, run_id=RUN, recipe=recipe,
        fresh_stage2=True, fresh_optimizer=True, source_stage2_checkpoint=None,
        initial_global_step=0, initial_optimizer_state_count=0,
        train_images=1281167, global_batch=2048, updates_per_epoch=626,
        stop_updates=3756, cosine_total_updates=62600, evaluation_epochs=[1, 2, 4, 6],
        initial_lr=.0005, auxiliary_loss_terms=0, sampled_trajectories=0,
        native_transformer=True, packed_id_rule='atom*512+coefficient_bin', time=time.time()))
    archive_sources()
    print('Prepared joint512 trial', RUN, flush=True)


def run():
    import yaml
    assert json.loads((OUT/'cpu-verification.json').read_text())['passed']
    cancelled = False
    child = None
    def stop(_signum, _frame):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None: child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if active.strip():
        metadata = json.loads((OLD/'checkpoint.json').read_text())
        checkpoint = (OLD/'checkpoints/last.pt').resolve()
        assert checkpoint.stat().st_size == metadata['bytes'] and metadata['global_step'] >= 626
        alias = checkpoint.parent.parent/'before-fine-joint512.pt'
        if not alias.exists(): alias.symlink_to(os.path.relpath(checkpoint, alias.parent))
        status = json.loads((OLD/'launch-status.json').read_text())
        pid = int(status['pid'])
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'controller.py run' in command and 'native-joint-aligned' in command
        launcher.write(OUT/'previous-run-handoff.json', dict(checkpoint=metadata,
            protected_alias=str(alias), controller_pid=pid,
            last_progress=json.loads((OLD/'train/progress-rank0.json').read_text()), time=time.time()))
        os.kill(pid, signal.SIGTERM)
        for _ in range(120):
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            if not active.strip(): break
            if cancelled: return 130
            time.sleep(1)
        else: raise RuntimeError('Previous run did not release GPUs')
        status.update(phase='paused_for_fine_joint512', preserved_checkpoint_step=metadata['global_step'],
                      replacement_run=RUN, time=time.time())
        launcher.write(OLD/'launch-status.json', status)
        import wandb
        previous = wandb.Api().run('helloimlixin-rutgers/laser/'+OLD.name)
        previous.summary['training/status'] = 'paused_for_fine_joint512'
        previous.summary['training/preserved_checkpoint_step'] = metadata['global_step']
        previous.summary.update()
    environment = dict(os.environ, **json.loads((BASE/'environment.json').read_text()))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4', str(BASE/'audit.py')]
    with (OUT/'tokenizer-audit.log').open('a') as log:
        child = subprocess.Popen(command, cwd=BASE, env=environment, stdout=log, stderr=subprocess.STDOUT)
        launcher.write(OUT/'launch-status.json', dict(phase='tokenizer_fidelity_audit', pid=os.getpid(),
            child_pid=child.pid, world_size=4, run_id=RUN, time=time.time()))
        result = child.wait()
    if result: raise RuntimeError(f'Tokenizer audit failed: {result}')
    if cancelled: return 130
    fidelity = json.loads((OUT/'tokenizer-fidelity-verification.json').read_text())
    policy = json.loads((OUT/'coefficient-policy-verification.json').read_text())
    assert fidelity['passed'] and policy['passed']
    recipe = yaml.safe_load((BASE/'inputs/recipe.yaml').read_text())
    recipe['targets']['physical_sigma'] = policy['selected_sigma']
    recipe['tokenizer']['measured_rfid'] = fidelity['fine512_rfid']
    recipe['tokenizer']['source_measured_rfid'] = fidelity['source_rfid']
    encoded = yaml.safe_dump(recipe, sort_keys=False)
    (BASE/'inputs/recipe.yaml').write_text(encoded); RECIPE.write_text(encoded)
    prepared = json.loads((OUT/'prepared.json').read_text()); prepared['recipe'] = recipe
    prepared['tokenizer_fidelity'] = fidelity; prepared['coefficient_policy'] = policy
    launcher.write(OUT/'prepared.json', prepared)
    archive_sources()
    return launcher.run(resume=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    else:
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        raise SystemExit(run())
