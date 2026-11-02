#!/usr/bin/env python3
"""Preserve the four-depth run and start a verified five-depth native RQ prior."""
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
RUN = 'imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
OUT = ROOT/'outputs'/RUN
BASE = Path('/tmp/laser-rq-joint20k-depth5-imagenet-20261017')
SOURCE = Path('/tmp/laser-rq-joint20k-imagenet-20261017')
OLD = ROOT/'outputs/imagenet-laser-joint20k-scratch2-4h100-20261017'
STUDY = ROOT/'outputs/imagenet-laser-joint20k-refinement-rfid-20261017'
RECIPE = ROOT/'configs/stage2/imagenet-laser-joint20k-depth5-scratch100-20261017.yaml'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, **values):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(values, time=time.time()), indent=2)+'\n')
    temporary.replace(path)


def matches(pid, token):
    try:
        return token.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def prepare():
    import numpy as np
    import torch
    import yaml
    recipe = yaml.safe_load(RECIPE.read_text())
    assert recipe['epochs'] == recipe['lr_schedule_epochs'] == 100
    assert recipe['tokenizer']['depth'] == 5 and recipe['auxiliary_loss_terms'] == 0
    assert len(recipe['targets']['temperature']) == 5
    calibration = json.loads((OUT/'target-temperature-calibration.json').read_text())
    assert calibration['passed'] and recipe['targets']['temperature'] == calibration['temperatures']
    report = json.loads((STUDY/'full-rfid.json').read_text())
    selected = report['results']['5']
    assert report['completed'] and report['real_images'] == report['reconstructed_images_per_candidate'] == 50000
    assert selected['measured_rfid'] == recipe['tokenizer']['measured_rfid']
    assert sha(Path(selected['codebook_path'])) == recipe['tokenizer']['codebook_sha256']
    assert sha(SOURCE/'inputs/stage1-model.pt') == recipe['tokenizer']['sha256']
    assert sha(SOURCE/'inputs/imagenet_256_train.npz') == recipe['evaluation']['reference_sha256']
    original = torch.load(SOURCE/'inputs/joint-codebook.pt', map_location='cpu', weights_only=True)
    book = torch.load(selected['codebook_path'], map_location='cpu', weights_only=True)
    assert book['atom_ids'].shape == book['coefficients'].shape == (5, 20481)
    assert torch.equal(book['dictionary'], original['dictionary'])
    for key in ('atom_ids', 'coefficients'):
        assert torch.equal(book[key][:4], original[key])
    manifest = json.loads((OLD/'reproducibility/code-manifest.json').read_text())['files']
    assert all(sha(SOURCE/name) == expected for name, expected in manifest.items())
    assert not BASE.exists(), 'Runtime already exists'
    (BASE/'inputs').mkdir(parents=True)
    # Reuse the exact prior runtime. Only the LASER factory's depth handling
    # changes; upstream RQ attention, transformer and training loop stay intact.
    for directory in ('source', 'runtime'):
        for path in (SOURCE/directory).rglob('*.py'):
            destination = BASE/path.relative_to(SOURCE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    shutil.copyfile(ROOT/'src/training/laser_joint_recipe.py', BASE/'source/src/training/laser_joint_recipe.py')
    for name in ('upstream', 'data', 'torch-cache'):
        (BASE/name).symlink_to((SOURCE/name).resolve(), target_is_directory=True)
    for name in ('stage1-model.pt', 'imagenet_256_train.npz'):
        (BASE/'inputs'/name).symlink_to((SOURCE/'inputs'/name).resolve())
    for name in ('stage2-training-config.yaml', 'stage2-archive-config.yaml', 'rq-upstream-core.json'):
        shutil.copyfile(SOURCE/'inputs'/name, BASE/'inputs'/name)
    shutil.copyfile(selected['codebook_path'], BASE/'inputs/joint-codebook.pt')
    shutil.copyfile(RECIPE, BASE/'inputs/recipe.yaml')
    fitting = json.loads((ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017/calibration-data.json').read_text())
    excluded = set(fitting['fit_indices']+fitting['heldout_training_indices'])
    indices = [int(i) for i in np.random.default_rng(26101720485).permutation(1281167)
               if int(i) not in excluded][:32]
    assert set(indices).isdisjoint(calibration['training_indices'])
    (BASE/'inputs/target-audit.json').write_text(json.dumps(dict(training_indices=indices,
        disjoint_from_fit=True, disjoint_from_temperature_calibration=True), indent=2)+'\n')
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss'] = dict(type='soft_target_cross_entropy', stochastic_codes=True,
                          temp=recipe['targets']['temperature'])
    config['experiment'].update(epochs=100, batch_size=64, total_batch_size=2048,
                                save_ckpt_freq=1, test_freq=2)
    assert config['optimizer']['init_lr'] == .0005 and config['optimizer']['warmup']['epoch'] == 0
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (BASE/'train.py').write_text("from runtime.rq_imagenet_entry import launch\nif __name__=='__main__':\n    launch('laser_joint')\n")
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_joint20k_depth5_smoke.py', BASE/'verify.py')
    shutil.copyfile(ROOT/'scripts/tools/configure_rq_sample_panels.py', BASE/'configure_samples.py')
    environment = json.loads((SOURCE/'environment.json').read_text())
    environment = {key: value.replace(str(SOURCE), str(BASE)) for key, value in environment.items()}
    (BASE/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    files = [BASE/name for name in ('train.py', 'verify.py', 'configure_samples.py', 'environment.json')]
    for directory in ('source', 'runtime'):
        files.extend((BASE/directory).rglob('*.py'))
    files += list((BASE/'inputs').glob('*.yaml'))+list((BASE/'inputs').glob('*.json'))
    files.append(BASE/'inputs/joint-codebook.pt')
    write('reproducibility/code-manifest.json', files={str(p.relative_to(BASE)): sha(p) for p in files})
    for name, source in [('recipe.yaml', BASE/'inputs/recipe.yaml'),
                         ('full-rfid.json', STUDY/'full-rfid.json'),
                         ('target-temperature-calibration.json', OUT/'target-temperature-calibration.json'),
                         ('stage2-training-config.yaml', BASE/'inputs/stage2-training-config.yaml')]:
        shutil.copyfile(source, OUT/'reproducibility'/name)
    with zipfile.ZipFile(OUT/'reproducibility/source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(BASE))
        for path in (Path(__file__), ROOT/'scripts/tools/calibrate_laser_joint20k_depth5.py'):
            archive.write(path, path.relative_to(ROOT))
    write('prepared.json', prepared=True, training_launched=False, run_id=RUN, recipe=recipe,
          fresh_stage2=True, fresh_optimizer=True, stage2_checkpoint_loaded=False,
          initial_global_step=0, world_size=4, global_batch=2048, microbatch=64,
          accumulation_steps=8, optimizer_updates_per_epoch=626, epochs=100,
          stop_after_updates=62600, cosine_total_updates=62600, tokens_per_image=320,
          measured_reconstruction_rfid=selected['measured_rfid'], vocabulary=20481,
          first_four_tokenizer_depths_unchanged=True, previous_training_run=OLD.name,
          code_manifest_sha256=sha(OUT/'reproducibility/code-manifest.json'))


def preserve_and_stop():
    import torch
    torch.set_num_threads(4)
    # Allow the epoch already in progress to finish while CPU preparation runs.
    # Stop only after the complete immutable epoch-three checkpoint is committed.
    deadline = time.monotonic()+1800
    while True:
        receipts = [json.loads(p.read_text()) for p in
                    (OLD/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
        receipt = max(receipts, key=lambda row: row['global_step'])
        if receipt['global_step'] >= 1878:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Waiting for the current run to finish epoch three')
        write('launch-status.json', phase='waiting_for_current_epoch_checkpoint',
              pid=os.getpid(), target_step=1878, latest_saved_step=receipt['global_step'])
        time.sleep(5)
    write('launch-status.json', phase='verifying_previous_checkpoint', pid=os.getpid(),
          saved_step=receipt['global_step'])
    for key in ('local', 'persistent'):
        path = Path(receipt[key])
        assert path.stat().st_size == receipt['bytes']
        assert sha(path) == receipt['sha256'], key
    saved = torch.load(receipt['local'], map_location='cpu', mmap=True, weights_only=False)
    step = receipt['global_step']
    assert saved['run_id'] == OLD.name and saved['global_step'] == step
    assert saved['next_microbatch'] == 0 and len(saved['rng_state_by_rank']) == 4
    assert all(int(state['step']) == step for state in saved['optimizer']['state'].values())
    del saved
    preserved = OLD/f'epoch{receipt["epoch"]:03d}-before-depth5.pt'
    preserved.symlink_to(Path(receipt['persistent']).relative_to(OLD))
    local = SOURCE/f'epoch{receipt["epoch"]:03d}-before-depth5.pt'
    os.link(receipt['local'], local)
    status = json.loads((OLD/'launch-status.json').read_text())
    controller = status['pid']
    progress = [json.loads((OLD/f'train/progress-rank{rank}.json').read_text()) for rank in range(4)]
    pids = [row['pid'] for row in progress]
    assert matches(controller, 'continue_laser_joint20k.py')
    assert all(matches(pid, str(SOURCE/'train.py')) for pid in pids)
    observed = max(row['train/global_step'] for row in progress)
    write('preserved-source-checkpoint.json', passed=True, receipt=receipt,
          preserved_alias=str(preserved), local_alias=str(local),
          local_and_persistent_sha256_verified=True, latest_observed_step=observed,
          unfinished_updates_not_in_checkpoint=observed-step)
    os.kill(controller, signal.SIGTERM)
    for second in range(45):
        live = [pid for pid in pids if matches(pid, str(SOURCE/'train.py'))]
        if not live and not matches(controller, 'continue_laser_joint20k.py'):
            break
        if second in (10, 25, 40):
            for pid in live:
                os.kill(pid, signal.SIGINT if second == 10 else signal.SIGTERM if second == 25 else signal.SIGKILL)
        time.sleep(1)
    assert not any(matches(pid, str(SOURCE/'train.py')) for pid in pids)
    assert not matches(controller, 'continue_laser_joint20k.py')
    paused = dict(phase='paused_for_five_token_variant', preserved_step=step,
                  latest_observed_step=observed, replacement_run=RUN, time=time.time())
    (OLD/'launch-status.json').write_text(json.dumps(paused, indent=2)+'\n')
    import wandb
    previous = wandb.Api(timeout=45).run(f'helloimlixin-rutgers/laser/{OLD.name}')
    previous.summary.update({'training/status': paused['phase'],
                             'training/preserved_checkpoint_step': step,
                             'training/replacement_run': RUN})
    previous.update()


def main():
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'prepared.json').exists(), 'Trial already prepared'
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    child = panel = None

    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
        if panel is not None and panel.poll() is None:
            panel.terminate()
        raise KeyboardInterrupt('Five-token trial interrupted')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(command, phase, env):
        nonlocal child
        with (OUT/(phase+'.log')).open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write('launch-status.json', phase=phase, pid=os.getpid(), child_pid=child.pid,
                  run_id=RUN, epochs=100, stop_step=62600)
            result = child.wait()
        if result:
            raise RuntimeError(f'{phase} failed with exit code {result}')

    prepare()
    launch([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            'tests/test_laser_joint_refinement.py', 'tests/test_laser_joint_compact_runtime.py',
            'tests/test_laser_joint_vocabulary.py',
            '--junitxml='+str(OUT/'reproducibility/tests.xml')], 'cpu_verification',
           dict(environment, PYTHONPATH='.:runtime:scripts/tools'))
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    preserve_and_stop()
    for _ in range(30):
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if not active:
            break
        time.sleep(2)
    assert not active, 'Unrelated GPU work remains'
    production = dict(environment, **json.loads((BASE/'environment.json').read_text()),
                      WANDB_API_KEY=os.environ['WANDB_API_KEY'])
    smoke = dict(production, LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke'))
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4']
    for name, expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text())['files'].items():
        assert sha(BASE/name) == expected
    launch([*distributed, str(BASE/'train.py'), '--microbatch', '64', '--smoke-updates', '2'],
           'gpu_preflight', smoke)
    launch([sys.executable, str(BASE/'verify.py')], 'checkpoint_verification', smoke)
    proof = json.loads((OUT/'smoke-reload-verification.json').read_text())
    assert proof['passed'] and proof['depth'] == 5 and proof['cached_events_verified'] == 320
    with (OUT/'sample-workspace.log').open('ab', buffering=0) as log:
        panel = subprocess.Popen([sys.executable, str(BASE/'configure_samples.py'), '--run', RUN, '--watch'],
                                 cwd=BASE, env=production, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    plan = json.loads((OUT/'prepared.json').read_text())
    plan.update(training_launched=True, preflight_passed=True, production_starts_from_zero=True)
    write('prepared.json', **plan)
    launch([*distributed, str(BASE/'train.py'), '--microbatch', '64', '--epochs', '100'],
           'training', production)
    completed = json.loads((OUT/'completed.json').read_text())
    assert completed['epochs'] == 100 and completed['global_step'] == 62600
    write('launch-status.json', phase='completed', epochs=100, updates=62600)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except BaseException as error:
        write('launch-status.json', phase='failed', error_type=type(error).__name__, error=str(error))
        raise
