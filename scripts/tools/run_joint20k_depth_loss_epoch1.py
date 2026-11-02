#!/usr/bin/env python3
"""Supervise two paired epoch-1-to-3 loss trials on all eight H100s."""
import argparse
import ast
import copy
import difflib
import fcntl
import hashlib
import importlib.util
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
MAIN_RUN = 'imagenet-laser-joint20k-depth5-earlierbest-continue100-8h100-20261018'
MAIN_OUT = ROOT / 'outputs' / MAIN_RUN
MAIN_BASE = Path('/tmp/laser-joint20k-earlier-best-8h100-20261018')
BASE = Path('/tmp/laser-joint20k-depth-loss-epoch1-20261018')
OUT = ROOT / 'outputs/imagenet-laser-joint20k-depth-loss-epoch1to3-20261018'
SOURCE_RUN = 'imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
ARMS = {
    'control': ('imagenet-laser-joint20k-depth5-ce-epoch1to3-8h100-20261018', [1.] * 5),
    'weighted': ('imagenet-laser-joint20k-depth5-depthwce-epoch1to3-8h100-20261018', [1.5, 1.25, 1., .75, .5]),
}
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key')
CURRENT_CHILD = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Unexpected frozen runtime around {before[:90]!r}')
    return text.replace(before, after)


def method(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ''.join(source.splitlines(True)[node.lineno - 1:node.end_lineno]) + '\n'
    raise ValueError(name)


def patch_recipe(source, implementation):
    old = method(source, 'loss')
    new = method(implementation, 'loss') + '\n' + method(implementation, 'depth_loss_weights')
    source = replace_once(source, old, new)
    return replace_once(source,
        "objective='native_soft_target_cross_entropy', auxiliary_loss_terms=0,",
        "objective=('native_soft_target_cross_entropy' if self.depth_loss_weights() == [1.] * self.shape[-1] "
        "else 'depth_weighted_soft_target_cross_entropy'), "
        "depth_loss_weights=self.depth_loss_weights(), auxiliary_loss_terms=0,")


def patch_driver(source):
    source = replace_once(source,
        'if epoch==0 or (epoch+1)%int(config.experiment.test_freq)==0:',
        'if epoch==0 or (epoch+1)%int(config.experiment.test_freq)==0 or epoch+1==args.epochs:')
    source = replace_once(source,
        "                if run is not None:\n                    uploads=BASE/'uploads/checkpoints'",
        "                if run is not None and os.environ.get('LASER_WANDB_CHECKPOINT_UPLOADS', '1') != '0':\n"
        "                    uploads=BASE/'uploads/checkpoints'")
    source = replace_once(source,
        "                    verified_targets=True\n",
        "                    from src.training.depth_loss_pilot import verify_first_batch\n"
        "                    write_json(folder/f'loss-pairing-rank{rank}.json',verify_first_batch(targets,codes,images))\n"
        "                    verified_targets=True\n")
    source = replace_once(source,
        "        if evaluation is not None or (epoch+1)%int(config.experiment.save_ckpt_freq)==0:\n",
        "        if epoch+1==args.epochs:\n"
        "            confirmation=generation(model,aux,device,rank,world,epoch+1,None,seed=261101)\n"
        "            assert PROFILE.digest(aux)==before\n"
        "            if rank==0:\n"
        "                write_json(OUT/'evaluations/confirmation-seed261101.json',dict(confirmation,\n"
        "                    epoch=epoch+1,global_step=global_step,seed=261101,generated_images=50000,\n"
        "                    real_images=1281167,real_split='train',rejection_sampling=False))\n"
        "                if run:run.log({'train/epoch':epoch+1,'train/global_step':global_step,\n"
        "                    'eval/fid_confirmation':confirmation['fid']})\n"
        "        if evaluation is not None or (epoch+1)%int(config.experiment.save_ckpt_freq)==0:\n")
    return source


def assert_recipe_pair(control, weighted):
    a, b = copy.deepcopy(control), copy.deepcopy(weighted)
    for value in (a, b):
        for key in ('run_id', 'name', 'lineage', 'loss'):
            value.pop(key, None)
    if a != b or control['loss']['depth_weights'] != [1.] * 5:
        raise ValueError('Trial recipes differ outside identity and depth loss weights')
    if weighted['loss']['depth_weights'] != ARMS['weighted'][1]:
        raise ValueError('Unexpected candidate weights')


def assert_depth_weights_match(actual, expected):
    # Normalization can turn 1.0 into 0.9999999999999999 in JSON metadata.
    if (len(actual) != len(expected) or
            any(not math.isfinite(a) or not math.isclose(a, e, rel_tol=1.e-12, abs_tol=0.)
                for a, e in zip(actual, expected))):
        raise ValueError(f'Preflight loss weights differ: {actual!r} versus {expected!r}')


def validate_preflight(base, weights):
    verification = json.loads((base / 'private-smoke/verification.json').read_text())
    assert verification['passed'] and verification['global_step'] == 627
    assert verification['optimizer_state_count'] == 780 and verification['all_adam_steps'] == 627
    assert verification['frozen_tokenizer_unchanged']
    for rank in range(8):
        row = json.loads((base / f'private-smoke/startup-rank{rank}.json').read_text())
        assert row['world_size'] == 8 and row['accumulation_steps'] == 4 and row['global_step'] == 626
        assert_depth_weights_match(row['depth_loss_weights'], weights)
        proof = json.loads((base / f'private-smoke/loss-pairing-rank{rank}.json').read_text())
        assert proof['passed'] and proof['cuda'] and proof['training_rng_unchanged']


def arm_completed(info):
    folder = Path(info['out'])
    path = folder / 'completed.json'
    if not path.exists():
        return False
    assert json.loads(path.read_text())['global_step'] == 1878
    for epoch in (2, 3):
        row = json.loads((folder / f'evaluations/epoch{epoch:03d}.json').read_text())
        assert row['train/global_step'] == epoch * 626 and row['generated_images'] == 50000
    row = json.loads((folder / 'evaluations/confirmation-seed261101.json').read_text())
    assert row['global_step'] == 1878 and row['generated_images'] == 50000
    return True


def prepare():
    import yaml
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        raise RuntimeError('Experiment already prepared')
    write(OUT / 'original-active.json', json.loads((ROOT / 'outputs/imagenet-laser-joint20k-active.json').read_text()))
    receipt = json.loads((BASE / 'source-receipt.json').read_text())
    assert receipt['global_step'] == 626 and receipt['epoch'] == 1
    assert sha(Path(receipt['local'])) == receipt['sha256']
    frozen_manifest = json.loads((MAIN_OUT / 'reproducibility/code-manifest.json').read_text())
    assert all(sha(MAIN_BASE / name) == digest for name, digest in frozen_manifest.items())
    recipes, arm_plans = {}, {}
    for arm, (run, weights) in ARMS.items():
        base, out = BASE / arm, ROOT / 'outputs' / run
        if base.exists() or out.exists():
            raise RuntimeError(f'Refusing to overwrite {arm}')
        base.mkdir(); out.mkdir()
        for directory in ('source', 'runtime'):
            shutil.copytree(MAIN_BASE / directory, base / directory,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for directory in ('upstream', 'data', 'torch-cache'):
            (base / directory).symlink_to(MAIN_BASE / directory, target_is_directory=True)
        (base / 'inputs').mkdir()
        for path in (MAIN_BASE / 'inputs').iterdir():
            if path.suffix in ('.pt', '.npz'):
                os.link(path, base / 'inputs' / path.name)
            else:
                shutil.copyfile(path, base / 'inputs' / path.name)
        shutil.copyfile(MAIN_BASE / 'train.py', base / 'train.py')
        recipe = yaml.safe_load((MAIN_BASE / 'inputs/recipe.yaml').read_text())
        recipe.update(run_id=run, name=f'LASER joint20k | {arm} CE | epoch1 to3 | 8 H100',
            lineage=dict(source_run=SOURCE_RUN, source_step=626, source_epoch=1,
                source_checkpoint_sha256=receipt['sha256'], hypothesis='Depth-weighted CE',
                stage2_initialization='continued', experiment=str(OUT)))
        recipe['loss'] = dict(depth_weights=weights)
        recipes[arm] = recipe
        (base / 'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
        config = yaml.safe_load((base / 'inputs/stage2-training-config.yaml').read_text())
        config['loss']['depth_weights'] = weights
        (base / 'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        changes = []
        for name, transform in (
            ('src/training/laser_joint_recipe.py', lambda s: patch_recipe(s, (ROOT / 'src/training/laser_joint_recipe.py').read_text())),
            ('src/training/rq_imagenet_shared.py', patch_driver),
        ):
            path = base / 'source' / name
            previous = path.read_text(); updated = transform(previous)
            ast.parse(updated)
            path.write_text(updated)
            changes.extend(difflib.unified_diff(previous.splitlines(True), updated.splitlines(True),
                fromfile='main/' + name, tofile=arm + '/' + name))
        for name in ('depth_weighted_ce.py', 'depth_loss_pilot.py'):
            shutil.copyfile(ROOT / 'src/training' / name, base / 'source/src/training' / name)
        env = {k: v.replace(str(MAIN_BASE), str(base)) for k, v in
               json.loads((MAIN_BASE / 'environment.json').read_text()).items()}
        env['LASER_WANDB_CHECKPOINT_UPLOADS'] = '0'
        write(base / 'environment.json', env)
        repro = out / 'reproducibility'; repro.mkdir()
        (repro / 'runtime.patch').write_text(''.join(changes))
        for name in ('recipe.yaml', 'stage2-training-config.yaml'):
            shutil.copyfile(base / 'inputs' / name, repro / name)
        files = [base / 'train.py', base / 'environment.json']
        for directory in ('source', 'runtime', 'upstream'):
            files.extend(p for p in (base / directory).rglob('*.py') if '__pycache__' not in p.parts)
        files.extend(p for p in (base / 'inputs').iterdir() if p.suffix in ('.yaml', '.json'))
        write(repro / 'code-manifest.json', {str(p.relative_to(base)): sha(p) for p in files})
        with zipfile.ZipFile(repro / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(base))
        arm_plans[arm] = dict(run_id=run, base=str(base), out=str(out), depth_weights=weights,
            url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run}')
    assert_recipe_pair(recipes['control'], recipes['weighted'])
    shutil.copyfile(__file__, OUT / Path(__file__).name)
    write(OUT / 'plan.json', dict(source=receipt, source_run=SOURCE_RUN, main_run=MAIN_RUN,
        arms=arm_plans, additional_epochs=2, initial_epoch=1, final_epoch=3,
        initial_step=626, final_step=1878, updates_per_arm=1252,
        world_size=8, microbatch=64, accumulation_steps=4, global_batch=2048,
        lr_schedule_epochs=100, cosine_updates=62600, sampling=dict(temperature=1., top_k=2048, top_p=.95),
        fid_images=50000, real_images=1281167, real_split='train', rejection_sampling=False,
        evaluate_epochs=[2, 3], endpoint_generation_seeds=[261001, 261101],
        only_training_change='Normalized per-depth CE weights', auto_promote=False,
        source_fid_not_reused='Source epoch-1 evaluation used K1024; trial sampling uses current-best K2048.',
        main_resume_after_trials=True, limits='One paired training seed; two-epoch continuation, not full convergence.',
        time=time.time()))
    write(OUT / 'status.json', dict(phase='prepared', time=time.time()))
    print(json.dumps(dict(prepared=True, arms=list(ARMS))), flush=True)


def support_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def migrate():
    import torch
    import yaml
    torch.set_num_threads(4)
    support = support_module('checkpoint_support', ROOT / 'scripts/tools/restore_joint20k_earlier_best.py')
    layout = support_module('layout_support', ROOT / 'src/training/rq_layout_rng.py')
    plan = json.loads((OUT / 'plan.json').read_text())
    saved = torch.load(BASE / 'source-epoch001.pt', map_location='cpu', mmap=True, weights_only=False)
    assert saved['run_id'] == SOURCE_RUN and saved['global_step'] == 626
    assert saved['next_epoch'] == 1 and saved['next_microbatch'] == 0
    assert len(saved['rng_state_by_rank']) == 4
    assert saved['joint_codebook_sha256'] == '922c33037770287facb9ba8f2c477224087155ff14af3b9d6d7390b5a40485a9'
    assert all(int(state['step']) == 626 for state in saved['optimizer']['state'].values())
    original_rng = support.tensor_hash(saved['rng_state_by_rank'])
    expanded, seeds = layout.expand_four_to_eight(saved['rng_state_by_rank'], plan['source']['sha256'], 626,
        lambda seed: torch.Generator(device='cuda:0').manual_seed(seed).get_state())
    assert support.tensor_hash(expanded[:4]) == original_rng
    saved['rng_state_by_rank'] = expanded
    keys = ('state_dict', 'optimizer', 'scheduler', 'scaler', 'rng_state_by_rank')
    protected = {key: support.tensor_hash(saved[key]) for key in keys}
    original_recipe = copy.deepcopy(saved['recipe'])
    original_evaluation = saved.get('evaluation')
    proofs = {}
    for arm, arm_plan in plan['arms'].items():
        base = Path(arm_plan['base'])
        recipe = yaml.safe_load((base / 'inputs/recipe.yaml').read_text())
        for key in ('tokenizer', 'targets'):
            assert recipe[key] == original_recipe[key], key
        saved['run_id'] = arm_plan['run_id']; saved['recipe'] = recipe
        saved['runtime'] = copy.deepcopy(saved['runtime'])
        saved['runtime'].update(world_size=8, accumulation_steps=4, microbatch=64, global_batch=2048)
        saved['evaluation'] = None; saved['best_fid'] = math.inf
        saved['source_evaluation_with_old_sampler'] = original_evaluation
        saved['config'] = copy.deepcopy(saved['config'])
        saved['config']['loss']['depth_weights'] = arm_plan['depth_weights']
        destination = base / 'resume-epoch001-eight-gpu.pt'
        torch.save(saved, destination)
        check = torch.load(destination, map_location='cpu', mmap=True, weights_only=False)
        assert {key: support.tensor_hash(check[key]) for key in keys} == protected
        proofs[arm] = dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha(destination))
        del check
    write(OUT / 'checkpoint-pairing.json', dict(passed=True, source_sha256=plan['source']['sha256'],
        initial_epoch=1, initial_step=626, original_rank_rng_states_preserved=4, added_rank_seeds=seeds,
        common_training_state_sha256=protected, arms=proofs, weights_optimizer_scheduler_scaler_identical=True,
        all_eight_rank_rng_states_identical_between_arms=True, time=time.time()))
    print(json.dumps(dict(migrated=True, initial_step=626, arms=list(proofs))), flush=True)


def records():
    folder = MAIN_OUT / 'persistent-train/checkpoints/.checkpoint-data'
    return [json.loads(p.read_text()) for p in folder.glob('*.json')]


def gpu_pids():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(line) for line in result.splitlines() if line.strip().isdigit()}


def pause_main():
    start = json.loads((MAIN_BASE / 'live-training/train/progress-rank0.json').read_text())['train/global_step']
    committed = max(records(), key=lambda record: record['global_step'])['global_step']
    threshold = committed if 0 <= start - committed <= 30 else math.ceil(start / 626) * 626
    if (OUT / 'pause-target.json').exists() and not (OUT / 'main-preservation.json').exists():
        threshold = int(json.loads((OUT / 'pause-target.json').read_text())['global_step'])
    while True:
        latest = max(records(), key=lambda record: record['global_step'])
        progress = json.loads((MAIN_BASE / 'live-training/train/progress-rank0.json').read_text())
        gap = progress['train/global_step'] - latest['global_step']
        write(OUT / 'status.json', dict(phase='waiting_for_main_checkpoint', main_step=progress['train/global_step'],
            committed_step=latest['global_step'], required_step=threshold, time=time.time()))
        if latest['global_step'] >= threshold and gap >= 0:
            local = Path(latest['local'])
            if local.is_file() and local.stat().st_size == latest['bytes']:
                break
        time.sleep(10)
    status = json.loads((MAIN_OUT / 'status.json').read_text())
    controller, child = int(status['pid']), int(status['child_pid'])
    assert str(MAIN_BASE / 'train.py') in Path(f'/proc/{child}/cmdline').read_bytes().decode().replace('\0', ' ')
    controller_command = Path(f'/proc/{controller}/cmdline').read_bytes().decode().split('\0')
    assert (any('scale_joint20k_earlier_best_to_eight.py' in arg for arg in controller_command) or
            (str(Path(__file__).resolve()) in controller_command and 'resume-main' in controller_command))
    pin = BASE / f'main-resume.{os.getpid()}.pending.pt'
    os.link(local, pin)
    assert sha(pin) == latest['sha256']
    pin.replace(BASE / 'main-resume.pt')
    progress = json.loads((MAIN_BASE / 'live-training/train/progress-rank0.json').read_text())
    gap = progress['train/global_step'] - latest['global_step']
    previous = OUT / 'main-preservation.json'
    if previous.exists():
        write(OUT / f'main-preservation-history/{time.time_ns()}.json', json.loads(previous.read_text()))
    write(OUT / 'main-preservation.json', dict(checkpoint=latest, local_pin=str(BASE / 'main-resume.pt'),
        observed_step=progress['train/global_step'], unsaved_updates_to_replay=gap,
        prior_controller_pid=controller, prior_torchrun_pid=child, time=time.time()))
    os.kill(controller, signal.SIGTERM)
    for _ in range(90):
        if not gpu_pids():
            write(MAIN_OUT / 'status.json', dict(phase='paused_for_depth_loss_test', experiment=str(OUT),
                preserved_step=latest['global_step'], time=time.time()))
            return
        time.sleep(2)
    raise RuntimeError('Main CUDA workers did not exit; refusing overlapping jobs')


def run_child(command, log, env=None):
    global CURRENT_CHILD
    with log.open('ab', buffering=0) as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        CURRENT_CHILD = child
        write(OUT / 'child.json', dict(pid=child.pid, command=command, log=str(log), time=time.time()))
        result = child.wait()
        CURRENT_CHILD = None
    if result:
        raise RuntimeError(f'Child exited {result}; see {log}')


def report():
    plan = json.loads((OUT / 'plan.json').read_text())
    result = dict(completed=True, final_epoch=3, final_step=1878, rows=[], auto_promoted=False)
    for epoch in (2, 3):
        values = {}
        for arm, p in plan['arms'].items():
            row = json.loads((Path(p['out']) / f'evaluations/epoch{epoch:03d}.json').read_text())
            assert row['train/global_step'] == 626 * epoch and row['generated_images'] == 50000
            assert row['real_images'] == 1281167 and row['real_split'] == 'train' and not row['rejection_sampling']
            values[arm] = row['eval/fid']
        result['rows'].append(dict(epoch=epoch, seed=261001, **values,
            improvement=values['control'] - values['weighted']))
    confirmation = {}
    for arm, p in plan['arms'].items():
        row = json.loads((Path(p['out']) / 'evaluations/confirmation-seed261101.json').read_text())
        assert row['global_step'] == 1878 and row['seed'] == 261101 and row['generated_images'] == 50000
        confirmation[arm] = row['fid']
        assert json.loads((Path(p['out']) / 'completed.json').read_text())['global_step'] == 1878
    result['rows'].append(dict(epoch=3, seed=261101, **confirmation,
        improvement=confirmation['control'] - confirmation['weighted']))
    fingerprints = {}
    for arm, p in plan['arms'].items():
        fingerprints[arm] = []
        for rank in range(8):
            proof = json.loads((Path(p['out']) / f'train/loss-pairing-rank{rank}.json').read_text())
            assert proof['passed'] and proof['cuda'] and proof['training_rng_unchanged']
            fingerprints[arm].append({key:proof[key] for key in ('image_fingerprint', 'code_fingerprint', 'target_fingerprint')})
    assert fingerprints['control'] == fingerprints['weighted']
    result['initial_images_codes_and_targets_match_all_eight_ranks'] = True
    endpoint = [row['improvement'] for row in result['rows'] if row['epoch'] == 3]
    result['screen_passed'] = all(gain >= .5 for gain in endpoint)
    result['limits'] = plan['limits']
    write(OUT / 'comparison.json', result)
    return result


def resume_main():
    log = (BASE / 'main-resume-supervisor.log').open('ab', buffering=0)
    child = subprocess.Popen([sys.executable, __file__, 'resume-main'], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(OUT / 'main-resume-supervisor.json', dict(pid=child.pid, time=time.time()))
    return child


def resume_main_worker():
    global CURRENT_CHILD
    if gpu_pids():
        raise RuntimeError('Cannot restore main run while another CUDA job is active')
    env = dict(os.environ, **json.loads((MAIN_BASE / 'environment.json').read_text()))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
        str(MAIN_BASE / 'train.py'), '--resume', str(BASE / 'main-resume.pt'), '--epochs', '100']
    log = (MAIN_OUT / 'depth-loss-test-resume.log').open('ab', buffering=0)
    child = subprocess.Popen(command, cwd=MAIN_BASE, env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    CURRENT_CHILD = child
    write(MAIN_OUT / 'status.json', dict(phase='training', pid=os.getpid(), child_pid=child.pid,
        command=command, resumed_after_depth_loss_test=True, time=time.time()))
    write(OUT / 'main-resumed.json', dict(pid=child.pid, command=command, time=time.time()))
    pointer = json.loads((OUT / 'original-active.json').read_text())
    pointer.update(updated=time.time(), active_experiment=None,
        last_experiment=str(OUT), experiments_cancelled=False)
    write(ROOT / 'outputs/imagenet-laser-joint20k-active.json', pointer)
    result = child.wait()
    CURRENT_CHILD = None
    write(MAIN_OUT / 'status.json', dict(phase='completed' if result == 0 else 'stopped_or_failed',
        returncode=result, pid=os.getpid(), child_pid=child.pid, time=time.time()))
    return result


def stop(*_):
    if CURRENT_CHILD is not None and CURRENT_CHILD.poll() is None:
        os.killpg(CURRENT_CHILD.pid, signal.SIGINT)
        try:
            CURRENT_CHILD.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(CURRENT_CHILD.pid, signal.SIGTERM)
            CURRENT_CHILD.wait(timeout=30)
    raise KeyboardInterrupt('Depth-loss controller interrupted')


def supervise():
    lock = (BASE / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paused = False
    try:
        plan = json.loads((OUT / 'plan.json').read_text())
        pending = {arm:p for arm,p in plan['arms'].items() if not arm_completed(p)}
        if not pending:
            result = report()
            write(OUT / 'status.json', dict(phase='completed', screen_passed=result['screen_passed'], time=time.time()))
            return
        pause_main(); paused = True
        if not (OUT / 'checkpoint-pairing.json').exists():
            run_child([sys.executable, __file__, 'migrate'], BASE / 'migration.log')
        pairing = json.loads((OUT / 'checkpoint-pairing.json').read_text())
        assert pairing['passed']
        for arm, p in pending.items():
            base, out = Path(p['base']), Path(p['out'])
            for name, expected in json.loads((out / 'reproducibility/code-manifest.json').read_text()).items():
                assert sha(base / name) == expected, name
            assert sha(base / 'resume-epoch001-eight-gpu.pt') == pairing['arms'][arm]['sha256']
            assert not gpu_pids()
            env = dict(os.environ, **json.loads((base / 'environment.json').read_text()))
            env['LASER_PRIVATE_SMOKE_DIR'] = str(base / 'private-smoke')
            distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8', str(base / 'train.py')]
            resume = ['--resume', str(base / 'resume-epoch001-eight-gpu.pt')]
            write(OUT / 'status.json', dict(phase='gpu_preflight', arm=arm, time=time.time()))
            if not (base / 'private-smoke/verification.json').exists():
                run_child([*distributed, *resume, '--smoke-updates', '1'], base / 'preflight.log', env)
            validate_preflight(base, p['depth_weights'])
            write(OUT / 'status.json', dict(phase='training', arm=arm, run_id=p['run_id'],
                final_epoch=3, final_step=1878, world_size=8, time=time.time()))
            pointer = json.loads((OUT / 'original-active.json').read_text())
            pointer.update(run_id=p['run_id'], url=p['url'], base=p['base'],
                recipe=f'Epoch-1-to-3 paired depth-loss test: {arm}', updated=time.time(),
                experiments_cancelled=False, active_experiment=str(OUT),
                main_run_preserved=MAIN_RUN, depth_loss_weights=p['depth_weights'])
            write(ROOT / 'outputs/imagenet-laser-joint20k-active.json', pointer)
            run_child([*distributed, *resume, '--epochs', '3', '--new-wandb-run'], base / 'training.log', env)
        result = report()
        write(OUT / 'status.json', dict(phase='completed', screen_passed=result['screen_passed'], time=time.time()))
    except BaseException as error:
        write(OUT / 'failure.json', dict(error_type=type(error).__name__, error=str(error), time=time.time()))
        write(OUT / 'status.json', dict(phase='failed', time=time.time()))
        raise
    finally:
        if (paused or (OUT / 'main-preservation.json').exists()) and not gpu_pids():
            resume_main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'migrate', 'supervise', 'report', 'resume-main'])
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = {'prepare':prepare, 'migrate':migrate, 'supervise':supervise, 'report':report,
              'resume-main':resume_main_worker}[args.phase]()
    if result is not None:
        print(json.dumps(result), flush=True)
