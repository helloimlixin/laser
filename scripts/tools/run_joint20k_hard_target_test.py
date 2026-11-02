#!/usr/bin/env python3
"""Paired fresh training test: remove soft labels and stochastic code prefixes."""
import argparse
import copy
import difflib
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
CONTROL = 'imagenet-laser-joint20k-depth5-rqnoise-k2048-rngfix-scratch100-8h100-20261017'
CONTROL_BASE = Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-rngfix-8h100-imagenet-20261017')
CONTROL_OUT = ROOT/'outputs'/CONTROL
RUN = 'imagenet-laser-joint20k-depth5-hardtargets-scratch4-8h100-20261018'
BASE = Path('/tmp/laser-joint20k-hard-target-test-20261018')
OUT = ROOT/'outputs'/RUN
KEY = Path('/root/.config/laser/imagenet-stage2-wandb.key')
INITIAL_SHA = '46386ed8076a513a67894cdeb591aead1167801c0c35cbca79063ecae98d4fc6'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assert_hard_target_only(control, trial):
    expected = copy.deepcopy(control)
    expected['targets'].update(policy='hard_deterministic', stochastic_prefixes=False)
    expected['targets'].pop('calibration', None)
    expected['objective'] = 'hard_target_cross_entropy'
    actual = copy.deepcopy(trial)
    for value in (expected, actual):
        for key in ('name', 'run_id', 'version', 'lineage'):
            value.pop(key, None)
        value.get('optimization', {}).pop('probe_initial_weight_sha256', None)
        if not value.get('optimization'):
            value.pop('optimization', None)
    if expected != actual:
        raise ValueError('Hard-target trial changes another training or sampling setting')
    if (trial['global_batch'] != 2048 or trial['microbatch'] != 64 or
            trial['lr_schedule_epochs'] != 100 or trial['epochs'] != 100):
        raise ValueError('Preserve global batch and original 100-epoch LR horizon')


def prepare():
    import yaml
    if BASE.exists() or OUT.exists():
        raise RuntimeError('Refusing to overwrite an existing trial')
    for name, expected in json.loads((CONTROL_OUT/'reproducibility/code-manifest.json').read_text()).items():
        if sha(CONTROL_BASE/name) != expected:
            raise RuntimeError(f'Control snapshot changed: {name}')
    BASE.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for name in ('source', 'runtime'):
        shutil.copytree(CONTROL_BASE/name, BASE/name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('upstream', 'data', 'torch-cache'):
        (BASE/name).symlink_to(CONTROL_BASE/name, target_is_directory=True)
    (BASE/'inputs').mkdir()
    for path in (CONTROL_BASE/'inputs').iterdir():
        destination = BASE/'inputs'/path.name
        if path.suffix in ('.pt', '.npz'):
            os.link(path, destination)
        else:
            shutil.copyfile(path, destination)
    for name in ('train.py', 'verify.py'):
        shutil.copyfile(CONTROL_BASE/name, BASE/name)
    # Keep the control runtime. Only add target-policy support, the verified
    # scalar-compatible sampler adapter, and an upload opt-out for this pilot.
    changed = ['src/training/laser_joint_recipe.py', 'src/training/rq_joint_sampling.py']
    for name in changed:
        shutil.copyfile(ROOT/name, BASE/'source'/name)
    shared = BASE/'source/src/training/rq_imagenet_shared.py'
    previous = shared.read_text()
    needle = "                if run is not None:\n                    uploads=BASE/'uploads/checkpoints'"
    replacement = "                if run is not None and os.environ.get('LASER_WANDB_CHECKPOINT_UPLOADS', '1') != '0':\n                    uploads=BASE/'uploads/checkpoints'"
    if previous.count(needle) != 1:
        raise RuntimeError('Unexpected control checkpoint upload code')
    shared.write_text(previous.replace(needle, replacement))
    control = yaml.safe_load((CONTROL_BASE/'inputs/recipe.yaml').read_text())
    recipe = copy.deepcopy(control)
    recipe.update(run_id=RUN, name='LASER joint20k depth5 | no target noise | paired fresh4',
        version='joint20k-hard-target-test-v1', objective='hard_target_cross_entropy',
        lineage=dict(control=CONTROL, stage2_initialization='scratch',
            hypothesis='Test whether soft labels and stochastic residual prefixes hurt generation'))
    recipe['targets'].update(policy='hard_deterministic', stochastic_prefixes=False)
    recipe['targets'].pop('calibration', None)
    recipe.setdefault('optimization', {})['probe_initial_weight_sha256'] = INITIAL_SHA
    assert_hard_target_only(control, recipe)
    (BASE/'inputs/recipe.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
    config = yaml.safe_load((BASE/'inputs/stage2-training-config.yaml').read_text())
    config['loss'].update(type='cross_entropy', stochastic_codes=False)
    config['loss'].pop('temp', None)
    (BASE/'inputs/stage2-training-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    verification = (BASE/'verify.py').read_text()
    old = '        assert .9<ratio<1.05 and all(0<e<10 for e in mean_entropy),(ratio,mean_entropy)'
    new = '        assert abs(ratio-1.)<1.e-5 and all(e==0. for e in mean_entropy),(ratio,mean_entropy)'
    if verification.count(old) != 1:
        raise RuntimeError('Unexpected target-noise verification code')
    verification = verification.replace(old, new)
    old = '            p,c=profile.targets(aux,batch)\n'
    verification = verification.replace(old, old+
        "            assert torch.equal(c,aux.quantizer.quantize(x)['codes'])\n")
    (BASE/'verify.py').write_text(verification)
    env = {key:value.replace(str(CONTROL_BASE), str(BASE)) for key,value in
           json.loads((CONTROL_BASE/'environment.json').read_text()).items()}
    # Checkpoints still commit to persistent storage with hashes. Avoid uploading
    # duplicate 16-GB aliases to W&B after every bounded pilot evaluation.
    env['LASER_WANDB_CHECKPOINT_UPLOADS'] = '0'
    write(BASE/'environment.json', env)
    repro = OUT/'reproducibility'
    repro.mkdir()
    patches = []
    for relative in ('source/src/training/laser_joint_recipe.py',
                     'source/src/training/rq_joint_sampling.py',
                     'source/src/training/rq_imagenet_shared.py', 'verify.py'):
        source = CONTROL_BASE/relative
        patches.extend(difflib.unified_diff(source.read_text().splitlines(True) if source.exists() else [],
            (BASE/relative).read_text().splitlines(True), fromfile='control/'+relative,
            tofile='hardtargets/'+relative))
    (repro/'runtime.patch').write_text(''.join(patches))
    for name in ('recipe.yaml', 'stage2-training-config.yaml'):
        shutil.copyfile(BASE/'inputs'/name, repro/name)
    shutil.copyfile(Path(__file__), repro/Path(__file__).name)
    shutil.copyfile(ROOT/'tests/test_joint_hard_targets.py', repro/'test_joint_hard_targets.py')
    files = [BASE/name for name in ('train.py', 'verify.py', 'environment.json')]
    for directory in ('source', 'runtime', 'upstream'):
        files.extend(path for path in (BASE/directory).rglob('*.py') if '__pycache__' not in path.parts)
    files.extend(path for path in (BASE/'inputs').iterdir() if path.suffix in ('.yaml', '.json'))
    write(repro/'code-manifest.json', {str(p.relative_to(BASE)):sha(p) for p in files})
    with zipfile.ZipFile(repro/'source.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(BASE))
    for path, expected in [('stage1-model.pt', recipe['tokenizer']['sha256']),
                           ('joint-codebook.pt', recipe['tokenizer']['codebook_sha256']),
                           ('imagenet_256_train.npz', recipe['evaluation']['reference_sha256'])]:
        assert sha(BASE/'inputs'/path) == expected
    plan = dict(run_id=RUN, control=CONTROL,
        url=f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{RUN}',
        fresh_stage2=True, fresh_optimizer=True, expected_initial_weight_sha256=INITIAL_SHA,
        world_size=8, global_batch=2048, microbatch=64, accumulation_steps=4,
        epochs=4, stop_step=2504, lr_schedule_epochs=100, cosine_updates=62600,
        target_policy='hard_deterministic', soft_labels=False, stochastic_prefixes=False,
        tokenizer_frozen=True, codebook_sha256=recipe['tokenizer']['codebook_sha256'],
        sampling=recipe['sampling'], compare_epochs=[1, 2, 4], generated_images=50000,
        reference_images=1281167, reference_split='train',
        primary_endpoint_epoch=4, auto_promote=False, automatic_control_resume=False,
        limits='One paired training seed. Removes soft labels and stochastic prefixes together; '
            'their separate effects need another ablation. No claim of long-horizon convergence.',
        checkpoints='Persistent output storage; W&B records metrics and samples without duplicate weights',
        time=time.time())
    write(OUT/'plan.json', plan)
    write(OUT/'status.json', dict(phase='prepared', time=time.time()))
    print(json.dumps(plan), flush=True)


def verify_initialization():
    import random
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
    sys.path[:0] = [str(BASE/'source'), str(BASE/'upstream'), str(BASE/'runtime')]
    from src.training.laser_joint_recipe import JointLaserRecipe
    from rqvae.utils.config import load_config, augment_arch_defaults
    torch.set_num_threads(8)
    random.seed(261001); np.random.seed(261001); torch.manual_seed(261001)
    profile = JointLaserRecipe(BASE)
    aux, _ = profile.tokenizer(torch.device('cpu'))
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch)
    arch.vocab_size = config.dataset.vocab_size
    model = profile.model(arch)
    profile.adapt(model, aux)
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    actual = digest.hexdigest()
    if actual != INITIAL_SHA:
        raise RuntimeError('Fresh hard-target model differs from the preserved control initialization')
    assert profile.sampling_settings() == dict(temperature=1., top_k=[2048], top_p=[.95])
    proof = dict(passed=True, initial_weight_sha256=actual,
        control_initial_weight_sha256=INITIAL_SHA, parameter_count=sum(p.numel() for p in model.parameters()),
        policy=profile.target_policy(), sampler_matches_control=True, time=time.time())
    write(OUT/'initialization-verification.json', proof)
    print(json.dumps(proof), flush=True)


def report():
    rows = []
    for epoch in (1, 2, 4):
        control = json.loads((CONTROL_OUT/f'evaluations/epoch{epoch:03d}.json').read_text())
        trial_path = OUT/f'evaluations/epoch{epoch:03d}.json'
        trial = json.loads(trial_path.read_text()) if trial_path.exists() else None
        for value in (control, trial):
            if value is not None and (value['train/global_step'] != epoch*626 or
                    value['generated_images'] != 50000 or value['real_images'] != 1281167 or
                    value['real_split'] != 'train' or value['rejection_sampling']):
                raise RuntimeError('Incompatible comparison protocol')
        rows.append(dict(epoch=epoch, control_fid=control['eval/fid'],
            hard_target_fid=None if trial is None else trial['eval/fid'],
            improvement=None if trial is None else control['eval/fid']-trial['eval/fid']))
    result = dict(completed=rows[-1]['hard_target_fid'] is not None, results=rows,
        observed_improvement_at_epoch4=bool(rows[-1]['improvement'] is not None and rows[-1]['improvement'] > 0),
        auto_promoted=False, final_convergence_improvement_established=False,
        limits=json.loads((OUT/'plan.json').read_text())['limits'])
    write(OUT/'comparison.json', result)
    return result


def run():
    lock = (BASE/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not json.loads((OUT/'initialization-verification.json').read_text())['passed']:
        raise RuntimeError('Initialization verification required')
    for name, expected in json.loads((OUT/'reproducibility/code-manifest.json').read_text()).items():
        if sha(BASE/name) != expected:
            raise RuntimeError(f'Frozen source changed: {name}')
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    if occupied:
        raise RuntimeError('Exclusive GPUs required; another CUDA job is active')
    if (OUT/'train').exists():
        raise RuntimeError('Refusing to overwrite a production trial')
    child = None

    def stop(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
        raise KeyboardInterrupt('Hard-target trial interrupted')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    env = dict(os.environ, **json.loads((BASE/'environment.json').read_text()),
        WANDB_API_KEY=KEY.read_text().strip(), LASER_PRIVATE_SMOKE_DIR=str(BASE/'private-smoke'))
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8']
    for phase, command in [
        ('gpu_preflight', [*distributed, str(BASE/'train.py'), '--smoke-updates', '2']),
        ('checkpoint_verification', [sys.executable, str(BASE/'verify.py')]),
        ('training', [*distributed, str(BASE/'train.py'), '--epochs', '4'])]:
        if phase == 'training':
            proof = json.loads((OUT/'smoke-reload-verification.json').read_text())
            if not proof['passed'] or proof['targets']['soft_targets'] or proof['targets']['stochastic_prefixes']:
                raise RuntimeError('No-noise preflight failed')
        with (OUT/f'{phase}.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=BASE, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            state = dict(phase=phase, pid=os.getpid(), child_pid=child.pid,
                command=command, time=time.time())
            write(OUT/'status.json', state)
            write(OUT/'launch-status.json', state)
            while child.poll() is None:
                if phase == 'training':
                    report()
                time.sleep(15)
            if child.returncode:
                raise RuntimeError(f'{phase} exited {child.returncode}; see {OUT/phase}.log')
    completed = json.loads((OUT/'completed.json').read_text())
    if completed['epochs'] != 4 or completed['global_step'] != 2504:
        raise RuntimeError('Unexpected training budget')
    result = report()
    write(OUT/'status.json', dict(phase='completed', result=result, time=time.time()))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'verify-initialization', 'run', 'report'])
    args = parser.parse_args()
    {'prepare':prepare, 'verify-initialization':verify_initialization, 'run':run,
     'report':lambda:print(json.dumps(report()))}[args.phase]()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            write(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error), time=time.time()))
        raise
