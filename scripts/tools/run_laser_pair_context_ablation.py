"""Checkpointed two-branch training/FID controller for the pair-context test."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-pair-context-ablation-20261015')
OUT = ROOT/'outputs/imagenet-laser-pair-context-ablation-20261015'
child = None
cancelled = False


def write(name, value):
    path = OUT/name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def record(**values):
    write('controller.json', dict(values, pid=os.getpid(), time=time.time()))


def stop(*_):
    global cancelled
    cancelled = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def launch(branch, phase, seed=261001):
    global child
    for name, digest in json.loads((OUT/'source-manifest.json').read_text()).items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == digest, name
    destination = OUT/branch/({'preflight':'preflight.json', 'training':'result.json', 'fid50k':'fid50k.json'}[phase])
    if phase == 'fid50k' and seed != 261001:
        destination = OUT/branch/f'fid50k-seed{seed}.json'
    if destination.exists():
        return
    if cancelled:
        raise KeyboardInterrupt
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
               str(BASE/'trial.py'), '--branch', branch]
    if phase == 'preflight':
        command += ['--stop-after', '2']
    elif phase == 'training':
        command += ['--resume', '--stop-after', '626']
    else:
        command += ['--fid-only', '--generation-seed', str(seed)]
    path = BASE/(branch+'-'+phase+('' if seed == 261001 else '-seed'+str(seed))+'.log')
    with path.open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT,
            env=dict(os.environ, **json.loads((BASE/'environment.json').read_text())),
            stdout=log, stderr=subprocess.STDOUT)
        record(phase=phase, branch=branch, child_pid=child.pid, log=str(path), command=command)
        code = child.wait()
    if code or not destination.exists():
        record(phase='failed', failed_phase=phase, branch=branch, returncode=code, log=str(path))
        raise RuntimeError(f'{branch} {phase} failed; inspect {path}')


def compare():
    results = {branch:json.loads((OUT/branch/'result.json').read_text()) for branch in ['control', 'context']}
    assert all(r['parent_step'] == 8764 and r['global_step'] == 9390 and r['additional_updates'] == 626
               and r['full_training_complete'] for r in results.values())
    assert results['control']['same_data_order_sha256_by_rank'] == results['context']['same_data_order_sha256_by_rank']
    assert abs(results['control']['before']['ce']-results['context']['before']['ce']) < 1.e-7
    assert results['context']['optimizer_age_groups'] == {'9390':782, '626':6}
    assert results['control']['optimizer_age_groups'] == {'9390':782}
    fids = {branch:json.loads((OUT/branch/'fid50k.json').read_text()) for branch in results}
    assert fids['control']['sampler'] == fids['context']['sampler']
    assert all(r['generated_images'] == 50000 and r['reference_images'] == 1281167 for r in fids.values())
    gain = fids['control']['fid']-fids['context']['fid']
    repeats = {}
    if abs(gain) < 1.:
        for branch in ['context', 'control']:
            launch(branch, 'fid50k', seed=261002)
            repeats[branch] = json.loads((OUT/branch/'fid50k-seed261002.json').read_text())['fid']
    scores = {branch:(case['fid']+repeats[branch])/2 if repeats else case['fid']
              for branch,case in fids.items()}
    measured_gain = scores['control']-scores['context']
    result = dict(passed=True, parent_step=8764, tested_step=9390, matched_additional_updates=626,
        fid={name:case['fid'] for name,case in fids.items()},
        context_fid_improvement_over_control=measured_gain,
        primary_seed_gain=gain, repeated_fid=repeats, mean_fid=scores,
        generation_seeds=[261001,261002] if repeats else [261001],
        recommended_branch='context' if measured_gain >= 1. else 'control',
        qualifies_for_adoption=measured_gain >= 1. and scores['context'] < 40.36652094676475,
        small_gain_requires_replication=False,
        auxiliary_loss_terms=0, sampled_omp_trajectories=0,
        generated_images_per_case=50000, reference_images=1281167, reference_split='train',
        shared_data_order_verified=True, exact_initial_logits_verified=True,
        production_replaced=False, continuation_checkpoint={name:r['checkpoint'] for name,r in results.items()})
    write('comparison.json', result)
    import wandb
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    api = wandb.Api(timeout=60)
    for branch in results:
        identity = json.loads((OUT/branch/'wandb.json').read_text())
        run = api.run('helloimlixin-rutgers/laser/'+identity['id'])
        run.summary.update({'comparison/control_fid':fids['control']['fid'],
            'comparison/context_fid':fids['context']['fid'],
            'comparison/context_improvement':measured_gain, 'comparison/complete':True})
    record(phase='complete', **result)


def main():
    lock = (BASE/'controller.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    for name, digest in json.loads((OUT/'source-manifest.json').read_text()).items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == digest, name
    assert json.loads((OUT/'cpu-verification.json').read_text())['passed']
    record(phase='waiting_for_gpu_handoff')
    while not (OUT/'paused-parent.json').exists():
        if cancelled:
            return
        time.sleep(4)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        if cancelled:
            return
        time.sleep(4)
    for branch in ['context', 'control']:
        launch(branch, 'preflight')
    for branch in ['context', 'control']:
        proof = json.loads((OUT/branch/'initialization-verification.json').read_text())
        assert proof['passed'] and proof['exact_initial_logits'] and proof['existing_adam_age'] == 8764
    for branch in ['context', 'control']:
        launch(branch, 'training')
    for branch in ['context', 'control']:
        launch(branch, 'fid50k')
    compare()


if __name__ == '__main__':
    main()
