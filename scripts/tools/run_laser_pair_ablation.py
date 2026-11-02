"""Run the frozen matched four-way trial and evaluate selected checkpoints."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-pair-ablation-20261015')
OUT = ROOT / 'outputs/imagenet-laser-pair-ablation-20261015'
child = None
cancelled = False


def record(**status):
    status.update(pid=os.getpid(), time=time.time())
    temporary = OUT / 'controller.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(OUT / 'controller.json')


def stop(*_):
    global cancelled
    cancelled = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def launch(decoder, coefficients, fid=False):
    global child
    branch = f'{decoder}-{coefficients}'
    destination = OUT / branch / ('fid50k.json' if fid else 'result.json')
    if destination.exists():
        return
    if cancelled:
        raise KeyboardInterrupt
    log_path = BASE / (branch + ('-fid50k' if fid else '') + '.log')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
               str(BASE / 'trial.py'), '--decoder', decoder, '--coefficients', coefficients, '--updates', '128']
    if fid:
        command.append('--fid-only')
    with log_path.open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record(phase='fid50k' if fid else 'training', branch=branch, child_pid=child.pid,
               log=str(log_path), command=command)
        code = child.wait()
    if code:
        record(phase='failed', branch=branch, returncode=code, log=str(log_path))
        raise RuntimeError(f'{branch} failed: {log_path}')
    assert destination.exists()


def main():
    lock = (BASE / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    record(phase='waiting_for_parent_checkpoint', target_step=3756)
    while not (OUT / 'paused-parent.json').exists():
        if cancelled:
            return
        time.sleep(4)
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        if cancelled:
            return
        time.sleep(4)
    for decoder, coefficients in [('scalar', 'hard'), ('scalar', 'soft'), ('pair', 'hard'), ('pair', 'soft')]:
        launch(decoder, coefficients)
    cases = {f'{d}-{c}': json.loads((OUT / f'{d}-{c}/result.json').read_text())
             for d in ['scalar', 'pair'] for c in ['hard', 'soft']}
    assert all(c['same_data_order_sha256_by_rank'] == cases['scalar-hard']['same_data_order_sha256_by_rank']
               for c in cases.values())
    # Coefficient smoothing can trade clean-token NLL for better generation.
    # Evaluate the complete small factorial instead of discarding that case.
    selected = ['scalar-hard', 'scalar-soft', 'pair-hard', 'pair-soft']
    (OUT / 'fid-selection.json').write_text(json.dumps(dict(
        criterion='All four predeclared decoder/coefficient-policy cases receive full FID',
        scores={k:v['after']['ce'] for k,v in cases.items()}, selected=selected,
        caveat='Likelihood is diagnostic; full 50000-sample FID determines adoption.'), indent=2) + '\n')
    for branch in selected:
        launch(*branch.split('-'), fid=True)
    results = {n: json.loads((OUT / n / 'fid50k.json').read_text()) for n in selected}
    winner = min(results, key=lambda n: results[n]['fid'])
    gain = results['scalar-hard']['fid'] - results[winner]['fid']
    summary = dict(passed=True, selected=selected, fid={n:r['fid'] for n,r in results.items()},
                   best_branch=winner, improvement_over_matched_control=gain,
                   qualifies_for_adoption=winner != 'scalar-hard' and gain >= 1.,
                   generated_images_per_case=50000, reference_images=1281167,
                   source_step=3756, tested_step=3884, matched_additional_updates=128,
                   auxiliary_loss_terms=0, omp_trajectory_banks=0, production_replaced=False)
    (OUT / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    record(phase='complete_awaiting_adoption', **summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
