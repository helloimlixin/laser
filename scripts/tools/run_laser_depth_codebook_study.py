#!/usr/bin/env python3
"""Complete depth-book tokenizer comparisons without interrupting prior training."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/Projects/laser')
OUT = ROOT/'outputs/imagenet-laser-depth-pair-codebooks-20261017'
BASE = Path('/tmp/laser-depth-pair-codebooks-20261017')
CODE = BASE/'source-evaluation-v2'
SCRIPT = CODE/'scripts/tools/fit_laser_depth_codebooks.py'


def write(name, values):
    value = dict(time=time.time(), stage2_training_launched=False, main_training_unchanged=True, **values)
    tmp = OUT/(name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(OUT/name)
    print(json.dumps(value), flush=True)


def verify_sources():
    manifest = json.loads((OUT/'evaluation-source-manifest.json').read_text())
    for rel, expected in manifest.items():
        assert hashlib.sha256((CODE/rel).read_bytes()).hexdigest() == expected, rel


def wait_for_fit(atoms, receipt):
    target = OUT/f'depth4-atoms{atoms}-bins512-fit-final.json'
    pid = json.loads((OUT/receipt).read_text())['pid']
    started = time.monotonic()
    while not target.exists():
        stat = Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split(') ', 1)[1].startswith('Z'):
            raise RuntimeError(f'Fitter {pid} exited without completion: {target}')
        if time.monotonic()-started > 1800:
            raise TimeoutError(f'Fitting took over 30 minutes: {atoms}')
        time.sleep(5)
    assert json.loads(target.read_text())['completed']


active = None


def run(command, name):
    global active
    verify_sources()
    env = os.environ.copy()
    env.update(LASER_DEPTH_SOURCE_ROOT=str(CODE), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    with (OUT/(name+'.log')).open('ab', buffering=0) as log:
        active = subprocess.Popen(command, cwd=BASE, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        write('study-status.json', dict(phase=name, child_pid=active.pid, command=command))
        result = active.wait()
    active = None
    if result != 0:
        raise RuntimeError(f'{name} failed with exit code {result}; see its log')


def evaluate(atoms, action):
    path = OUT/f'depth4-atoms{atoms}-bins512-codebook.pt'
    receipt = OUT/f'depth4-atoms{atoms}-bins512-{action}-rfid.json'
    if receipt.exists():
        report = json.loads(receipt.read_text())
        with path.open('rb') as stream:
            expected = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert report['completed'] and report['codebook_sha256'] == expected
        assert report['reconstructed_images'] == (50000 if action == 'full' else 2048)
        return
    run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
        str(SCRIPT), action, '--codebook', str(path), '--batch-size', '2', '--memory-fraction', '.05'],
        f'{action}-atoms{atoms}')


def main():
    wait_for_fit(2048, 'process.json')
    wait_for_fit(4096, 'control-process.json')
    if not (OUT/'depth4-atoms1024-bins512-fit-final.json').exists():
        run([sys.executable, str(SCRIPT), 'fit', '--atoms', '1024', '--passes', '12',
            '--device', 'cuda:0', '--memory-fraction', '.05'], 'fit-atoms1024')
    for atoms in (2048, 4096, 1024):
        evaluate(atoms, 'screen')
    screens = {atoms:json.loads((OUT/f'depth4-atoms{atoms}-bins512-screen-rfid.json').read_text())
               for atoms in (1024, 2048, 4096)}
    best = min((1024, 2048), key=lambda atoms:screens[atoms]['measured_rfid'])
    write('screen-comparison.json', dict(candidates={str(k):v for k,v in screens.items()},
        selected_reduced_atom_candidate=best, same_total_atom_control=4096,
        selection_only=True, full_50000_reconstruction_fid_required=True))
    evaluate(best, 'full')
    full = json.loads((OUT/f'depth4-atoms{best}-bins512-full-rfid.json').read_text())
    baseline = json.loads((OUT/'source-heldout-training-baseline.json').read_text())
    summary = dict(completed=True, selected_atoms_per_book=best, total_atoms=4*best,
        source_shared_atoms=16384, coefficient_bins=512, vocabulary_per_book=best*512,
        measured_full_rfid=full['measured_rfid'], reconstructed_images=50000,
        comparison_to_published_stage1_rfid_473=full['measured_rfid']-4.73,
        published_stage1_rfid_not_remeasured_here=True,
        source_heldout_training_latent_mse=baseline['quantized_latent_mse'],
        published_rfid_gate_passed=full['measured_rfid']<=4.73,
        encoder_decoder_frozen=True, prior_launch_requires_review_of_measured_reconstruction_tradeoff=True)
    write('study-report.json', summary)
    write('study-status.json', dict(phase='completed', **summary))


def stop(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    except BaseException as error:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
        write('study-status.json', dict(phase='failed', error=repr(error)))
        raise
