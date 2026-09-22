"""Compare a completed stage-2 pilot and sampling fix, then confirm at 50k."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--pilot-checkpoints', type=Path, required=True)
    args = parser.parse_args()
    work, pilot = args.work, args.work / 'scale-weighted'
    runtime = work / 'evaluation-runtime'
    deadline = time.monotonic() + 2400
    write_json(work / 'confirmation-status.json', dict(phase='waiting_for_pilot', time=time.time()))
    while True:
        status_path = pilot / 'train/status.json'
        if (pilot / 'train/complete.json').exists():
            break
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get('phase') == 'checkpoint_upload_wait' and status.get('epoch') == 60:
                break
        if time.monotonic() > deadline:
            raise RuntimeError('Pilot did not finish updates within 40 minutes')
        pipeline_path = pilot / 'pipeline-status.json'
        if pipeline_path.exists() and json.loads(pipeline_path.read_text()).get('phase') == 'failed':
            raise RuntimeError('Pilot supervisor failed')
        time.sleep(5)
    import torch
    snapshots = work / 'confirmation-checkpoints'
    snapshots.mkdir(exist_ok=True)
    for name in ('prior-last.pt', 'prior-best-fid.pt'):
        os.link(args.pilot_checkpoints / name, snapshots / name)
    last = torch.load(snapshots / 'prior-last.pt', map_location='cpu', weights_only=False, mmap=True)
    assert last['progress']['epoch'] == 60
    del last
    best = torch.load(snapshots / 'prior-best-fid.pt', map_location='cpu', weights_only=False, mmap=True)
    best_epoch = best['progress']['epoch']
    del best
    baseline = Path('/tmp/laser-var-checkpoints/ffhq256-var341-scratch-20260921/history/epoch050/prior-best-fid.pt')
    candidates = [('original', baseline), ('pilot-last', snapshots / 'prior-last.pt')]
    if best_epoch not in (50, 60):
        candidates.append(('pilot-best', snapshots / 'prior-best-fid.pt'))
    plan = [dict(name='baseline'), dict(name='atom-temperature-060', atom_temperature=.6)]
    write_json(work / 'validation-plan.json', plan)

    def evaluate(label, checkpoint, plan_path, samples, seed):
        output = work / label
        run_id = 'ffhq256-var341-fix-' + label + '-20260922'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
            str(runtime / 'scripts/tools/investigate_ffhq_var_sampling.py'),
            '--config', str(runtime / 'configs/experiments/ffhq256-var341-compound.yaml'),
            '--checkpoint', str(checkpoint), '--output', str(output), '--plan', str(plan_path),
            '--reference', '/tmp/laser-rqvae-reference/ffhq_256_train.npz', '--samples', str(samples),
            '--seed', str(seed), '--run-id', run_id, '--online']
        write_json(work / 'confirmation-status.json', dict(phase=label, time=time.time(), command=command))
        with (work / (label + '.log')).open('a', buffering=1) as log:
            subprocess.run(command, cwd=runtime, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=1800)
        if not (output / 'complete.json').exists():
            raise RuntimeError('Evaluation exited without a completion receipt')
        return json.loads((output / 'results.json').read_text())

    results = []
    for label, checkpoint in candidates:
        for record in evaluate('validation-' + label, checkpoint, work / 'validation-plan.json', 10000, 173000):
            if record['conditional_on_real_codes']:
                raise RuntimeError('Conditional diagnostics are not eligible for model selection')
            record.update(candidate=label, checkpoint=str(checkpoint))
            results.append(record)
        write_json(work / 'confirmation-results.json', results)
    original = next(r for r in results if r['candidate'] == 'original' and r['case']['name'] == 'baseline')
    winner = min(results, key=lambda r: r['fid'])
    write_json(work / 'selected-fix.json', dict(winner=winner, original=original,
        validation_improvement=original['fid'] - winner['fid'], selection_samples=10000, selection_seed=173000))
    if winner['fid'] < original['fid'] - .5:
        write_json(work / 'final-plan.json', [dict(winner['case'], name='selected-fix')])
        final = evaluate('final-50k', Path(winner['checkpoint']), work / 'final-plan.json', 50000, 73000)
        write_json(work / 'final-comparison.json', dict(original_fid50k=29.479616766958998,
            selected=final[0], winner=winner,
            improvement=29.479616766958998 - final[0]['fid'],
            promoted=final[0]['fid'] < 29.479616766958998,
            reference='RQ released FFHQ train statistics; training splits differ',
            selection_protocol='2000 screening, fresh-seed 10000 validation, 50000 confirmation'))
    write_json(work / 'confirmation-status.json', dict(phase='complete', time=time.time()))


if __name__ == '__main__':
    main()
