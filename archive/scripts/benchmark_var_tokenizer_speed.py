"""Compare tokenizer execution settings on isolated copies of a saved state."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['baseline', 'autotune', 'channels-last-autotune'])
    args = parser.parse_args()
    base = args.base
    runtime = base/'runtime'
    output = base/'speed-tuning'
    local = Path('/tmp/laser-var-checkpoints')/base.name
    environment = json.loads(args.environment.read_text())
    environment.update(WANDB_MODE='disabled', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    checkpoint = torch.load(local/'tokenizer/tokenizer-last.pt', map_location='cpu', weights_only=False, mmap=True)
    original_progress = checkpoint['progress'].copy()
    assert len(checkpoint['rng']) == 3 and checkpoint['optimizer']['state']
    checkpoint['progress'] = dict(epoch=0, batch=0, step=0)
    (output/'saved-production-progress.json').write_text(json.dumps(original_progress, indent=2))
    result_path = output/'benchmark-results.json'
    results = ([r for r in json.loads(result_path.read_text()) if r['case'] not in args.cases]
               if result_path.exists() else [])
    for name, channels_last, autotune in [('baseline', False, False),
                                          ('autotune', False, True),
                                          ('channels-last-autotune', True, True),
                                          ('lpips-channels-last', False, True)]:
        if name not in args.cases:
            continue
        directory = output/name
        directory.mkdir(parents=True, exist_ok=True)
        checkpoints = local/'speed-tuning'/name
        checkpoints.mkdir(parents=True, exist_ok=True)
        with (checkpoints/'tokenizer-last.pt').open('wb') as stream:
            torch.save(checkpoint, stream)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
            str(runtime/'train.py'), '--config', str(runtime/'configs/experiments/ffhq256-var341-tokenizer.yaml'),
            'smoke_steps=16', 'logging.every_steps=1', 'wandb.mode=disabled', 'tokenizer.adversarial_start=0',
            f'output_dir={directory}', f'execution.checkpoint_dir={checkpoints}',
            f'++execution.channels_last={str(channels_last).lower()}',
            f'++execution.lpips_channels_last={str(name == "lpips-channels-last").lower()}',
            f'++execution.cudnn_benchmark={str(autotune).lower()}']
        started = time.time()
        print(json.dumps(dict(case=name, status='started', time=started)), flush=True)
        with (directory/'run.log').open('w') as log:
            completed = subprocess.run(command, cwd=runtime, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=360)
        rows = []
        for line in (directory/'run.log').read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('phase') == 'tokenizer':
                rows.append(row)
        valid = completed.returncode == 0 and len(rows) == 16
        result = dict(case=name, channels_last=channels_last, cudnn_benchmark=autotune,
                      returncode=completed.returncode, valid=valid, elapsed_seconds=time.time()-started)
        if valid:
            result.update(median_images_per_second=statistics.median(r['images_per_second'] for r in rows[5:]),
                          first_loss=rows[0]['loss'], first_reconstruction=rows[0]['reconstruction'],
                          first_dictionary_loss=rows[0]['dictionary_loss'],
                          peak_memory_gib=max(r['peak_memory_gib'] for r in rows),
                          finite_gradients=True, steps=len(rows))
        results.append(result)
        (output/'benchmark-results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
