#!/usr/bin/env python3
"""After successful stage 1 completion, benchmark its best ViSQOL checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


def process_running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--training-dir', type=Path, default=Path('outputs/vctk_mdctcodec_stage1_6kbps'))
    parser.add_argument('--baseline-dir', type=Path, action='append', default=[],
                        help='Additional completed baseline evaluations on the same locked test set')
    args = parser.parse_args()
    if args.pid:
        while process_running(args.pid):
            time.sleep(30)
    finals = list(args.training_dir.glob('checkpoints/**/final.ckpt'))
    if not finals:
        raise RuntimeError('Training did not finish successfully; no final checkpoint exists')
    final = max(finals, key=lambda p: p.stat().st_mtime_ns)
    state = torch.load(final, map_location='cpu', weights_only=False)
    callback = next(v for v in state['callbacks'].values()
                    if isinstance(v, dict) and v.get('monitor') == 'val/audio_visqol_audio48k')
    best = Path(callback['best_model_path'])
    if not best.is_file(): raise FileNotFoundError(best)
    hparams = state['hyper_parameters']
    sparsity = int(hparams['sparsity_level'])
    bits = int(hparams['coefficient_quantization_bits'])
    if sparsity != 2 or bits != 7 or hparams['coefficient_quantization_max'] is None:
        raise ValueError('Final evaluation requires the trained K=2, Q=7 6 kbps model')
    output = args.training_dir / 'final_comparison'
    baseline_dirs = ['outputs/mdctcodec_benchmark_vctk200',
                     'outputs/mdctcodec_benchmark_native48',
                     'outputs/mdctcodec_benchmark_dac8model_6kbps']
    baseline_dirs.extend(str(path) for path in args.baseline_dir)
    command = [sys.executable, '-u', 'scripts/benchmark_mdctcodec_vctk.py', '--checkpoint', str(best),
               '--output', str(output), '--num-items', '200', '--mode', 'online',
               '--laser-sparsity', '2', '--quantizer', 'signed127',
               '--systems', 'laser_float_coefficients', 'laser_q7_6kbps',
               '--name', 'mdctcodec-laser-6kbps-final-best-checkpoint',
               '--baseline-dir', *baseline_dirs]
    (args.training_dir / 'final_evaluation.json').write_text(json.dumps({
        'checkpoint': str(best), 'validation_visqol': float(callback['best_model_score']),
        'command': command,
    }, indent=2))
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, '-u', 'scripts/report_mdctcodec_vctk.py',
                    '--directories', str(output),
                    *baseline_dirs,
                    '--reference', 'laser_q7_6kbps', '--max-kbps', '6.2',
                    '--label', 'trained 6 kbps checkpoint with best validation ViSQOL',
                    '--output', str(args.training_dir / 'final_report')], check=True)


if __name__ == '__main__':
    main()
