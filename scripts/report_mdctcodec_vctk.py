#!/usr/bin/env python3
"""Publish the matched VCTK comparison and preserve its implementation on W&B."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directories', type=Path, nargs='+', default=[
        Path('outputs/mdctcodec_benchmark_6kbps_initial'),
        Path('outputs/mdctcodec_benchmark_vctk200'), Path('outputs/mdctcodec_benchmark_native48')])
    parser.add_argument('--reference', default='laser_q7_6kbps')
    parser.add_argument('--max-kbps', type=float, default=6.2)
    parser.add_argument('--label', default='initial 6 kbps checkpoint')
    parser.add_argument('--output', type=Path, default=Path('outputs/mdctcodec_comparison_6kbps_report'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows, records, urls = [], {}, []
    manifest = json.loads((args.directories[0] / 'manifest.json').read_text())
    for directory in args.directories:
        other = json.loads((directory / 'manifest.json').read_text())
        if other['test'] != manifest['test']: raise ValueError('Test manifests differ')
        urls.append(json.loads((directory / 'run.json').read_text())['url'])
        for row in json.loads((directory / 'results.json').read_text()):
            name = row['system']
            if name.startswith('laser') and (name != args.reference or directory != args.directories[0]):
                continue
            if row['payload_kbps'] is None or row['payload_kbps'] > args.max_kbps:
                continue
            if name in records:
                previous = next(r for r in rows if r['system'] == name)
                if previous['visqol_audio48k'] != row['visqol_audio48k']:
                    raise ValueError('Conflicting duplicate system: ' + name)
                continue
            # A benchmark may reuse verified baseline summaries; locate their
            # original per-utterance measurements and source W&B run.
            source = next((d for d in args.directories if (d / (name + '.json')).is_file()), None)
            if source is None: raise FileNotFoundError(name + '.json')
            row['model_sample_rate'] = (44100 if name.startswith('dac') else
                24000 if name.startswith('encodec') and '48khz' not in name else 48000)
            row['source_run'] = json.loads((source / 'run.json').read_text())['url']
            rows.append(row)
            data = json.loads((source / (name + '.json')).read_text())
            records[name] = {r['utterance']: r['visqol_audio48k'] for r in data}
    reference = records[args.reference]
    calibration = json.loads((args.directories[0] / 'calibration.json').read_text())
    rng = np.random.default_rng(20260911)
    paired = []
    ordered = sorted(reference)
    speakers = np.array([name.split('_')[0] for name in ordered])
    speaker_names = np.unique(speakers)
    # Resample speakers together to retain within-speaker dependence.
    draws = rng.integers(0, len(speaker_names), size=(5000, len(speaker_names)))
    for name, values in records.items():
        if set(values) != set(reference): raise ValueError('Unpaired test records: ' + name)
        delta = np.array([reference[k] - values[k] for k in ordered])
        groups = [delta[speakers == speaker] for speaker in speaker_names]
        totals = np.array([group.sum() for group in groups])
        counts = np.array([len(group) for group in groups])
        bootstrap = totals[draws].sum(1) / counts[draws].sum(1)
        paired.append({'baseline': name, 'laser_q7_minus_baseline': float(delta.mean()),
                       'ci95_low': float(np.quantile(bootstrap, .025)),
                       'ci95_high': float(np.quantile(bootstrap, .975)),
                       'bootstrap_unit': 'speaker', 'clusters': len(speaker_names),
                       'bootstrap_resamples': 5000})
    lines = ['# MDCTCodec–LASER: ' + args.label, '',
        f"Measured on the same {len(manifest['test'])} VCTK mic2 utterances, balanced across eight speakers, "
        'with full recordings of at least two seconds. Checkpoint: ' + manifest['checkpoint'] + '. '
        'Validation and the legacy validation subset are excluded.', '',
        '| System | Model sample rate | Token payload kbps | ViSQOL audio 48 kHz | STOI |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        rate = 'Unquantized upper bound' if r['payload_kbps'] is None else f"{r['payload_kbps']:.3f}"
        lines.append(f"| {r['system']} | {r['model_sample_rate']} | {rate} | {r['visqol_audio48k']:.4f} | {r['stoi']:.4f} |")
    lines += ['', 'Token rates exclude headers, EnCodec scale values, and shared coefficient calibration. '
        f"LASER uses 13-bit atom IDs and 7-bit coefficients, {calibration['sparsity_level']} atoms per 150 Hz latent frame. "
        'The 6 kbps payload stores each frame in five bytes; file padding explains the measured 6.008 kbps. '
        'Its clipping bound comes only from training crops. EnCodec 48 kHz uses its official stereo model '
        'with duplicated mono input and channel-averaged output. The 24 kHz EnCodec rows have narrower '
        'bandwidth and should be read separately. The released baselines use official pretrained weights '
        'with different training corpora. When present, `mdctcodec_trained_rvq_6kbps` is the archived '
        'VCTK-trained control from report oyg7smih, reevaluated on this test manifest. Its selected '
        'checkpoint has a different update count from the current LASER model. These are checkpoint '
        'comparisons, not experiments with equal training budgets. Paired 95% confidence intervals '
        'resample the eight speakers together, using 5,000 bootstrap draws.', '',
        'Both DAC rows use seven codebooks near 6 kbps. The row `dac_8kbps_model_6kbps` '
        'uses DAC’s standard 8 kbps checkpoint; `dac_6kbps` uses its 16 kbps checkpoint.', '',
        'Google ViSQOL v3.3.3 audio mode scores every system at 48 kHz. '
        'Runtime measurements were collected with concurrent GPU jobs and are diagnostic only. '
        'These comparisons do not establish a claim of current state of the art.', '',
        'The original MDCTCodec paper reports 4.18 ViSQOL at 48 kHz / 6 kbps on its own test protocol: '
        '[MDCTCodec paper](https://arxiv.org/abs/2411.00464). '
        'Implementations: [MDCTCodec](https://github.com/PB20000090/MDCTCodec), '
        '[EnCodec](https://github.com/facebookresearch/encodec), '
        '[DAC](https://github.com/descriptinc/descript-audio-codec), '
        '[ViSQOL](https://github.com/google/visqol).', '',
        'Source evaluation runs: ' + ', '.join(urls), '']
    (args.output / 'comparison.md').write_text('\n'.join(lines))
    (args.output / 'results.json').write_text(json.dumps(rows, indent=2))
    (args.output / 'paired_deltas.json').write_text(json.dumps(paired, indent=2))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    run = wandb.init(project='laser', entity='helloimlixin-rutgers', mode='online',
        name='mdctcodec-laser-6kbps-vctk-comparison-report', group='mdctcodec-vctk-6kbps-20260911',
        job_type='report', dir=str(args.output), config={'source_runs': urls, 'manifest': manifest,
            'reference': args.reference, 'calibration': calibration, 'label': args.label})
    columns = ['system', 'model_sample_rate', 'payload_kbps', 'visqol_audio48k', 'stoi']
    run.log({'comparison/results': wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in rows]),
             'comparison/paired_deltas': wandb.Table(columns=list(paired[0]), data=[list(r.values()) for r in paired])})
    for r in rows:
        run.summary[f"comparison/{r['system']}/visqol_audio48k"] = r['visqol_audio48k']
    artifact = wandb.Artifact(f'mdctcodec-vctk-report-{run.id}', type='benchmark')
    for path in args.output.glob('*.json'): artifact.add_file(str(path), name=path.name)
    artifact.add_file(str(args.output / 'comparison.md'), name='comparison.md')
    run.log_artifact(artifact).wait()
    # The source snapshot also includes untracked audio configs/scripts, which
    # W&B's automatic git diff alone would otherwise miss on cluster loss.
    source = wandb.Artifact(f'mdctcodec-laser-stage1-source-{run.id}', type='code', metadata={
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'recovered_from': 'helloimlixin-rutgers/laser/s2er91dm'})
    selected = subprocess.check_output(['git', 'ls-files', 'src', 'configs', 'train.py', 'requirements.txt'], text=True).splitlines()
    selected += ['requirements-audio.txt', 'src/mdctcodec_bitstream.py', 'docs/mdctcodec-stage1-2026-09-11.md']
    selected += [str(p) for p in Path('configs').rglob('*mdctcodec*.yaml')]
    selected += [str(p) for p in Path('scripts').glob('*mdctcodec*') if p.is_file()]
    selected += ['tests/test_mdctcodec_audio.py']
    for name in dict.fromkeys(selected): source.add_file(name, name=name)
    for path in (Path('outputs/vctk_mdctcodec_stage1_6kbps/environment.json'),
                 Path('outputs/mdctcodec_6kbps_init/provenance.json')):
        if path.is_file(): source.add_file(str(path), name='reproducibility/' + path.name)
    run.log_artifact(source).wait()
    (args.output / 'run.json').write_text(json.dumps({'id': run.id, 'url': run.url}))
    print(run.url, flush=True); run.finish()


if __name__ == '__main__':
    main()
