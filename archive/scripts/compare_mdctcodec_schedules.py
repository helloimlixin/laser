#!/usr/bin/env python3
"""Compare fixed-epoch schedule checkpoints on a locked, separate validation set."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import soundfile as sf
import torch
import wandb
from hydra import compose, initialize_config_dir

from archive.scripts.benchmark_mdctcodec_vctk import align_mdct, measure
from src.mdctcodec_bitstream import pack_frames, unpack_frames
from src.models.laser import LASER
from src.stage1_setup import build_stage1_datamodule, data_config_from_section


def prepare_manifest(path: Path) -> dict:
    locked = json.loads(Path('outputs/mdctcodec_benchmark_vctk200/manifest.json').read_text())
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base='1.2'):
            cfg = compose(config_name='vctk_mdctcodec_stage1_6kbps')
        cfg.data.num_workers = 0
        dm = build_stage1_datamodule(data_config_from_section(cfg.data)); dm.setup()
        chosen = list(dm.val_dataset.paths)
        forbidden = set(locked['test']) | {str(p) for p in chosen}
        rng = random.Random(20260911)
        for speaker in sorted({p.parent.name for p in chosen}):
            pool = [p for p in dm.test_dataset.paths if p.parent.name == speaker
                    and str(p) not in forbidden and int(p.stem.split('_')[1]) >= 25
                    and sf.info(p).duration >= 2.0]
            chosen.extend(rng.sample(pool, 12))
        manifest = {'seed': 20260911, 'purpose': 'schedule comparison, validation only',
                    'selection': 'original 4 + 12 randomly selected later utterances per speaker',
                    'validation': [str(p) for p in chosen], 'locked_test': locked['test'],
                    'sample_rate': 48000, 'metric': 'Google ViSQOL v3.3.3 audio mode',
                    'coefficient_calibration': 'use checkpoint bound; no recalibration'}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
    selected = manifest['validation']
    assert len(selected) == len(set(selected)) == 128
    assert not set(selected).intersection(locked['test'])
    assert set(manifest['locked_test']) == set(locked['test'])
    counts = Counter(Path(p).parent.name for p in selected)
    assert len(counts) == 8 and set(counts.values()) == {16}
    return manifest


def committed_artifacts(run_paths, epoch, timeout_hours):
    api = wandb.Api(timeout=45)
    for run_path in run_paths.values():
        api.run(run_path)  # Fail promptly for invalid runs or credentials.
    deadline = time.monotonic() + timeout_hours * 3600
    while True:
        artifacts = {}
        for label, run_path in run_paths.items():
            entity, project, run_id = run_path.split('/')
            try:
                artifact = api.artifact(f'{entity}/{project}/model-{run_id}-selected-checkpoints:epoch-{epoch:03d}')
                if artifact.state == 'COMMITTED': artifacts[label] = artifact
            except (wandb.errors.CommError, ValueError):
                pass  # The scheduled artifact does not exist until this epoch completes.
        if len(artifacts) == len(run_paths): return artifacts
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Committed epoch-{epoch} checkpoint artifacts were not available')
        time.sleep(30)


def evaluate(checkpoint, paths, device, workers):
    model = LASER.load_from_checkpoint(str(checkpoint), map_location='cpu').to(device).eval()
    assert model.bottleneck.sparsity_level == 2
    assert model.bottleneck.coefficient_quantization_bits == 7
    bound = model.bottleneck.coefficient_quantization_max
    jobs, payload_bits, total_samples = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.inference_mode():
        for path in paths:
            reference, rate = sf.read(path, dtype='float32')
            if rate != 48000 or reference.ndim != 1:
                raise ValueError(f'Expected mono 48 kHz: {path}')
            waveform = torch.from_numpy(reference).to(device)[None, None]
            padded, length = align_mdct(waveform)
            _, _, codes = model.encode(padded)
            integers = codes.values.float().div(bound / 63).round().long()
            payload = pack_frames(codes.support.cpu().numpy(), integers.cpu().numpy())
            atoms, decoded_integers = unpack_frames(payload)
            support = torch.from_numpy(atoms).to(device).reshape_as(codes.support)
            values = torch.from_numpy(decoded_integers).to(device).reshape_as(codes.values).float() * (bound / 63)
            decoded = model.decode_from_atoms_and_coeffs(support, values)[0, 0, :length].float().cpu().numpy()
            if decoded.shape != reference.shape or not np.isfinite(decoded).all():
                raise RuntimeError(f'Invalid decoded waveform: {path}')
            jobs.append(pool.submit(measure, (Path(path).name, reference, decoded.clip(-1, 1))))
            payload_bits += len(payload) * 8; total_samples += length
        rows = [job.result() for job in jobs]
    del model
    torch.cuda.empty_cache()
    return rows, payload_bits / (total_samples / 48000) / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/vctk_mdctcodec_stage1_6kbps_low_lr/validation128.json'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--candidate-run')
    parser.add_argument('--control-run', default='helloimlixin-rutgers/laser/2ng5thjx')
    parser.add_argument('--epoch', type=int, default=265)
    parser.add_argument('--timeout-hours', type=float, default=6)
    parser.add_argument('--output', type=Path, default=Path('outputs/vctk_mdctcodec_stage1_6kbps_low_lr/schedule_comparison'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--metric-workers', type=int, default=8)
    args = parser.parse_args()
    manifest = prepare_manifest(args.manifest)
    if args.prepare_only:
        print(f'Locked {len(manifest["validation"])} validation files; zero test overlap', flush=True)
        return
    if not args.candidate_run: parser.error('--candidate-run is required')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    run_paths = {'control': args.control_run, 'low_lr': args.candidate_run}
    print(f'Waiting for committed epoch-{args.epoch} artifacts from both schedules', flush=True)
    artifacts = committed_artifacts(run_paths, args.epoch, args.timeout_hours)
    sources = {}
    for label, artifact in artifacts.items():
        ranked = [Path(p).name for p in artifact.metadata['checkpoint_paths'] if Path(p).name != 'last.ckpt']
        if not ranked: raise RuntimeError(f'{label}: no ranked checkpoint')
        checkpoint = Path(artifact.get_entry(ranked[0]).download(root=str(args.output / label)))
        sources[label] = {'run': run_paths[label], 'artifact': artifact.qualified_name,
                          'checkpoint': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    summaries, records = {}, {}
    for label, source in sources.items():
        rows, rate = evaluate(source['checkpoint'], manifest['validation'], torch.device(args.device), args.metric_workers)
        records[label] = rows
        summaries[label] = {'visqol': float(np.mean([r['visqol_audio48k'] for r in rows])),
                            'stoi': float(np.mean([r['stoi'] for r in rows])), 'payload_kbps': rate}
        (args.output / f'{label}.json').write_text(json.dumps(rows, indent=2))
        print(label, summaries[label], flush=True)
    control = {r['utterance']: r['visqol_audio48k'] for r in records['control']}
    candidate = {r['utterance']: r['visqol_audio48k'] for r in records['low_lr']}
    assert control.keys() == candidate.keys()
    delta = np.array([candidate[k] - control[k] for k in sorted(control)])
    rng = np.random.default_rng(20260911)
    bootstrap = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
    ci = np.quantile(bootstrap, [.025, .975]).tolist()
    assessment = 'low_lr_better' if ci[0] > 0 else 'control_better' if ci[1] < 0 else 'inconclusive'
    result = {'epoch_boundary': args.epoch, 'sources': sources, 'results': summaries,
              'low_lr_minus_control': float(delta.mean()), 'utterance_bootstrap_ci95': ci,
              'assessment': assessment, 'training_automatically_modified': False,
              'selection': 'best checkpoints by original val32 through the same epoch; compared on fixed val128'}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    run = wandb.init(entity='helloimlixin-rutgers', project='laser', mode='online',
                     name=f'mdctcodec-6kbps-schedule-comparison-epoch{args.epoch}',
                     group='mdctcodec-vctk-6kbps-20260911', job_type='validation-comparison',
                     dir=str(args.output), config={'manifest': manifest, 'sources': sources})
    columns = ['schedule', 'visqol', 'stoi', 'payload_kbps']
    run.log({'validation128/comparison': wandb.Table(columns=columns, data=[
        [label, values['visqol'], values['stoi'], values['payload_kbps']] for label, values in summaries.items()])})
    for label, values in summaries.items():
        for key, value in values.items(): run.summary[f'validation128/{label}/{key}'] = value
    run.summary['validation128/low_lr_minus_control'] = float(delta.mean())
    run.summary['assessment'] = assessment
    artifact = wandb.Artifact(f'mdctcodec-schedule-comparison-{run.id}', type='benchmark', metadata=result)
    for path in args.output.glob('*.json'): artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact).wait()
    (args.output / 'run.json').write_text(json.dumps({'id': run.id, 'url': run.url}))
    print(json.dumps(result, indent=2), flush=True); print(run.url, flush=True)
    run.finish()


if __name__ == '__main__':
    main()
