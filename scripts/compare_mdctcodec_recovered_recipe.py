#!/usr/bin/env python3
"""Compare the recovered q8 recipe on validation, with optional bottleneck probes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
import wandb

from scripts.benchmark_mdctcodec_vctk import align_mdct, measure
from src.mdctcodec_bitstream import pack_frames, unpack_frames
from src.models.laser import LASER
from src.sparse_token_codec import build_coeff_bin_values, quantize_sparse_coefficients


def q8_payload_roundtrip(support, bins):
    """Two (12-bit atom, 8-bit bin) pairs per frame; raw payload, no header."""
    atoms = np.asarray(support)
    values = np.asarray(bins)
    if atoms.shape != values.shape or atoms.shape[-1] != 2:
        raise ValueError('Expected two equally shaped tokens per frame')
    for array, upper in [(atoms, 4095), (values, 255)]:
        if not np.issubdtype(array.dtype, np.integer) or np.any((array < 0) | (array > upper)):
            raise ValueError('Token outside its fixed-width field')
    atoms, values = atoms.astype(np.uint64).reshape(-1, 2), values.astype(np.uint64).reshape(-1, 2)
    words = (atoms[:, 0] << 28) | (values[:, 0] << 20) | (atoms[:, 1] << 8) | values[:, 1]
    shifts = np.array([32, 24, 16, 8, 0], dtype=np.uint64)
    payload = ((words[:, None] >> shifts) & 255).astype(np.uint8).tobytes()
    octets = np.frombuffer(payload, dtype=np.uint8).astype(np.uint64).reshape(-1, 5)
    decoded = np.bitwise_or.reduce(octets << shifts, axis=1)
    decoded_atoms = np.stack([(decoded >> 28) & 4095, (decoded >> 8) & 4095], axis=-1).astype(np.int64)
    decoded_bins = np.stack([(decoded >> 20) & 255, decoded & 255], axis=-1).astype(np.int64)
    if not np.array_equal(decoded_atoms, atoms) or not np.array_equal(decoded_bins, values):
        raise AssertionError('Payload round trip failed')
    return payload, decoded_atoms, decoded_bins


def evaluate(checkpoint, paths, device, mode, workers):
    model = LASER.load_from_checkpoint(str(checkpoint), map_location='cpu', strict=True).to(device).eval()
    bound = float(model.bottleneck.coefficient_quantization_max)
    expected = (4096, 8) if mode == 'recovered_q8' else (8192, 7)
    assert (model.bottleneck.num_embeddings, model.bottleneck.coefficient_quantization_bits) == expected
    assert model.bottleneck.sparsity_level == 2
    # The recovered recipe uses the uniform256 quantizer, unlike the current
    # signed127 recipe. Apply each exact quantizer externally before packing.
    model.bottleneck.coefficient_quantization_bits = 0
    if mode == 'current_dense_diagnostic':
        model.bypass_bottleneck = True
    jobs, payload_bits, samples = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.inference_mode():
        for index, path in enumerate(paths):
            ref, rate = sf.read(path, dtype='float32')
            if rate != 48000 or ref.ndim != 1:
                raise ValueError(f'Expected mono 48 kHz: {path}')
            padded, length = align_mdct(torch.from_numpy(ref).to(device)[None, None])
            latent, _, codes = model.encode(padded)
            if mode == 'current_q7':
                integers = codes.values.float().clamp(-bound, bound).div(bound / 63).round().long()
                payload = pack_frames(codes.support.cpu().numpy(), integers.cpu().numpy())
                atoms, integers = unpack_frames(payload)
                support = torch.from_numpy(atoms).to(device).reshape_as(codes.support)
                values = torch.from_numpy(integers).to(device).reshape_as(codes.values).float() * (bound / 63)
                reconstructed = model.decode_from_atoms_and_coeffs(support, values)
                payload_bits += len(payload) * 8
            elif mode == 'recovered_q8':
                bins, _ = quantize_sparse_coefficients(codes.values, coeff_vocab_size=256, coeff_max=bound)
                payload, atoms, bins = q8_payload_roundtrip(codes.support.cpu().numpy(), bins.cpu().numpy())
                support = torch.from_numpy(atoms).to(device).reshape_as(codes.support)
                indices = torch.from_numpy(bins).to(device).reshape_as(codes.values)
                levels = build_coeff_bin_values(coeff_vocab_size=256, coeff_max=bound, device=device)
                reconstructed = model.decode_from_atoms_and_coeffs(support, levels[indices])
                payload_bits += len(payload) * 8
            else:
                reconstructed = model.decode(latent)
            decoded = reconstructed[0, 0, :length].float().cpu().numpy()
            if decoded.shape != ref.shape or not np.isfinite(decoded).all():
                raise RuntimeError(f'Invalid reconstruction: {path}')
            jobs.append(pool.submit(measure, (Path(path).name, ref, decoded.clip(-1, 1))))
            samples += length
            if (index + 1) % 16 == 0:
                print(f'{mode}: reconstructed {index + 1}/{len(paths)}', flush=True)
        rows = [job.result() for job in jobs]
    del model
    torch.cuda.empty_cache()
    rate = payload_bits / (samples / 48000) / 1000 if payload_bits else None
    return rows, rate


def paired_comparison(candidate, reference):
    a = {r['utterance']: r['visqol_audio48k'] for r in candidate}
    b = {r['utterance']: r['visqol_audio48k'] for r in reference}
    assert a.keys() == b.keys()
    keys = sorted(a)
    delta = np.array([a[k] - b[k] for k in keys])
    speakers = np.array([k.split('_')[0] for k in keys])
    unique = np.unique(speakers)
    groups = [delta[speakers == speaker] for speaker in unique]
    rng = np.random.default_rng(20260911)
    draws = rng.integers(0, len(groups), size=(5000, len(groups)))
    totals = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    means = totals[draws].sum(1) / counts[draws].sum(1)
    return {'mean': float(delta.mean()), 'speaker_bootstrap_ci95': np.quantile(means, [.025, .975]).tolist(),
            'speakers': len(groups), 'resamples': 5000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/mdctcodec_recovery/oyg7smih/validation_comparison'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--metric-workers', type=int, default=8)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--mode', choices=['online', 'disabled'], default='online')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads(Path('outputs/mdctcodec_recovery/oyg7smih/benchmark/manifest.json').read_text())
    old_test = {Path(p['path']).name for p in old_manifest['items']}
    current = json.loads(Path('outputs/vctk_mdctcodec_stage1_6kbps_low_lr/validation128.json').read_text())
    current_test = {Path(p).name for p in current['locked_test']}
    paths = [p for p in current['validation'] if Path(p).name not in old_test | current_test]
    names = {Path(p).name for p in paths}
    assert len(paths) == len(names) and names.isdisjoint(old_test | current_test)
    assert len({Path(p).parent.name for p in paths}) == 8
    manifest = {'purpose': 'validation comparison; excludes both historical and current test sets',
                'validation': paths, 'excluded_historical_test': sorted(old_test),
                'excluded_current_test': sorted(current_test), 'source_validation_count': 128}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    sources = {
        'current': json.loads(Path('outputs/vctk_mdctcodec_stage1_6kbps_low_lr/schedule_comparison/results.json').read_text())['sources']['control'],
        'recovered': json.loads(Path('outputs/mdctcodec_recovery/ds6thjoj/recovery.json').read_text()),
    }
    for source in sources.values():
        assert hashlib.sha256(Path(source['checkpoint']).read_bytes()).hexdigest() == source['sha256']
    modes = ['current_q7', 'recovered_q8']
    if args.diagnostics:
        modes += ['current_float_diagnostic', 'current_dense_diagnostic']
    records, summaries = {}, {}
    for mode in modes:
        source = sources['recovered' if mode == 'recovered_q8' else 'current']
        rows, rate = evaluate(source['checkpoint'], paths, torch.device(args.device), mode, args.metric_workers)
        records[mode] = rows
        summaries[mode] = {'visqol': float(np.mean([r['visqol_audio48k'] for r in rows])),
                           'stoi': float(np.mean([r['stoi'] for r in rows])), 'payload_kbps': rate,
                           'valid_6kbps_codec': rate is not None}
        (args.output / f'{mode}.json').write_text(json.dumps(rows, indent=2))
        print(mode, summaries[mode], flush=True)
    deltas = {mode: paired_comparison(rows, records['current_q7']) for mode, rows in records.items()
              if mode != 'current_q7'}
    result = {'utterances': len(paths), 'sources': sources, 'results': summaries,
              'minus_current_q7': deltas, 'training_modified': False,
              'notes': ['Current and recovered models selected by their original validation metrics.',
                        'Float and dense bypass modes are diagnostics with no 6 kbps bitrate claim.',
                        'Bypass changes decoder input distribution; its score is not an achievable upper bound.',
                        'Payload rates exclude headers and shared coefficient bounds.',
                        'Recovered q8 uses uniform256; current q7 uses signed127.']}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    if args.mode == 'online':
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', mode='online',
                         name='mdctcodec-6kbps-recovered-q8-vs-current-q7-validation',
                         job_type='validation-comparison', dir=str(args.output),
                         config={'manifest': manifest, 'sources': sources})
        for mode, values in summaries.items():
            for key, value in values.items():
                if value is not None: run.summary[f'validation/{mode}/{key}'] = value
        artifact = wandb.Artifact(f'mdctcodec-recovered-recipe-comparison-{run.id}', type='benchmark', metadata=result)
        for path in args.output.glob('*.json'): artifact.add_file(str(path), name=path.name)
        artifact.add_file(__file__, name='compare_mdctcodec_recovered_recipe.py')
        run.log_artifact(artifact).wait()
        (args.output / 'run.json').write_text(json.dumps({'id': run.id, 'url': run.url}))
        print(run.url, flush=True)
        run.finish()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
