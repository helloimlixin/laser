#!/usr/bin/env python3
"""Evaluate the archived trained RVQ control on the locked current VCTK test set."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import soundfile as sf
import torch
import wandb

from archive.scripts.benchmark_mdctcodec_vctk import align_mdct, load_reference, measure


def payload_roundtrip(codes):
    """Four 10-bit RVQ IDs per 150 Hz frame, serialized in five bytes."""
    source = np.asarray(codes)
    if source.ndim != 3 or source.shape[:2] != (1, 4):
        raise ValueError('Expected code indices [1,4,T]')
    if not np.issubdtype(source.dtype, np.integer) or np.any((source < 0) | (source > 1023)):
        raise ValueError('RVQ IDs must be integers in [0,1023]')
    fields = source[0].T.astype(np.uint64)
    offsets = np.array([30, 20, 10, 0], dtype=np.uint64)
    byte_offsets = np.array([32, 24, 16, 8, 0], dtype=np.uint64)
    words = np.bitwise_or.reduce(fields << offsets, axis=1)
    payload = ((words[:, None] >> byte_offsets) & 255).astype(np.uint8).tobytes()
    octets = np.frombuffer(payload, dtype=np.uint8).astype(np.uint64).reshape(-1, 5)
    parsed = np.bitwise_or.reduce(octets << byte_offsets, axis=1)
    restored = ((parsed[:, None] >> offsets) & 1023).astype(np.int64).T[None]
    if not np.array_equal(source, restored):
        raise AssertionError('Serialized RVQ payload changed the code indices')
    return payload, restored


def load_trained_reference(checkpoint, reference_root, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hp, weights = state['hyper_parameters'], state['state_dict']
    expected = {'bottleneck_type': 'mdctcodec_rvq', 'audio_backbone': 'mdctcodec_official',
                'embedding_dim': 32, 'audio_mdct_num_coefficients': 40,
                'audio_mdct_vae_temporal_downsample_factor': 8,
                'audio_mdct_vae_hidden_channels': 256,
                'audio_mdct_vae_convnext_intermediate_channels': 512,
                'audio_mdct_vae_num_residual_layers': 8, 'bypass_bottleneck': False}
    for key, value in expected.items():
        if hp.get(key) != value:
            raise ValueError(f'Unsupported trained RVQ setting {key}={hp.get(key)!r}')
    if any(k.startswith(('pre_bottleneck.', 'post_bottleneck.')) for k in weights):
        raise ValueError('Checkpoint includes nonidentity bottleneck adapters')
    # Build the verified authors' architecture; replace every inference weight
    # with the archived trained checkpoint using strict loads on all modules.
    modules = load_reference(reference_root, device)
    for module, prefix in zip(modules, ['encoder.', 'bottleneck.quantizer.', 'decoder.']):
        module.load_state_dict({k.removeprefix(prefix): v for k, v in weights.items()
                                if k.startswith(prefix)}, strict=True)
    return modules, {'epoch': state['epoch'], 'global_step': state['global_step'],
                     'generator_updates': int(weights['_manual_train_step']),
                     'strict_load': True, 'backbone_settings': expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, default=Path('outputs/mdctcodec_recovery/trained_rvq_recovery.json'))
    parser.add_argument('--manifest', type=Path, default=Path('outputs/mdctcodec_benchmark_vctk200/manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/mdctcodec_benchmark_trained_rvq'))
    parser.add_argument('--reference-root', type=Path, default=Path('outputs/mdctcodec_reference/MDCTCodec'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--metric-workers', type=int, default=8)
    parser.add_argument('--mode', choices=['online', 'disabled'], default='online')
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text())
    checkpoint = Path(provenance['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == provenance['sha256']
    manifest = json.loads(args.manifest.read_text())
    assert manifest['sample_rate'] == 48000
    assert len(manifest['test']) == len(set(manifest['test'])) == 200
    assert not set(manifest['test']).intersection(manifest['validation'])
    manifest.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=provenance['sha256'],
                    baseline_training='archived trained VCTK RVQ control; not equal training budget to current LASER')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    device = torch.device(args.device)
    (encoder, quantizer, decoder), verification = load_trained_reference(checkpoint, args.reference_root, device)
    verification['quantizer_source_sha256'] = hashlib.sha256((args.reference_root / 'quantize.py').read_bytes()).hexdigest()
    (args.output / 'verification.json').write_text(json.dumps(verification, indent=2))
    system = 'mdctcodec_trained_rvq_6kbps'
    run = wandb.init(entity='helloimlixin-rutgers', project='laser', mode=args.mode,
                     name='mdctcodec-trained-rvq-6kbps-locked-vctk200', job_type='benchmark',
                     dir=str(args.output), config={'manifest': manifest, 'provenance': provenance,
                                                  'verification': verification})
    jobs, bits, samples, elapsed = [], 0, 0, 0.0
    with ThreadPoolExecutor(max_workers=args.metric_workers) as pool, torch.inference_mode():
        for index, path in enumerate(manifest['test']):
            reference, rate = sf.read(path, dtype='float32')
            if rate != 48000 or reference.ndim != 1:
                raise ValueError(f'Expected mono 48 kHz: {path}')
            inputs, length = align_mdct(torch.from_numpy(reference).to(device)[None, None])
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            latent, codes, _, _, _ = quantizer(encoder(inputs), n_quantizers=4)
            payload, restored = payload_roundtrip(codes.cpu().numpy())
            decoded_latent = quantizer.from_codes(torch.from_numpy(restored).to(device))[0]
            if index == 0:
                torch.testing.assert_close(decoded_latent, latent, rtol=1e-4, atol=1e-5)
            decoded = decoder(decoded_latent)[0, 0, :length].float().cpu().numpy()
            torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            if decoded.shape != reference.shape or not np.isfinite(decoded).all():
                raise RuntimeError(f'Invalid decoded waveform: {path}')
            decoded = decoded.clip(-1, 1)
            jobs.append(pool.submit(measure, (Path(path).name, reference, decoded)))
            bits += len(payload) * 8
            samples += length
            if index < 2 and args.mode == 'online':
                run.log({f'audio/{system}/{index}': wandb.Audio(decoded, sample_rate=48000),
                         f'audio/reference/{index}': wandb.Audio(reference, sample_rate=48000)})
            if (index + 1) % 25 == 0:
                print(f'{system}: reconstructed {index + 1}/200', flush=True)
        rows = [job.result() for job in jobs]
    summary = {'system': system, 'utterances': len(rows),
               'visqol_audio48k': float(np.mean([r['visqol_audio48k'] for r in rows])),
               'visqol_standard_error': float(np.std([r['visqol_audio48k'] for r in rows], ddof=1) / np.sqrt(len(rows))),
               'stoi': float(np.mean([r['stoi'] for r in rows])), 'payload_kbps': bits / (samples / 48000) / 1000,
               'inference_rtf': elapsed / (samples / 48000), 'model_sample_rate': 48000,
               'rate_scope': 'serialized four 10-bit IDs per frame; excludes headers',
               'training': 'archived VCTK RVQ control; different update count from current LASER'}
    (args.output / f'{system}.json').write_text(json.dumps(rows, indent=2))
    (args.output / 'results.json').write_text(json.dumps([summary], indent=2))
    run.log({f'comparison/{system}/{k}': v for k, v in summary.items() if isinstance(v, (int, float))})
    if args.mode == 'online':
        artifact = wandb.Artifact(f'mdctcodec-trained-rvq-vctk200-{run.id}', type='benchmark', metadata=summary)
        for path in args.output.glob('*.json'): artifact.add_file(str(path), name=path.name)
        artifact.add_file(__file__, name='benchmark_mdctcodec_trained_rvq.py')
        run.log_artifact(artifact).wait()
        (args.output / 'run.json').write_text(json.dumps({'id': run.id, 'url': run.url}))
        print(run.url, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    run.finish()


if __name__ == '__main__':
    main()
