#!/usr/bin/env python3
"""Evaluate LASER, released MDCTCodec, EnCodec, and DAC on identical VCTK files.

No test data are used for checkpoint selection or coefficient calibration.
ViSQOL uses Google's native binary in 48 kHz audio mode for every system.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from pystoi import stoi
from torchaudio.functional import resample
import wandb

from src.audio_logging import _measure_visqol, is_visqol_available
from src.models.audio_codec import build_mdctcodec_official_backbone
from src.models.laser import LASER
from src.mdctcodec_bitstream import pack_frames, unpack_frames
from src.stage1_setup import build_stage1_datamodule, data_config_from_section


def align_mdct(x):
    length = x.shape[-1]
    aligned = (math.ceil((length / 40 + 1) / 8) * 8 - 1) * 40
    return F.pad(x, (0, aligned - length)), length


def pad_encodec_segment_tail(x, segment_length, stride):
    """Avoid EnCodec 0.1.1's short final overlap-add allocation bug.

    Its output allocation assumes the final frame reaches beyond every earlier
    frame. A tail shorter than the 10 ms overlap violates that assumption.
    At most 479 padding samples restore it; all comparisons trim to the original
    length and bitrate accounting includes the additional encoded samples.
    """
    tail = (x.shape[-1] - 1) % stride + 1
    return F.pad(x, (0, max(0, segment_length - stride - tail)))


def load_reference(root, device):
    spec = importlib.util.spec_from_file_location('mdctcodec_reference_quantize', root / 'quantize.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder, decoder = build_mdctcodec_official_backbone()
    quantizer = module.ResidualVectorQuantize(input_dim=32, codebook_dim=32, n_codebooks=4, codebook_size=1024)
    state = torch.load(root / 'encoder_00200000_onlyvctk', map_location='cpu', weights_only=False)['encoder']
    encoder.load_state_dict({k: v for k, v in state.items() if not k.startswith('quantizer.')}, strict=True)
    quantizer.load_state_dict({k.removeprefix('quantizer.'): v for k, v in state.items() if k.startswith('quantizer.')}, strict=True)
    decoder.load_state_dict(torch.load(root / 'decoder_00200000_onlyvctk', map_location='cpu', weights_only=False)['decoder'], strict=True)
    return [m.to(device).eval() for m in (encoder, quantizer, decoder)]


def measure(item):
    name, reference, output = item
    score = _measure_visqol(torch.from_numpy(reference), torch.from_numpy(output), sample_rate=48000)
    if score is None or not np.isfinite(score):
        raise RuntimeError(f'Official ViSQOL failed for {name}; no proxy metric is substituted')
    return {'utterance': name, 'visqol_audio48k': score,
            'stoi': float(stoi(reference, output, 48000)),
            'snr_db': float(10 * np.log10((np.square(reference).sum() + 1e-12) /
                                        (np.square(reference - output).sum() + 1e-12)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, default=Path('outputs/mdctcodec_reference/MDCTCodec'))
    parser.add_argument('--num-items', type=int, default=200)
    parser.add_argument('--metric-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--systems', nargs='+', default=None)
    parser.add_argument('--name', default='mdctcodec-laser-vctk-matched-test')
    parser.add_argument('--baseline-dir', nargs='*', type=Path, default=[])
    parser.add_argument('--laser-sparsity', type=int, choices=[2, 4])
    parser.add_argument('--coefficient-max', type=float)
    parser.add_argument('--quantizer', choices=['uniform128', 'signed127'], default='signed127')
    args = parser.parse_args()
    if not is_visqol_available():
        raise RuntimeError('Official ViSQOL is required')
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base='1.2'):
        cfg = compose(config_name='vctk_mdctcodec_stage1')
    cfg.data.num_workers = 0
    dm = build_stage1_datamodule(data_config_from_section(cfg.data)); dm.setup()
    by_speaker = {}
    for path in dm.test_dataset.paths:
        if sf.info(path).duration >= 2:
            by_speaker.setdefault(path.parent.name, []).append(path)
    rng = random.Random(args.seed)
    for paths in by_speaker.values(): rng.shuffle(paths)
    selected = []
    # Interleave speakers so even small smoke comparisons cover the same pool.
    for index in range(max(map(len, by_speaker.values()))):
        for speaker in sorted(by_speaker):
            if index < len(by_speaker[speaker]): selected.append(by_speaker[speaker][index])
    selected = selected[:args.num_items]
    if len(selected) != args.num_items:
        raise ValueError('Insufficient eligible test utterances')
    references = []
    for path in selected:
        audio, rate = sf.read(path, dtype='float32')
        if rate != 48000 or audio.ndim != 1: raise ValueError(f'Expected mono 48 kHz: {path}')
        references.append(audio)
    manifest = {'seed': args.seed, 'sample_rate': 48000, 'level_policy': 'original',
                'split': 'mdctcodec_disjoint', 'validation': [str(p) for p in dm.val_dataset.paths],
                'test': [str(p) for p in selected], 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'metric': 'Google ViSQOL v3.3.3, audio mode, 48 kHz',
                'selection': 'balanced speakers, >=2 seconds, full utterances',
                'baseline_training': 'official pretrained weights; training corpora differ'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    run = wandb.init(project='laser', entity='helloimlixin-rutgers', mode=args.mode,
                     name=args.name, group='mdctcodec-vctk-stage1-20260911',
                     job_type='benchmark', config=manifest, dir=str(args.output))
    (args.output / 'run.json').write_text(json.dumps({'id': run.id, 'url': run.url}))
    overrides = {'sparsity_level': args.laser_sparsity} if args.laser_sparsity else {}
    model = LASER.load_from_checkpoint(str(args.checkpoint), map_location='cpu', **overrides).to(device).eval()
    # Benchmark the float upper bound and the transmitted quantized values
    # separately, even when the checkpoint was trained with quantization.
    stored_bound = model.bottleneck.coefficient_quantization_max
    stored_bits = model.bottleneck.coefficient_quantization_bits
    model.bottleneck.coefficient_quantization_bits = 0
    # Calibrate on randomly selected training crops only; store reproducibly.
    indices = torch.randperm(len(dm.train_dataset))[:1024].tolist()
    values = []
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            x = torch.stack([dm.train_dataset[i][0] for i in indices[start:start+32]]).to(device)
            values.append(model.encode(x)[2].values.detach().float().cpu().reshape(-1))
    coefficient_max = (args.coefficient_max or (stored_bound if stored_bits else None)
                       or float(torch.quantile(torch.cat(values).abs(), 0.999)))
    calibration = {'split': 'train', 'indices': indices, 'percentile': 99.9, 'coefficient_max': coefficient_max,
                   'bits': 7, 'quantizer': args.quantizer, 'sparsity_level': model.bottleneck.sparsity_level,
                   'checkpoint_sha256': manifest['checkpoint_sha256']}
    (args.output / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    run.config.update({'coefficient_max': coefficient_max, 'quantizer': args.quantizer,
                       'laser_sparsity': model.bottleneck.sparsity_level,
                       'checkpoint_coefficient_bits': stored_bits})
    summaries = []
    laser_rate_name = 'laser_q7_6kbps' if model.bottleneck.sparsity_level == 2 else 'laser_q7_12kbps'
    systems = args.systems or ['laser_float_coefficients', laser_rate_name, 'mdctcodec_rvq_6kbps',
               'encodec_6kbps', 'encodec_12kbps', 'encodec_48khz_6kbps', 'encodec_48khz_12kbps',
               'dac_6kbps', 'dac_12kbps']
    allowed = {'laser_float_coefficients', 'laser_q7_6kbps', 'laser_q7_12kbps', 'mdctcodec_rvq_6kbps',
               'encodec_6kbps', 'encodec_12kbps', 'encodec_48khz_6kbps', 'encodec_48khz_12kbps',
               'dac_6kbps', 'dac_12kbps', 'dac_8kbps_model_6kbps'}
    if not set(systems) <= allowed: raise ValueError('Unknown benchmark system')
    if 'laser_q7_6kbps' in systems and model.bottleneck.sparsity_level != 2:
        raise ValueError('6 kbps requires K=2 with this 150 Hz, 8192-atom architecture')
    if 'laser_q7_6kbps' in systems and args.quantizer != 'signed127':
        raise ValueError('6 kbps payload uses the same signed127 quantizer as training')
    if 'laser_q7_12kbps' in systems and model.bottleneck.sparsity_level != 4:
        raise ValueError('12 kbps label requires K=4')
    with ThreadPoolExecutor(max_workers=args.metric_workers) as pool:
        for system in systems:
            if system.startswith('mdctcodec'):
                baseline = load_reference(args.reference_root, device)
            elif system.startswith('encodec'):
                from encodec import EncodecModel
                factory = EncodecModel.encodec_model_48khz if '48khz' in system else EncodecModel.encodec_model_24khz
                baseline = factory().to(device).eval()
                baseline.set_target_bandwidth(6.0 if '6kbps' in system else 12.0)
            elif system.startswith('dac'):
                import dac
                pretrained_rate = '8kbps' if system == 'dac_8kbps_model_6kbps' else '16kbps'
                baseline = dac.DAC.load(dac.utils.download(model_type='44khz', model_bitrate=pretrained_rate)).to(device).eval()
            else: baseline = None
            jobs = []; inference_seconds = 0.; payload_bits = 0
            with torch.inference_mode():
                for i, (path, ref) in enumerate(zip(selected, references)):
                    x = torch.from_numpy(ref).to(device)[None, None]
                    torch.cuda.synchronize(device); before = time.perf_counter()
                    if system.startswith('laser'):
                        padded, length = align_mdct(x)
                        z, _, codes = model.encode(padded)
                        if system.startswith('laser_q7'):
                            if args.quantizer == 'signed127':
                                q = codes.values.float().clamp(-coefficient_max, coefficient_max).div(coefficient_max / 63).round().long()
                                if system == 'laser_q7_6kbps':
                                    # Decode the actual five-byte-per-frame payload.
                                    payload = pack_frames(codes.support.cpu().numpy(), q.cpu().numpy())
                                    atoms, integers = unpack_frames(payload)
                                    supports = torch.from_numpy(atoms).to(device).reshape_as(codes.support)
                                    q = torch.from_numpy(integers).to(device).reshape_as(q)
                                    payload_bits += len(payload) * 8
                                else:
                                    supports = codes.support
                                    payload_bits += codes.support.numel() * (13 + 7)
                                coefficients = q.float() * (coefficient_max / 63)
                            else:
                                supports = codes.support
                                q = ((codes.values / coefficient_max).clamp(-1, 1) + 1).mul(63.5).round().long()
                                coefficients = (q.float() / 63.5 - 1) * coefficient_max
                                payload_bits += codes.support.numel() * (13 + 7)
                            # Reconstruct using discrete supports and quantized coefficients.
                            y = model.decode_from_atoms_and_coeffs(supports, coefficients)[..., :length]
                        else: y = model.decode(z)[..., :length]
                    elif system.startswith('mdctcodec'):
                        padded, length = align_mdct(x)
                        encoder, quantizer, decoder = baseline
                        q = quantizer(encoder(padded))
                        y = decoder(q[0])[..., :length]
                        payload_bits += q[1].numel() * 10
                    elif system.startswith('encodec'):
                        source = x.repeat(1, 2, 1) if '48khz' in system else resample(x, 48000, 24000)
                        if baseline.segment_length is not None:
                            source = pad_encodec_segment_tail(source, baseline.segment_length, baseline.segment_stride)
                        frames = baseline.encode(source)
                        decoded = baseline.decode(frames)
                        y = (decoded.mean(dim=1, keepdim=True) if '48khz' in system else
                             resample(decoded, 24000, 48000))[..., :x.shape[-1]]
                        payload_bits += sum(frame[0].numel() * 10 for frame in frames)
                    else:
                        x44 = resample(x, 48000, 44100)
                        # 44.1 kHz / 512 * 10 bits: 7/14 codebooks give
                        # 6.03/12.06 kbps; report the actual payload below.
                        q = baseline.encode(baseline.preprocess(x44, 44100), n_quantizers=7 if '6kbps' in system else 14)
                        y44 = baseline.decode(q[0])[..., :x44.shape[-1]]
                        y = resample(y44, 44100, 48000)[..., :x.shape[-1]]
                        payload_bits += q[1].numel() * 10
                    torch.cuda.synchronize(device); inference_seconds += time.perf_counter() - before
                    out = y[0, 0].float().cpu().numpy()
                    if len(out) != len(ref) or not np.isfinite(out).all():
                        raise RuntimeError(f'Invalid reconstruction: {system}/{path.name}')
                    # The same PCM saturation is applied to all systems for ViSQOL.
                    out = out.clip(-1, 1)
                    jobs.append(pool.submit(measure, (path.name, ref, out)))
                    if i < 2:
                        run.log({f'audio/{system}/{i}': wandb.Audio(out, sample_rate=48000),
                                 f'audio/reference/{i}': wandb.Audio(ref, sample_rate=48000)})
                    if (i+1) % 25 == 0: print(system, i+1, '/', len(selected), flush=True)
            rows = [job.result() for job in jobs]
            (args.output / f'{system}.json').write_text(json.dumps(rows, indent=2))
            duration = sum(len(x) / 48000 for x in references)
            summary = {'system': system, 'utterances': len(rows),
                       'visqol_audio48k': float(np.mean([r['visqol_audio48k'] for r in rows])),
                       'visqol_standard_error': float(np.std([r['visqol_audio48k'] for r in rows], ddof=1) / np.sqrt(len(rows))),
                       'stoi': float(np.mean([r['stoi'] for r in rows])),
                       'payload_kbps': payload_bits / duration / 1000 if payload_bits else None,
                       'inference_rtf': inference_seconds / duration,
                       'model_sample_rate': 44100 if system.startswith('dac') else
                           (24000 if system.startswith('encodec') and '48khz' not in system else 48000),
                       'rate_scope': 'fixed-width tokens; excludes headers and shared calibration',
                       'training': 'recovered LASER' if baseline is None else 'official pretrained'}
            if system.startswith('dac'):
                summary['pretrained_model_bitrate'] = pretrained_rate
            summaries.append(summary)
            run.log({f'comparison/{system}/{key}': value for key, value in summary.items() if isinstance(value, (int, float))})
            (args.output / 'results.json').write_text(json.dumps(summaries, indent=2))
            print(json.dumps(summary), flush=True)
            del baseline
            torch.cuda.empty_cache()
    for directory in args.baseline_dir:
        baseline_manifest = json.loads((directory / 'manifest.json').read_text())
        if baseline_manifest['test'] != manifest['test'] or baseline_manifest['sample_rate'] != 48000:
            raise ValueError(f'Baseline test manifest does not match: {directory}')
        baseline_rows = json.loads((directory / 'results.json').read_text())
        for row in baseline_rows:
            if not row['system'].startswith('laser'):
                summaries.append(row)
    columns = list(dict.fromkeys(k for row in summaries for k in row))
    summaries = [{k: row.get(k) for k in columns} for row in summaries]
    (args.output / 'results.json').write_text(json.dumps(summaries, indent=2))
    run.log({'comparison/results': wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in summaries])})
    with (args.output / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in summaries:
        if r['payload_kbps'] is not None:
            ax.errorbar(r['payload_kbps'], r['visqol_audio48k'], yerr=1.96*r['visqol_standard_error'], fmt='o', label=r['system'])
    ax.set(xlabel='Token payload (kbps), excluding headers', ylabel='ViSQOL audio 48 kHz', title='Same VCTK test utterances; official pretrained baselines')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(args.output / 'comparison.png', dpi=160); plt.close(fig)
    run.log({'comparison/plot': wandb.Image(str(args.output / 'comparison.png'))})
    artifact = wandb.Artifact(f'mdctcodec-vctk-comparison-{run.id}', type='benchmark', metadata=manifest)
    for p in args.output.iterdir():
        if p.suffix in {'.json', '.csv', '.png'}: artifact.add_file(str(p), name=p.name)
    logged = run.log_artifact(artifact)
    if args.mode == 'online':
        logged.wait()
    print('W&B:', run.url, flush=True); run.finish()


if __name__ == '__main__':
    main()
