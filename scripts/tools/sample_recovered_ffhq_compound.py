"""Sample the recovered FFHQ compound-v4 trainer in an isolated runtime.

Outputs are qualitative comparisons, not new FID measurements. Each setting
uses the same seed, sampling batch size, and frozen checkpoint pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def settings():
    baseline = dict(atom_temperature=1.0, atom_top_k=250, atom_top_p=1.0,
                    coeff_temperature=1.0, coeff_top_p=0.85)
    result = []
    for temperature in (0.70, 0.85, 1.00, 1.15, 1.30):
        for top_k in (64, 128, 250, 512, 2048):
            name = f'atom_t{temperature:.2f}_k{top_k:04d}'
            policy = dict(baseline, atom_temperature=temperature, atom_top_k=top_k)
            result.append(dict(name=name, family='atom', policy=policy))
    for temperature in (0.70, 0.85, 1.00, 1.15, 1.30):
        for top_p in (0.70, 0.85, 0.95, 1.00):
            if temperature == 1.0 and top_p == 0.85:
                continue  # Reuse the atom sweep's identical baseline.
            name = f'coeff_t{temperature:.2f}_p{top_p:.2f}'
            policy = dict(baseline, coeff_temperature=temperature, coeff_top_p=top_p)
            result.append(dict(name=name, family='coefficient', policy=policy))
    return result


def load_models(args):
    sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    from src import ffhq_v4_archived as training
    assert Path(training.__file__).resolve().is_relative_to(args.runtime.resolve())
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    checkpoint = (args.weights_directory / args.checkpoint.name
                  if args.weights_directory else args.checkpoint)
    stage1_checkpoint = (args.weights_directory / args.stage1_checkpoint.name
                         if args.weights_directory else args.stage1_checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = payload['config']
    assert payload['epoch'] == 200 and payload['global_step'] == 109200
    assert config['compound_tokens'] and config['num_atoms'] == 2048
    assert config['coeff_vocab_size'] == 2048
    model = training.build_model(
        config['num_atoms'] + config['coeff_vocab_size'], config['num_atoms'],
        compound=True, coeff_vocab_size=config['coeff_vocab_size'],
        compound_refiner_layers=config['compound_refiner_layers'],
        compound_geometry_head=(config['geometry_loss_weight'] > 0 and
                                not config['compound_distribution_geometry']),
        compound_micro_transformer_layers=config['compound_micro_transformer_layers'],
        compound_depth_specific_coeff_heads=config['compound_depth_specific_coeff_heads'],
        architecture=config['architecture'], num_classes=1,
    )
    compatibility = model.load_state_dict(payload['state_dict'], strict=True)
    assert not compatibility.missing_keys and not compatibility.unexpected_keys
    model.to(args.device).eval().requires_grad_(False)
    aux = training.LaserAux(
        stage1_checkpoint, config['num_atoms'], config['coeff_vocab_size'],
        config['coeff_max'], config['coeff_scale'],
        attn_resolutions=config['stage1_attn_resolutions'],
        coeff_scales=config['coeff_scales'],
    ).to(args.device).eval().requires_grad_(False)
    assert torch.isfinite(aux.dictionary).all()
    assert tuple(aux.dictionary.shape) == (256, 2048)
    info = dict(epoch=payload['epoch'], global_step=payload['global_step'],
                historical_fid=payload['fid'], strict_stage2_load=True,
                stage1_non_quantizer_keys_match=True,
                parameters=sum(p.numel() for p in model.parameters()),
                coeff_scales=aux.coeff_scales.tolist(),
                block_size=list(model.block_size),
                optimizer_present='optimizer' in payload,
                scheduler_present=payload.get('scheduler') is not None,
                loaded_stage2=str(checkpoint.resolve()),
                loaded_stage1=str(stage1_checkpoint.resolve()),
                runtime=str(args.runtime.resolve()), torch_version=torch.__version__,
                cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(args.device),
                sampler_autocast_dtype=str(torch.get_autocast_dtype('cuda')),
                decoder_dtype=str(next(aux.decoder.parameters()).dtype))
    del payload
    return model, aux, info


def worker(args):
    import torch
    from torchvision.utils import save_image
    model, aux, info = load_models(args)
    write_json(args.output / f'load-worker-{args.worker_index}.json', info)
    chosen = settings()
    if args.baseline_only:
        chosen = [s for s in chosen if s['name'] == 'atom_t1.00_k0250']
    for index, setting in enumerate(chosen):
        if index % args.worker_count != args.worker_index:
            continue
        destination = args.output / setting['name']
        destination.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        all_images, all_atoms, all_coefficients = [], [], []
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, args.samples, args.batch_size):
                count = min(args.samples - start, args.batch_size)
                atoms, coefficients = model.sample_compound(
                    count, aux, amp=True, **setting['policy'])
                assert atoms.min() >= 0 and atoms.max() < 2048
                assert coefficients.min() >= 0 and coefficients.max() < 2048
                assert (atoms[..., 0] != atoms[..., 1]).all()
                all_atoms.append(atoms.cpu())
                all_coefficients.append(coefficients.cpu())
                for offset in range(0, count, args.decode_batch_size):
                    decoded = aux.decode_compound(
                        atoms[offset:offset + args.decode_batch_size],
                        coefficients[offset:offset + args.decode_batch_size])
                    assert decoded.dtype == torch.float32 and torch.isfinite(decoded).all()
                    all_images.append(((decoded + 1) * 0.5).clamp(0, 1).cpu())
        images = torch.cat(all_images)
        assert images.shape == (args.samples, 3, 256, 256)
        atoms, coefficients = torch.cat(all_atoms), torch.cat(all_coefficients)
        save_image(images, destination / 'grid.png', nrow=8, padding=0)
        save_image(images[:16], destination / 'preview.png', nrow=4, padding=0)
        torch.save(dict(atoms=atoms, coefficient_ids=coefficients), destination / 'codes.pt')
        image_dir = destination / 'images'
        image_dir.mkdir(exist_ok=True)
        for index, img in enumerate(images):
            save_image(img, image_dir / f'{index:03d}.png')
        result = dict(**setting, seed=args.seed, samples=args.samples,
                      batch_size=args.batch_size, decode_batch_size=args.decode_batch_size,
                      seconds=time.monotonic() - started,
                      pixel_mean=float(images.mean()), pixel_std=float(images.std()),
                      codes_sha256=hashlib.sha256((destination / 'codes.pt').read_bytes()).hexdigest(),
                      all_pixels_finite=True, all_supports_distinct=True)
        write_json(destination / 'result.json', result)
        print(json.dumps(result), flush=True)


def contact_sheet(args, family):
    from PIL import Image, ImageDraw
    temperatures = (0.70, 0.85, 1.00, 1.15, 1.30)
    columns = (64, 128, 250, 512, 2048) if family == 'atom' else (0.70, 0.85, 0.95, 1.00)
    panel, label_height, margin = 400, 30, 12
    sheet = Image.new('RGB', (len(columns) * (panel + margin) + margin,
                              len(temperatures) * (panel + label_height + margin) + 65), 'white')
    draw = ImageDraw.Draw(sheet)
    title = f'FFHQ recovered epoch 200 | {family} sweep | seed {args.seed}'
    draw.text((margin, 8), title, fill='black', font_size=22)
    subtitle = ('Coefficient T=1.0, p=0.85; atom p=1.0' if family == 'atom'
                else 'Atom T=1.0, top-k=250, p=1.0')
    draw.text((margin, 36), subtitle + ' | First 16 samples shown per setting', fill='black', font_size=17)
    for row, temperature in enumerate(temperatures):
        for column, value in enumerate(columns):
            if family == 'atom':
                name = f'atom_t{temperature:.2f}_k{value:04d}'
                label = f'T={temperature:.2f}, top-k={value}'
                baseline = temperature == 1 and value == 250
            else:
                name = f'coeff_t{temperature:.2f}_p{value:.2f}'
                label = f'T={temperature:.2f}, top-p={value:.2f}'
                baseline = temperature == 1 and value == .85
                if baseline:
                    name = 'atom_t1.00_k0250'
            x, y = margin + column * (panel + margin), 65 + row * (panel + label_height + margin)
            draw.text((x, y), label + (' [original]' if baseline else ''), fill='black', font_size=18)
            with Image.open(args.output / name / 'preview.png') as preview:
                sheet.paste(preview.resize((panel, panel), Image.Resampling.LANCZOS), (x, y + label_height))
    target = args.output / f'{family}_contact_sheet.jpg'
    sheet.save(target, quality=95)
    return target.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'checkpoint', 'stage1-checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--weights-directory', type=Path,
                        help='Optional directory containing verified local copies of both checkpoints')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--decode-batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--worker-count', type=int, default=1)
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.decode_batch_size) <= 0:
        parser.error('Sample and batch counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        worker(args)
        return
    devices = args.devices.split(',')
    if args.baseline_only:
        devices = devices[:1]
    processes, handles = [], []
    for index, device in enumerate(devices):
        log = (args.output / f'worker-{index}.log').open('w')
        handles.append(log)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                   '--worker-index', str(index), '--worker-count', str(len(devices)),
                   '--device', f'cuda:{device}']
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   MPLCONFIGDIR='/tmp/laser-matplotlib')
        processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env))
    try:
        while any(p.poll() is None for p in processes):
            done = len(list(args.output.glob('*/result.json')))
            print(json.dumps(dict(completed_settings=done,
                                  workers_running=sum(p.poll() is None for p in processes))), flush=True)
            failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Sweep worker failed; inspect worker logs: {failed}')
            time.sleep(15)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Sweep worker failed; inspect worker logs')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in handles:
            log.close()
    results = [json.loads(p.read_text()) for p in sorted(args.output.glob('*/result.json'))]
    expected = 1 if args.baseline_only else len(settings())
    assert len(results) == expected, (len(results), expected)
    sheets = [] if args.baseline_only else [contact_sheet(args, f) for f in ('atom', 'coefficient')]
    write_json(args.output / 'manifest.json', dict(
        source_run='helloimlixin-rutgers/laser/ffhqcmp0804205803',
        checkpoint=str(args.checkpoint.resolve()), stage1_checkpoint=str(args.stage1_checkpoint.resolve()),
        runtime=str(args.runtime.resolve()), seed=args.seed,
        settings=results, total_images=len(results) * args.samples,
        contact_sheets=sheets, qualitative_only=True, new_fid_computed=False))
    print(f'Complete: {args.output / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
