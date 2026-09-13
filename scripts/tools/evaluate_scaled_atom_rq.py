#!/usr/bin/env python3
"""Frozen Church scaled-atom RQ study; never updates the source tokenizer/prior."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0] = [str(UPSTREAM), str(ROOT)]
os.environ.setdefault('TORCH_HOME', '/workspace/tmp/official-rqvae-eval-cache')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw
from rqvae.img_datasets.lsun import LSUNClass
from rqvae.metrics.fid import get_inception_model, frechet_distance
from src.scaled_atom_rq import (FrozenSparseBackbone, ScaledAtomRQ,
    continuous_matching_pursuit, orthogonal_matching_pursuit, fit_signed_levels)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_contact_sheet(images, output):
    n = min(6, len(next(iter(images.values()))))
    canvas = Image.new('RGB', (140+256*n, 280*len(images)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (name, xs) in enumerate(images.items()):
        draw.text((8, row*280+16), name, fill='black')
        grid = make_grid(xs[:n], nrow=n, padding=0)
        pixels = (grid.permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8')
        canvas.paste(Image.fromarray(pixels), (140, row*280))
    canvas.save(output)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    parser.add_argument('--data-root', default='/tmp/laser-sign-data')
    parser.add_argument('--calibration-images', type=int, default=2048)
    parser.add_argument('--calibration-start', type=int, default=60000)
    parser.add_argument('--screen-images', type=int, default=4096)
    parser.add_argument('--screen-start', type=int, default=0)
    parser.add_argument('--levels', type=int, nargs='+', default=[2,4,8,16,32])
    parser.add_argument('--levels-file', type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--skip-fid', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    start = time.time()
    def status(phase, **kwargs):
        payload = dict(phase=phase, elapsed_seconds=time.time()-start, **kwargs)
        print(json.dumps(payload), flush=True)
        write_json(args.output/'status.json', payload)
    status('loading_frozen_backbone')
    checkpoint_hash = sha256(args.checkpoint)
    expected = '93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388'
    if checkpoint_hash != expected:
        raise ValueError(f'Unexpected frozen Church checkpoint hash: {checkpoint_hash}')
    model = FrozenSparseBackbone(args.checkpoint).to(args.device).eval()
    before_hash = fingerprint(model)
    gram = model.dictionary.T @ model.dictionary
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
    train = LSUNClass(args.data_root, 'church', transform)
    def loader(dataset, indices):
        return DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    calibration_indices = list(range(args.calibration_start, args.calibration_start+args.calibration_images))
    screen_indices = list(range(args.screen_start, args.screen_start+args.screen_images))
    assert min(calibration_indices+screen_indices) >= 0
    assert max(calibration_indices+screen_indices) < len(train)
    assert not set(calibration_indices).intersection(screen_indices)
    if args.levels_file:
        calibrated = json.loads(args.levels_file.read_text())
        assert calibrated['checkpoint_sha256'] == checkpoint_hash
        assert not set(calibrated['calibration_indices']).intersection(screen_indices)
        levels = {b: torch.tensor(calibrated['levels'][str(b)], device=args.device) for b in args.levels}
    else:
        coefficients = []
        for images, _ in loader(train, calibration_indices):
            z = model.encode(images.to(args.device, non_blocking=True))
            mp = continuous_matching_pursuit(z, model.dictionary)
            coefficients.append(mp['coefficients'].cpu())
            done = sum(len(x) for x in coefficients)
            if done % 128 == 0 or done == len(calibration_indices):
                status('calibrating', images=done, total=len(calibration_indices))
        coefficients = torch.cat(coefficients).to(args.device)
        levels = {b: fit_signed_levels(coefficients, b) for b in args.levels}
        calibrated = dict(checkpoint_sha256=checkpoint_hash, calibration_indices=calibration_indices,
            method='symmetric scalar Lloyd on continuous MP4 coefficients; fixed zero; shared across depths',
            coefficient_abs_quantiles=torch.quantile(coefficients.abs().flatten(),
                torch.tensor([0., .5, .9, .99, .999, 1.], device=args.device)).cpu().tolist(),
            levels={b: value.cpu().tolist() for b, value in levels.items()})
    write_json(args.output/'levels.json', calibrated)
    quantizers = {f'rq{b}': ScaledAtomRQ(model.dictionary, level).to(args.device) for b, level in levels.items()}
    names = ['omp4', 'continuous_mp4', *quantizers]
    manifest = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        checkpoint_sha256=checkpoint_hash, frozen_state_before_sha256=before_hash,
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__), ROOT/'src/scaled_atom_rq.py']},
        code_shape=[8,8,4], latent_shape=[8,8,256], dictionary_shape=list(model.dictionary.shape),
        precision='FP32 throughout; CUDA matmul and cuDNN TF32 disabled',
        vocabularies={name: q.vocab_size for name,q in quantizers.items()},
        zero_token=0, token_formula='1 + atom_index * signed_level_count + coefficient_bin',
        screen_indices=screen_indices, calibration_indices=calibrated['calibration_indices'],
        no_optimizer=True, no_stage1_training=True)
    write_json(args.output/'manifest.json', manifest)
    inception = None if args.skip_fid else get_inception_model().to(args.device).eval().requires_grad_(False)
    reference_path = ROOT/'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz'
    reference = np.load(reference_path) if inception is not None else None
    results = {}

    def evaluate(dataset, indices, split, fid):
        totals = {name: dict(pixel_mse=0., latent_mse=0., latent_energy=0., feature_l2=0.) for name in names}
        usage = {name: torch.zeros(4,q.vocab_size, dtype=torch.long, device=args.device) for name,q in quantizers.items()}
        feats = {name: [] for name in ['original', *names]} if fid else {}
        done = 0
        for images, _ in loader(dataset, indices):
            images = images.to(args.device, non_blocking=True)
            z = model.encode(images)
            original = (images*.5+.5).clamp(0,1)
            original_features = inception(original) if fid else None
            if fid:
                feats['original'].append(original_features.cpu().numpy())
            outputs = {'omp4': orthogonal_matching_pursuit(z, model.dictionary, gram),
                       'continuous_mp4': continuous_matching_pursuit(z, model.dictionary)}
            outputs.update({name:q.quantize(z) for name,q in quantizers.items()})
            contact = {'original': original.cpu()} if done == 0 else None
            for name, value in outputs.items():
                reconstruction = (model.decode(value['quantized'])*.5+.5).clamp(0,1)
                totals[name]['pixel_mse'] += (reconstruction-original).square().flatten(1).mean(1).sum().item()
                totals[name]['latent_mse'] += (z-value['quantized']).square().flatten(1).mean(1).sum().item()
                totals[name]['latent_energy'] += z.square().flatten(1).mean(1).sum().item()
                if fid:
                    feature = inception(reconstruction)
                    totals[name]['feature_l2'] += (feature-original_features).square().sum(1).sum().item()
                    feats[name].append(feature.cpu().numpy())
                if name in usage:
                    for d in range(4):
                        usage[name][d] += torch.bincount(value['codes'][...,d].reshape(-1), minlength=usage[name].shape[-1])
                if contact is not None:
                    contact[name] = reconstruction.cpu()
            if contact is not None:
                make_contact_sheet(contact, args.output/f'{split}-reconstructions.png')
            done += len(images)
            if done % 128 == 0 or done == len(indices):
                status('reconstructing', split=split, images=done, total=len(indices),
                       running_latent_mse={name: val['latent_mse']/done for name,val in totals.items()})
        metrics = {}
        for name, total in totals.items():
            metrics[name] = {key: value/done for key,value in total.items() if key != 'feature_l2' or fid}
            metrics[name]['psnr_from_mean_mse'] = -10*np.log10(metrics[name]['pixel_mse'])
            metrics[name]['latent_relative_mse'] = total['latent_mse']/total['latent_energy']
            if name in usage:
                counts = usage[name].cpu().numpy()
                np.save(args.output/f'{split}-{name}-token-counts.npy', counts)
                probs = counts/counts.sum(1, keepdims=True)
                metrics[name].update(vocab_size=counts.shape[1],
                    used_tokens=int((counts.sum(0)>0).sum()),
                    used_tokens_per_depth=(counts>0).sum(1).tolist(),
                    zero_fraction_per_depth=probs[:,0].tolist(),
                    entropy_bits_per_depth=(-(probs*np.log2(np.maximum(probs,1e-30))).sum(1)).tolist())
        results[split] = dict(images=done, metrics=metrics)
        write_json(args.output/'metrics.json', results)
        if fid:
            stats = {}
            for name, chunks in feats.items():
                features = np.concatenate(chunks)
                np.save(args.output/f'{split}-{name}-features.npy', features)
                stats[name] = (features.mean(0), np.cov(features, rowvar=False))
            for name in names:
                status('computing_fid', split=split, variant=name)
                mu,cov = stats[name]
                metrics[name]['rfid_matched'] = float(frechet_distance(mu,cov,*stats['original']))
                metrics[name]['rfid_published_reference'] = float(frechet_distance(mu,cov,reference['mu'],reference['sigma']))
                write_json(args.output/'metrics.json', results)
        return metrics

    if not args.skip_validation:
        if 'church_val' not in LSUNClass.valid_categories:
            LSUNClass.valid_categories.append('church_val')
        validation = LSUNClass(args.data_root, 'church_val', transform)
        evaluate(validation, list(range(len(validation))), 'validation', False)
    evaluate(train, screen_indices, 'screen', not args.skip_fid)
    after_hash = fingerprint(model)
    assert before_hash == after_hash, 'Frozen source state changed during evaluation'
    manifest['frozen_state_after_sha256'] = after_hash
    manifest['frozen_state_unchanged'] = True
    manifest['elapsed_seconds'] = time.time()-start
    write_json(args.output/'manifest.json', manifest)
    status('complete', metrics_file=str(args.output/'metrics.json'))


if __name__ == '__main__':
    main()
