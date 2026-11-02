"""Bounded 4096-symbol joint-token pilot on a frozen epoch-10 VAR tokenizer.

Calibration and screening use disjoint images from the TRAIN archive. The
screen reports paired PSNR, not validation rFID or a generation benchmark.
No source checkpoint or source dataset is modified.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from src.models.multiscale_laser_var import LaserVQVAE, PATCH_NUMS
from src.models.scratch_var import ScratchVQVAE
from src.models.joint_multiscale_var import JointMultiScaleQuantizer, JointLaserVAR
from src.training.laser_joint_vocabulary import fit_coefficient_candidates


def report(path, payload):
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


def sample_training_archive(archive, cache, classes):
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    with tarfile.open(archive, 'r:') as outer:
        members = sorted((m for m in outer.getmembers() if m.isfile() and m.name.endswith('.tar')),
                         key=lambda m: m.name)
        if len(members) != 1000:
            raise ValueError(f'Expected 1000 ImageNet class archives, found {len(members)}')
        indices = np.linspace(0, 999, classes, dtype=int).tolist()
        for position, label in enumerate(indices):
            member = members[label]
            with tarfile.open(fileobj=outer.extractfile(member), mode='r:') as inner:
                chosen = 0
                for image in inner:
                    if not image.isfile() or not image.name.lower().endswith(('.jpeg', '.jpg')):
                        continue
                    data = inner.extractfile(image).read()
                    with Image.open(io.BytesIO(data)) as im:
                        im.verify()
                    path = cache / f'{label:04d}-{chosen}.JPEG'
                    path.write_bytes(data)
                    rows.append(dict(path=str(path), label=label,
                                     split='fit' if chosen == 0 else 'screen',
                                     archive_class=member.name, archive_image=image.name))
                    chosen += 1
                    if chosen == 2:
                        break
                if chosen != 2:
                    raise ValueError(f'Missing usable training images for {member.name}')
            if position % 16 == 0:
                print(json.dumps(dict(phase='extract_train', classes=position+1)), flush=True)
    manifest_path.write_text(json.dumps(rows, indent=2))
    return rows


def load_source(path, kind, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint['progress'] != dict(epoch=10, batch=0, step=100090):
        raise ValueError('This pilot requires the verified epoch-10 checkpoint')
    if kind == 'laser':
        model = LaserVQVAE(pretrained=None, sparsity=2, coefficient_bins=257)
    else:
        model = ScratchVQVAE(channels=32, atoms=4096, ch=160, patch_nums=PATCH_NUMS)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def cache_latents(model, rows, cache, device):
    transform = transforms.Compose([transforms.Resize(288, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(.5, .5)])
    result = {}
    for split in ('fit', 'screen'):
        path = cache / f'{split}-latents.pt'
        if path.exists():
            result[split] = torch.load(path, weights_only=True)
            continue
        images, latents, labels = [], [], []
        selected = [r for r in rows if r['split'] == split]
        for first in range(0, len(selected), 8):
            batch = []
            for row in selected[first:first+8]:
                with Image.open(row['path']) as im:
                    batch.append(transform(im.convert('RGB')))
                labels.append(row['label'])
            x = torch.stack(batch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model.quant_conv(model.encoder(x.to(device)))
            images.append(x); latents.append(z.float().cpu())
            if first % 32 == 0:
                print(json.dumps(dict(phase='encode', split=split, images=first+len(x))), flush=True)
        result[split] = dict(images=torch.cat(images), latents=torch.cat(latents), labels=torch.tensor(labels))
        torch.save(result[split], path)
    return result


@torch.inference_mode()
def fit_book(source, latents, levels, device):
    q = source.quantize
    codes = q.decompose(latents.to(device))
    score = torch.zeros(q.vocab_size, device=device)
    residuals, accumulated, offset = [], torch.zeros_like(latents, device=device), 0
    for scale, pn in enumerate(q.v_patch_nums):
        atoms = codes['atoms'][:, offset:offset+pn*pn]
        bins = codes['coefficients'][:, offset:offset+pn*pn]
        values = q.coefficient_values(bins, scale)
        score.scatter_add_(0, atoms.flatten(), values.square().flatten())
        residuals.append(F.interpolate(latents.to(device)-accumulated, (pn, pn), mode='area')
                         .permute(0, 2, 3, 1).reshape(-1, q.Cvae))
        accumulated.add_(q.contribution(q.embed(atoms, bins, scale), scale))
        offset += pn*pn
    directions = 4096 // levels
    selected = score.topk(directions).indices.sort().values
    dictionary = q.normalized_dictionary()[:, selected].contiguous()
    tables = []
    for scale, signals in enumerate(residuals):
        pieces = [q.dictionary.batch_omp_with_support(x.T, dictionary) for x in signals.split(2048)]
        atoms = torch.cat([x[0] for x in pieces])
        values = torch.cat([x[1] for x in pieces])
        books, _ = fit_coefficient_candidates(atoms, values, directions,
                                              counts=(levels,), iterations=15, prior_weight=4.)
        tables.append(books[levels])
    return JointMultiScaleQuantizer(q, dictionary, torch.stack(tables)).to(device), selected.cpu()


@torch.inference_mode()
def screen(source, quantizer, data, device, vq=None):
    totals = dict(psnr=0., latent_mse=0., image_mse=0.)
    preview = []
    for first in range(0, len(data['latents']), 8):
        images = data['images'][first:first+8].to(device)
        z = data['latents'][first:first+8].to(device)
        if vq is not None:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                vq_latent = vq.quant_conv(vq.encoder(images)).float()
            # The upstream training forward consults the distributed usage
            # tracker even in eval mode. Use its inference quantization path.
            with torch.autocast('cuda', enabled=False):
                vq_quantized = vq.quantize.f_to_idxBl_or_fhat(vq_latent, to_fhat=True)[-1]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                reconstructed = vq.decoder(vq.post_quant_conv(vq_quantized))
            quantized = None
        else:
            result = quantizer.decompose(z)
            quantized = result['latent']
            if 'codes' in result:
                decoded, inputs = quantizer.from_codes(result['codes'])
                torch.testing.assert_close(decoded, quantized)
                torch.testing.assert_close(inputs, result['inputs'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                reconstructed = source.decoder(source.post_quant_conv(quantized))
        mse = ((reconstructed.float().clamp(-1, 1)-images) / 2).square().flatten(1).mean(-1)
        totals['psnr'] += float((-10*mse.log10()).sum())
        totals['image_mse'] += float(mse.sum())
        if quantized is not None:
            totals['latent_mse'] += float((quantized-z).square().flatten(1).mean(-1).sum())
        if first == 0:
            preview = reconstructed[:8].float().cpu()
    return {k:v/len(data['latents']) for k, v in totals.items()}, preview


def learning_check(q, data, device, steps, out):
    # A deliberately bounded fixed-batch fit, not evidence of generalization.
    with torch.inference_mode():
        encoded = q.decompose(data['latents'][:8].to(device))
    inputs, codes = encoded['inputs'].clone(), encoded['codes'].clone()
    labels = data['labels'][:8].to(device)
    torch.manual_seed(20261102)
    model = JointLaserVAR(q).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.e-4, betas=(.9, .95), weight_decay=.05)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        initial = float(model(labels, inputs, codes))
    trace = []
    model.train()
    for step in range(steps):
        start = step % 4 * 2
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(labels[start:start+2], inputs[start:start+2], codes[start:start+2])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
        optimizer.step()
        row = dict(step=step+1, loss=float(loss.detach()), grad_norm=float(norm))
        trace.append(row)
        if step % 10 == 0:
            print(json.dumps(dict(phase='fixed_batch_learning', **row)), flush=True)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        final = float(model(labels, inputs, codes))
        sampled = model.sample(labels[:2], seed=3)
    if not torch.isfinite(sampled).all():
        raise FloatingPointError('Nonfinite sampled latent')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), steps=steps,
                    vocabulary=4096, source_stage1_step=100090), out/'pilot-prior.pt')
    return dict(initial_fixed_batch_nll=initial, final_fixed_batch_nll=final,
                steps=steps, images=8, microbatch=2, architecture='VAR-d16',
                note='Repeated fixed-batch fit only; no heldout generation claim', trace=trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--vq-checkpoint', type=Path)
    parser.add_argument('--train-archive', required=True, type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classes', type=int, default=128)
    parser.add_argument('--levels', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--learning-steps', type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    if any(level not in (2, 4, 8, 16) for level in args.levels):
        raise ValueError('Choose 2, 4, 8 or 16 levels in a 4096-symbol vocabulary')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    with args.checkpoint.open('rb') as handle:
        source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    if source_hash != '3f0c1b79272e8d6177e284f45e9acd923b5bc678890b2b9766a14e07ce92b6aa':
        raise ValueError('Source checkpoint differs from the verified epoch-10 LASER model')
    rows = sample_training_archive(args.train_archive, args.cache/'images', args.classes)
    source = load_source(args.checkpoint, 'laser', device)
    cached = cache_latents(source, rows, args.cache, device)
    results = dict(source_checkpoint_sha256=source_hash, vocabulary=4096,
        fitting_split='train', screen_split='disjoint heldout TRAIN images',
        fit_images=args.classes, screen_images=args.classes, spatial_sites=680,
        contributions_per_site=2, categorical_events_per_image=1360,
        nominal_bits_per_image=16320, vq_nominal_bits_per_image=8160,
        coefficient_tables='scale-specific; shared across sparse depths',
        validation_fid_measured=False, variants={})
    baseline, preview = screen(source, source.quantize, cached['screen'], device)
    results['source_laser'] = baseline
    save_image(preview.mul(.5).add(.5), args.output/'source-laser.png', nrow=4)
    report(args.output/'results.json', results)
    if args.vq_checkpoint:
        vq = load_source(args.vq_checkpoint, 'vq', device)
        result, preview = screen(source, None, cached['screen'], device, vq=vq)
        results['vq_control'] = result
        save_image(preview.mul(.5).add(.5), args.output/'vq-control.png', nrow=4)
        del vq
        report(args.output/'results.json', results)
    best, best_psnr = None, -float('inf')
    for levels in args.levels:
        with torch.inference_mode():
            q, selected = fit_book(source, cached['fit']['latents'], levels, device)
            result, preview = screen(source, q, cached['screen'], device)
        name = f'directions{4096//levels}-levels{levels}'
        spec = dict(kind='multiscale_joint_atom_coefficient', format_version=1,
                    dictionary=q.dictionary.cpu(), coefficients=q.coefficients.cpu(),
                    source_atom_ids=selected, source_checkpoint_sha256=source_hash,
                    patch_nums=list(PATCH_NUMS), sparsity=2, vocabulary=4096,
                    encoder_decoder_frozen=True, original_directions_frozen=True)
        torch.save(spec, args.output/f'{name}-codebook.pt')
        save_image(preview.mul(.5).add(.5), args.output/f'{name}.png', nrow=4)
        results['variants'][name] = result
        report(args.output/'results.json', results)
        if result['psnr'] > best_psnr:
            best, best_psnr = name, result['psnr']
    results['screen_selected_variant'] = best
    results['selection_note'] = 'Pilot selection uses heldout training PSNR; validation not accessed'
    report(args.output/'results.json', results)
    if args.learning_steps:
        spec = torch.load(args.output/f'{best}-codebook.pt', weights_only=True)
        q = JointMultiScaleQuantizer(source.quantize, spec['dictionary'], spec['coefficients']).to(device)
        results['learning_check'] = learning_check(q, cached['fit'], device,
                                                   args.learning_steps, args.output)
        report(args.output/'results.json', results)


if __name__ == '__main__':
    main()
