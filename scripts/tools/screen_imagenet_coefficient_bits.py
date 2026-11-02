"""Compare coefficient bit widths with one frozen FP32 encoder/OMP pass.

The 512-bin reference and every candidate reconstruct the same real images
and exactly the same supports. Small screens select candidates; the selected
codec must pass a full 50k-image paired reconstruction check before training.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
from src.training.imagenet_compound_runtime import make_aux, STAGE1_SHA
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training.rqtransformer import val_image_transform


def frozen_digest(aux):
    digest = hashlib.sha256()
    for name, value in aux.state_dict().items():
        if name != 'coeff_bins':
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def balanced_indices(labels, samples):
    if samples % 1000:
        raise ValueError('Use an equal number of validation images per class')
    count = samples // 1000
    result = []
    for label in range(1000):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < count:
            raise ValueError(f'Insufficient validation images for class {label}')
        result.extend(candidates[:count].tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bins', type=int, nargs='+', default=[16, 32, 64, 128, 256, 512])
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--memory-fraction', type=float, default=0.)
    args = parser.parse_args()
    bins = sorted(set(args.bins + [512]))
    if any(n < 2 or n & (n - 1) for n in bins):
        raise ValueError('Coefficient bins must be powers of two')
    rank, world, local = (int(os.environ.get(k, default))
                          for k, default in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    torch.set_num_threads(4)
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    aux = make_aux(args.base/'inputs/stage1-tokenizer.pt', device)
    before = frozen_digest(aux)
    aux.clamp_coeffs = False
    dataset = TarImageNet(args.base/'data', 'val', val_image_transform())
    indices = balanced_indices(dataset.labels, args.samples)
    loader = DataLoader(Subset(dataset, indices[rank::world]), batch_size=args.batch_size,
                        num_workers=2, multiprocessing_context='spawn', pin_memory=True)
    real = DistributedOriginalRQVAEMetrics(device)
    metrics = {key: DistributedOriginalRQVAEMetrics(device, inception=real.inception)
               for key in ['continuous'] + [str(n) for n in bins]}
    centers = {n: torch.linspace(-3., 3., n, device=device) for n in bins}
    errors = torch.zeros(len(bins), 3, device=device, dtype=torch.float64)
    clipped = torch.zeros(2, device=device, dtype=torch.float64)
    seen, started = 0, time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            atoms, values = aux.encode_sparse_components(images)
            clipped += torch.stack(((values.abs() > 3).sum(), values.new_tensor(values.numel())))
            atom_vectors = aux.dictionary.T[atoms]
            scales = aux.coeff_scales.view(1, 1, 1, -1)

            def decode(coefficients):
                physical = coefficients*scales
                z = (atom_vectors * physical[..., None]).sum(-2)
                z = z.permute(0, 3, 1, 2).contiguous()
                return aux.decoder(aux.post_quant_conv(z)).clamp(-1, 1)

            real.update(images.mul(.5).add(.5), real=True)
            metrics['continuous'].update(decode(values).mul(.5).add(.5), real=False)
            for index, n in enumerate(bins):
                ids = (values[..., None] - centers[n]).abs().argmin(-1)
                quantized = centers[n][ids]
                reconstruction = decode(quantized)
                if batch_index == 0:
                    old = aux.coeff_bins, aux.coeff_vocab_size, aux.vocab_size
                    aux.coeff_bins, aux.coeff_vocab_size, aux.vocab_size = centers[n], n, 16384+n
                    tokens = torch.stack((atoms, ids+16384), -1).flatten(-2)
                    expected = aux.decode_tokens(tokens)
                    torch.testing.assert_close(reconstruction, expected, rtol=1e-5, atol=1e-5)
                    aux.coeff_bins, aux.coeff_vocab_size, aux.vocab_size = old
                metrics[str(n)].update(reconstruction.mul(.5).add(.5), real=False)
                delta = quantized-values
                errors[index] += torch.stack((delta.square().sum(),
                    (delta*aux.coeff_scales).square().sum(), delta.new_tensor(delta.numel())))
            seen += len(images)
            if rank == 0 and batch_index % 10 == 0:
                record = dict(phase='reconstruction', images_per_rank=seen,
                              samples=args.samples, world_size=world, bins=bins,
                              seconds=time.monotonic()-started, pid=os.getpid())
                write_json(args.output/'progress.json', record)
                print(json.dumps(record), flush=True)
    if world > 1:
        dist.all_reduce(errors)
        dist.all_reduce(clipped)
    measured = {}
    for key, metric in metrics.items():
        metric.real_sum.copy_(real.real_sum)
        metric.real_cross.copy_(real.real_cross)
        metric.real_count.copy_(real.real_count)
        measured[key] = metric.compute()[0]
        assert int(metric.real_count) == int(metric.fake_count) == args.samples
        if rank == 0:
            print(json.dumps(dict(phase='metric', codec=key, rfid=measured[key])), flush=True)
    assert frozen_digest(aux) == before, 'Frozen encoder/dictionary/decoder changed'
    rows = []
    for index, n in enumerate(bins):
        delta = measured[str(n)]-measured['512']
        rows.append(dict(bins=n, coefficient_bits=int(math.log2(n)),
            nominal_bits_per_image=64*4*(14+int(math.log2(n))),
            rfid=measured[str(n)], delta_rfid_vs_512=delta,
            coefficient_mse=float(errors[index, 0]/errors[index, 2]),
            physical_coefficient_mse=float(errors[index, 1]/errors[index, 2]),
            full_gate_passed=(args.samples == 50000 and measured[str(n)] < 4.4 and delta < .1)))
    report = dict(samples=args.samples, world_size=world, stage1_sha256=STAGE1_SHA,
        frozen_weights_unchanged=True, continuous_rfid=measured['continuous'],
        reference='same selected validation images for every codec', metric_backend='original-rqvae',
        normalized_coefficients_range=[-3., 3.], per_depth_scales=aux.coeff_scales.tolist(),
        source_clipping_fraction=float(clipped[0]/clipped[1]), results=rows,
        token_codec_parity_passed=True, seconds=time.monotonic()-started,
        training_updates=0, full_gate=dict(samples=50000, maximum_rfid=4.4, maximum_delta_vs_512=.1))
    if rank == 0:
        write_json(args.output/'result.json', report)
        print(json.dumps(report), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
