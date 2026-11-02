"""Measure continuous, 512-bin, and previous soft-target reconstruction quality."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from src.training.imagenet_compound_runtime import make_aux, STAGE1_SHA
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training.rqtransformer import val_image_transform
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


def main():
    base = Path(os.environ['LASER_RUN_BASE'])
    torch.set_num_threads(4)
    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(261026+rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    aux = make_aux(base/'inputs/stage1-tokenizer.pt', device)
    def digest():
        h = hashlib.sha256()
        for name, value in aux.state_dict().items():
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before = digest()
    dataset = TarImageNet(base/'data', 'val', val_image_transform())
    assert len(dataset) == 50000 and len(dataset.classes) == 1000
    loader = DataLoader(Subset(dataset, range(rank, len(dataset), world)), batch_size=16,
                        num_workers=4, multiprocessing_context='spawn', pin_memory=True)
    hard = DistributedOriginalRQVAEMetrics(device)
    continuous = DistributedOriginalRQVAEMetrics(device, inception=hard.inception)
    soft = DistributedOriginalRQVAEMetrics(device, inception=hard.inception)
    errors = torch.zeros(4, dtype=torch.float64, device=device)
    started = time.monotonic()
    with torch.inference_mode():
        for index, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            aux.clamp_coeffs = False
            atoms, values = aux.encode_sparse_components(images)
            aux.clamp_coeffs = True
            tokens, targets = aux.sparse_targets(atoms, values.clamp(-3, 3),
                                                compact=True, hard=True, stochastic=False)
            assert ((tokens[..., 0::2] >= 0) & (tokens[..., 0::2] < 16384)).all()
            assert ((tokens[..., 1::2] >= 16384) & (tokens[..., 1::2] < 16896)).all()
            assert torch.equal(tokens[..., 0::2], targets[0])
            decoded = aux.decode_tokens(tokens)
            old_tokens, _ = aux.sparse_targets(atoms, values.clamp(-3, 3), compact=True,
                                              hard=False, stochastic=True, temp=.0078125)
            old_decoded = aux.decode_tokens(old_tokens)
            z = aux.physical_contributions(atoms, values).sum(-2).permute(0, 3, 1, 2).contiguous()
            control = aux.decoder(aux.post_quant_conv(z)).clamp(-1, 1)
            hard_values = aux.coeff_bins[tokens[..., 1::2] - 16384]
            soft_values = aux.coeff_bins[old_tokens[..., 1::2] - 16384]
            errors += torch.stack(((hard_values-values).square().sum(),
                (soft_values-values).square().sum(), values.square().sum(),
                torch.tensor(values.numel(), device=device)))
            assert torch.isfinite(decoded).all() and torch.isfinite(control).all()
            hard.update(images.mul(.5).add(.5), real=True)
            hard.update(decoded.mul(.5).add(.5), real=False)
            continuous.update(control.mul(.5).add(.5), real=False)
            soft.update(old_decoded.mul(.5).add(.5), real=False)
            if rank == 0 and index % 50 == 0:
                write_json(base/'records/quantization-progress.json', dict(images_per_rank=(index+1)*16,
                    seconds=time.monotonic()-started))
                print('Reconstruction verification rank0 images', (index+1)*16, flush=True)
    # Copy local real moments before compute() performs distributed reduction.
    for metric in (continuous, soft):
        metric.real_sum.copy_(hard.real_sum)
        metric.real_cross.copy_(hard.real_cross)
        metric.real_count.copy_(hard.real_count)
    qr, _, _ = hard.compute()
    cr, _, _ = continuous.compute()
    sr, _, _ = soft.compute()
    dist.all_reduce(errors)
    assert int(hard.real_count) == int(hard.fake_count) == 50000
    assert digest() == before, 'Frozen tokenizer changed during evaluation'
    passed = qr < 4.4 and abs(qr - cr) < .1
    if rank == 0:
        report = dict(passed=passed, continuous_rfid=cr, quantized_rfid=qr,
            previous_soft_target_rfid=sr, quantization_rfid_delta=qr-cr,
            hard_coefficient_mse=float(errors[0]/errors[3]),
            previous_soft_coefficient_mse=float(errors[1]/errors[3]),
            images=50000, reference_split='val', metric_backend='original-rqvae',
            atom_range_half_open=[0, 16384], coefficient_range_half_open=[16384, 16896],
            coefficient_bins=512, stage1_sha256=STAGE1_SHA,
            frozen_tokenizer_digest=before, seconds=time.monotonic()-started)
        write_json(base/'records/quantization.json', report)
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError('512-bin tokenizer verification failed; training must not start')


if __name__ == '__main__':
    main()
