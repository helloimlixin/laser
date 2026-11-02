#!/usr/bin/env python3
"""Matched full-validation reconstruction audit and training-only sigma calibration."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(os.environ['LASER_RQ_BASE'])
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
sys.path[:0] = [str(BASE/'source'), str(BASE/'upstream'), str(BASE)]
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src import rq_upstream, rqvae_metrics as metrics
from src.training.laser_fine_joint_recipe import FineJointLaserRecipe
from src.training.laser_hard_ar import nearest_coefficient_ids
from src.training.rq_imagenet_runtime import TarImageNet, write_json, sha256
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms


@torch.inference_mode()
def main():
    rank, world, local = [int(os.environ.get(k, v)) for k, v in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    torch.set_num_threads(4); torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    profile = FineJointLaserRecipe(BASE)
    out = Path('/workspace/Projects/laser/outputs')/profile.run_id
    aux, _ = profile.tokenizer(device)
    metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    source = metrics.DistributedOriginalRQVAEMetrics(device, inception=inception)
    converted = metrics.DistributedOriginalRQVAEMetrics(device, inception=inception)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    loader = DataLoader(Subset(data, range(rank, len(data), world)), batch_size=8,
                        num_workers=4, pin_memory=True)
    source_bins = torch.linspace(-3, 3, 2048, device=device)
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for index, (images, _) in enumerate(loader):
        images = images.to(device)
        latent = aux.encode(images)
        codes, gamma = aux.latent_codes(latent)
        atom_ids = codes // aux.coeff_vocab_size
        basis = aux.get_code_emb_with_depth(codes)[0]
        # Reconstruct both from exactly the same source support / gamma.
        from src.orthogonal_sparse_codec import ordered_support_basis
        q, _ = ordered_support_basis(aux.dictionary.T[atom_ids])
        ids = nearest_coefficient_ids(gamma/aux.coeff_scales, source_bins)
        reference = (q * (source_bins[ids]*aux.coeff_scales)[..., None]).sum(-2)
        fine = basis.sum(-2)
        for metric, z in [(source, reference), (converted, fine)]:
            reconstruction = aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous()))
            metric.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
            metric.update(reconstruction.mul(.5).add(.5).clamp(0, 1), real=False)
        totals[:2] += torch.stack(((latent-reference).square().sum().double(),
                                   (latent-fine).square().sum().double()))
        totals[2] += latent.numel()
        if rank == 0 and index % 25 == 0:
            write_json(out/'tokenizer-audit-progress.json', dict(phase='matched_rfid',
                images_on_rank=min((index+1)*8, 12500), world_size=world, time=time.time()))
    source_fid, _, _ = source.compute(); fine_fid, _, _ = converted.compute()
    dist.all_reduce(totals)
    assert int(source.fake_count) == int(converted.fake_count) == 50000
    assert int(source.real_count) == int(converted.real_count) == 50000
    report = dict(source_rfid=source_fid, fine512_rfid=fine_fid, drift=fine_fid-source_fid,
        real_images=50000, reconstructed_images=50000,
        reference='same complete validation set / original RQ Inception',
        source_latent_mse=float(totals[0]/totals[2]), fine512_latent_mse=float(totals[1]/totals[2]),
        source_sha256=sha256(BASE/'inputs/stage1-model.pt'),
        passed=fine_fid <= source_fid+.05, maximum_drift=.05, time=time.time())
    if rank == 0: write_json(out/'tokenizer-fidelity-verification.json', report)
    assert report['passed'], report
    del inception, source, converted
    torch.cuda.empty_cache()
    latents = np.load('/tmp/laser-rq-joint-vocabulary-study-20261015/temperature-latents.npy', mmap_mode='r')
    sigmas = [.0625, .07, .075, .08, .1, .125]
    errors = torch.zeros(1+len(sigmas), device=device, dtype=torch.float64)
    for first in range(rank*4, len(latents), world*4):
        x = torch.from_numpy(latents[first:first+4].copy()).to(device)
        codes, gamma = aux.latent_codes(x)
        atoms = codes // aux.coeff_vocab_size
        from src.orthogonal_sparse_codec import ordered_support_basis
        q, _ = ordered_support_basis(aux.dictionary.T[atoms])
        hard = aux.get_code_emb_with_depth(codes)[0].sum(-2)
        errors[0] += (x-hard).square().sum().double()
        values = aux.coeff_scales[:, None]*aux.coeff_bins[None, :]
        for i, sigma in enumerate(sigmas, 1):
            p = (-(gamma[..., None]-values).square()/(2*sigma*sigma)).softmax(-1)
            mean = (p*values).sum(-1)
            variance = (p*(values-mean[..., None]).square()).sum(-1)
            expected = (q*mean[..., None]).sum(-2)
            # Exact expectation, using the orthonormal source-support basis.
            errors[i] += ((x-expected).square().sum().double() + variance.sum().double())
    dist.all_reduce(errors)
    ratios = {s: float(errors[i]/errors[0]) for i, s in enumerate(sigmas, 1)}
    original = json.loads(Path('/workspace/Projects/laser/outputs/imagenet-laser-joint-vocabulary-study-20261015/selected.json').read_text())
    reference_ratio = original['original_rq_measured_distortion_ratio']
    selected = min(sigmas, key=lambda s: abs(ratios[s]-reference_ratio))
    if rank == 0:
        write_json(out/'coefficient-policy-verification.json', dict(passed=True,
            selected_sigma=selected, sigma_ratios=ratios, original_rq_distortion_ratio=reference_ratio,
            original_reference='same 128 held-out training images, native RQ temperature 0.5',
            expectation='exact categorical expectation in orthonormal source basis',
            coefficient_center='continuous source gamma', training_images=len(latents),
            training_only=True, disjoint_from_fitting=True, time=time.time()))
        print(json.dumps(dict(fidelity=report, selected_sigma=selected, ratios=ratios)), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
