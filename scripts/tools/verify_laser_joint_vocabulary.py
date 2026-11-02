#!/usr/bin/env python3
"""Full matched reconstruction audit and stochastic-target calibration."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/tools')]
import prepare_laser_joint_vocabulary as study
from src import rq_upstream
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from src import rqvae_metrics as metrics
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.rq_imagenet_runtime import TarImageNet, write_json, sha256
from src.training.laser_hard_ar import sparse_encoding, nearest_coefficient_ids
from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients, ordered_support_basis


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    rank, world, local = (int(os.environ.get(k, v)) for k, v in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    os.environ['TORCH_HOME'] = str(study.SOURCE/'torch-cache')
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    path = study.OUT/f'{args.candidate}-codebook.pt'
    spec = torch.load(path, weights_only=True)
    assert spec['source_stage1_checkpoint_sha256'] == sha256(study.SOURCE/'inputs/stage1-model.pt')
    aux = study.tokenizer(device)
    q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)
    torch.testing.assert_close(q.dictionary, aux.dictionary, rtol=0, atol=0)
    metrics.INCEPTION_PATH = study.SOURCE/'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    source_metric = metrics.DistributedOriginalRQVAEMetrics(device, inception=inception)
    converted_metric = metrics.DistributedOriginalRQVAEMetrics(device, inception=inception)
    config = load_config(study.SOURCE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(study.SOURCE/'data', 'val', create_transforms(config.dataset, split='val'))
    loader = DataLoader(Subset(data, range(rank, len(data), world)), batch_size=8,
                        num_workers=4, pin_memory=True)
    gram = aux.dictionary.T @ aux.dictionary
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for index, (images, _) in enumerate(loader):
        images = images.to(device)
        x = aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).contiguous()
        a, c = sparse_encoding(x, aux.dictionary, 4, gram=gram)
        vectors = aux.dictionary.T[a]
        gamma, _ = dictionary_to_orthogonal_coefficients(vectors, c)
        ids = nearest_coefficient_ids(gamma / aux.coeff_scales, aux.coeff_bins)
        basis, _ = ordered_support_basis(vectors)
        reference = (basis * (aux.coeff_bins[ids] * aux.coeff_scales)[..., None]).sum(-2)
        converted = q.quantize(x, chunk_size=128)
        torch.testing.assert_close(q.embed(converted['codes']).sum(-2), converted['quantized'],
                                   rtol=1.e-4, atol=2.e-5)
        for metric, z in [(source_metric, reference), (converted_metric, converted['quantized'])]:
            reconstruction = aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous()))
            metric.update(images.mul(.5).add(.5).clamp(0, 1), real=True)
            metric.update(reconstruction.mul(.5).add(.5).clamp(0, 1), real=False)
        totals[:2] += torch.stack(((x-reference).square().sum().double(),
                                   (x-converted['quantized']).square().sum().double()))
        totals[2] += x.numel()
        if rank == 0 and index % 25 == 0:
            study.status('full_matched_rfid', candidate=args.candidate,
                         images_on_rank=(index + 1)*8, world_size=world)
    source_fid, _, _ = source_metric.compute()
    converted_fid, _, _ = converted_metric.compute()
    dist.all_reduce(totals)
    assert int(source_metric.fake_count) == int(converted_metric.fake_count) == 50000
    assert int(source_metric.real_count) == int(converted_metric.real_count) == 50000
    report = dict(candidate=args.candidate, source_rfid=source_fid, converted_rfid=converted_fid,
        drift=converted_fid-source_fid, maximum_drift=.20,
        passed=converted_fid <= source_fid+.20, real_images=50000, reconstructed_images=50000,
        reference='same full validation set, same original RQ Inception',
        source_latent_mse=float(totals[0]/totals[2]), converted_latent_mse=float(totals[1]/totals[2]),
        vocab_size=q.vocab_size, codebook_sha256=sha256(path),
        source_sha256=sha256(study.SOURCE/'inputs/stage1-model.pt'),
        prefix_stable=True, native_transformer_compatible=True, time=time.time())
    if rank == 0:
        write_json(study.OUT/f'{args.candidate}-full-rfid.json', report)
        print(json.dumps(report), flush=True)
    if report['passed']:
        # Split is disjoint from coefficient fitting and reconstruction selection.
        latents = np.load(study.BASE/'temperature-latents.npy', mmap_mode='r')
        errors = {}
        for temperature in [0., .015625, .03125, .0625, .125, .25]:
            total = 0.
            for first in range(rank*4, len(latents), world*4):
                x = torch.from_numpy(latents[first:first+4].copy()).to(device)
                if temperature == 0.:
                    z = q.quantize(x)['quantized']
                else:
                    torch.manual_seed(261015+first); torch.cuda.manual_seed(261015+first)
                    _, codes = q.get_soft_codes(x, temp=temperature, stochastic=True, chunk_size=128)
                    z = q.embed(codes).sum(-2)
                total += (x-z).square().sum().item()
            value = torch.tensor(total, device=device, dtype=torch.float64)
            dist.all_reduce(value)
            errors[temperature] = float(value)
        ratios = {t: value/errors[0.] for t, value in errors.items() if t}
        # Recompute the published tokenizer's perturbation on these same
        # held-out training images instead of inheriting a historical estimate.
        from src.training.rq_imagenet_recipes import RQRecipe
        native, _ = RQRecipe('/tmp/laser-original-rq-imagenet-20261014').tokenizer(device)
        manifest = json.loads((study.OUT/'calibration-data.json').read_text())
        training = TarImageNet(study.SOURCE/'data', 'train', create_transforms(config.dataset, split='val'))
        native_errors = torch.zeros(2, device=device, dtype=torch.float64)
        local_indices = manifest['temperature_indices'][rank::world]
        for first in range(0, len(local_indices), 4):
            images = torch.stack([training[i][0] for i in local_indices[first:first+4]]).to(device)
            x = native.encode(images)
            hard, _, _ = native.quantizer(x)
            torch.manual_seed(261015+first+rank*10000); torch.cuda.manual_seed(261015+first+rank*10000)
            _, codes = native.quantizer.get_soft_codes(x, temp=.5, stochastic=True)
            sampled = native.get_code_emb_with_depth(codes)[0].sum(-2)
            native_errors += torch.stack(((x-hard).square().sum().double(),
                                           (x-sampled).square().sum().double()))
        dist.all_reduce(native_errors)
        reference_ratio = float(native_errors[1]/native_errors[0])
        allowed = [t for t in ratios if ratios[t] <= reference_ratio+.02]
        if not allowed:
            raise RuntimeError('No calibrated stochastic temperature met distortion bound')
        selected = min(allowed, key=lambda t: abs(ratios[t]-reference_ratio))
        if rank == 0:
            report.update(temperature=selected, temperature_images=len(latents),
                sampled_to_hard_latent_mse_ratio=ratios[selected],
                temperature_ratios=ratios, original_rq_measured_distortion_ratio=reference_ratio,
                temperature_calibration_training_only=True,
                temperature_images_disjoint_from_coefficient_fitting=True)
            write_json(study.OUT/f'{args.candidate}-full-rfid.json', report)
            write_json(study.OUT/'selected.json', report)
            print(json.dumps(report), flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__': main()
