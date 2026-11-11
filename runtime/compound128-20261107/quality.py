"""Full 50k validation rFID of the exact compound training codec, with source references."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/tmp/laser-compound128-20261107')
BASE = ROOT / 'frozen'
sys.path[:0] = [str(BASE / 'source'), str(BASE)]
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
os.environ['TORCH_HOME'] = str(ROOT / 'torch-cache')

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src import rq_upstream
from src import rqvae_metrics as metrics
from src.training.rq_imagenet_runtime import TarImageNet, write_json, sha256
from src.training.laser_hard_ar import nearest_coefficient_ids
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from runtime.compound_recipe import Compound128Recipe


@torch.inference_mode()
def main():
    rank, world, local = [int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']]
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl', device_id=device)
    profile = Compound128Recipe(BASE)
    assert sha256(BASE / 'inputs/stage1-model.pt') == profile.recipe.tokenizer.sha256
    aux, _ = profile.tokenizer(device)
    before = profile.digest(aux)
    config = load_config(BASE / 'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(BASE / 'data', 'val', create_transforms(config.dataset, split='val'))
    assert len(dataset) == 50000
    loader = DataLoader(Subset(dataset, range(rank, 50000, world)), batch_size=8,
        num_workers=4, pin_memory=True, prefetch_factor=1, multiprocessing_context='spawn')
    metrics.INCEPTION_PATH = BASE / 'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    accum = {name: metrics.DistributedOriginalRQVAEMetrics(device, inception=inception)
             for name in ['continuous', 'compound512', 'compound128']}
    centers512 = torch.linspace(-3., 3., 512, device=device)
    error = torch.zeros(3, dtype=torch.float64, device=device)
    started = time.monotonic()
    seen = 0
    for index, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        atoms, values = profile.clean_components(aux, images)
        ids = nearest_coefficient_ids(values, aux.coeff_bins)
        codes = torch.stack((atoms, ids + 16384), -1).flatten(-2)
        coeff128 = aux.coeff_bins[ids]
        coeff512 = centers512[nearest_coefficient_ids(values, centers512)]
        vectors = aux.dictionary.T[atoms]
        real = images.mul(.5).add(.5).clamp(0, 1)
        features, _ = inception(real)
        features = features.double()
        for name, coefficients in [('continuous', values), ('compound512', coeff512), ('compound128', coeff128)]:
            latent = (vectors * (coefficients * aux.coeff_scales)[..., None]).sum(-2).permute(0, 3, 1, 2).contiguous()
            decoded = aux.decoder(aux.post_quant_conv(latent)).clamp(-1, 1)
            if index == 0 and name == 'compound128':
                # Use the production encoder chunk size: different cuDNN batch
                # shapes can move a coefficient across a quantization boundary.
                direct, context = profile.targets(aux, images)
                torch.testing.assert_close(direct, codes, rtol=0, atol=0)
                assert torch.equal(direct, context)
                torch.testing.assert_close(decoded, profile.decode(aux, context), rtol=0, atol=0)
                assert profile.verify_targets(aux, direct, context)['training_and_evaluation_codec_identical']
            metric = accum[name]
            metric.real_sum.add_(features.sum(0))
            metric.real_cross.add_(features.T @ features)
            metric.real_count.add_(len(images))
            metric.update(decoded.mul(.5).add(.5).clamp(0, 1), real=False)
        error[0] += (coeff128 - values).square().sum()
        error[1] += (values.abs() > 3).sum()
        error[2] += values.numel()
        seen += len(images)
        if rank == 0 and index % 25 == 0:
            row = dict(images_on_rank=seen, target_on_rank=12500, seconds=time.monotonic() - started, time=time.time())
            write_json(ROOT / 'quality/progress.json', row)
            print(json.dumps(row), flush=True)
    results = {}
    for name, metric in accum.items():
        rfid, _, _ = metric.compute()
        results[name] = float(rfid)
    dist.all_reduce(error)
    assert profile.digest(aux) == before
    passed = results['compound128'] < 4.73
    if rank == 0:
        report = dict(passed=passed, measured_validation_rfid=results['compound128'], results=results,
            maximum_rfid=4.73, delta_vs_continuous=results['compound128']-results['continuous'],
            delta_vs_compound512=results['compound128']-results['compound512'],
            real_images=50000, reconstructed_images=50000, reference_split='val',
            original_encoder_dictionary_decoder_frozen=True, frozen_weights_unchanged=True,
            atoms=16384, depth=4, coefficient_bins=128, coefficient_bits=7,
            coefficient_range=[-3., 3.], coefficient_scales=list(profile.recipe.tokenizer.coefficient_scales),
            coefficient_noise=False, training_and_evaluation_codec_identical=True,
            measured_codec='actual deterministic compound training targets and prefixes',
            coefficient_mse=float(error[0]/error[2]), coefficient_clipping_fraction=float(error[1]/error[2]),
            source_stage1_checkpoint_sha256=str(profile.recipe.tokenizer.sha256), training_updates_added=0,
            seconds=time.monotonic()-started)
        for p in [ROOT/'quality/results.json', ROOT/'tokenizer-verification.json', BASE/'tokenizer-verification.json']:
            write_json(p, report)
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    assert passed, '128-bin tokenizer failed the reconstruction gate; do not launch AR training'


if __name__ == '__main__':
    main()
