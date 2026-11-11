"""Full validation rFID of actual support-plus-pattern codes and matched references."""
from common import BASE, ROOT, distributed, sha, write
import json
import math
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from src import rq_upstream
from src import rqvae_metrics as metrics
from src.training.rq_imagenet_runtime import TarImageNet
from src.training.laser_hard_ar import nearest_coefficient_ids
from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from runtime.compound_recipe import Compound128Recipe
from coefficient_pattern_codec import assign_coefficient_patterns, reconstruct_pattern_latents, selected_support_grams


@torch.inference_mode()
def main():
    rank, world, device = distributed()
    fit = json.loads((ROOT/'fit-results.json').read_text())
    assert fit['passed'] and not fit['validation_used_for_fitting']
    profile = Compound128Recipe(BASE)
    aux, _ = profile.tokenizer(device)
    before = profile.digest(aux)
    books = {}
    for key, row in fit['selected'].items():
        from pathlib import Path
        path = Path(row['path'])
        assert sha(path) == row['sha256']
        payload = torch.load(path, map_location='cpu', weights_only=False)
        books['pattern'+key] = payload['patterns'].to(device)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    assert len(dataset) == 50000
    loader = DataLoader(Subset(dataset, range(rank, 50000, world)), batch_size=8, num_workers=2,
        pin_memory=True, prefetch_factor=1, multiprocessing_context='spawn')
    metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    names = ['continuous', 'compound128', *books]
    accum = {key:metrics.DistributedOriginalRQVAEMetrics(device, inception=inception) for key in names}
    errors = {key:torch.zeros(2, dtype=torch.float64, device=device) for key in names}
    counts = {key:torch.zeros(len(p), dtype=torch.long, device=device) for key, p in books.items()}
    started = time.monotonic()
    seen = 0
    first_proof = None
    for index, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        atoms, normalized = profile.clean_components(aux, images)
        physical = normalized*aux.coeff_scales
        selected_vectors = aux.dictionary.T[atoms]
        original_latent = (selected_vectors*physical[..., None]).sum(-2)
        ids = nearest_coefficient_ids(normalized, aux.coeff_bins)
        quantized = aux.coeff_bins[ids]*aux.coeff_scales
        latents = dict(continuous=original_latent,
            compound128=(selected_vectors*quantized[..., None]).sum(-2))
        flat = physical.reshape(-1, 4)
        grams = selected_support_grams(atoms.reshape(-1, 4), aux.dictionary.T)
        for name, patterns in books.items():
            pattern_ids, distance = assign_coefficient_patterns(flat, patterns, grams=grams, chunk_size=1024)
            latents[name] = reconstruct_pattern_latents(atoms, pattern_ids.reshape(atoms.shape[:-1]),
                patterns, aux.dictionary.T)
            counts[name].add_(torch.bincount(pattern_ids, minlength=len(patterns)))
            if index == 0:
                # Verify all candidate distances directly, avoiding the expanded
                # quadratic form used by the efficient assignment kernel.
                delta = flat[:16, None, :]-patterns[None, :, :]
                direct = torch.einsum('nvi,nij,nvj->nv', delta, grams[:16], delta)
                expected = direct.min(-1).values
                actual = direct.gather(-1, pattern_ids[:16, None]).squeeze(-1)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-4)
                direct_latent_error = (latents[name]-original_latent).square().sum(-1).reshape(-1)
                torch.testing.assert_close(distance, direct_latent_error, rtol=5e-3, atol=3e-4)
        real = images.mul(.5).add(.5).clamp(0, 1)
        features, _ = inception(real)
        features = features.double()
        real_sum, real_cross = features.sum(0), features.T@features
        for name, latent in latents.items():
            decoded = aux.decoder(aux.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)
            if index == 0 and name == 'compound128':
                codes, context = profile.targets(aux, images)
                assert torch.equal(codes, context)
                torch.testing.assert_close(decoded, profile.decode(aux, context), rtol=0, atol=0)
                first_proof = dict(passed=True, baseline_decoder_bitwise_equal=True,
                    pattern_lookup_reconstructs_actual_latent=True, assignment_matches_direct_latent_error=True,
                    full_candidate_geometry_checked=True, frozen_dictionary=True, reconstruction_only=True,
                    stage2_training_launched=False)
            errors[name][0] += (latent-original_latent).square().sum()
            errors[name][1] += latent.numel()
            metric = accum[name]
            metric.real_sum.add_(real_sum)
            metric.real_cross.add_(real_cross)
            metric.real_count.add_(len(images))
            metric.update(decoded.mul(.5).add(.5).clamp(0, 1), real=False)
        seen += len(images)
        if rank == 0 and index % 25 == 0:
            row = dict(phase='full50k_rfid', images_on_rank=seen, target_on_rank=12500,
                seconds=time.monotonic()-started, time=time.time(), candidates=list(books))
            write(ROOT/'progress.json', row)
            print(row, flush=True)
    results = {}
    for name, metric in accum.items():
        rfid, _, _ = metric.compute()
        dist.all_reduce(errors[name])
        results[name] = dict(rfid=float(rfid), real_images=50000, reconstructed_images=50000,
            mean_squared_latent_error_per_coordinate=float(errors[name][0]/errors[name][1]))
        if name in books:
            dist.all_reduce(counts[name])
            probability = counts[name].double()/counts[name].sum()
            entropy = -(probability*probability.clamp_min(1e-300).log2()).sum()
            size = len(books[name])
            results[name].update(patterns=size, decisions_per_site=5,
                categorical_predictions_per_image=320, nominal_bits_per_site=56+int(math.log2(size)),
                marginal_pattern_entropy_bits=float(entropy), patterns_used=int((counts[name]>0).sum()))
    assert profile.digest(aux) == before
    if rank == 0:
        for name in books:
            results[name].update(delta_rfid_vs_compound128=results[name]['rfid']-results['compound128']['rfid'],
                beats_original_rq_rfid=results[name]['rfid']<4.73,
                within_0p05_of_compound128=results[name]['rfid']<=results['compound128']['rfid']+.05)
        report = dict(passed=True, results=results, reference_split='val',
            reference_backend='original RQ-VAE Inception', validation_images=50000,
            original_rq_published_rfid=4.73, original_atoms=16384,
            original_encoder_dictionary_decoder_unchanged=True, frozen_weights_unchanged=True,
            physical_coefficient_patterns=True, coefficient_noise=False, actual_pattern_lookup_decode=True,
            fitting_images=16000, seed_selection_images=2000, fitting_and_seed_selection_split='train',
            validation_used_for_fitting_or_seed_selection=False, stage2_training_launched=False,
            generation_quality_measured=False, verification=first_proof,
            seconds=time.monotonic()-started)
        write(ROOT/'results.json', report)
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
