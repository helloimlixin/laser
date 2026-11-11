"""Verify the selected teacher on 50,000 class-balanced training views."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(os.environ['LASER_BEAM_BASE'])
OUT = Path(os.environ['LASER_STAGE2_OUTPUT'])
sys.path[:0] = [str(BASE / 'source'), str(BASE)]
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
from src import rq_upstream
from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config
from runtime.training_acceleration import install
install()
from runtime import beam_teacher
from runtime.mixture_teacher import get_targets, verify_bayes
from runtime.mixture_recipe import MixtureRecipe
from runtime.preservation_guard import verify_frozen_codec


@torch.inference_mode()
def main():
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(261108250 + rank)
    torch.cuda.manual_seed(261108250 + rank)
    assert verify_bayes()['passed']
    profile = MixtureRecipe(BASE)
    aux, _ = profile.tokenizer(device)
    codec = verify_frozen_codec(profile, aux)
    config = load_config(BASE / 'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(BASE / 'data', 'train', create_transforms(config.dataset, split='train'),
                         seed=261108250)
    rng = np.random.default_rng(261108250)
    indices = np.concatenate([rng.choice(np.flatnonzero(dataset.labels == label), 50, replace=False)
                              for label in range(1000)])
    rng.shuffle(indices)
    assert len(indices) == 50000 and len(np.unique(indices)) == 50000
    assert np.array_equal(np.bincount(dataset.labels[indices]), np.full(1000, 50))
    if rank == 0:
        np.save(OUT / 'teacher-calibration-indices.npy', indices)
    loader = DataLoader(Subset(dataset, indices[rank::world].tolist()), batch_size=8,
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        prefetch_factor=1, multiprocessing_context='spawn')
    stats = torch.zeros(14, dtype=torch.float64, device=device)
    start = time.monotonic()
    seen = 0
    for batch, (images, _) in enumerate(loader):
        latent = aux.encode(images.to(device, non_blocking=True).float())
        pool = beam_teacher.beam_pool(aux.physical_book, latent)
        row = torch.arange(len(pool['costs']), device=device)
        beam = pool['candidates'][row, pool['canonical_index']].reshape(*latent.shape[:-1], 4)
        best = pool['costs'][row, pool['canonical_index']].clamp_min(1e-8)
        relative = (pool['costs'] - best[:, None]).clamp_min(0) / best[:, None]
        for column, bound in [(2, .01), (3, .05)]:
            probability = beam_teacher.posterior(pool, 2., max_relative_error=bound)
            stats[column] += -(probability * probability.clamp_min(1e-30).log()).sum() / 4
            stats[column + 10] += (probability * relative).sum()
        targets, codes, detail = get_targets(aux, latent, .25, temperature=.5,
                                             beam_codes=beam, return_details=True)
        if batch == 0:
            torch.testing.assert_close(detail['joint_probability'], detail['expected_joint_probability'],
                                       atol=3e-7, rtol=3e-5)
            reference = aux.quantize_reconstruction(latent[:2])['codes']
            assert torch.equal(beam[:2], reference)
            with torch.random.fork_rng(devices=[device.index]):
                torch.cuda.manual_seed(261108250)
                p0, c0 = aux.quantizer.get_soft_codes(latent[:2], temp=.5, stochastic=True)
            with torch.random.fork_rng(devices=[device.index]):
                torch.cuda.manual_seed(261108250)
                p1, c1 = get_targets(aux, latent[:2], 1., temperature=.5, beam_codes=beam[:2])
            assert torch.equal(p0, p1) and torch.equal(c0, c1)
            del p0, p1, c0, c1
        flat = targets.reshape(-1, 16384)
        entropy = torch.cat([-(p * p.clamp_min(1e-30).log()).sum(-1) for p in flat.split(256)])
        stats[6:10] += entropy.reshape(-1, 4).sum(0).double()
        quantized = aux.physical_book.embed(codes.reshape(-1, 4)).sum(-2)
        error = (latent.reshape(-1, latent.shape[-1]) - quantized).square().sum(-1)
        stats[0] += len(best)
        stats[1] += len(images)
        stats[4] += ((error - best).clamp_min(0) / best).sum()
        stats[5] += detail['native_branch'].sum()
        stats[10] += error.sum()
        stats[11] += best.sum()
        seen += len(images)
        del targets, codes, detail, flat, entropy, quantized, pool, latent
        if rank == 0 and batch % 25 == 0:
            progress = dict(images_on_rank=seen, target_on_rank=len(indices[rank::world]),
                            elapsed_seconds=time.monotonic() - start, time=time.time())
            write_json(OUT / 'calibration-progress.json', progress)
            print(json.dumps(progress), flush=True)
    dist.all_reduce(stats)
    assert int(stats[1]) == 50000
    result = dict(
        passed=True, training_images=int(stats[1]), sites=int(stats[0]), classes=1000,
        images_per_class=50, seed=261108250, split='train', training_views=True,
        indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        native_weight=.25, native_temperature=.5,
        mean_entropy_nats_per_token=float(stats[6:10].sum() / (4 * stats[0])),
        entropy_by_depth=(stats[6:10] / stats[0]).tolist(),
        baseline_cutoff01_entropy=float(stats[2] / stats[0]),
        baseline_cutoff05_entropy=float(stats[3] / stats[0]),
        baseline_cutoff01_expected_excess_latent_error=float(stats[12] / stats[0]),
        baseline_cutoff05_expected_excess_latent_error=float(stats[13] / stats[0]),
        observed_native_fraction=float(stats[5] / stats[0]),
        mean_relative_excess_latent_error=float(stats[4] / stats[0]),
        mean_absolute_latent_error=float(stats[10] / stats[0]),
        mean_canonical_latent_error=float(stats[11] / stats[0]),
        native_endpoint_bitwise_equal=True, beam_endpoint_matches_saved_encoder=True,
        exact_mixture_joint_probability_verified=True, **codec)
    assert result['mean_entropy_nats_per_token'] > .2
    assert abs(result['observed_native_fraction'] - .25) < .002
    if rank == 0:
        write_json(OUT / 'teacher-calibration.json', result)
        recipe = OmegaConf.load(BASE / 'inputs/recipe.yaml')
        recipe.targets.measured_entropy = result['mean_entropy_nats_per_token']
        OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
