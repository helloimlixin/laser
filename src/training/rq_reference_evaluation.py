"""Additional FID evaluation using RQ-Transformer's released FFHQ statistics."""
import math
import time

import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image

from src.original_rq_training import FeatureMoments, atomic_json, file_sha256
from src.training.var_laser import get_inception_model, frechet_distance


@torch.no_grad()
def evaluate_rq_reference(experiment, model, epoch, count, reference, expected_sha256, label):
    if file_sha256(reference) != expected_sha256:
        raise ValueError('RQ reference statistics checksum mismatch')
    if count < 2:
        raise ValueError('FID requires at least two generated images')
    matched_contract = None
    manifest = experiment.cfg.evaluation.get('fid_reference_manifest')
    if manifest:
        from src.training.ffhq_fid_protocol import validate_matched_reference
        matched_contract = validate_matched_reference(reference, manifest, experiment.cfg.data.root,
            dict(train=experiment.train.dataset._fingerprint, validation=experiment.val.dataset._fingerprint))
        if count != 50000:
            raise ValueError('Matched FFHQ benchmark requires 50000 generated samples')
    started = time.monotonic()
    model.eval()
    if experiment.inception is None:
        experiment.inception = get_inception_model().eval().requires_grad_(False).to(experiment.device)
    moments = FeatureMoments(experiment.device)
    cfg = experiment.cfg.prior
    indices = list(range(experiment.rank, count, experiment.world))
    for begin in range(0, len(indices), experiment.cfg.evaluation.batch_size):
        selected = indices[begin:begin + experiment.cfg.evaluation.batch_size]
        labels = torch.zeros(len(selected), device=experiment.device, dtype=torch.long)
        with experiment.amp():
            latent = model.sample(labels, cfg=cfg.cfg, top_k=cfg.top_k, top_p=cfg.top_p,
                                  seed=73000 + experiment.rank + begin * experiment.world,
                                  **experiment.sampling_options())
        # Released RQ evaluation decodes in FP32 and passes continuous [0,1]
        # pixels directly to Inception, without PNG/uint8 rounding.
        with torch.autocast(experiment.device.type, enabled=False):
            pixels = experiment.vae.fhat_to_img(latent.float()).mul(.5).add(.5).clamp(0, 1)
            moments.update(experiment.inception(pixels))
        if begin == 0 and experiment.rank == 0:
            grid_path = experiment.media_path(f'rq-reference-{label}-epoch{epoch:03d}.png')
            save_image(pixels[:experiment.cfg.evaluation.preview_samples],
                       grid_path,
                       nrow=int(experiment.cfg.evaluation.get('grid_columns', 8)))
            if getattr(experiment, 'run', None):
                import wandb
                experiment.run.log({f'rq_reference_{label}/samples': wandb.Image(str(grid_path))})
        if begin % (experiment.cfg.evaluation.batch_size * 25) == 0:
            experiment.log('rq_reference_sampling', epoch=epoch,
                           generated=min((begin + len(selected)) * experiment.world, count), count=count)
    generated, mean, covariance = moments.finish()
    if generated != count:
        raise RuntimeError('RQ reference evaluation sample count mismatch')
    record = None
    if experiment.rank == 0:
        with np.load(reference, allow_pickle=False) as real:
            score = float(frechet_distance(mean, covariance, real['mu'], real['sigma']))
        if not math.isfinite(score):
            raise RuntimeError('RQ reference FID is not finite')
        record = dict(epoch=epoch, count=count, fid=score, label=label,
                      reference=str(reference), reference_sha256=expected_sha256,
                      real_reference_images=60000, cfg=cfg.cfg, top_k=cfg.top_k, top_p=cfg.top_p,
                      sampling_options=experiment.sampling_options(),
                      elapsed_seconds=time.monotonic() - started,
                      pixel_protocol='FP32 decoder, continuous float32 [0,1], no uint8 rounding',
                      feature_backend='released RQ-Transformer pytorch-fid Inception; float64 streaming moments',
                      published_rq_transformer_fid=10.38,
                      comparison='Same released FID reference and image range; training splits and architectures differ. '
                                 'LASER uses official contiguous FFHQ split; RQ uses its released shuffled split.')
        if matched_contract:
            record.pop('published_rq_transformer_fid')
            record.update(reference_contract=matched_contract, reference_manifest_sha256=file_sha256(manifest),
                          comparison='Exact LASER full training images and verified preprocessing. '
                                     'Do not compare this score directly with published RQ FID using different reference statistics.')
        atomic_json(experiment.out / f'rq-reference-{label}-epoch{epoch:03d}-{count}.json', record)
        experiment.log(f'rq_reference_{label}', epoch=epoch, count=count, fid=score)
    dist.barrier()
    return record
