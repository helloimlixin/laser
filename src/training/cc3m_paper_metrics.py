"""Released RQ-Transformer CC3M evaluation, with distributed FID moments."""
import hashlib
from pathlib import Path

import torch

from src.rqvae_metrics import frechet_distance, load_reference_statistics


PROTOCOL = 'rq_transformer_cc3m_paper_v1'
PAPER_SAMPLING = dict(temperature=1.0, top_k=1024, top_p=0.9)
REFERENCE_URL = ('https://twg.kakaocdn.net/brainrepo/etc/RQVAE/'
                 '8b325b628f49bf60a3094fcf9419398c/fid_stats.tar.gz')
REFERENCE_SHA256 = '49603d7520746d63a92036be30a28d5e0717b99b0d62e0a26e091ef89886b564'


def load_reference(cfg):
    evaluation = cfg.get('evaluation', {})
    if evaluation.get('protocol') != PROTOCOL:
        raise ValueError('CC3M evaluation requires an explicit released-paper protocol')
    if cfg['sampling'] != PAPER_SAMPLING:
        raise ValueError('Released CC3M sampling requires temperature=1, k=1024, p=0.9')
    path = Path(evaluation['fid_reference'])
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != evaluation['fid_reference_sha256']:
        raise ValueError('CC3M FID reference SHA256 mismatch')
    return load_reference_statistics(path)


def load_clip(device):
    # The released evaluator constructs CLIP on CPU, then moves its FP32
    # weights to CUDA. Loading directly on CUDA would instead select FP16.
    from rqvae.metrics.clip_score import get_clip
    model, preprocess = get_clip()
    return model.to(device).eval(), preprocess


def clip_scores(pixels, captions, model, preprocess, device):
    # Delegate the uint8/PIL conversion, official CLIP preprocessing,
    # tokenization, and unscaled cosine similarity to the released code.
    from rqvae.metrics.clip_score import clip_score
    return clip_score(pixels, captions, model, preprocess, device=device)


def fid_from_moments(total, cross, count, reference):
    if count < 2:
        raise ValueError('FID needs at least two generated images')
    mean = total.double() / count
    covariance = (cross.double() - torch.outer(total.double(), total.double()) / count) / (count - 1)
    return frechet_distance(reference[0], reference[1],
                            mean.cpu().numpy(), covariance.cpu().numpy())
