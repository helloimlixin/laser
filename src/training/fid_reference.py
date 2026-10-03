"""Validated sufficient statistics for a complete TorchMetrics real corpus."""
from pathlib import Path
from contextlib import contextmanager
from functools import wraps

import torch

SCHEMA = 'laser-torchmetrics-real-reference-v1'


def load_torchmetrics_reference(path, *, feature_dim=2048, expected_samples=None):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if payload.get('schema') != SCHEMA or payload.get('feature_dim') != feature_dim:
        raise ValueError('FID reference schema or feature dimension mismatch')
    total = payload['real_features_sum']
    cross = payload['real_features_cov_sum']
    count = payload['real_features_num_samples']
    if (total.shape != (feature_dim,) or cross.shape != (feature_dim, feature_dim)
            or total.dtype != torch.float64 or cross.dtype != torch.float64
            or count.shape != () or count.dtype != torch.int64
            or int(count) < 2 or not torch.isfinite(total).all()
            or not torch.isfinite(cross).all()):
        raise ValueError('Invalid FID reference moments')
    metadata = payload['metadata']
    if metadata['samples'] != int(count):
        raise ValueError('FID reference sample count disagrees with provenance')
    if expected_samples is not None and int(count) != expected_samples:
        raise ValueError('FID reference does not include the complete expected corpus')
    return payload


def seed_torchmetrics_reference(metric, path, *, rank=0):
    """Seed only rank zero: distributed FID sums real moments across ranks."""
    payload = load_torchmetrics_reference(path, feature_dim=metric.real_features_sum.numel())
    for name in ('real_features_sum', 'real_features_cov_sum', 'real_features_num_samples'):
        target = getattr(metric, name)
        target.zero_()
        if rank == 0:
            target.copy_(payload[name].to(target.device))
    return payload['metadata']


@contextmanager
def fixed_evaluation_rng(seed, device, *, rank=0):
    """Reuse evaluation randomness without advancing the training RNG streams."""
    if seed is None:
        yield
        return
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed) + int(rank))
        if devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed) + int(rank))
        yield


def with_fixed_evaluation_rng(evaluator):
    @wraps(evaluator)
    def evaluate(model, *args, **kwargs):
        process_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        device = next(model.parameters()).device
        was_training = model.training
        try:
            with fixed_evaluation_rng(kwargs.get('fid_seed'), device, rank=process_rank):
                return evaluator(model, *args, **kwargs)
        finally:
            model.train(was_training)
    return evaluate


def compute_reference_fids(metric, reference_paths, *, expected_generated_samples):
    """Score one globally synchronized fake corpus against distinct real corpora."""
    from torchmetrics.image.fid import _compute_fid

    references = {name: load_torchmetrics_reference(path,
        feature_dim=metric.fake_features_sum.numel()) for name, path in reference_paths.items()}
    fingerprints = {ref['metadata'].get('inception_state_sha256') for ref in references.values()}
    if len(fingerprints) != 1:
        raise ValueError('FID references use different Inception feature extractors')
    results = {}
    with metric.sync_context(should_sync=metric.sync_on_compute):
        fake_count = int(metric.fake_features_num_samples)
        if fake_count != expected_generated_samples or fake_count < 2:
            raise ValueError('FID generated sample count differs from the requested corpus')
        fake_mean = metric.fake_features_sum / fake_count
        fake_cov = (metric.fake_features_cov_sum - fake_count * torch.outer(fake_mean, fake_mean)) / (fake_count - 1)
        for name, reference in references.items():
            count = int(reference['real_features_num_samples'])
            real_mean = reference['real_features_sum'].to(fake_mean.device) / count
            real_cov = (reference['real_features_cov_sum'].to(fake_mean.device)
                        - count * torch.outer(real_mean, real_mean)) / (count - 1)
            score = _compute_fid(real_mean, real_cov, fake_mean, fake_cov).to(metric.orig_dtype)
            results[name] = dict(fid=float(score), real_images=count, generated_images=fake_count,
                                 real_split=reference['metadata'].get('real_split', 'unknown'))
    return results


def fid_log_values(reference_metrics, *, dataset):
    return {f'eval/fid_{dataset}_{name}': result['fid']
            for name, result in reference_metrics.items()}
