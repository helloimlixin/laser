#!/usr/bin/env python3
"""Read-only CPU audit of the loss in a frozen ImageNet training snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def numerics(torch, soft_ce, native_ce):
    """Check actual vocabulary, noncontiguous input and the live AMP scale."""
    torch.manual_seed(261018503)
    results = []
    vocabulary = 20481
    for dtype in (torch.float32, torch.float16):
        for spread, concentration in ((1.5, .99), (4., .5), (15., .9999)):
            values = (torch.randn(2, 3, 5, vocabulary) * spread).to(dtype)
            # Deliberately noncontiguous, with complete groups of five depths.
            values = values.transpose(0, 1)
            targets = torch.rand(values.shape).softmax(-1) * (1 - concentration)
            winners = torch.randint(vocabulary, (*values.shape[:-1], 1))
            targets.scatter_add_(-1, winners, torch.full_like(winners, concentration, dtype=torch.float32))
            # Exercise the target-row-sum term in the analytical derivative.
            targets *= .9999
            for chunk in (7, 128):
                for amp_scale in (1., 524288.):
                    actual = values.detach().requires_grad_()
                    dense = values.detach().clone().requires_grad_()
                    native = values.detach().clone().requires_grad_()
                    losses = [soft_ce(actual, targets, chunk_size=chunk),
                              -(targets * dense.float().log_softmax(-1)).sum(-1).mean(),
                              native_ce(native.float().reshape(-1, vocabulary), targets.reshape(-1, vocabulary))]
                    # A real microbatch has 64*8*8*5 target rows. Match its
                    # individual scaled gradients while probing fewer rows.
                    upstream_scale = amp_scale * .25 * (values.numel() // vocabulary) / 20480
                    for loss in losses:
                        (loss * upstream_scale).backward()
                    errors = []
                    for reference in (dense, native):
                        error = (actual.grad.float() - reference.grad.float()).norm()
                        relative = float(error / reference.grad.float().norm().clamp_min(1.e-30))
                        assert relative < (1.e-3 if dtype == torch.float16 else 2.e-5), relative
                        errors.append(relative)
                    torch.testing.assert_close(losses[0], losses[1], atol=2.e-5, rtol=2.e-6)
                    torch.testing.assert_close(losses[0], losses[2], atol=2.e-5, rtol=2.e-6)
                    assert torch.isfinite(actual.grad).all()
                    results.append(dict(dtype=str(dtype), spread=spread, target_peak=concentration,
                        chunk_size=chunk, amp_scale=amp_scale, loss=float(losses[0].detach()),
                        relative_gradient_error_dense=errors[0], relative_gradient_error_upstream=errors[1]))

    # Also propagate through FP16 logits into FP32 classifier parameters.
    features = torch.randn(30, 16)
    weight = torch.randn(vocabulary, 16) * .4
    target = torch.zeros(30, vocabulary).scatter_(1, torch.randint(vocabulary, (30, 1)), 1.)
    parameter_gradients = []
    for custom in (False, True):
        parameter = weight.clone().requires_grad_()
        logits = (features @ parameter.T).half()
        loss = (soft_ce(logits, target, 7) if custom else native_ce(logits.float(), target))
        (loss * (524288. * .25 * 30 / 20480)).backward()
        parameter_gradients.append(parameter.grad / 524288.)
    relative = float((parameter_gradients[0] - parameter_gradients[1]).norm() /
                     parameter_gradients[0].norm())
    assert relative < 1.e-3
    return dict(passed=True, cases=results, classifier_fp32_gradient_relative_error=relative,
        vocabulary=vocabulary, device='cpu',
        limits='FP16 logits and FP32 arithmetic are checked on CPU; GPU kernels are not re-executed.')


def accumulation(torch, soft_ce):
    from src.training.exact_global_batch import ExactGlobalBatchSampler
    from src.training.rq_imagenet_shared import training_backward

    class Profile:
        def targets(self, aux, images):
            return images[:, 2:].reshape(-1, 5, 13), images[:, :2]

        def loss(self, model, logits, targets):
            return soft_ce(logits, targets, 7)

    class Model(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone())

        def forward(self, codes, **kwargs):
            return (codes @ self.weight).reshape(-1, 5, 13)

    class Scaler:
        def scale(self, loss):
            return loss * 524288.

    torch.manual_seed(261018504)
    features = torch.randn(5263, 2)
    targets = torch.randn(5263, 5, 13).softmax(-1)
    data = torch.cat((features, targets.flatten(1)), 1)
    initial = torch.randn(2, 65)
    layouts = []
    for world, microbatches in ((4, 8), (8, 4)):
        samplers = [ExactGlobalBatchSampler(data, 2048, world, rank, microbatches, 261001)
                    for rank in range(world)]
        for sampler in samplers:
            sampler.set_epoch(9)
        batches = [list(sampler) for sampler in samplers]
        layout = []
        for step in range(3):
            gradients, indices = [], []
            for sampler, rank_batches in zip(samplers, batches):
                model = Model(initial)
                for micro in range(microbatches):
                    absolute = step * microbatches + micro
                    chosen = rank_batches[absolute]
                    indices.extend(chosen)
                    training_backward(Profile(), model, None, data[chosen], None, Scaler(),
                        world * len(chosen) / sampler.global_count(absolute), amp=False)
                gradients.append(model.weight.grad / 524288.)
            # DDP averages the ranks' accumulated gradients once per update.
            actual = torch.stack(gradients).mean(0)
            reference = Model(initial)
            y, codes = Profile().targets(None, data[indices])
            native = -(y * reference(codes).log_softmax(-1)).sum(-1).mean()
            native.backward()
            torch.testing.assert_close(actual, reference.weight.grad, rtol=3.e-5, atol=2.e-8)
            layout.append(actual)
        layouts.append(layout)
    for four, eight in zip(*layouts):
        torch.testing.assert_close(four, eight, rtol=3.e-5, atol=2.e-8)
    return dict(passed=True, actual_frozen_training_backward=True, worlds=[4, 8],
                global_batch=2048, final_batch=1167, soft_target_depths=5,
                limits='CPU simulation of DDP averaging, not a new distributed GPU run.')


def probe(torch, base, checkpoint, output, image_count):
    import numpy as np
    from omegaconf import OmegaConf
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.utils.config import load_config
    from rqvae.img_datasets.transforms import create_transforms

    profile = JointLaserRecipe(base)
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    profile.verify_resume(saved)
    step = int(saved['global_step'])
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).eval()
    del saved
    aux, _ = profile.tokenizer(torch.device('cpu'))
    config = load_config(base / 'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(base / 'data', 'val', create_transforms(config.dataset, split='val'))
    indices = np.random.default_rng(261018505).permutation(len(dataset))[:image_count].tolist()
    torch.manual_seed(261018505)
    depth = profile.shape[-1]
    names = ('ce', 'target_entropy', 'predictive_entropy', 'target_top1_mass',
             'logit_gradient_squared_norm', 'codeword_squared_norm',
             'latent_error_reduction', 'expected_codeword_error_squared')
    sums = {key: torch.zeros(depth, dtype=torch.float64) for key in names}
    books = [aux.quantizer.expanded_codebook(d) for d in range(depth)]
    rows = 0
    with torch.inference_mode():
        for first in range(0, image_count, 4):
            batch = [dataset[index] for index in indices[first:first + 4]]
            images = torch.stack([entry[0] for entry in batch])
            labels = torch.tensor([entry[1] for entry in batch])
            targets, codes = profile.targets(aux, images)
            logits = model(codes, model_aux=aux, cond=labels, amp=False)
            assert targets.shape == logits.shape
            logp = logits.log_softmax(-1)
            probability = logp.exp()
            def add(name, values):
                sums[name].add_(values.reshape(-1, depth).double().sum(0))
            add('ce', -(targets * logp).sum(-1))
            add('target_entropy', -(targets * targets.clamp_min(1.e-30).log()).sum(-1))
            add('predictive_entropy', -(probability * logp).sum(-1))
            add('target_top1_mass', targets.amax(-1))
            add('logit_gradient_squared_norm',
                (probability * targets.sum(-1, keepdim=True) - targets).square().sum(-1))
            latents = aux.encode(images)
            embeddings = aux.quantizer.embed(codes)
            residual = latents.clone()
            for d in range(depth):
                physical = embeddings[..., d, :]
                sums['codeword_squared_norm'][d] += physical.square().sum().double()
                before = residual.square().sum().double()
                residual -= physical
                sums['latent_error_reduction'][d] += before - residual.square().sum().double()
                predicted_mean = probability[..., d, :] @ books[d]
                target_mean = targets[..., d, :] @ books[d]
                sums['expected_codeword_error_squared'][d] += (predicted_mean - target_mean).square().sum().double()
            rows += len(images) * 64
            write(output / 'status.json', dict(phase='validation_probe', images=first + len(batch),
                target_images=image_count, checkpoint_step=step, pid=os.getpid(), time=time.time()))
    values = {key: (value / rows).tolist() for key, value in sums.items()}
    values['kl_target_to_prediction'] = ((sums['ce'] - sums['target_entropy']) / rows).tolist()
    values['ce_share'] = (sums['ce'] / sums['ce'].sum()).tolist()
    values['latent_error_reduction_share'] = (sums['latent_error_reduction'] /
        sums['latent_error_reduction'].sum()).tolist()
    return dict(checkpoint=str(checkpoint), checkpoint_step=step, images=image_count,
        validation_indices=indices, rows_per_depth=rows, per_depth=values,
        target_temperatures=profile.target_temperature(), model_updated=False,
        limits='Small teacher-forced FP32 validation probe with stochastic targets, no dropout. '
               'Latent error reductions do not measure decoder sensitivity or generated-image FID. '
               'CE and KL across different tokenizers are not comparable image likelihoods.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--images', type=int, default=32)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['RQ_UPSTREAM_ROOT'] = str(args.base / 'upstream')
    sys.path[:0] = [str(args.base / 'source'), str(args.base / 'upstream'),
                   str(args.base), str(args.base / 'runtime')]
    import torch
    from src.scaled_atom_training import soft_cross_entropy
    from rqvae.optimizer.loss import soft_target_cross_entropy
    torch.set_num_threads(16)
    sources = [args.base / 'source/src/scaled_atom_training.py',
               args.base / 'source/src/training/rq_imagenet_shared.py',
               args.base / 'source/src/training/laser_joint_recipe.py',
               args.base / 'upstream/rqvae/optimizer/loss.py']
    report = dict(base=str(args.base), source_sha256={str(path):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        training_recipe_modified=False, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    write(args.output / 'status.json', dict(phase='numerics', pid=os.getpid(), time=time.time()))
    report['numerics'] = numerics(torch, soft_cross_entropy, soft_target_cross_entropy)
    report['accumulation'] = accumulation(torch, soft_cross_entropy)
    write(args.output / 'numerics.json', report)
    print(json.dumps(dict(numerics_passed=True, accumulation_passed=True)), flush=True)
    if args.checkpoint:
        write(args.output / 'status.json', dict(phase='loading_checkpoint', pid=os.getpid(), time=time.time()))
        report['validation_probe'] = probe(torch, args.base, args.checkpoint, args.output, args.images)
    report['completed'] = True
    write(args.output / 'report.json', report)
    write(args.output / 'status.json', dict(phase='completed', pid=os.getpid(), time=time.time()))
    print(json.dumps(report.get('validation_probe', {}).get('per_depth', {})), flush=True)


if __name__ == '__main__':
    main()
