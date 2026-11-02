#!/usr/bin/env python3
"""Matched, resumable FID50k comparison without interrupting stage-2 training."""
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict
import fcntl
import gc
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-imagenet-duplicate-20261014')
TRAIN_RUN = 'imagenet-rfid421-tiny-jitter-fresh-4h100-20261014'
EVAL_RUN = 'imagenet-tiny-jitter-epoch8-ancestral-joint32-fid50k-20261014'
OUT = ROOT / 'outputs' / TRAIN_RUN / 'joint-fid50k-epoch8'
WORK = BASE / 'joint-fid50k-epoch8'
sys.path[:0] = [str(BASE / 'source'), str(BASE / 'source/runtime')]
os.environ['TORCH_HOME'] = str(BASE / 'torch-cache')

import numpy as np
import torch
from src.training import rqtransformer as training
from src.rqvae_metrics import (
    OriginalRQVAEInception, frechet_distance, inception_score,
    load_reference_statistics,
)

_CPU_AUX = _CPU_INCEPTION = None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            sha.update(chunk)
    return sha.hexdigest()


def load_joint(path):
    spec = importlib.util.spec_from_file_location('laser_matched_joint_sampling', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cpu_initialize(scales, threads):
    global _CPU_AUX, _CPU_INCEPTION
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _CPU_AUX = training.LaserAux(
        BASE / 'inputs/stage1-tokenizer.pt', 16384, 2048, 3., 6.4,
        coeff_scales=scales, soft_target_physical=False,
        clamp_coeffs=False, sparsity_level=4,
    ).eval()
    _CPU_INCEPTION = OriginalRQVAEInception().eval()


@torch.inference_mode()
def decode_and_measure(task):
    """Persist every code and feature chunk before acknowledging completion."""
    sampler, first, atoms, coefficients, labels, seed = task
    features, probabilities = [], []
    for offset in range(0, len(labels), 8):
        decoded = _CPU_AUX.decode_compound(
            torch.from_numpy(atoms[offset:offset + 8].astype(np.int64)),
            torch.from_numpy(coefficients[offset:offset + 8].astype(np.int64)),
        )
        images = ((decoded.float() + 1.) * .5).clamp(0, 1)
        acts, logits = _CPU_INCEPTION(images, return_logits=True)
        assert torch.isfinite(decoded).all() and torch.isfinite(acts).all()
        assert torch.isfinite(logits).all()
        features.append(acts.numpy())
        probabilities.append(logits.softmax(-1).numpy())
        if first == 0 and offset == 0:
            from torchvision.utils import save_image
            save_image(images, OUT / f'{sampler}-preview.png', nrow=4)
    features = np.concatenate(features)
    probabilities = np.concatenate(probabilities)
    assert features.shape == (len(labels), 2048)
    assert probabilities.shape == (len(labels), 1008)
    directory = WORK / 'chunks' / sampler
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{first:06d}.npz'
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, first=np.int64(first), seed=np.int64(seed), labels=labels,
                 atoms=atoms, coefficient_ids=coefficients,
                 features=features, probabilities=probabilities)
    temporary.replace(path)
    return dict(sampler=sampler, first=first, samples=len(labels), path=str(path))


class SamplingAux(torch.nn.Module):
    compound_embeddings = training.LaserAux.compound_embeddings

    def __init__(self, dictionary, bins, scales, device):
        super().__init__()
        self.num_atoms, self.coeff_vocab_size, self.sparsity_level = 16384, 2048, 4
        for name, value in [('dictionary', dictionary), ('coeff_bins', bins),
                            ('coeff_scales', torch.tensor(scales, dtype=torch.float32))]:
            self.register_buffer(name, value.to(device))


def load_sampling_model(device):
    payload = torch.load(WORK / 'checkpoint.pt', map_location='cpu', mmap=True,
                         weights_only=False)
    assert payload['epoch'] == 8 and payload['global_step'] == 5008
    assert payload['config']['compound_pair_autoregressive']
    scales = payload['config']['coeff_scales']
    with torch.device('meta'):
        model = training.build_model(
            18432, 16384, compound=True, coeff_vocab_size=2048,
            compound_pair_autoregressive=True,
            model_preset='imagenet-1400m', sparsity_level=4,
        )
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    converted = 0
    # These linears already execute under FP16 CUDA autocast in production.
    # Store their autocast values directly; retain FP32 residuals, layer norms,
    # embeddings, and both classification heads outside the autocast scope.
    for name in ('body_transformer', 'head_transformer', 'input_mlp',
                 'head_mlp', 'pair_embedding_adapter'):
        for module in getattr(model, name).modules():
            if isinstance(module, torch.nn.Linear):
                converted += sum(p.numel() for p in module.parameters())
                module.half()
    model.to(device).eval()
    stage1 = training.load_stage1_checkpoint(BASE / 'inputs/stage1-tokenizer.pt')
    state = stage1['state_dict']
    key = 'quantizer.dictionary' if 'quantizer.dictionary' in state else 'bottleneck.dictionary'
    dictionary = torch.nn.functional.normalize(state[key].float(), dim=0)
    aux = SamplingAux(dictionary, torch.linspace(-3., 3., 2048), scales, device)
    assert sum(p.numel() for p in model.parameters()) == 1401311744
    assert all(p.dtype == torch.float32 for p in model.classifier.parameters())
    assert all(p.dtype == torch.float32 for p in model.coeff_fusion.parameters())
    del payload, stage1, state
    gc.collect()
    return model, aux, scales, converted


def chunk_path(sampler, first):
    return WORK / 'chunks' / sampler / f'{first:06d}.npz'


def valid_chunk(path, first, count, seed):
    if not path.exists():
        return False
    with np.load(path) as chunk:
        assert int(chunk['first']) == first and int(chunk['seed']) == seed
        assert np.array_equal(chunk['labels'], np.arange(first, first + count) % 1000)
        assert chunk['atoms'].shape == chunk['coefficient_ids'].shape == (count, 8, 8, 4)
        assert chunk['features'].shape == (count, 2048)
        assert chunk['probabilities'].shape == (count, 1008)
        assert np.isfinite(chunk['features']).all() and np.isfinite(chunk['probabilities']).all()
    return True


def compute_result(sampler, protocol):
    total = torch.zeros(2048, dtype=torch.float64)
    cross = torch.zeros(2048, 2048, dtype=torch.float64)
    probabilities, class_counts = [], np.zeros(1000, dtype=np.int64)
    count = 0
    for first in range(0, 50000, protocol['generation_batch']):
        current = min(protocol['generation_batch'], 50000 - first)
        seed = protocol['seed'] + first
        path = chunk_path(sampler, first)
        assert valid_chunk(path, first, current, seed)
        with np.load(path) as chunk:
            features = torch.from_numpy(chunk['features']).double()
            total += features.sum(0)
            cross.addmm_(features.T, features)
            probabilities.append(torch.from_numpy(chunk['probabilities']))
            class_counts += np.bincount(chunk['labels'], minlength=1000)
            count += len(features)
    assert count == 50000 and np.all(class_counts == 50)
    mean = total / count
    covariance = (cross - torch.outer(total, total) / count) / (count - 1)
    real_mean, real_covariance = load_reference_statistics(BASE / 'inputs/imagenet_256_train.npz')
    fid = frechet_distance(real_mean, real_covariance, mean.numpy(), covariance.numpy())
    probabilities = torch.cat(probabilities)
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(protocol['seed']))
    is_mean, is_std = inception_score(probabilities[permutation])
    result = dict(sampler=sampler, generated_images=count, training_reference_images=1281167,
                  checkpoint_epoch=8, checkpoint_step=5008, fid=fid,
                  inception_score=is_mean, inception_score_std=is_std,
                  class_counts_min=int(class_counts.min()), class_counts_max=int(class_counts.max()),
                  protocol=protocol, completed_at=time.time())
    assert np.isfinite([fid, is_mean, is_std]).all()
    np.savez(OUT / f'{sampler}-generated-statistics.npz', mu=mean.numpy(), sigma=covariance.numpy())
    write_json(OUT / f'{sampler}-result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cpu-workers', type=int, default=6)
    parser.add_argument('--cpu-threads', type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    limit = 4.5 * 1024**3
    torch.cuda.set_per_process_memory_fraction(limit / torch.cuda.get_device_properties(device).total_memory, device)
    model, aux, scales, converted = load_sampling_model(device)
    frozen_joint = OUT / 'compound_joint_sampling.py'
    if not frozen_joint.exists(): shutil.copyfile(ROOT / 'src/compound_joint_sampling.py', frozen_joint)
    joint = load_joint(frozen_joint)
    policy = joint.CompoundJointPolicy()
    protocol = dict(
        checkpoint_epoch=8, checkpoint_step=5008,
        checkpoint_sha256=digest(WORK / 'checkpoint.pt'),
        generated_images_per_sampler=50000, training_reference_images=1281167,
        reference='published ImageNet-256 full training statistics',
        reference_sha256=digest(BASE / 'inputs/imagenet_256_train.npz'),
        tokenizer_sha256=digest(BASE / 'inputs/stage1-tokenizer.pt'),
        metric_backend='original-rqvae', rejection_sampling=False,
        generation_batch=args.batch_size, cpu_decode_batch=8, seed=261001,
        native=dict(atom_temperature=1., atom_top_k=256, atom_top_p=.95,
                    coeff_temperature=1., coeff_top_k=0, coeff_top_p=.95),
        joint32_t1=asdict(policy),
        sampler_source_sha256=digest(frozen_joint),
        cuda_memory_limit_gib=4.5, autocast='production CUDA FP16',
        autocast_linear_parameters_stored_fp16=converted,
        classification_heads_and_residuals='FP32',
        cpu_workers=args.cpu_workers, cpu_threads_per_worker=args.cpu_threads,
    )
    protocol = json.loads(json.dumps(protocol))
    protocol_path = WORK / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Resume requires identical protocol'
    else:
        write_json(protocol_path, protocol)
    write_json(OUT / 'protocol.json', protocol)
    if not args.smoke:
        shutil.copyfile(Path(__file__), OUT / 'evaluate_compound_joint_fid.py')
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        os.environ['WANDB_DIR'] = str(WORK)
        import wandb
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=EVAL_RUN,
            name='ImageNet epoch 8 | ancestral vs joint32 | matched FID50k',
            job_type='sampler-evaluation', group=TRAIN_RUN, resume='allow',
            config=dict(protocol, parent_run=TRAIN_RUN, wandb_id=EVAL_RUN))
        run.define_metric('evaluation/generated_images')
        run.define_metric('progress/*', step_metric='evaluation/generated_images')
        run.define_metric('eval/*', summary='last')
        run.summary.update({'evaluation/status':'running', 'evaluation/checkpoint_epoch':8})
        for path in [OUT / 'protocol.json', frozen_joint, OUT / 'evaluate_compound_joint_fid.py']:
            run.save(str(path), base_path=str(OUT), policy='now')
        write_json(OUT / 'launch.json', dict(pid=os.getpid(), wandb_url=run.url,
                   wandb_id=EVAL_RUN, started_at=time.time(), protocol=protocol))
    else:
        run = None
    completed = {'ancestral':0, 'joint32_t1':0}
    futures = {}
    start = time.monotonic()

    def finish(future):
        item = future.result()
        completed[item['sampler']] += item['samples']
        status = dict(phase='smoke' if args.smoke else 'generating', pid=os.getpid(),
                      completed=completed.copy(), elapsed_seconds=time.monotonic()-start,
                      cuda_peak_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                      time=time.time())
        write_json(OUT / ('smoke-progress.json' if args.smoke else 'progress.json'), status)
        if run:
            run.log({'evaluation/generated_images':min(completed.values()),
                     **{f'progress/{name}_images':n for name,n in completed.items()}})
        print(json.dumps(status), flush=True)
        del futures[future]

    limit_samples = args.batch_size if args.smoke else 50000
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.cpu_workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=cpu_initialize, initargs=(scales,args.cpu_threads),
    ) as pool, torch.inference_mode():
        for first in range(0, limit_samples, args.batch_size):
            count = min(args.batch_size, limit_samples-first)
            seed = protocol['seed'] + first
            labels = torch.arange(first, first+count, device=device).remainder(1000)
            for sampler in completed:
                if valid_chunk(chunk_path(sampler,first),first,count,seed):
                    completed[sampler] += count
                    continue
                while len(futures) >= args.cpu_workers * 2:
                    ready, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in ready: finish(future)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                if sampler == 'ancestral':
                    atoms, coefficients = model.sample_compound(count,aux,labels,amp=True,**protocol['native'])
                else:
                    atoms, coefficients = joint.sample_compound_joint(model,count,aux,labels,amp=True,policy=policy)
                assert atoms.shape == coefficients.shape == (count,8,8,4)
                for depth in range(1,4):
                    assert not (atoms[...,depth,None] == atoms[...,:depth]).any()
                assert int(atoms.min()) >= 0 and int(atoms.max()) < 16384
                assert int(coefficients.min()) >= 0 and int(coefficients.max()) < 2048
                task = (sampler, first, atoms.cpu().numpy().astype(np.int16),
                        coefficients.cpu().numpy().astype(np.int16),
                        labels.cpu().numpy().astype(np.int16), seed)
                future = pool.submit(decode_and_measure, task)
                futures[future] = (sampler,first)
                del atoms,coefficients,task
                torch.cuda.empty_cache()
                for future in list(futures):
                    if future.done(): finish(future)
        for future in concurrent.futures.as_completed(list(futures)):
            finish(future)
    if args.smoke:
        assert completed == {'ancestral':args.batch_size,'joint32_t1':args.batch_size}
        write_json(OUT / 'smoke-verification.json', dict(passed=True,completed=completed,
                   cuda_peak_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                   full_fid_computed=False, protocol=protocol))
        return
    assert completed == {'ancestral':50000,'joint32_t1':50000}
    del model,aux
    torch.cuda.empty_cache()
    torch.set_num_threads(16)
    results = {}
    for sampler in completed:
        result = compute_result(sampler,protocol)
        results[sampler] = result
        run.log({f'eval/{sampler}_fid':result['fid'],
                 f'eval/{sampler}_inception_score':result['inception_score'],
                 f'eval/{sampler}_inception_score_std':result['inception_score_std']})
        run.save(str(OUT / f'{sampler}-result.json'),base_path=str(OUT),policy='now')
        run.save(str(OUT / f'{sampler}-generated-statistics.npz'),base_path=str(OUT),policy='now')
        run.log({f'samples/{sampler}':wandb.Image(str(OUT / f'{sampler}-preview.png'))})
    delta = results['joint32_t1']['fid']-results['ancestral']['fid']
    comparison = dict(completed=True,checkpoint_epoch=8,checkpoint_step=5008,
                      ancestral_fid=results['ancestral']['fid'],joint32_t1_fid=results['joint32_t1']['fid'],
                      joint_minus_ancestral_fid=delta,joint_improved=delta<0,
                      generated_images_per_sampler=50000,training_reference_images=1281167,
                      wandb_url=run.url,protocol=protocol,elapsed_seconds=time.monotonic()-start)
    write_json(OUT / 'comparison.json',comparison)
    run.log({'eval/joint_minus_ancestral_fid':delta})
    run.summary.update({'evaluation/status':'completed','evaluation/comparison':comparison})
    run.save(str(OUT / 'comparison.json'),base_path=str(OUT),policy='now')
    run.finish()
    print(json.dumps(comparison),flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_json(OUT / 'failure.json',dict(error=repr(error),time=time.time(),pid=os.getpid()))
        raise
