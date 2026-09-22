"""Controlled prefix diagnostics and sampling ablations with a frozen tokenizer.

Real-code prefix experiments are conditional diagnostics, never unconditional
benchmark scores. All cases use identical sample counts, seeds and FID reference.
"""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import save_image

from src.training.cli import load_config
from src.training.compound_var import CompoundExperiment
from src.training.var_laser import get_inception_model, frechet_distance
from src.original_rq_training import FeatureMoments, atomic_json, file_sha256
from src.data.var_token_cache import VARTokenCache
from src.training.distributed_failure import fatal_worker_error


@torch.no_grad()
def measure(experiment, model, cache, case, count, seed, reference, output):
    rank, world, device = experiment.rank, experiment.world, experiment.device
    forced = int(case.get('teacher_scales', 0))
    if forced and count > len(cache):
        raise ValueError('Prefix diagnostics require distinct validation images')
    indices = np.arange(rank, count, world)
    real_indices = np.random.default_rng(123).permutation(len(cache))
    moments = FeatureMoments(device)
    telemetry = torch.zeros(len(model.patch_nums), 3, dtype=torch.float64, device=device)
    options = {k: v for k, v in case.items() if k != 'name'}
    batch_size = int(experiment.cfg.evaluation.batch_size)
    started = time.monotonic()
    q = experiment.vae.quantize
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin:begin + batch_size]
        teacher = None
        if forced:
            positions = real_indices[selected]
            teacher = {name: torch.from_numpy(np.array(cache.arrays[name][positions, 0, 0])).long().to(device)
                       for name in ('atoms', 'coefficients')}
        labels = torch.zeros(len(selected), dtype=torch.long, device=device)
        with experiment.amp():
            result = model.sample(labels, cfg=0., seed=seed + rank + begin * world,
                                  teacher_codes=teacher, return_codes=True, **options)
        with torch.autocast('cuda', enabled=False):
            pixels = experiment.vae.fhat_to_img(result['latent'].float()).mul(.5).add(.5).clamp(0, 1)
            moments.update(experiment.inception(pixels))
            offset = 0
            for scale, pn in enumerate(model.patch_nums):
                end = offset + pn * pn
                ids = result['coefficients'][:, offset:end]
                atoms = result['atoms'][:, offset:end]
                values = q.coefficient_values(ids, scale)
                vectors = torch.nn.functional.embedding(atoms, q.normalized_dictionary().T)
                contribution = q.contribution(q.embed(atoms, ids, scale), scale)
                telemetry[scale, 0] += values.square().mean().double() * len(selected)
                telemetry[scale, 1] += (vectors[..., 0, :] * vectors[..., 1, :]).sum(-1).abs().mean().double() * len(selected)
                telemetry[scale, 2] += contribution.square().mean().double() * len(selected)
                offset = end
        if rank == 0 and begin == 0:
            save_image(pixels[:64], output / f'{case["name"]}-grid.png', nrow=8)
        if rank == 0 and begin % (batch_size * 25) == 0:
            atomic_json(output / 'status.json', dict(phase='sampling', case=case['name'],
                approximate_images=min((begin + len(selected)) * world, count), count=count, time=time.time()))
    generated, mean, covariance = moments.finish()
    dist.all_reduce(telemetry)
    if generated != count:
        raise RuntimeError('Incorrect generated sample count')
    record = None
    if rank == 0:
        score = float(frechet_distance(mean, covariance, reference['mu'], reference['sigma']))
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite FID')
        telemetry = telemetry.cpu().numpy() / count
        record = dict(case=case, samples=count, seed=seed, world_size=world, batch_size=batch_size,
                      fid=score, conditional_on_real_codes=bool(forced),
                      interpretation='Real-code conditional diagnostic; not unconditional generation FID' if forced else
                                     'Unconditional generation; compare only equal sample counts and reference',
                      elapsed_seconds=time.monotonic() - started,
                      per_scale=[dict(resolution=pn, coefficient_rms=float(row[0] ** .5),
                                      pair_abs_cosine=float(row[1]), contribution_rms=float(row[2] ** .5))
                                 for pn, row in zip(model.patch_nums, telemetry)])
        atomic_json(output / f'{case["name"]}.json', record)
        np.savez(output / f'{case["name"]}-moments.npz', mu=mean, sigma=covariance)
        print(json.dumps(record), flush=True)
        if experiment.run:
            import wandb
            experiment.run.log({'case': case['name'], 'fid': score, 'samples': count,
                               'conditional_on_real_codes': bool(forced),
                               'samples_grid': wandb.Image(str(output / f'{case["name"]}-grid.png'))})
    dist.barrier()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--reference-manifest', type=Path)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=73000)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    experiment = None
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        cfg = load_config(args.config)
        cfg.compound.mode = str(args.output.resolve())
        cfg.execution.media_dir = str(args.output)
        cfg.execution.wandb_dir = str(args.output / 'wandb')
        experiment = CompoundExperiment(cfg)
        if experiment.num_classes != 1:
            raise ValueError('This investigation expects unconditional FFHQ')
        saved = torch.load(args.checkpoint, weights_only=False, map_location='cpu', mmap=True)
        if saved['tokenizer_sha256'] != experiment.tokenizer_sha256:
            raise ValueError('Wrong tokenizer for checkpoint')
        model = experiment.build_prior().eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        epoch = int(saved['progress']['epoch'])
        del saved
        cache = VARTokenCache(cfg.execution.token_cache_dir, 'validation')
        if cache.manifest['tokenizer_sha256'] != experiment.tokenizer_sha256:
            raise ValueError('Wrong tokenizer for real-code prefixes')
        if cache.manifest['dataset_fingerprints']['validation'] != experiment.val.dataset._fingerprint:
            raise ValueError('Validation cache dataset mismatch')
        reference_manifest = args.reference_manifest or cfg.evaluation.get('fid_reference_manifest')
        reference_contract = None
        if reference_manifest:
            from src.training.ffhq_fid_protocol import validate_matched_reference
            reference_contract = validate_matched_reference(args.reference, reference_manifest, cfg.data.root,
                dict(train=experiment.train.dataset._fingerprint, validation=experiment.val.dataset._fingerprint))
            if args.samples != 50000:
                raise ValueError('Matched FFHQ benchmark requires 50000 generated samples')
        experiment.inception = get_inception_model().eval().requires_grad_(False).to(experiment.device)
        with np.load(args.reference, allow_pickle=False) as data:
            reference = {name: data[name] for name in ('mu', 'sigma')}
        cases = json.loads(args.plan.read_text())
        if reference_contract is not None and any(case.get('teacher_scales', 0) for case in cases):
            raise ValueError('Matched FFHQ benchmark requires unconditional generation')
        if len({case['name'] for case in cases}) != len(cases):
            raise ValueError('Case names must be unique')
        if experiment.rank == 0:
            metadata = dict(checkpoint=args.checkpoint, checkpoint_sha256=file_sha256(args.checkpoint), epoch=epoch,
                tokenizer_sha256=experiment.tokenizer_sha256, reference=args.reference,
                reference_sha256=file_sha256(args.reference), samples=args.samples, seed=args.seed,
                world_size=experiment.world, batch_size=cfg.evaluation.batch_size, cases=cases,
                pixel_protocol='FP32 decoder; continuous float [0,1]; no uint8 rounding',
                reference_contract=reference_contract,
                reference_manifest_sha256=file_sha256(reference_manifest) if reference_manifest else None)
            atomic_json(args.output / 'experiment.json', metadata)
            if args.online:
                import wandb
                Path(cfg.execution.wandb_dir).mkdir(parents=True, exist_ok=True)
                experiment.run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                    id=args.run_id, name=args.run_id, group=cfg.wandb.group, resume='allow',
                    mode='online', dir=cfg.execution.wandb_dir, config=metadata)
        results = []
        for case in cases:
            record = measure(experiment, model, cache, case, args.samples, args.seed, reference, args.output)
            if record:
                results.append(record)
        if experiment.rank == 0:
            atomic_json(args.output / 'results.json', results)
            if experiment.run:
                import wandb
                artifact = wandb.Artifact(args.run_id + '-evaluation', type='evaluation', metadata=metadata)
                for pattern in ('*.json', '*.yaml', '*-grid.png'):
                    for path in args.output.glob(pattern):
                        artifact.add_file(str(path), name=path.name)
                for path in (Path(__file__), Path(__file__).resolve().parents[2] / 'src/models/compound_var.py'):
                    artifact.add_file(str(path), name='source/' + path.name)
                if reference_manifest:
                    artifact.add_file(str(reference_manifest), name='reference/manifest.json')
                    artifact.add_file(str(args.reference), name='reference/statistics.npz')
                    artifact.add_file(str(Path(reference_manifest).parent/'laser-train-ids.npy'), name='reference/laser-train-ids.npy')
                experiment.run.log_artifact(artifact).wait()
                atomic_json(args.output / 'upload.json', dict(artifact=artifact.qualified_name, state=artifact.state))
                experiment.run.finish()
            atomic_json(args.output / 'complete.json', dict(cases=len(results), time=time.time()))
        dist.barrier()
    except BaseException as error:
        fatal_worker_error(error, args.output, dist.get_rank())
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
