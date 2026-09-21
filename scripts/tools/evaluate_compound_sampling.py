"""Matched compound-token sampling sweep using an immutable training runtime."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(training, config, checkpoint, device):
    import torch
    model = training.build_model(
        config['num_atoms'] + config['coeff_vocab_size'], config['num_atoms'],
        compound=True, coeff_vocab_size=config['coeff_vocab_size'],
        compound_refiner_layers=config['compound_refiner_layers'],
        compound_geometry_head=config['compound_distribution_geometry'],
        compound_micro_transformer_layers=config['compound_micro_transformer_layers'],
        compound_depth_specific_coeff_heads=config['compound_depth_specific_coeff_heads'],
        compound_causal_prefix_state=config['causal_prefix_state'],
        compound_pair_autoregressive=config['compound_pair_autoregressive'],
        sparsity_level=config['sparsity_level'], model_preset=config['model_preset'],
    )
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(payload['state_dict'], strict=True)
    assert payload['config']['coeff_scales'] == config['coeff_scales']
    aux = training.LaserAux(
        Path(config['checkpoint']), config['num_atoms'], config['coeff_vocab_size'],
        config['coeff_max'], config['coeff_scale'], attn_resolutions=(8,),
        coeff_scales=config['coeff_scales'], soft_target_physical=True,
        sparsity_level=config['sparsity_level'],
    )
    return model.to(device).eval().requires_grad_(False), aux.to(device).eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    request = read(directory / 'request.json')
    screen_seeds = {request['screen_seed'] + rank for rank in range(request['world_size'])}
    confirm_seeds = {request['confirm_seed'] + rank for rank in range(request['world_size'])}
    assert screen_seeds.isdisjoint(confirm_seeds), 'Screening and confirmation rank seeds overlap'
    parent = Path(request['training_directory'])
    runtime = parent / 'runtime'
    sys.path[:0] = [str(runtime), str(runtime / 'third_party/rq-vae-transformer')]
    import numpy as np
    import torch
    import torch.distributed as dist
    from torchvision.utils import save_image
    from fast_attention import install
    from src.training import rqtransformer as training
    torch.set_num_threads(8)
    install()
    config = read(parent / 'train/request.json')['config']
    if args.cpu_preflight:
        candidates = sorted((parent / 'train/checkpoints').glob('best_fid_*.pt'))
        assert len(candidates) == 1
        model, aux = build(training, config, candidates[0], torch.device('cpu'))
        write(directory / 'cpu-preflight.json', dict(
            passed=True, strict_state_dict=True, gpu_used=False,
            checkpoint=str(candidates[0]), parameters=sum(p.numel() for p in model.parameters()),
            model_type=type(model).__name__, auxiliary_type=type(aux).__name__,
            evaluator_sha256=sha(__file__), checked_unix=time.time(),
        ))
        return
    world = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert world == request['world_size'] == 8
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    frozen = read(directory / 'frozen-checkpoint.json')
    model, aux = build(training, config, frozen['path'], device)
    assert torch.get_autocast_dtype('cuda') == torch.float16
    wb = None
    if rank == 0:
        import wandb
        wb = wandb.init(
            entity='helloimlixin-rutgers', project='laser', id=request['run_id'],
            name=request['run_id'], resume='allow', mode='online', dir=str(directory),
            job_type='sampling-sweep',
            tags=['lsun-church', 'compound', 'sampling-sweep', '8xH200'],
        )
        wb.config.update(dict(**request, frozen_checkpoint=frozen), allow_val_change=True)
        wb.use_artifact(frozen['artifact'])
        assets = read(parent / 'train/token_cache_artifact.json')['input_artifacts']
        for artifact in assets.values():
            wb.use_artifact(artifact)
        write(directory / 'wandb-run.json', dict(id=wb.id, url=wb.url, mode='online'))
    dist.barrier()

    @torch.no_grad()
    def evaluate(phase, name, count, seed):
        destination = directory / phase / name
        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / 'result.json'
        identity = dict(phase=phase, setting=name, num_samples=count, seed=seed,
                        parameters=request['settings'][name], checkpoint_sha256=frozen['sha256'],
                        reference_sha256=request['reference_sha256'],
                        batch_size_per_gpu=request['batch_size_per_gpu'], world_size=world)
        if result_path.exists():
            result = read(result_path)
            assert all(result[key] == value for key, value in identity.items())
            assert (destination / 'samples.png').is_file()
            dist.barrier()
            return result
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        start = time.monotonic()
        seen = 0
        first_images = None
        first_codes = None
        original_decode = aux.decode_compound

        def capture_decode(atoms, coefficients):
            nonlocal seen, first_images, first_codes
            decoded = original_decode(atoms, coefficients)
            assert decoded.dtype == torch.float32
            if first_images is None:
                first_images = ((decoded[:8].float() + 1.) * .5).clamp(0, 1).detach()
                assert torch.isfinite(first_images).all()
                # NCCL collectives support int32, but not int16.
                first_codes = torch.stack([atoms[:8], coefficients[:8]], dim=-1).to(torch.int32)
            seen += decoded.shape[0]
            if rank == 0:
                progress = dict(**identity, local_generated=seen, expected_global_generated=seen * world,
                                elapsed_seconds=time.monotonic() - start, updated_unix=time.time())
                write(directory / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
            return decoded

        aux.decode_compound = capture_decode
        try:
            fid, _, _ = training.evaluate_generation_metrics(
                model, aux, None, count, batch_size=request['batch_size_per_gpu'],
                num_condition_classes=1, compute_inception_score=False,
                metric_backend='original-rqvae', fid_reference_stats=config['fid_reference_stats'],
                **request['settings'][name],
            )
        finally:
            aux.decode_compound = original_decode
        assert seen == count // world + (rank < count % world)
        assert math.isfinite(fid)
        collected = [torch.empty_like(first_images) for _ in range(world)]
        collected_codes = [torch.empty_like(first_codes) for _ in range(world)]
        dist.all_gather(collected, first_images)
        dist.all_gather(collected_codes, first_codes)
        memory = torch.tensor(torch.cuda.max_memory_allocated(), device=device)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        elapsed = time.monotonic() - start
        result = dict(**identity, fid=fid, elapsed_seconds=elapsed,
                      images_per_second=count / elapsed, max_gpu_allocated_bytes=memory.item(),
                      fid_reference_images=126227, inference_autocast='float16',
                      decoder_dtype='float32', inception_dtype='float32', tf32=True,
                      native_evaluator=str(Path(training.__file__).resolve()), completed_unix=time.time())
        if rank == 0:
            save_image(torch.stack(collected, dim=1).flatten(0, 1).cpu(), destination / 'samples.png', nrow=8)
            torch.save(dict(codes=torch.stack(collected_codes, dim=1).flatten(0, 1).cpu().to(torch.int16),
                            global_sample_ids=torch.arange(64), metadata=identity), destination / 'preview-codes.pt')
            write(result_path, result)
            wb.log({f'{phase}/{name}/fid': fid, f'{phase}/{name}/images_per_second': count / elapsed,
                    f'{phase}/{name}/samples': wandb.Image(str(destination / 'samples.png'))})
            print(json.dumps(dict(event='setting_complete', **result)), flush=True)
        dist.barrier()
        return result

    screen = [evaluate('screen', name, request['screen_samples'], request['screen_seed'])
              for name in request['settings']]
    baseline = request['baseline']
    challenger = min((r for r in screen if r['setting'] != baseline), key=lambda r: (r['fid'], r['setting']))['setting']
    confirmation = [evaluate('confirm', name, request['confirm_samples'], request['confirm_seed'])
                    for name in [baseline, challenger]]
    winner = min(confirmation, key=lambda r: (r['fid'], r['setting']))
    if rank == 0:
        report = dict(screen=screen, confirmation=confirmation, baseline=baseline, challenger=challenger,
                      selected_setting=winner['setting'], selected_fid_50000=winner['fid'],
                      baseline_fid_50000=confirmation[0]['fid'],
                      fid_improvement=confirmation[0]['fid'] - winner['fid'],
                      frozen_checkpoint=frozen,
                      caution='One independent confirmation seed; no statistical significance claim. '
                              'FID4096 and FID50000 have different sample-size bias and must not be compared directly.')
        write(directory / 'report.json', report)
        write(directory / 'selected-sampling.json', dict(**winner['parameters'],
              checkpoint=frozen['path'], checkpoint_sha256=frozen['sha256'],
              fid_50000=winner['fid'], seed=request['confirm_seed'], run_url=wb.url))
        wb.summary.update(dict(selected_setting=winner['setting'], selected_fid_50000=winner['fid'],
                               baseline_fid_50000=confirmation[0]['fid'], fid_improvement=report['fid_improvement']))
        table = wandb.Table(columns=['phase', 'setting', 'samples', 'seed', 'fid'])
        for row in screen + confirmation:
            table.add_data(row['phase'], row['setting'], row['num_samples'], row['seed'], row['fid'])
        wb.log({'sampling_results': table})
        artifact = wandb.Artifact(f'{wb.id}-evaluation', type='evaluation', metadata=dict(
            checkpoint_sha256=frozen['sha256'], selected_setting=winner['setting'],
            selected_fid_50000=winner['fid'], baseline_fid_50000=confirmation[0]['fid']))
        sources = [directory / filename for filename in ['request.json', 'frozen-checkpoint.json',
                   'report.json', 'selected-sampling.json', 'source-manifest.json', 'evaluate.py', 'supervise.py']]
        sources += [path for phase in ['screen', 'confirm'] for path in sorted((directory / phase).glob('*/*'))]
        import base64
        files = []
        for path in sources:
            name = str(path.relative_to(directory))
            with path.open('rb') as stream:
                digest = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
            files.append(dict(name=name, bytes=path.stat().st_size, md5=digest))
            artifact.add_file(str(path), name=name, policy='immutable', skip_cache=True)
        committed = wb.log_artifact(artifact, aliases=['latest', 'selected-sampling'])
        committed.wait()
        remote = wandb.Api(timeout=180).artifact(committed.qualified_name)
        assert str(remote.state).upper() == 'COMMITTED'
        for item in files:
            entry = remote.manifest.entries[item['name']]
            assert entry.size == item['bytes'] and entry.digest == item['md5']
        receipt = dict(verified_online=True, artifact=committed.qualified_name, files=files, url=wb.url)
        write(directory / 'upload.json', receipt)
        wb.summary['evaluation_artifact'] = committed.qualified_name
        wb.summary['verified_online'] = True
        wb.finish()
        write(directory / 'complete.json', dict(passed=True, **report, upload=receipt))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
