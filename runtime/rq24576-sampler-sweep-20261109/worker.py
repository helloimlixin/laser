"""Frozen native RQ inference sharing spare VRAM with training, without NCCL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    base, local = Path(plan['base']), Path(plan['local'])
    rank, world = args.rank, plan['world_size']
    os.environ.update(RQ_UPSTREAM_ROOT=str(base/'upstream'), TORCH_HOME=str(base/'torch-cache'))
    sys.path[:0] = [str(base/'source'), str(base), str(base/'upstream')]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from runtime.precision import install_bf16
    from runtime.native_recipe import NativeRecipe
    from src import rqvae_metrics as metrics
    install_bf16()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(plan['memory_gib']*2**30/properties.total_memory, device)
    status_path = local/f'worker-{rank}.json'

    def status(**values):
        write(status_path, dict(rank=rank, pid=os.getpid(), time=time.time(), **values))

    status(phase='loading_frozen_checkpoint')
    checkpoint = Path(plan['checkpoint'])
    identity = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    assert identity[0] == plan['checkpoint_bytes']
    profile = NativeRecipe(base)
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    profile.verify_resume(payload)
    assert int(payload['global_step']) == plan['source_step']
    assert {int(v['step']) for v in payload['optimizer']['state'].values()} == {plan['source_step']}
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(payload['config']['arch']))
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).eval().to(device)
    del payload
    import gc
    gc.collect()
    aux, _ = profile.tokenizer(device)
    preservation = profile.verify_codec(aux)
    frozen_tokenizer = profile.digest(aux)
    versions = [(name, p._version) for name, p in model.named_parameters()]
    assert all(not p.requires_grad for p in model.parameters())
    assert all(not p.requires_grad for p in aux.parameters())
    metrics.INCEPTION_PATH = base/'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().eval().requires_grad_(False).to(device)
    mean, covariance = metrics.load_reference_statistics(base/'inputs/imagenet_256_train.npz')
    mean_t = torch.as_tensor(mean, dtype=torch.float64, device=device)
    std_t = torch.as_tensor(np.sqrt(np.maximum(np.diag(covariance), 1e-8)), dtype=torch.float64, device=device)

    def native_settings(setting):
        return dict(temperature=float(setting['temperature']), top_k=setting['top_k'], top_p=setting['top_p'])

    @torch.inference_mode()
    def generate(setting, n, first):
        labels = (torch.arange(first, first+n, device=device)*world+rank)%1000
        codes = model.sample(torch.zeros(n, 8, 8, 4, dtype=torch.long, device=device),
            model_aux=aux, cond=labels, amp=True, **native_settings(setting))
        assert codes.shape == (n, 8, 8, 4) and codes.dtype == torch.long
        assert bool(((codes >= 0) & (codes < 24576)).all())
        for chunk in codes.split(plan['decode_batch']):
            images = profile.decode(aux, chunk).float().mul(.5).add(.5).clamp(0, 1)
            features, logits = inception(images, return_logits=True)
            assert bool(torch.isfinite(features).all()) and bool(torch.isfinite(logits).all())
            yield features.double(), logits.softmax(-1).cpu()

    def verify():
        assert versions == [(name, p._version) for name, p in model.named_parameters()]
        assert profile.digest(aux) == frozen_tokenizer
        assert identity == (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        assert hashlib.sha256((base/'inputs/recipe.yaml').read_bytes()).hexdigest() == plan['active_recipe_sha256']

    if args.preflight:
        status(phase='memory_preflight')
        torch.manual_seed(26110900)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        with torch.inference_mode():
            for _ in generate(plan['policies']['baseline'], plan['batch'], 0):
                pass
        torch.cuda.synchronize()
        verify()
        result = dict(passed=True, batch=plan['batch'], seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            memory_cap_gib=plan['memory_gib'], tokenizer_digest=frozen_tokenizer,
            source_step=plan['source_step'], stage1_preservation=preservation,
            weights_unmodified=True, training_recipe_unchanged=True)
        write(local/'memory-preflight.json', result)
        print(json.dumps(result), flush=True)
        return

    def evaluate(stage, names):
        samples, seed = plan[f'{stage}_images'], plan[f'{stage}_seed']
        count = samples//world + (rank < samples%world)
        for name in names:
            destination = local/stage/name/f'rank-{rank}.npz'
            receipt = destination.with_suffix('.json')
            if receipt.exists():
                prior = json.loads(receipt.read_text())
                assert prior['checkpoint_sha256'] == plan['checkpoint_sha256']
                assert prior['seed'] == seed and prior['count'] == count and destination.exists()
                assert prior['sampling'] == plan['policies'][name]
                continue
            torch.manual_seed(seed+rank)
            torch.cuda.manual_seed(seed+rank)
            total = torch.zeros(2048, dtype=torch.float64, device=device)
            cross = torch.zeros(2048, 2048, dtype=torch.float64, device=device)
            fourth = torch.zeros((), dtype=torch.float64, device=device)
            probabilities = []
            started = time.monotonic()
            status(phase=stage, policy=name, generated=0, target=count)
            with torch.inference_mode():
                for first in range(0, count, plan['batch']):
                    n = min(plan['batch'], count-first)
                    for features, probs in generate(plan['policies'][name], n, first):
                        total.add_(features.sum(0))
                        cross.addmm_(features.T, features)
                        if stage == 'screen':
                            normalized = (features-mean_t)/std_t
                            fourth += normalized.square().sum(-1).square().sum()
                        probabilities.append(probs)
                    status(phase=stage, policy=name, generated=first+n, target=count,
                        seconds=time.monotonic()-started)
            assert sum(len(p) for p in probabilities) == count
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.pending')
            with temporary.open('wb') as stream:
                np.savez(stream, total=total.cpu().numpy(), cross=cross.cpu().numpy(),
                    fourth=float(fourth), probabilities=torch.cat(probabilities).numpy(), count=count)
            temporary.replace(destination)
            write(receipt, dict(rank=rank, checkpoint_sha256=plan['checkpoint_sha256'],
                seed=seed, count=count, sampling=plan['policies'][name], seconds=time.monotonic()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))
            print(json.dumps(dict(event='policy_complete', stage=stage, policy=name, rank=rank)), flush=True)
            del total, cross, fourth, probabilities
            torch.cuda.empty_cache()

    evaluate('screen', list(plan['policies']))
    status(phase='waiting_for_confirmation_selection')
    while not (local/'confirm-selection.json').exists():
        time.sleep(2)
    selection = json.loads((local/'confirm-selection.json').read_text())
    assert selection['checkpoint_sha256'] == plan['checkpoint_sha256']
    evaluate('confirm', selection['policies'])
    verify()
    write(local/f'verification-rank-{rank}.json', dict(passed=True, rank=rank,
        model_weights_unmodified=True, tokenizer_unchanged=True, tokenizer_digest=frozen_tokenizer,
        active_recipe_unchanged=True, checkpoint_unchanged=True, stage1_preservation=preservation))
    status(phase='completed')


if __name__ == '__main__':
    main()
