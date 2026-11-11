"""Frozen epoch78 sampler search with depth temperatures and matched class images."""
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

    from rqvae.models.rqtransformer import transformers as native_module
    original_draw = native_module.sample_from_logits

    @torch.inference_mode()
    def sample_codes(setting, labels):
        temperature = setting['temperature']
        args = dict(model_aux=aux, cond=labels, amp=True, top_k=setting['top_k'], top_p=setting['top_p'])
        partial = torch.zeros(len(labels), 8, 8, 4, dtype=torch.long, device=device)
        assert native_module.sample_from_logits is original_draw
        if not isinstance(temperature, list):
            return model.sample(partial, temperature=float(temperature), **args)
        assert len(temperature) == 4 and all(float(t)>0 for t in temperature)
        counts = [0,0,0,0]
        calls = 0
        def draw(logits, *positional, **kwargs):
            nonlocal calls
            depth = calls % 4
            calls += 1
            counts[depth] += 1
            kwargs['temperature'] = float(temperature[depth])
            return original_draw(logits, *positional, **kwargs)
        native_module.sample_from_logits = draw
        try:
            result = model.sample(partial, temperature=1.0, **args)
        finally:
            native_module.sample_from_logits = original_draw
        assert calls == 256 and counts == [64,64,64,64]
        return result

    def save_preview(name, setting):
        from PIL import Image
        from torchvision.utils import make_grid
        base_policy = plan['case_base_policy'].get(name,name)
        seed = int(plan['policy_seeds'].get(name,plan['previews']['primary_seed']))
        key = base_policy+'__s'+str(seed)
        destination = local/'previews'/key/f'rank-{rank}.png'
        receipt = destination.with_suffix('.json')
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['checkpoint_sha256']==plan['checkpoint_sha256'] and prior['sampling']==setting and destination.exists()
            return
        class_info = plan['previews']['classes'][rank]
        count = plan['previews']['images_per_class']
        preview_seed = int(plan['previews']['seed_offset'])+seed+rank
        before = torch.cuda.get_rng_state(device)
        status(phase='preview',policy=name,preview_images=count)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(preview_seed)
            torch.cuda.manual_seed(preview_seed)
            labels = torch.full((count,),int(class_info['imagenet_id']),dtype=torch.long,device=device)
            with torch.inference_mode():
                codes = sample_codes(setting,labels)
                pixels = profile.decode(aux,codes).float().mul(.5).add(.5).clamp(0,1)
                grid = make_grid(pixels,nrow=count,padding=2)
                array = grid.mul(255).round().byte().permute(1,2,0).cpu().numpy()
        assert torch.equal(before,torch.cuda.get_rng_state(device))
        destination.parent.mkdir(parents=True,exist_ok=True)
        pending = destination.with_suffix('.pending')
        Image.fromarray(array).save(pending,format='PNG')
        pending.replace(destination)
        write(receipt,dict(rank=rank,key=key,base_policy=base_policy,generation_seed=seed,
            preview_seed=preview_seed,class_info=class_info,images=count,sampling=setting,
            checkpoint_sha256=plan['checkpoint_sha256'],unfiltered=True,excluded_from_metric_count=True,
            sampling_rng_restored=True,sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))

    @torch.inference_mode()
    def generate(setting, n, first):
        labels = (torch.arange(first, first+n, device=device)*world+rank)%1000
        codes = sample_codes(setting,labels)
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

    status(phase='verifying_depth_temperature_adapter')
    with torch.random.fork_rng(devices=[device.index]):
        labels = torch.full((4,),plan['previews']['classes'][rank]['imagenet_id'],dtype=torch.long,device=device)
        seed = 861104+rank
        current = plan['policies']['validated']
        torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        native_codes = sample_codes(current,labels)
        native_rng = torch.cuda.get_rng_state(device)
        uniform = dict(current,temperature=[float(current['temperature'])]*4)
        torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        uniform_codes = sample_codes(uniform,labels)
        assert torch.equal(native_codes,uniform_codes)
        assert torch.equal(native_rng,torch.cuda.get_rng_state(device))
        torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        varied_codes = sample_codes(plan['adapter_probe'],labels)
        assert not torch.equal(native_codes,varied_codes)
    write(local/f'adapter-verification-rank-{rank}.json',dict(passed=True,rank=rank,
        native_uniform_temperature_codes_identical=True,native_uniform_rng_consumption_identical=True,
        nonuniform_temperatures_change_codes=True,depth_calls=[64,64,64,64],model_architecture_unmodified=True))
    torch.cuda.empty_cache()

    def evaluate(stage, names):
        samples, default_seed = plan[f'{stage}_images'], plan[f'{stage}_seed']
        count = samples//world + (rank < samples%world)
        for name in names:
            seed = int(plan.get('policy_seeds', {}).get(name, default_seed))
            save_preview(name,plan['policies'][name])
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
            initial_cuda_rng_sha256 = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
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
                seed=seed, initial_cuda_rng_sha256=initial_cuda_rng_sha256, count=count, sampling=plan['policies'][name], seconds=time.monotonic()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30))
            print(json.dumps(dict(event='policy_complete', stage=stage, policy=name, rank=rank)), flush=True)
            del total, cross, fourth, probabilities
            torch.cuda.empty_cache()

    evaluate('screen', plan['screen_policies'])
    for round_name in ['primary', 'replicate']:
        status(phase='waiting_for_'+round_name+'_selection')
        selection_path = local/(round_name+'-selection.json')
        while not selection_path.exists():
            time.sleep(2)
        selection = json.loads(selection_path.read_text())
        assert selection['checkpoint_sha256'] == plan['checkpoint_sha256']
        evaluate('confirm', selection['policies'])
    verify()
    write(local/f'verification-rank-{rank}.json', dict(passed=True, rank=rank,
        model_weights_unmodified=True, tokenizer_unchanged=True, tokenizer_digest=frozen_tokenizer,
        active_recipe_unchanged=True, checkpoint_unchanged=True, stage1_preservation=preservation))
    status(phase='completed')


if __name__ == '__main__':
    main()
