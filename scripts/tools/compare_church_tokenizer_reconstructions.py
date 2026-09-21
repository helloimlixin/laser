#!/usr/bin/env python3
"""Paired held-out reconstruction audit of native LASER, compact LASER, and RQVAE."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-dir', type=Path, default=ROOT/'outputs/church-consistent-rqvae-20260914')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not 1 <= args.images <= 300 or args.batch_size < 1:
        parser.error('Use 1..300 held-out images and a positive batch size')
    base = args.pipeline_dir.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()

    # Import the active frozen driver so the compact tokenizer is restored by
    # exactly the same implementation and full-state hash check as production.
    runtime = ROOT/'outputs/church-ft3ep-scratch-adaptive-20260916/runtime'
    if '--pipeline-dir' not in sys.argv:
        sys.argv.extend(['--pipeline-dir', str(base)])
    driver = load_module('church_reconstruction_production', runtime/'scripts/tools/continue_church_stage2.py')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset
    from torchvision.utils import save_image
    from rqvae.img_datasets.lsun import LSUNClass
    from rqvae.img_datasets.transforms import create_transforms
    from src.original_rq_training import atomic_json, file_sha256, state_sha256, load_tokenizer, load_stage2_config
    from src.scaled_atom_rq import continuous_matching_pursuit
    import rqvae.losses.vqgan.lpips as lpips_module

    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.08, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20260917)

    def status(phase, **kwargs):
        row = dict(phase=phase, elapsed_seconds=time.time()-started, **kwargs)
        atomic_json(out/'status.json', row)
        print(json.dumps(row), flush=True)

    manifest = json.loads((base/'stage2-source-manifest.json').read_text())
    for path, expected in manifest.items():
        assert file_sha256(base/'stage2-source'/path) == expected, path
    launch_manifest = json.loads((base/'launch-manifest.json').read_text())
    native_source = base/'stage1-source/src/models'
    source_files = ['stage1-source/src/models/dictionary_learner.py',
                    'stage1-source/src/models/bottleneck_utils.py']
    for path in source_files:
        assert file_sha256(base/path) == launch_manifest[path], path
    for name in ['modules.py', 'layers.py']:
        native = base/'stage1-source/third_party/rq-vae-transformer/rqvae/models/rqvae'/name
        frozen = driver.UPSTREAM/'rqvae/models/rqvae'/name
        assert native.read_bytes() == frozen.read_bytes(), name

    cache = json.loads((base/'preparation/cache/complete.json').read_text())
    for name in ['checkpoint', 'codebook']:
        assert file_sha256(cache[name]) == cache[name+'_sha256']
    compact = driver.load_frozen_tokenizer(cache).to(device).eval()
    compact_hash = state_sha256(compact)
    # A private package isolates the historical native OMP implementation from
    # current working-tree edits and the production snapshot's src namespace.
    package = types.ModuleType('church_native_models')
    package.__path__ = [str(native_source)]
    sys.modules[package.__name__] = package
    native_module = load_module('church_native_models.dictionary_learner', native_source/'dictionary_learner.py')
    stage1 = json.loads((base/'stage1/complete.json').read_text())
    config = OmegaConf.load(stage1['config'])
    hparams = OmegaConf.to_container(config.arch.hparams, resolve=True)
    hparams.update(num_embeddings=hparams['n_embed'], embedding_dim=hparams['embed_dim'])
    accepted = inspect.signature(native_module.DictionaryLearning.__init__).parameters
    native_quantizer = native_module.DictionaryLearning(**{k:v for k,v in hparams.items() if k in accepted})
    checkpoint = torch.load(cache['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    native_quantizer.load_state_dict({k.removeprefix('quantizer.'):v for k,v in checkpoint['state_dict'].items()
                                     if k.startswith('quantizer.')}, strict=True)
    native_quantizer.requires_grad_(False).eval().to(device)
    native_hash = state_sha256(native_quantizer)
    del checkpoint

    published = Path('/workspace/tmp/original-rqvae-church-published/extracted/church/stage1')
    published_hash = file_sha256(published/'model.pt')
    assert published_hash == 'ba008ec2e192a6d4084a8fd511927a789c68a8459d6e8bfc22122ae02887b800'
    reference, _ = load_tokenizer(published/'model.pt', published/'config.yaml', device)
    reference_hash = state_sha256(reference)
    lpips_weights = base/'stage1-source/vgg_lpips/vgg.pth'
    assert hashlib.md5(lpips_weights.read_bytes()).hexdigest() == 'd507d7349b931f0638a25a48a722f98a'
    lpips_module.get_ckpt_path = lambda name: str(lpips_weights)
    perceptual = lpips_module.LPIPS().requires_grad_(False).eval().to(device)

    # The released class already defines the validation LMDB path; extend only
    # its category allowlist, retaining its exact decoding and pixel transform.
    LSUNClass.valid_categories = [*LSUNClass.valid_categories, 'church_val']
    transform = create_transforms(load_stage2_config(driver.UPSTREAM).dataset, split='val')
    dataset = LSUNClass('/tmp/laser-sign-data', 'church_val', transform)
    assert len(dataset) == 300
    protocol = json.loads((base/'reference/data-protocol.json').read_text())
    # Pixel probes bind this evaluation to the audited original RQVAE loader.
    def find_validation(value):
        if isinstance(value, dict):
            if 'church_val' in value: return value['church_val']
            for child in value.values():
                result = find_validation(child)
                if result is not None: return result
        return None
    expected = find_validation(protocol)
    assert expected and expected['images'] == 300
    checked_probes = {}
    for index, digest in expected['pixel_probes'].items():
        pixels = dataset[int(index)][0]
        actual = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        assert actual == digest, index
        checked_probes[index] = actual
    loader = DataLoader(Subset(dataset, range(args.images)), batch_size=args.batch_size, num_workers=0)
    methods = ['native_laser', 'continuous_greedy_control', 'compact_laser', 'released_rqvae']
    rows = []
    for name in ['original', *methods]: (out/name).mkdir()
    futures = []
    verification = {}
    status('ready', images=args.images, methods=methods)
    with ThreadPoolExecutor(max_workers=4) as writers, torch.inference_mode():
        offset = 0
        for xs, _ in loader:
            xs = xs.to(device)
            original = xs.mul(.5).add(.5).clamp(0, 1)
            z = compact.encode(xs)
            native_q, _, native_codes = native_quantizer(z.permute(0,3,1,2).contiguous())
            native_q = native_q.permute(0,2,3,1).contiguous()
            greedy_q = continuous_matching_pursuit(z, compact.backbone.dictionary, depth=4)['quantized']
            compact_codes = compact.quantizer.quantize(z)['codes']
            compact_q = compact.quantizer.embed(compact_codes).sum(-2)
            rz = reference.encode(xs)
            rq, _, _ = reference.quantizer(rz)
            latents = {'native_laser':native_q, 'continuous_greedy_control':greedy_q,
                       'compact_laser':compact_q, 'released_rqvae':rq}
            batch_rows = [dict(index=offset+i, lmdb_key=dataset.keys[offset+i].decode(), metrics={})
                          for i in range(len(xs))]
            for name in methods:
                decoder = reference if name == 'released_rqvae' else compact
                decoded = decoder.decode(latents[name]).mul(.5).add(.5).clamp(0,1)
                assert decoded.shape == original.shape and torch.isfinite(decoded).all()
                mse = (decoded-original).double().square().flatten(1).mean(1).cpu().numpy()
                mae = (decoded-original).double().abs().flatten(1).mean(1).cpu().numpy()
                lpips = perceptual(decoded*2-1, original*2-1, reduction='none').flatten().cpu().numpy()
                latent_mse = (latents[name]-(rz if name=='released_rqvae' else z)).double().square().flatten(1).mean(1).cpu().numpy()
                for i in range(len(xs)):
                    batch_rows[i]['metrics'][name] = dict(mse=float(mse[i]), mae=float(mae[i]),
                        psnr_db=float(-10*np.log10(mse[i])), lpips=float(lpips[i]), latent_mse=float(latent_mse[i]))
                cpu = decoded.cpu()
                for i in range(len(xs)):
                    futures.append(writers.submit(save_image, cpu[i], out/name/f'{offset+i:03d}.png'))
                if offset == 0 and name == 'compact_laser':
                    direct = compact.decode_code(compact_codes).mul(.5).add(.5).clamp(0,1)
                    torch.testing.assert_close(decoded,direct,rtol=0,atol=0)
                    verification['compact_decode_matches_production'] = True
            if offset == 0:
                identity = perceptual(xs, xs, reduction='none')
                assert float(identity.abs().max()) < 1e-7
                verification['lpips_identity_max_abs'] = float(identity.abs().max())
                # Compare fresh encoder latents with the actual production
                # validation cache. Validation files use rank-strided ordering.
                shards = [torch.load(base/f'preparation/cache/validation-rank{r}.pt',
                          map_location='cpu', weights_only=False) for r in range(2)]
                verification['validation_cache_types'] = [type(s).__name__ for s in shards]
                if all(isinstance(s, torch.Tensor) for s in shards):
                    cached = torch.stack([shards[i%2][i//2] for i in range(len(xs))]).to(device)
                    torch.testing.assert_close(z,cached,atol=2e-5,rtol=2e-5)
                    verification['fresh_vs_cached_latent_max_abs'] = float((z-cached).abs().max())
            for i, pixels in enumerate(original.cpu()):
                futures.append(writers.submit(save_image, pixels, out/'original'/f'{offset+i:03d}.png'))
            rows.extend(batch_rows)
            offset += len(xs)
            if offset % 20 == 0 or offset == args.images:
                atomic_json(out/'per-image.json', rows)
                status('reconstructing', completed=offset, total=args.images)
        for future in futures: future.result()
    dataset.env.close()
    assert len(rows) == args.images
    assert state_sha256(compact) == compact_hash
    assert state_sha256(native_quantizer) == native_hash
    assert state_sha256(reference) == reference_hash
    summary = {}
    for name in methods:
        summary[name] = {metric:float(np.mean([row['metrics'][name][metric] for row in rows]))
                         for metric in ['mse','mae','psnr_db','lpips','latent_mse']}
        summary[name]['psnr_from_mean_mse_db'] = -10*math.log10(summary[name]['mse'])
    rng = np.random.default_rng(20260917)
    draws = rng.integers(len(rows), size=(10000,len(rows)))
    paired = {}
    for left, right in [('compact_laser','native_laser'),('native_laser','released_rqvae'),
                        ('compact_laser','released_rqvae'),('continuous_greedy_control','native_laser'),
                        ('compact_laser','continuous_greedy_control')]:
        stats = {}
        for metric in ['mse','lpips']:
            difference = np.array([r['metrics'][left][metric]-r['metrics'][right][metric] for r in rows])
            interval = np.quantile(difference[draws].mean(1), [.025,.975]).tolist()
            stats[metric] = dict(mean_difference=float(difference.mean()), bootstrap_95_percent_interval=interval,
                                 left_worse_images=int((difference>0).sum()), total_images=len(rows))
        paired[left+'_minus_'+right] = stats
    report = dict(images=len(rows), split='all 300 official validation images' if len(rows)==300 else 'validation smoke subset',
        summary=summary, paired=paired, verification=verification,
        checkpoints=dict(laser=cache['checkpoint'],laser_sha256=cache['checkpoint_sha256'],
                         compact_codebook=cache['codebook'],compact_codebook_sha256=cache['codebook_sha256'],
                         released_rqvae=str(published/'model.pt'),released_rqvae_sha256=published_hash),
        native_source_hashes={p:file_sha256(base/p) for p in source_files},
        codec_modules_identical_to_stage1=True, original_loader_pixel_probes=checked_probes,
        tokenizers_unchanged=True, precision='FP32; TF32 disabled; evaluation mode; deterministic hard codes',
        perceptual_metric='released RQVAE VGG LPIPS, RGB [-1,1]; metrics measured before PNG saving',
        lpips_linear_weights_sha256=file_sha256(lpips_weights),
        elapsed_seconds=time.time()-started, peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        script_sha256=file_sha256(__file__),
        limitations=['Paired reconstruction metrics, not generation FID.',
                     'Validation contains 300 images; bootstrap intervals are descriptive and assume independent images.',
                     'Native OMP versus compact RQ changes both coefficient precision and support selection.',
                     'Latent MSE has a common scale only among the three LASER paths.'])
    atomic_json(out/'result.json', report)
    atomic_json(out/'per-image.json', rows)
    (out/'audit-script.py').write_text(Path(__file__).read_text())
    status('complete', summary=summary, paired=paired, peak_gpu_allocated_gib=report['peak_gpu_allocated_gib'])


if __name__ == '__main__':
    main()
