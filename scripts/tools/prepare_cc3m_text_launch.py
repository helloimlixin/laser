"""Verify existing physical caches and benchmark the new CC3M text prior."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf

from scripts.tools.build_cc3m_compound_cache import STAGE1_SHA, text_tokenizer, write_json
from src.training.cc3m_text import build_model, configure_performance, objective
from src.training.cc3m_compound import make_aux


def prepare(options, source, base):
    scales = torch.tensor(options['coeff_scales'])
    receipts = []
    for split, expected in [('train', options['train_items']), ('validation', options['validation_items'])]:
        destination = Path(options['token_cache'] if split == 'train' else options['validation_cache'])
        if destination.is_file():
            continue
        parts, offset = [], 0
        atoms = torch.empty((expected, 8, 8, 4), dtype=torch.int16)
        coeffs = torch.empty((expected, 8, 8, 4), dtype=torch.float32)
        text = torch.empty((expected, 32), dtype=torch.int16)
        captions = []
        shards = sorted(source.glob(f'cc3m-{split}-*.pt'))
        assert len(shards) == (576 if split == 'train' else 16)
        for index, path in enumerate(shards):
            # Copy small shards to local storage before loading or mmaping.
            import shutil
            local = base / 'merge-shard.pt'
            shutil.copyfile(path, local)
            payload = torch.load(local, weights_only=True, map_location='cpu')
            meta = payload['meta']
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert meta == receipt
            assert meta['stage1_sha256'] == STAGE1_SHA and not meta['clip_coefficients']
            assert meta['encoder_precision'] == 'fp32'
            count = meta['items']
            assert payload['atoms'].shape == (count, 8, 8, 4)
            assert payload['coeffs'].dtype == torch.float32 and torch.isfinite(payload['coeffs']).all()
            assert payload['text_ids'].shape == (count, 32)
            supports = payload['atoms'].sort(-1).values
            assert (supports.diff(dim=-1) > 0).all()
            assert 0 <= int(supports.min()) <= int(supports.max()) < 16384
            assert len(payload['captions']) == count
            atoms[offset:offset+count] = payload['atoms']
            coeffs[offset:offset+count] = payload['coeffs'] / scales
            text[offset:offset+count] = payload['text_ids']
            captions.extend(payload['captions'])
            offset += count
            receipts.append(dict(shard=path.name, **meta))
            if index % 50 == 0:
                print(json.dumps(dict(phase='merge', split=split, shards=index+1, items=offset)), flush=True)
        assert offset == expected
        metadata = dict(format='laser_cc3m_physical_pair_scalar_v1', stage1_sha256=STAGE1_SHA,
            shape=[8, 8, 4], split=split, items=expected, clip_coefficients=False,
            coefficient_storage='fp32', encoder_precision='fp32',
            coeff_scales=options['coeff_scales'], coeff_vocab_size=2048, coeff_max=3.,
            dataset_revision=options['dataset_revision'], transform='resize256_center_crop256',
            text_tokenizer='bpe16k_huggingface', text_length=32,
            source_cache=str(source), source_shards=[p.name for p in shards])
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(atoms=atoms, coeffs=coeffs, text_ids=text, captions=captions, meta=metadata), destination)
        del atoms, coeffs, text, captions, payload
        gc.collect()
        with destination.open('rb') as reader:
            digest = hashlib.file_digest(reader, 'sha256').hexdigest()
        options['cache_sha256'][split] = digest
        write_json(destination.with_suffix('.json'), dict(sha256=digest, **metadata))
        print(json.dumps(dict(phase='merge_complete', split=split, items=expected, sha256=digest)), flush=True)
    write_json(base / 'assets/cache-verification.json', dict(passed=True, shards=receipts,
        stage1_sha256=STAGE1_SHA, cache_sha256=options['cache_sha256']))


def benchmark(options, base):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(719)
    cache = Path(options['validation_cache'])
    if cache.is_file():
        data = torch.load(cache, weights_only=True, map_location='cpu', mmap=True)
    else:
        import shutil
        source = Path(options['benchmark_physical_shard'])
        cache = base / 'benchmark-validation.pt'
        shutil.copyfile(source, cache)
        data = torch.load(cache, weights_only=True, map_location='cpu')
        data['coeffs'] /= torch.tensor(options['coeff_scales'])
    aux = make_aux(options, options['coeff_scales'], torch.device('cuda', 0))
    model = build_model(options).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], betas=(.9, .95), weight_decay=1e-4, fused=True)
    rows = []

    def trial(batch, compiled):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        atoms = data['atoms'][:batch].long().cuda()
        coeffs = data['coeffs'][:batch].cuda()
        text = data['text_ids'][:batch].long().cuda()
        times = []
        for index in range(4):
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = objective(model, aux, atoms, coeffs, text, options['coeff_target_temperature'])
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if index:
                times.append(time.monotonic()-start)
        row = dict(batch_size=batch, compile_transformer_blocks=compiled,
            mean_seconds=sum(times)/len(times), images_per_second=batch/(sum(times)/len(times)),
            peak_gib=torch.cuda.max_memory_allocated()/2**30, loss=float(loss.detach()), passed=True)
        rows.append(row)
        print(json.dumps(dict(phase='benchmark', **row)), flush=True)

    for batch in [32, 64, 128, 256]:
        try:
            trial(batch, False)
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
            rows.append(dict(batch_size=batch, passed=False, error='CUDA out of memory'))
            break
    candidates = [row for row in rows if row['passed'] and row['peak_gib'] < 69]
    selected = max(candidates, key=lambda row: row['images_per_second'])
    options['compile_transformer_blocks'] = True
    options['compiled_depth_attention'] = True
    configure_performance(model, options)
    for batch in [selected['batch_size'], 256]:
        try:
            trial(batch, True)
            if rows[-1]['peak_gib'] < 72.5 and rows[-1]['images_per_second'] > selected['images_per_second']:
                selected = rows[-1]
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
            rows.append(dict(batch_size=batch, compile_transformer_blocks=True, passed=False, error='CUDA out of memory'))
            break
    options['batch_size'] = selected['batch_size']
    options['accumulation'] = 2048 // (8 * selected['batch_size'])
    options['compile_transformer_blocks'] = selected['compile_transformer_blocks']
    options['compiled_depth_attention'] = selected['compile_transformer_blocks']
    # Round-trip the real full-size model/Adam state and reproduce a CUDA
    # update, including stochastic coefficient targets and model dropout.
    batch = selected['batch_size']
    atoms = data['atoms'][:batch].long().cuda()
    coeffs = data['coeffs'][:batch].cuda()
    text = data['text_ids'][:batch].long().cuda()
    from src.training.cc3m_text import cpu_snapshot, generate
    snapshot = cpu_snapshot(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state()))
    torch.save(snapshot, base / 'preflight-roundtrip.pt')

    def update():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = objective(model, aux, atoms, coeffs, text, options['coeff_target_temperature'])
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        return loss.detach().cpu()

    expected_loss = update()
    expected = cpu_snapshot(model.state_dict())
    restored = torch.load(base / 'preflight-roundtrip.pt', weights_only=True, map_location='cpu', mmap=True)
    model.load_state_dict(restored['model'], strict=True)
    optimizer.load_state_dict(restored['optimizer'])
    torch.set_rng_state(restored['torch_cpu'])
    torch.cuda.set_rng_state(restored['torch_cuda'])
    actual_loss = update()
    torch.testing.assert_close(expected_loss, actual_loss, rtol=0, atol=0)
    actual = cpu_snapshot(model.state_dict())
    assert all(torch.equal(expected[key], actual[key]) for key in expected), 'CUDA continuation differs after resume'
    del snapshot, expected, restored, actual
    (base / 'preflight-roundtrip.pt').unlink()
    gc.collect()
    model.eval()
    pixels = generate(model, aux, text[:4], options)
    assert pixels.shape == (4, 3, 256, 256) and torch.isfinite(pixels).all()
    from torchvision.utils import save_image
    save_image(pixels, base / 'preflight-generation.png', nrow=2)
    from src.rqvae_metrics import OriginalRQVAEInception
    import clip
    from PIL import Image
    metric = OriginalRQVAEInception().cuda().eval()
    with torch.inference_mode():
        features, _ = metric(pixels)
        assert features.shape == (4, 2048) and torch.isfinite(features).all()
    del metric
    clip_model, preprocess = clip.load('ViT-B/32', device='cuda')
    images = (pixels * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    with torch.inference_mode():
        image_features = clip_model.encode_image(torch.stack([preprocess(Image.fromarray(x)) for x in images]).cuda())
        text_features = clip_model.encode_text(clip.tokenize(data['captions'][:4], truncate=True).cuda())
        scores = torch.nn.functional.cosine_similarity(image_features.float(), text_features.float())
        assert torch.isfinite(scores).all()
    write_json(base / 'gpu-preflight.json', dict(passed=True, full_size_adam_roundtrip=True,
        next_update_bitwise_identical=True, conditioned_generation=True, fid_network=True,
        clip_network=True, clip_scores=scores.cpu().tolist()))
    write_json(base / 'throughput.json', dict(gpu='NVIDIA H100 80GB', world_size=8,
        trials=rows, selected=selected, global_batch=2048, accumulation=options['accumulation'],
        reserved_for_ddp_gib=7, benchmark_weights_discarded=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'benchmark'], required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    options = OmegaConf.to_container(config.options, resolve=True)
    torch.set_num_threads(8)
    if args.mode == 'prepare':
        prepare(options, args.source_cache, args.base)
    else:
        benchmark(options, args.base)
    config.options = options
    OmegaConf.save(config, args.config)


if __name__ == '__main__':
    main()
