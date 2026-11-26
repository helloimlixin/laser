"""Record native ImageNet samples and test coefficient-tail clipping locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--distribution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--reference-preview', type=Path)
    p.add_argument('--memory-limit-gib', type=float, default=16.)
    p.add_argument('--sampling-support-policy', choices=['clean-q995', 'observed-max'])
    args = p.parse_args()
    sys.path[:0] = [str(args.base/'source'), str(args.base/'source/runtime')]
    import torch
    from src.training.rqtransformer import LaserAux, save_class_labeled_grid, class_names_for_dataset
    from src.training.imagenet_ffhq_adapter import build_model
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    from torchvision.utils import save_image

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(
        args.memory_limit_gib*2**30/torch.cuda.get_device_properties(device).total_memory,
        device)
    # Match live sampling, which re-enables matmul TF32 after FP32 encoding.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if args.cache_dir:
        os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(args.cache_dir)
        os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    source = args.checkpoint.resolve(strict=True)
    local = _checkpoint_upload_source(source)
    pin = args.output/'.diagnostic-checkpoint-pin.pt'
    if local != source:
        os.link(local, pin)
        local = pin
    payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    config = payload['config']
    assert config['ffhq_adapted_to_imagenet'] and config['sparsity_level'] == 4
    model = build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
        sparsity_level=4, compound_micro_transformer_layers=2,
        compound_depth_specific_coeff_heads=True, compound_pair_autoregressive=True,
        physical_pair_context=False, model_preset='imagenet-1400m')
    model.load_state_dict(payload['state_dict'], strict=True)
    for block in model.head_transformer.blocks:
        block.attn.short_attention_backend = 'compiled'
    model.to(device).eval()
    step = payload['global_step']
    del payload
    aux = LaserAux(args.base/'inputs/stage1-tokenizer.pt', 16384, 2048, 3.,
                   coeff_scales=config['coeff_scales'], soft_target_physical=False,
                   clamp_coeffs=False, sparsity_level=4).to(device).eval()
    distribution = json.loads(args.distribution.read_text())
    tail = torch.tensor([d['normalized_absolute_quantiles']['0.995']
                         for d in distribution['depths']], device=device)
    clean_rms = torch.tensor([d['normalized_rms'] for d in distribution['depths']],
                            device=device)
    sampling_limits = None
    if args.sampling_support_policy:
        key = '0.995' if args.sampling_support_policy == 'clean-q995' else '1.0'
        sampling_limits = [d['normalized_absolute_quantiles'][key]
                           for d in distribution['depths']]
        model.coefficient_sampling_limits = sampling_limits
    sample_atoms, sample_ids = [], []
    decoder_clipping, limited_decoder_clipping = [], []
    pictures = []
    torch.manual_seed(261001)
    labels = torch.randperm(1000, device=device)[:8].repeat_interleave(8)
    with torch.inference_mode():
        batch_size = config['sample_grid_batch_size']
        for start in range(0, 64, batch_size):
            current = min(batch_size, 64-start)
            atoms, ids = model.sample_compound(current, aux, cond=labels[start:start+current],
                atom_temperature=.9, atom_top_k=16384, atom_top_p=.9,
                coeff_temperature=1., coeff_top_k=2048, coeff_top_p=.85,
                amp=True)
            assert (ids >= 0).all() and (ids < 2048).all()
            c = aux.coeff_bins[ids]
            limited = c.clamp(-tail, tail)
            clean_z = aux.physical_contributions(atoms, c).sum(-2)
            limited_z = aux.physical_contributions(atoms, limited).sum(-2)

            def decode(z):
                raw = aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous()))
                assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
                return raw.clamp(-1, 1).add(1).mul(.5), (raw.abs() > 1).float().mean((1, 2, 3))

            native, clipping = decode(clean_z)
            clipped, limited_clipping = decode(limited_z)
            assert torch.equal(native, aux.decode_compound(atoms, ids).add(1).mul(.5))
            sample_atoms.append(atoms.cpu()); sample_ids.append(ids.cpu())
            decoder_clipping.append(clipping.cpu()); limited_decoder_clipping.append(limited_clipping.cpu())
            # Native and tail-limited copies share exactly the same generated atoms.
            pictures.extend(torch.stack((native[i].cpu(), clipped[i].cpu())) for i in range(current))
            print(json.dumps(dict(label=args.label, completed=start+current, total=64,
                elapsed_seconds=time.time()-started)), flush=True)
    atoms = torch.cat(sample_atoms)
    ids = torch.cat(sample_ids)
    coefficients = aux.coeff_bins.cpu()[ids]
    tail = tail.cpu(); clean_rms = clean_rms.cpu()
    rows = []
    for d in range(4):
        c = coefficients[..., d].flatten()
        rows.append(dict(depth=d, normalized_mean=c.mean().item(),
            normalized_rms=c.square().mean().sqrt().item(), clean_rms=clean_rms[d].item(),
            absolute_quantiles=dict(zip(['0.5', '0.9', '0.95', '0.99', '0.995', '1.0'],
                torch.quantile(c.abs(), torch.tensor([.5, .9, .95, .99, .995, 1.])).tolist())),
            clean_absolute_q995=tail[d].item(), outside_clean_q995_fraction=(c.abs() > tail[d]).float().mean().item(),
            outside_bin_range_fraction=(c.abs() > 3).float().mean().item(),
            end_bin_fraction=((ids[..., d] == 0) | (ids[..., d] == 2047)).float().mean().item(),
            negative_fraction=(c < 0).float().mean().item()))
    clipping = torch.cat(decoder_clipping)
    limited_clipping = torch.cat(limited_decoder_clipping)
    report = dict(label=args.label, checkpoint=str(source), step=step, images=64,
        classes=labels[::8].cpu().tolist(), seed=261001, generation_batch_size=batch_size,
        sampling_precision='Native per-function AMP; no enclosing autocast; FP32 decoding',
        sampler=dict(atom_temperature=.9, atom_top_p=.9, atom_top_k=16384,
            coefficient_temperature=1., coefficient_top_p=.85, coefficient_top_k=2048),
        target_sigma_bins=config['coefficient_noise_sigma_bins'],
        sampling_support_policy=args.sampling_support_policy,
        sampling_coefficient_limits=sampling_limits,
        coefficient_scales=config['coeff_scales'], depths=rows,
        raw_decoder_clipped_fraction_mean=clipping.mean().item(),
        tail_limited_raw_decoder_clipped_fraction_mean=limited_clipping.mean().item(),
        decoder_clipping_by_image=clipping.tolist(),
        tail_limited_decoder_clipping_by_image=limited_clipping.tolist(),
        coefficient_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        limitations='Local 64-image diagnostic; optional sampling support masks logits before token feedback. '
            'The paired tail-limited decode changes only decoded coefficients, '
            'not autoregressive token history. It is not a proposed sampler or FID evaluation. '
            'Beyond q99.5 is a statistical tail, not intrinsically invalid. '
            'Comparisons at different checkpoint ages do not isolate the noise effect.',
        training_modified=False, wandb_metrics_logged=False,
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.time()-started)
    native_images = torch.stack([pair[0] for pair in pictures])
    wnids = sorted(folder.name for folder in (args.base/'imagenet/train').iterdir()
                   if folder.is_dir())
    preview = args.output/'reproduced-native-preview.png'
    save_class_labeled_grid(native_images, labels[::8], class_names_for_dataset('imagenet', wnids),
                            preview, samples_per_class=8)
    if args.reference_preview:
        import numpy as np
        from PIL import Image
        actual = np.asarray(Image.open(preview)).astype(np.int16)
        reference = np.asarray(Image.open(args.reference_preview)).astype(np.int16)
        report['preview_reproduction'] = dict(reference=str(args.reference_preview),
            reference_sha256=hashlib.sha256(args.reference_preview.read_bytes()).hexdigest(),
            reproduced_sha256=hashlib.sha256(preview.read_bytes()).hexdigest(),
            pixels_exact=bool(np.array_equal(actual, reference)),
            pixel_mae_255=float(np.abs(actual-reference).mean()))
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    torch.save(dict(atoms=atoms, coefficient_ids=ids, labels=labels.cpu(),
                    coefficients=coefficients), args.output/'sampled-tokens.pt')
    # Four native/tail-limited pairs per row; eight classes, eight samples each.
    save_image(torch.cat(pictures), args.output/'native-versus-tail-limited.png', nrow=8)
    pin.unlink(missing_ok=True)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
