"""Inspect sampled coefficients from the recovered exact FFHQ FID8.17 model."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.source), str(args.source/'runtime')]
    import torch
    from src import ffhq_v4_archived as archive
    from src.training.imagenet_ffhq_adapter import verify_archive
    from torchvision.utils import save_image

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    verify_archive()
    checkpoint = args.recovery/'best_fid_8.1744_epoch_200.pt'
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert payload['epoch'] == 200 and payload['global_step'] == 109200
    assert abs(payload['fid']-8.174392700195312) < 1e-7
    model = archive.build_model(4096, 2048, compound=True, coeff_vocab_size=2048,
        compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
        architecture='ffhq-350m', num_classes=1)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.cuda().eval()
    del payload
    aux = archive.LaserAux(args.recovery/'best_rfid_slot3_model.pt', 2048, 2048, 3.,
        attn_resolutions=(16,), coeff_scales=[36.208333333333336, 8.583333333333334],
        soft_target_physical=False, clamp_coeffs=False).cuda().eval()
    images, all_atoms, all_ids, reencoded, clipping = [], [], [], [], []
    torch.manual_seed(261001)
    with torch.inference_mode():
        for start in range(0, 64, 8):
            atoms, ids = model.sample_compound(8, aux, cond=torch.zeros(8, dtype=torch.long, device='cuda'),
                atom_temperature=1., atom_top_k=250, atom_top_p=1.,
                coeff_temperature=1., coeff_top_p=.85, amp=True)
            latent = aux.compound_embeddings(atoms, ids).sum(-2)
            raw = aux.decoder(aux.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous()))
            assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
            decoded = raw.clamp(-1, 1)
            assert torch.equal(decoded, aux.decode_compound(atoms, ids))
            clipping.append((raw.abs() > 1).float().mean((1, 2, 3)).cpu())
            # This measures generated-image re-encoding, not the real FFHQ cache.
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            _, coefficients = aux.encode_sparse_components(decoded)
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            images.append(decoded.cpu().add(1).mul(.5))
            all_atoms.append(atoms.cpu()); all_ids.append(ids.cpu()); reencoded.append(coefficients.cpu())
            print(json.dumps(dict(images=start+8, total=64, elapsed_seconds=time.time()-started)), flush=True)
    atoms = torch.cat(all_atoms); ids = torch.cat(all_ids)
    coefficients = aux.coeff_bins.cpu()[ids]; reencoded = torch.cat(reencoded)
    rows = []
    for d in range(2):
        c = coefficients[..., d].flatten(); clean = reencoded[..., d].flatten()
        counts = torch.bincount(atoms[..., d].flatten(), minlength=2048).double()
        probability = counts/counts.sum(); positive = probability[probability > 0]
        entropy = -(positive*positive.log()).sum().item()
        rows.append(dict(depth=d, coefficient_scale=float(aux.coeff_scales[d]),
            sampled_normalized_rms=c.square().mean().sqrt().item(),
            sampled_normalized_absolute_quantiles=dict(zip(['0.5', '0.9', '0.95', '0.99', '0.995', '1.0'],
                torch.quantile(c.abs(), torch.tensor([.5, .9, .95, .99, .995, 1.])).tolist())),
            sampled_negative_fraction=(c < 0).float().mean().item(),
            sampled_end_bin_fraction=((ids[..., d] == 0) | (ids[..., d] == 2047)).float().mean().item(),
            sampled_outside_bin_range_fraction=(c.abs() > 3).float().mean().item(),
            generated_image_reencoded_normalized_rms=clean.square().mean().sqrt().item(),
            generated_image_reencoded_outside_bin_range_fraction=(clean.abs() > 3).float().mean().item(),
            sampled_atom_empirical_entropy_nats=entropy,
            sampled_atom_empirical_effective_vocabulary=float(torch.exp(torch.tensor(entropy))),
            sampled_atom_observed_unique_count=int((counts > 0).sum())))
    report = dict(checkpoint=str(checkpoint), epoch=200, global_step=109200,
        checkpoint_fid=8.174392700195312, prior_archive_sha256=verify_archive(),
        tokenizer=str(args.recovery/'best_rfid_slot3_model.pt'), images=64, seed=261001,
        sampler=dict(atom_temperature=1., atom_top_k=250, atom_top_p=1.,
            coefficient_temperature=1., coefficient_top_p=.85),
        uniform_bins=dict(count=2048, minimum=-3., maximum=3., width=6/2047),
        target_temperature=.5, target_sigma_normalized=.5, target_sigma_bins=.5/(6/2047),
        depths=rows, raw_decoder_clipped_fraction_mean=torch.cat(clipping).mean().item(),
        original_cache_validation_clipped_fraction=.004971986636519432,
        limitations='64 new samples from the exact final checkpoint using archived sampler settings, '
            'not the original saved preview RNG. Generated-image re-encoding is not a measurement '
            'of the real FFHQ training cache. Marginal atom entropy is sample-size biased. '
            'Decoder clipping alone does not establish invalid coefficients.',
        training_modified=False, wandb_metrics_logged=False,
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.time()-started)
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    torch.save(dict(atoms=atoms, coefficient_ids=ids, generated_image_reencoded_coefficients=reencoded),
               args.output/'sampled-tokens.pt')
    save_image(torch.cat(images), args.output/'samples.png', nrow=8)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
