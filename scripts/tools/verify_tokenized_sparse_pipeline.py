"""Verify the actual stage-1 decoder input against stage-2/cache token decoding."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.training.cli import load_config
from src.training.var_laser import HFSquareImages
from src.models.scratch_var import build_scratch_tokenizer
from src.models.compound_var import compound_decompose
from src.models.sparse_token_codec import token_temperatures, validate_tokenized_checkpoint
from src.data.var_token_cache import restore_cached_codes
from src.original_rq_training import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(args.config)
    vae = build_scratch_tokenizer(cfg.model, cfg.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    validate_tokenized_checkpoint(vae.quantize, checkpoint)
    if checkpoint['initialization']['initialization'] != 'scratch':
        raise ValueError('The new pipeline requires scratch tokenizer initialization')
    vae.load_state_dict(checkpoint['model'], strict=True)
    progress = checkpoint['progress']
    del checkpoint
    vae.cuda().train()
    q = vae.quantize
    data = HFSquareImages(cfg.data.root, 'validation', False, cfg.seed, 256, resize_crop=False)
    images = torch.stack([data[i][0] for i in range(4)]).cuda()
    observed = {}
    hook = q.register_forward_hook(lambda module, inputs, output: observed.update(latent=output[0].detach()))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        original, _, _ = vae(images)
    hook.remove()
    with torch.no_grad():
        decoded, _ = q.from_codes(q.last_atom_ids, q.last_coefficient_ids)
        torch.testing.assert_close(decoded, observed['latent'], rtol=0, atol=0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            roundtrip = vae.fhat_to_img(decoded)
        torch.testing.assert_close(roundtrip, original.clamp(-1, 1), rtol=0, atol=0)
        vae.eval()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            latent = vae.quant_conv(vae.encoder(images))
        atom_t, coefficient_t = token_temperatures(q)
        implicit = compound_decompose(q, latent, stochastic=True,
            generator=torch.Generator(device='cuda').manual_seed(887))
        explicit = compound_decompose(q, latent, stochastic=True, atom_temperatures=atom_t,
            coefficient_temperatures=coefficient_t, generator=torch.Generator(device='cuda').manual_seed(887))
        restored = restore_cached_codes(q, implicit, coefficient_t)
        for name in ('atoms', 'coefficients', 'inputs', 'latent', 'coefficient_probabilities'):
            torch.testing.assert_close(implicit[name], explicit[name], rtol=0, atol=0)
            torch.testing.assert_close(implicit[name], restored[name], rtol=0, atol=0)
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
        progress=progress, images=4, initialization='scratch', tokenized_sparse_policy=q.tokenized_sparse_policy,
        decoder_input_max_error=0., reconstructed_pixel_max_error=0., stage2_cache_max_error=0.,
        decoder_uses_hard_atom_coefficient_ids=True, continuous_coefficients_bypass_decoder=False,
        device=torch.cuda.get_device_name())
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
