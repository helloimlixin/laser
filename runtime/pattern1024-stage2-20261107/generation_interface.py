"""Verify full-size native generation and exact pattern-lookup decoding."""
import torch
from runtime.pattern_model import unpack, verify_codes
from runtime.pattern_codec import reconstruct_pattern_latents


@torch.no_grad()
def verify_generation(profile,model,aux,device):
    was_training=model.training
    model.eval()
    def forbidden(*args,**kwargs):
        raise AssertionError('Pattern generation invoked the image encoder')
    hook=aux.encoder.register_forward_pre_hook(forbidden)
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(2611071024)
            torch.cuda.manual_seed(2611071024)
            codes=profile.sample(model,aux,2,torch.tensor([933,849],device=device))
            assert codes.shape==(2,8,8,4)
            verify_codes(codes)
            atoms,patterns=unpack(codes,1024)
            latent=reconstruct_pattern_latents(atoms,patterns,aux.coefficient_patterns,aux.dictionary.T)
            expected=aux.decoder(aux.post_quant_conv(latent.permute(0,3,1,2).contiguous())).clamp(-1,1)
            pixels=profile.decode(aux,codes)
            torch.testing.assert_close(pixels,expected,rtol=0,atol=0)
            assert torch.isfinite(pixels).all()
            # Match the native full forward and cache in real FP16 execution.
            labels=torch.tensor([933,849],device=device)
            full=model(codes,model_aux=aux,cond=labels,amp=True)
            model.init_cache()
            max_error=0.
            for depth in range(4):
                cached=model.cached_forward(codes,model_aux=aux,cond=labels,amp=True,sample_loc=(0,0,depth))
                reference=full['atom_logits'][:,0,0,depth]
                torch.testing.assert_close(cached,reference,rtol=3e-2,atol=3e-2)
                finite=torch.isfinite(reference)
                max_error=max(max_error,float((cached[finite]-reference[finite]).abs().max()))
            cached=model.cached_pattern_logits(atoms[:,0,0],aux,amp=True)
            torch.testing.assert_close(cached,full['pattern_logits'][:,0,0],rtol=3e-2,atol=3e-2)
            return dict(passed=True,patterns=1024,atoms=16384,native_cached_sampling=True,
                image_encoder_used=False,direct_pattern_lookup_decode=True,finite_images=True,
                categorical_events_per_site=5,native_depth_steps=4,fp16_cache_parity=True,
                maximum_atom_logit_absolute_error=max_error,generation_quality_measured=False)
    finally:
        hook.remove()
        model.init_cache()
        model.train(was_training)
