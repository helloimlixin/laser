"""Native generation must decode directly and must not invoke an image encoder."""
import torch


@torch.no_grad()
def verify_generation(profile, model, aux, device):
    training=model.training;model.eval()
    original=aux.quantize_reconstruction
    def forbidden(*args,**kwargs):raise AssertionError('Generation used reconstruction encoding')
    aux.quantize_reconstruction=forbidden
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(261105451);torch.cuda.manual_seed(261105451)
            codes=profile.sample(model,aux,2,torch.tensor([933,849],device=device))
            assert codes.shape==(2,8,8,4) and codes.dtype==torch.long
            assert int(codes.min())>=0 and int(codes.max())<16384
            pixels=profile.decode(aux,codes)
            assert pixels.shape==(2,3,256,256) and torch.isfinite(pixels).all()
            torch.testing.assert_close(pixels,aux.decode(aux.quantizer.embed_code(codes)),rtol=0,atol=0)
            return dict(passed=True,native_cached_autoregressive_sampling=True,
                vocabulary=16384,finite_images=True,reconstruction_beam_used=False,
                fixed_codeword_decoding=True,generation_quality_measured=False)
    finally:
        aux.quantize_reconstruction=original;model.train(training)
