"""Verify native cached sampling and decoding without reconstruction search."""
import torch


@torch.no_grad()
def verify_generation(profile, model, aux, device):
    training = model.training
    model.eval()
    original = aux._reconstruction_search
    def forbidden(*args, **kwargs):
        raise AssertionError('AR generation must not call reconstruction beam search')
    aux._reconstruction_search = forbidden
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(261105473); torch.cuda.manual_seed(261105473)
            labels = torch.tensor([933, 849], device=device)
            codes = profile.sample(model, aux, 2, labels)
            assert codes.shape == (2, 8, 8, 4) and codes.dtype == torch.long
            assert int(codes.min()) >= 0 and int(codes.max()) < 16385
            pixels = profile.decode(aux, codes)
            assert pixels.shape == (2, 3, 256, 256) and torch.isfinite(pixels).all()
            embedded = aux.quantizer.embed(codes).sum(-2)
            torch.testing.assert_close(pixels, aux.decode(embedded), rtol=0, atol=0)
            result = dict(passed=True, native_cached_autoregressive_sampling=True,
                generated_code_shape=list(codes.shape), decoded_image_shape=list(pixels.shape),
                code_min=int(codes.min()), code_max=int(codes.max()), vocabulary=16385,
                finite_images=True, reconstruction_beam_used=False,
                fixed_codeword_decoding=True, sampler=profile.sampling_settings(),
                generation_quality_measured=False)
    finally:
        aux._reconstruction_search = original
        model.train(training)
    return result
