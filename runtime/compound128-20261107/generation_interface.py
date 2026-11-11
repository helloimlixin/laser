"""Check that generation samples compound codes and decodes without running the encoder."""
import torch

from src.training.laser_hard_ar import verify_tokens


@torch.no_grad()
def verify_generation(profile, model, aux, device):
    was_training = model.training
    model.eval()

    def forbidden(*args, **kwargs):
        raise AssertionError('Generation invoked the image encoder')

    handle = aux.encoder.register_forward_pre_hook(forbidden)
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(261107128)
            torch.cuda.manual_seed(261107128)
            codes = profile.sample(model, aux, 2, torch.tensor([933, 849], device=device))
            assert codes.shape == (2, 8, 8, 8)
            verify_tokens(aux, codes)
            pixels = profile.decode(aux, codes)
            assert pixels.shape == (2, 3, 256, 256) and torch.isfinite(pixels).all()
            torch.testing.assert_close(pixels, aux.decode_tokens(codes), rtol=0, atol=0)
            return dict(passed=True, native_cached_autoregressive_sampling=True, coefficient_bins=128,
                atoms=16384, finite_images=True, image_encoder_used=False,
                direct_compound_decoding=True, generation_quality_measured=False)
    finally:
        handle.remove()
        model.train(was_training)
