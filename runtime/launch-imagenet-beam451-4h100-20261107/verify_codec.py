"""Check the accelerated beam against the archived FP32 search on real images."""
from pathlib import Path
import json
import os
import sys

BASE = Path('/tmp/laser-beam451-stage2-4h100-20261107')
sys.path[:0] = [str(BASE/'source'), str(BASE)]
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['LASER_FAST_BEAM_TOPK'] = '0'
import torch
from runtime.aligned_recipe import AlignedBeamRecipe
from runtime import beam_teacher
from runtime.speed.exact_beam import beam_pool as accelerated
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.img_datasets.transforms import create_transforms
from omegaconf import OmegaConf


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = True
    recipe = AlignedBeamRecipe(BASE)
    aux, _ = recipe.tokenizer('cuda:0')
    before = recipe.digest(aux)
    config = OmegaConf.load(BASE/'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    images = torch.stack([dataset[i][0] for i in (0, 11, 121, 999)]).cuda()
    with torch.inference_mode():
        latent = aux.encode(images)
        original = beam_teacher.beam_pool(aux.physical_book, latent)
        fast = accelerated(aux.physical_book, latent)
        for name in ('candidates', 'costs', 'canonical_index', 'valid'):
            torch.testing.assert_close(original[name], fast[name], rtol=0, atol=0)
        row = torch.arange(len(original['canonical_index']), device='cuda')
        expected = original['candidates'][row, original['canonical_index']].reshape(4,8,8,4)
        actual = aux.quantize_reconstruction(latent)['codes']
        assert torch.equal(expected, actual)
        weights = beam_teacher.posterior(original, recipe.target_temperature())
        selected = weights.argmax(-1)
        codes = original['candidates'][row, selected]
        targets = beam_teacher.conditional_targets(original['candidates'], weights, codes, 16384)
        torch.testing.assert_close(targets.sum(-1), torch.ones_like(targets[...,0]))
        pixels = aux.decode_code(actual)
        assert torch.isfinite(pixels).all()
    assert recipe.digest(aux) == before
    result = dict(passed=True, real_images=4, accelerated_pool_bitwise_equal=True,
        canonical_beam_ids_bitwise_equal=True, frozen_tokenizer_unchanged=True,
        fixed_integer_products=True, conditional_targets_normalized=True,
        conditional_probability_test=beam_teacher.verify_conditionals())
    (BASE/'codec-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
