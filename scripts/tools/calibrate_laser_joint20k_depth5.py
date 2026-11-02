#!/usr/bin/env python3
"""Calibrate only the new residual depth on cached training-image latents."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    torch.set_num_threads(4)
    out = ROOT/'outputs/imagenet-laser-joint20k-depth5-scratch100-4h100-20261017'
    out.mkdir(parents=True, exist_ok=True)
    book = ROOT/'outputs/imagenet-laser-joint20k-refinement-rfid-20261017/joint20k-depth5-codebook.pt'
    latent_path = Path('/tmp/laser-rq-joint-vocabulary-study-20261015/temperature-latents.npy')
    metadata = json.loads((ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017/calibration-data.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(latent_path) == metadata['heldout_latents_sha256']
    assert digest(book) == '922c33037770287facb9ba8f2c477224087155ff14af3b9d6d7390b5a40485a9'
    spec = torch.load(book, map_location='cpu', weights_only=True)
    quantizer = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients'])
    assert quantizer.depth == 5 and quantizer.vocab_size == 20481
    previous = json.loads((ROOT/'outputs/imagenet-laser-joint20k-scratch2-4h100-20261017/prepared.json').read_text())
    temperatures = list(previous['recipe']['targets']['temperature'])
    native = json.loads((ROOT/'outputs/imagenet-laser-joint-vocabulary-improvement-20261016/target-profile-comparison.json').read_text())
    # RQ has four depths. Extend its final-depth relative corruption budget to
    # the fifth level, rather than reusing a temperature in different units.
    budget = native['original_rq_target_profiles'][-1]['expected_extra_error_over_nearest_error']
    x = torch.from_numpy(np.load(latent_path)[:32].copy()).reshape(-1, 256)
    residual = x.clone()
    generator = torch.Generator().manual_seed(26101720485)
    trace = []
    with torch.inference_mode():
        for depth in range(5):
            scores = quantizer.scores(residual, depth)
            maximum = scores.max(-1).values
            error = (residual.square().sum(-1)-maximum).clamp_min(1e-8).sum()
            gap = maximum[:, None]-scores
            del scores, maximum

            def ratio(temperature):
                return float(((-gap/temperature).softmax(-1)*gap).sum()/error)

            if depth == 4:
                low, high = 1e-5, 1.
                assert ratio(low) < budget < ratio(high)
                for _ in range(20):
                    midpoint = (low+high)/2
                    if ratio(midpoint) < budget:
                        low = midpoint
                    else:
                        high = midpoint
                temperatures.append((low+high)/2)
            probability = (-gap/temperatures[depth]).softmax(-1)
            code = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
            residual.sub_(quantizer.embed_depth(code, depth))
            row = dict(depth=depth+1, temperature=temperatures[depth],
                       temperature_fitted=depth == 4,
                       mean_entropy=float(-(probability*probability.clamp_min(1e-30).log()).sum()/len(x)),
                       expected_extra_error_over_nearest_error=ratio(temperatures[depth]))
            trace.append(row)
            print(json.dumps(row), flush=True)
            del gap, probability
        hard = quantizer.quantize(x)['quantized']
        hard_mse = float((x-hard).square().mean())
        soft_mse = float(residual.square().mean())
    assert 0 < trace[-1]['mean_entropy'] < 5
    assert soft_mse/hard_mse < 1.05
    report = dict(passed=True, temperatures=temperatures, trace=trace,
                  calibration_split='train', calibration_images=32,
                  training_indices=metadata['heldout_training_indices'][:32],
                  latents_sha256=digest(latent_path), codebook_sha256=digest(book),
                  first_four_temperatures_unchanged=True,
                  matching='fifth-depth expected excess quantization error / nearest-code error matches original RQ final-depth budget',
                  reference_budget=budget, source_reference_depth=4,
                  sampled_to_hard_latent_mse_ratio=soft_mse/hard_mse,
                  hard_latent_mse=hard_mse, sampled_latent_mse=soft_mse,
                  independent_training_view_validation_required=True, time=time.time())
    (out/'target-temperature-calibration.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
