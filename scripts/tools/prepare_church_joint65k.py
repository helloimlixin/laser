"""Recover a frozen 32k book and expand its levels without fitting any values.

The optional GPU probe compares matched stage-one reconstructions only. It
does not load a transformer, optimize weights, or change the training sources.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path('/tmp/laser-church-joint65k-20260929')
REPO = Path('/workspace/Projects/laser')
TOKENIZER = Path('/tmp/laser-church-combination-20260928/tokenizer.pt')
SOURCE_SHA = '812c10ba93167663cb56c19da0c1673f6aa88feb0bc264cf9d1f2e2f13d4b1cb'
TOKENIZER_SHA = '762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, payload):
    (ROOT / name).write_text(json.dumps(payload, indent=2) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    ROOT.mkdir(exist_ok=True)
    source = ROOT / 'recovery/compact-codebook.pt'
    assert sha(source) == SOURCE_SHA
    assert sha(TOKENIZER) == TOKENIZER_SHA
    old = torch.load(source, map_location='cpu', weights_only=True)
    stage1 = torch.load(TOKENIZER, map_location='cpu', mmap=True, weights_only=False)
    current_normalized = F.normalize(stage1['state_dict']['quantizer.dictionary'].float(), dim=0)
    # Preserve the authenticated old book exactly. Its historical device's
    # normalization differs from current CPU normalization by FP32 rounding.
    dictionary = old['dictionary']
    torch.testing.assert_close(dictionary, current_normalized, rtol=0, atol=5e-8)
    normalization_rounding_max = float((dictionary - current_normalized).abs().max())
    assert old['source_stage1_checkpoint_sha256'] == TOKENIZER_SHA
    levels = old['levels']
    assert tuple(dictionary.shape) == (256, 16384) and tuple(levels.shape) == (16384, 2)
    assert torch.isfinite(levels).all() and (levels[:, 0] < 0).all() and (levels[:, 1] > 0).all()
    expanded = torch.stack((levels[:, 0], levels[:, 0] / 2,
                            levels[:, 1] / 2, levels[:, 1]), dim=-1)
    assert (expanded[:, 1:] > expanded[:, :-1]).all() and not (expanded == 0).any()
    result = dict(old, levels=expanded, vocabulary_size=65537,
        coefficient_levels_per_atom=4, levels_shared_across_depth=True,
        coefficient_units='physical', coefficient_normalization='none',
        depth_specific_scales=False, coefficient_refitting=False,
        expansion='[old_negative, old_negative/2, old_positive/2, old_positive]',
        source_codebook_sha256=SOURCE_SHA, source_codebook=str(source),
        source_artifact='helloimlixin-rutgers/laser/church-laser-integer32769-raw-noclip-scratch300-h200x5-20260922-provenance:v0',
        integer_storage_dtype='uint32', zero_token=0, repeated_atoms_allowed=True)
    destination = ROOT / 'joint65k-codebook.pt'
    if destination.exists():
        saved = torch.load(destination, map_location='cpu', weights_only=True)
        assert torch.equal(saved['levels'], expanded) and torch.equal(saved['dictionary'], dictionary)
    else:
        torch.save(result, destination)
    module = load_module('joint65k_preparation_quantizer', REPO / 'src/adaptive_scaled_atom_rq.py')
    q_old = module.AdaptiveScaledAtomRQ(dictionary, levels, depth=4)
    q_new = module.AdaptiveScaledAtomRQ(dictionary, expanded, depth=4)
    ids = torch.arange(65537)
    expected = torch.cat((dictionary.new_zeros(1, 256),
                          (dictionary.T[:, None] * expanded[..., None]).flatten(0, 1)))
    assert torch.equal(q_new.embed(ids), expected)
    old_ids = torch.arange(32769)
    old_nonzero = (old_ids - 1).clamp_min(0)
    mapped_ids = torch.where(old_ids == 0, 0,
        1 + 4 * (old_nonzero // 2) + 3 * (old_nonzero % 2))
    assert torch.equal(q_old.embed(old_ids), q_new.embed(mapped_ids))
    # 65,536 is a valid class and cannot be represented by a uint16 cache.
    stored = ids.numpy().astype(np.uint32)
    assert np.array_equal(stored.astype(np.int64), ids.numpy()) and stored[-1] == 65536
    receipt = dict(passed=True, codebook=str(destination), codebook_sha256=sha(destination),
        source_codebook_sha256=SOURCE_SHA, tokenizer_sha256=TOKENIZER_SHA,
        dictionary_shape=list(dictionary.shape), levels_shape=list(expanded.shape),
        code_shape=[8, 8, 4], vocabulary=65537, raw_physical_coefficients=True,
        levels_shared_across_depth=True, depth_specific_scales=False,
        coefficients_refitted=False, codebook_fitting_performed=False,
        source_dictionary_bitwise_equal=True,
        current_tokenizer_normalization_max_abs_difference=normalization_rounding_max,
        all_65537_embeddings_exact=True,
        every_old_codeword_preserved_exactly=True, uint32_roundtrip_exact=True,
        repeated_atoms_allowed=True, coefficient_min=float(expanded.min()),
        coefficient_max=float(expanded.max()), expansion=result['expansion'],
        script_sha256=sha(__file__), quantizer_source_sha256=sha(REPO / 'src/adaptive_scaled_atom_rq.py'))
    write('book-preparation.json', receipt)
    print(json.dumps(receipt), flush=True)
    return q_old, q_new


@torch.inference_mode()
def stochastic_probe(quantizer, values, seed, temperature=.125):
    generator = torch.Generator(device=values.device).manual_seed(seed)
    residual = values.reshape(-1, 256).clone()
    entropies, codes = [], []
    penalty = quantizer.norms[:, None] * quantizer.levels.square()
    for _ in range(4):
        depth_entropy, depth_codes = [], []
        for start in range(0, len(residual), 128):
            r = residual[start:start + 128]
            corr = r @ quantizer.dictionary
            scores = (2 * corr[..., None] * quantizer.levels - penalty).flatten(1)
            scores = torch.cat((scores.new_zeros(len(r), 1), scores), dim=1)
            probs = (scores / temperature).softmax(-1)
            picked = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            depth_entropy.append(-(probs * probs.clamp_min(1e-38).log()).sum(-1))
            depth_codes.append(picked)
            r.sub_(quantizer.embed(picked))
        entropies.append(torch.cat(depth_entropy))
        codes.append(torch.cat(depth_codes))
    ids = torch.stack(codes, -1)
    energy = values.reshape(-1, 256).square().sum(-1).mean()
    return dict(temperature=temperature, sites=len(residual),
        target_entropy_by_depth=torch.stack(entropies, -1).mean(0).cpu().tolist(),
        mean_vector_squared_error=float(residual.square().sum(-1).mean()),
        latent_mse=float(residual.square().mean()),
        relative_vector_distortion=float(residual.square().sum(-1).mean() / energy),
        zero_fraction=float((ids == 0).float().mean()), seed=seed)


@torch.inference_mode()
def probe(q_old, q_new):
    started = time.monotonic()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    q_old, q_new = q_old.cuda(), q_new.cuda()
    recovery = load_module('joint65k_probe_recovery',
        Path('/tmp/laser-church-shared-physical-scratch-20260929/recovery_helpers.py'))
    aux = recovery.build_aux(TOKENIZER, [1.] * 4, decoder=True, device='cuda')
    torch.testing.assert_close(aux.dictionary, q_new.dictionary, rtol=0, atol=5e-8)
    provenance = json.loads(Path('/tmp/laser-church-combination-20260928/validation-upload/assets/signals-validation.json').read_text())
    assert provenance['tokenizer_sha256'] == TOKENIZER_SHA
    assert sha('/tmp/laser-church-combination-20260928/probe-signals.pt') == provenance['probe_signals_sha256']
    assert sha('/tmp/laser-church-combination-20260928/probe.pt') == provenance['probe_sha256']
    pixels = torch.load('/tmp/laser-church-combination-20260928/probe.pt',
                        map_location='cpu', mmap=True, weights_only=False)
    signals = torch.load('/tmp/laser-church-combination-20260928/probe-signals.pt',
                         map_location='cpu', mmap=True, weights_only=False)
    z_cpu = signals['val'][:128]
    originals = pixels['datasets']['val']['images'][:128].float() / 255
    assert z_cpu.shape == (128, 8, 8, 256) and z_cpu.dtype == torch.float32
    assert pixels['tokenizer_sha256'] == TOKENIZER_SHA
    fixed_omp = load_module('joint65k_reference_quantizer', REPO / 'src/scaled_atom_rq.py')
    gram = aux.dictionary.T @ aux.dictionary
    totals = {key:dict(latent_mse=0., pixel_mse=0., pixel_mse_to_native=0.)
              for key in ['continuous_omp', 'old32769', 'new65537']}
    previews = None
    with torch.inference_mode():
        for start in range(0, 128, 8):
            z = z_cpu[start:start + 8].cuda()
            baseline = fixed_omp.orthogonal_matching_pursuit(z, aux.dictionary, gram, depth=4)['quantized']
            old_z = q_old.quantize(z)['quantized']
            new_z = q_new.quantize(z)['quantized']
            decoded = {}
            for key, value in [('continuous_omp', baseline), ('old32769', old_z), ('new65537', new_z)]:
                image = aux.decoder(aux.post_quant_conv(value.permute(0, 3, 1, 2).contiguous()))
                decoded[key] = (image.clamp(-1, 1) + 1) / 2
                totals[key]['latent_mse'] += float((value - z).double().square().mean()) * len(z)
                totals[key]['pixel_mse'] += float((decoded[key].cpu() - originals[start:start + 8]).double().square().mean()) * len(z)
            for key in totals:
                totals[key]['pixel_mse_to_native'] += float((decoded[key] - decoded['continuous_omp']).double().square().mean()) * len(z)
            if start == 0:
                previews = [originals[:8]] + [decoded[key].cpu() for key in totals]
    for values in totals.values():
        for key in values:
            values[key] /= 128
    latent_array = np.load('/tmp/laser-church-combination-20260928/signals.npy', mmap_mode='r')
    indices = np.random.default_rng(2026092975).choice(len(latent_array), 16, replace=False)
    training_z = torch.from_numpy(np.array(latent_array[indices], copy=True)).float().cuda()
    teachers = {key:stochastic_probe(q, training_z, 2026092977)
                for key, q in [('old32769', q_old), ('new65537', q_new)]}
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (8 * 256, 4 * 280), 'white')
    pen = ImageDraw.Draw(canvas)
    for row, (name, images) in enumerate(zip(['Original validation pixels', 'Native continuous OMP control',
                                            'Recovered 32769 joint book', 'Expanded 65537 joint book'], previews)):
        pen.text((4, row * 280 + 4), name, fill='black')
        for column, image in enumerate(images):
            array = (image.permute(1, 2, 0) * 255).round().clamp(0, 255).byte().numpy()
            canvas.paste(Image.fromarray(array), (column * 256, row * 280 + 24))
    canvas.save(ROOT / 'joint65k-reconstruction.png')
    receipt = dict(passed=True, images=128, split='first128 fixed official validation probe entries',
        metrics=totals, teacher_probe=teachers, training_probe_indices=indices.tolist(),
        codebook_sha256=sha(ROOT / 'joint65k-codebook.pt'), tokenizer_sha256=TOKENIZER_SHA,
        stage2_loaded=False, optimizer_updates=0, codebook_fitted=False,
        continuous_omp_solve_used_for_reference_only=True, generation_FID_measured=False,
        input_provenance=provenance,
        validation_indices=pixels['datasets']['val']['indices'][:128].tolist(),
        precision='FP32 encoder latents, geometry and decoder; TF32 disabled',
        elapsed_seconds=time.monotonic() - started, script_sha256=sha(__file__))
    write('reconstruction-preflight.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    old, new = prepare()
    if args.probe:
        probe(old, new)
