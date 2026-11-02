#!/usr/bin/env python3
"""Measure sign conditioning at the immutable FID16.2197 compound checkpoint."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-compound1622-sign-20261015')
OUT = ROOT/'outputs/imagenet-compound1622-sign-diagnosis-20261015'
FINE = Path('/tmp/laser-rq-fine-joint512-imagenet-20261015')
REFERENCE = ROOT/'outputs/imagenet-laser-fine-joint512-scratch6-4h100-20261015/diagnosis-epoch001/teacher-forcing.json'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def initialize(args):
    source = BASE/'archived/source'
    sys.path[:0] = [str(source), str(source/'runtime'),
                   str(source/'third_party/rq-vae-transformer')]
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*2**30/total, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf
    historical = ROOT/'outputs/imagenet-rfid421-ffhq-q995-clipped-sigma42p65-scratch-4h200-20261009/resume-best-fid-refine-4h200-20261012'
    config = OmegaConf.to_container(OmegaConf.load(historical/'resume.yaml'))['options']
    return torch, device, config


def prepare(args):
    torch, device, config = initialize(args)
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from src.training.rqtransformer import LaserAux
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config
    # Load only the JPEG reader from the maintained runtime; all model and
    # tokenizer imports above come from the checkpoint's archived source.
    spec = importlib.util.spec_from_file_location('diagnostic_jpeg_reader', ROOT/'src/training/rq_imagenet_runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    reference = json.loads(REFERENCE.read_text())
    indices = reference['validation_indices']
    assert len(indices) == len(set(indices)) == 128
    stage1 = FINE/'inputs/stage1-model.pt'
    stage1_sha = digest(stage1)
    assert stage1_sha == 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
    aux = LaserAux(stage1, 16384, 2048, 3., coeff_scales=config['coeff_scales'],
                   soft_target_physical=False, clamp_coeffs=True, sparsity_level=4).to(device).eval()
    transform = create_transforms(load_config(FINE/'inputs/stage2-training-config.yaml').dataset, split='val')
    dataset = runtime.TarImageNet(FINE/'data', 'val', transform)
    loader = DataLoader(Subset(dataset, indices), batch_size=4, num_workers=0)
    parts = dict(atoms=[], coefficients=[], latent=[], labels=[], hard_ids=[])
    image_hash = hashlib.sha256()
    started = time.monotonic()
    with torch.inference_mode():
        gram = aux.dictionary.T@aux.dictionary
        for images, labels in loader:
            image_hash.update(images.numpy().tobytes())
            images = images.to(device)
            atoms, coefficients = aux.encode_sparse_components(images, dictionary_gram=gram)
            latent = aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).float()
            hard_ids = (coefficients[..., None]-aux.coeff_bins).abs().argmin(-1)
            assert (coefficients.abs() <= 3).all()
            assert torch.isfinite(latent).all()
            for key, value in dict(atoms=atoms, coefficients=coefficients,
                                   latent=latent, labels=labels, hard_ids=hard_ids).items():
                parts[key].append(value.cpu())
            write_json(OUT/'progress.json', dict(phase='prepare', images=sum(len(x) for x in parts['atoms']), time=time.time()))
    cache = {key:torch.cat(values) for key, values in parts.items()}
    # Shuffle current atom identity within each residual depth. Prefixes and
    # transformer context stay fixed during this coefficient-head intervention.
    permutation = np.random.default_rng(261015).permutation(128*64)
    cache['shuffled_atoms'] = cache['atoms'].reshape(-1, 4)[permutation].reshape_as(cache['atoms'])
    metadata = dict(images=128, validation_indices=indices,
        same_validation_indices_as_failed_head=True, image_tensor_sha256=image_hash.hexdigest(),
        stage1_sha256=stage1_sha, coefficient_scales=config['coeff_scales'],
        tokenizer='historical deterministic OMP support and final dictionary-coordinate least-squares coefficients',
        coefficient_clipping=True, coefficient_bins=2048, coefficient_range=[-3., 3.],
        encoder_precision='float32; TF32 disabled', historical_source=str(BASE/'archived/source'),
        source_archive_sha256=digest(ROOT/'outputs/imagenet-rfid421-ffhq-q995-clipped-sigma42p65-scratch-4h200-20261009/resume-best-fid-refine-4h200-20261012/source-code.tar.gz'),
        elapsed_seconds=time.monotonic()-started)
    cache['metadata'] = metadata
    torch.save(cache, BASE/'held-out-probe.pt')
    write_json(OUT/'probe-verification.json', metadata)
    print(json.dumps(metadata), flush=True)


def evaluate(args):
    torch, device, config = initialize(args)
    from src.training.rqtransformer import LaserAux
    from src.training.imagenet_ffhq_adapter import build_model, verify_archive
    recovery = json.loads((BASE/'checkpoint-recovery.json').read_text())
    assert recovery['sha256'] == 'a3392c62eefe01ea378d09a1f4a671574d73365e1e589a1d59344f8e92e4ca61'
    assert recovery['verified_all_chunk_and_full_sha256']
    checkpoint = Path(recovery['path'])
    assert checkpoint.stat().st_size == recovery['bytes']
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert payload['global_step'] == 61348
    saved_config = payload['config']
    for key in ['coeff_vocab_size', 'num_atoms', 'model_preset']:
        assert saved_config[key] == config[key], key
    # Runtime saves the actual FP32 tokenizer scales; YAML retains their
    # calibration precision. Require exact equality at the tokenizer dtype.
    assert torch.equal(torch.tensor(saved_config['coeff_scales'], dtype=torch.float32),
                       torch.tensor(config['coeff_scales'], dtype=torch.float32))
    config['coeff_scales'] = saved_config['coeff_scales']
    assert saved_config['ffhq_adapted_to_imagenet']
    with torch.device('meta'):
        model = build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
            sparsity_level=4, compound_micro_transformer_layers=2,
            compound_depth_specific_coeff_heads=True, compound_pair_autoregressive=True,
            physical_pair_context=False, model_preset='imagenet-1400m')
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    proof = dict(passed=True, checkpoint=str(checkpoint), source=recovery['source'],
        checkpoint_sha256=recovery['sha256'], all_chunks_and_full_checkpoint_verified=True,
        checkpoint_bytes=recovery['bytes'], global_step=payload['global_step'],
        saved_epoch=payload['epoch'], fid_evaluation_epoch=98, recorded_fid=16.219738094672664,
        strict_model_reload=True, model_state_tensors=len(payload['state_dict']),
        parameters=sum(p.numel() for p in model.parameters()),
        archived_decoder_sha256=verify_archive(), historical_source=True,
        training_updates_added=0)
    write_json(OUT/'checkpoint-verification.json', proof)
    model.requires_grad_(False).eval().to(device)
    del payload
    aux = LaserAux(FINE/'inputs/stage1-model.pt', 16384, 2048, 3.,
        coeff_scales=config['coeff_scales'], soft_target_physical=False,
        clamp_coeffs=True, sparsity_level=4).to(device).eval()
    cache = torch.load(BASE/'held-out-probe.pt', map_location='cpu', weights_only=False)
    hidden_cache = {}
    def capture_hidden(module, inputs, output):
        hidden_cache['hidden'] = output
    hook = model.head_transformer.register_forward_hook(capture_hidden)
    results = {}
    started = time.monotonic()
    with torch.inference_mode():
        for mode in ['clean_nearest_prefix', 'training_noise_prefix_sigma010']:
            sums, latent_sums, per_image = {}, {}, []
            generator = torch.Generator(device=device).manual_seed(261015)
            def accumulate(key, values):
                sums[key] = sums.get(key, torch.zeros(4, dtype=torch.float64)) + values.reshape(-1, 4).double().sum(0).cpu()
            for index in range(128):
                atoms = cache['atoms'][index:index+1].to(device)
                coefficients = cache['coefficients'][index:index+1].to(device)
                ids = cache['hard_ids'][index:index+1].to(device)
                latent = cache['latent'][index:index+1].to(device)
                labels = cache['labels'][index:index+1].to(device)
                targets = (-(coefficients[..., None]-aux.coeff_bins).square()/(2*.1**2)).softmax(-1)
                if mode != 'clean_nearest_prefix':
                    ids = torch.multinomial(targets.reshape(-1, 2048), 1, generator=generator).reshape_as(ids)
                packed = atoms*2048+ids
                output = model(packed, aux, labels, amp=True)
                logits = output['coeff_logits'].float()
                assert torch.isfinite(logits).all()
                logp = logits.log_softmax(-1)
                p = logp.exp()
                hidden = hidden_cache.pop('hidden').reshape(1, 8, 8, 4, -1)
                shuffled = cache['shuffled_atoms'][index:index+1].to(device)
                with torch.autocast('cuda', dtype=torch.float16):
                    shuffled_logits = model.coefficient_logits(hidden, aux.dictionary.T[shuffled])
                shuffled_logp = shuffled_logits.float().log_softmax(-1)
                sign_match = (aux.coeff_bins > 0) == (coefficients[..., None] > 0)
                correct_sign = (p*sign_match).sum(-1)
                per_image.append(correct_sign.mean((0, 1, 2)).cpu())
                accumulate('correct_sign_probability', correct_sign)
                accumulate('binary_sign_accuracy', correct_sign > .5)
                accumulate('correct_sign_nll', -correct_sign.clamp_min(1e-30).log())
                accumulate('coefficient_positive_fraction', coefficients > 0)
                accumulate('coefficient_entropy', -(p*logp).sum(-1))
                accumulate('coefficient_soft_ce_sigma010', -(targets*logp).sum(-1))
                accumulate('coefficient_target_entropy_sigma010', -(targets*targets.clamp_min(1e-30).log()).sum(-1))
                accumulate('shuffled_atom_correct_sign_probability', (shuffled_logp.exp()*sign_match).sum(-1))
                accumulate('correct_vs_shuffled_atom_kl', (p*(logp-shuffled_logp)).sum(-1))
                accumulate('shuffled_atom_changed_fraction', shuffled != atoms)
                atom_logp = output['atom_logits'].float().log_softmax(-1)
                accumulate('atom_ce', -atom_logp.gather(-1, atoms[..., None]).squeeze(-1))
                physical = aux.coeff_bins[None, :]*aux.coeff_scales[:, None]
                oracle_sign_p = (p+p.flip(-1))*sign_match
                torch.testing.assert_close(oracle_sign_p.sum(-1), torch.ones_like(correct_sign), atol=2e-5, rtol=2e-5)
                torch.testing.assert_close((oracle_sign_p*physical.abs()).sum(-1), (p*physical.abs()).sum(-1), atol=2e-5, rtol=2e-5)
                vectors = aux.dictionary.T[atoms]
                hard = aux.coeff_bins[cache['hard_ids'][index:index+1].to(device)]*aux.coeff_scales
                hard_latent = (vectors*hard[..., None]).sum(-2)
                latent_sums['hard_source'] = latent_sums.get('hard_source', 0.) + float((latent-hard_latent).square().sum())
                for name, probabilities in [('predicted_coefficients', p), ('oracle_sign', oracle_sign_p)]:
                    mean = (probabilities*physical).sum(-1)
                    variance = (probabilities*(physical-mean[..., None]).square()).sum(-1)
                    predicted_latent = (vectors*mean[..., None]).sum(-2)
                    error = (latent-predicted_latent).square().sum()+variance.sum()
                    latent_sums[name] = latent_sums.get(name, 0.) + float(error)
                if (index+1) % 8 == 0:
                    progress = dict(phase='evaluate', mode=mode, images=index+1,
                        correct_sign_probability=(sums['correct_sign_probability']/((index+1)*64)).tolist(), time=time.time())
                    write_json(OUT/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                del output, logits, p, logp, shuffled_logits, shuffled_logp, hidden
            report = {k:(v/(128*64)).tolist() for k,v in sums.items()}
            report['expected_latent_mse'] = {k:v/(128*64*256) for k,v in latent_sums.items()}
            report['oracle_sign_fraction_error_removed'] = 1-latent_sums['oracle_sign']/latent_sums['predicted_coefficients']
            image_scores = torch.stack(per_image)
            report['correct_sign_probability_image_standard_error'] = (image_scores.double().std(0)/128**.5).tolist()
            report['per_image_correct_sign_probability'] = image_scores.tolist()
            results[mode] = report
            write_json(OUT/'partial-results.json', results)
    hook.remove()
    comparison = json.loads((REFERENCE.parent/'earlier-head-sign-comparison.json').read_text())
    result = dict(passed=True, checkpoint=proof, probe=cache['metadata'], results=results,
        earlier_epoch_one_comparison=comparison,
        limitations=['Teacher forcing with ground-truth atoms and prior pairs; not free-generation sign accuracy.',
            'Historical compound is epoch98; comparison checkpoints are epoch1.',
            'Historical compound uses final dictionary-coordinate coefficients; fine512 uses orthogonal coordinates.',
            'Expected latent MSE samples coefficient distributions independently at fixed teacher-forced prefixes.',
            'No FID recomputed; 16.2197 is the recorded 50k/full-training-set evaluation.'],
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        elapsed_seconds=time.monotonic()-started, diagnostic_only=True, training_updates_added=0)
    write_json(OUT/'sign-diagnostic.json', result)
    print(json.dumps(dict(passed=True, results={k:{m:v for m,v in r.items() if m != 'per_image_correct_sign_probability'} for k,r in results.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'evaluate'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--memory-limit-gib', type=float, default=11.)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    else:
        evaluate(args)
