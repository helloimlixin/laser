#!/usr/bin/env python3
"""Read-only matched-checkpoint audit of compound atoms and coefficients."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-compound512-native-imagenet-20261016')
OUT = ROOT/'outputs/imagenet-compound512-gap-20261016'
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
sys.path[:0] = [str(BASE/'source'), str(BASE/'runtime'), str(BASE/'upstream'), str(ROOT)]

import torch
from omegaconf import OmegaConf
from src import rq_upstream
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.rq_imagenet_recipes import RQRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training import k4_checkpoint_io
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.utils.utils import top_k_logits, top_p_probs

TEMPERATURES = [.7, .85, .9, 1., 1.15, 1.3]


def checkpoint(case, epoch):
    step = epoch*626
    if case == 'compound512':
        directory = ROOT/'outputs'/CompoundReplayRecipe(BASE).run_id/'persistent-train/checkpoints'
        alias = directory/f'epoch{epoch:03d}-gap-audit.pt'
        source = (alias if alias.exists() else directory/'last.pt').resolve()
        # Saving has already materialized a verified immutable local copy.
        # Avoid mmap over the workspace's remote FUSE filesystem.
        for receipt in (BASE/'checkpoint-cache/objects').glob('*.json'):
            metadata = json.loads(receipt.read_text())
            if Path(metadata['source']) == source:
                path = receipt.with_suffix('.pt')
                assert path.stat().st_size == metadata['identity']['st_size']
                saved = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
                assert saved['global_step'] == step, (saved['global_step'], step)
                if not alias.exists(): alias.symlink_to(os.path.relpath(source, directory))
                return path, saved
        raise RuntimeError('The immutable local checkpoint copy is not ready')
    if case == 'original_rq':
        directory = Path('/tmp/laser-original-rq-imagenet-20261014/checkpoint-cache/objects')
    else:
        directory = Path('/tmp/laser-compound-early-replay-20261016/chunk-cache')
    for path in sorted(directory.glob('*.pt')):
        saved = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
        if saved.get('global_step') == step:
            return path, saved
        del saved
    raise RuntimeError(f'No {case} checkpoint at {step}')


def average(values):
    # All probes use [batch, height, width, depth] metrics.
    return values.double().mean((0, 1, 2)).cpu()


def ce(logits, probabilities):
    lp = logits.float().log_softmax(-1)
    return -torch.where(probabilities > 0, probabilities*lp, 0.).sum(-1)


@torch.inference_mode()
def audit(case, epoch, count, device, cache_only=False):
    path, saved = checkpoint(case, epoch)
    assert saved['global_step'] == epoch*626
    is_rq = case == 'original_rq'
    base = Path('/tmp/laser-original-rq-imagenet-20261014') if is_rq else BASE
    profile = RQRecipe(base) if is_rq else CompoundReplayRecipe(base)
    if case == 'compound2048':
        profile.recipe.tokenizer.coefficient_bins = 2048
        profile.recipe.targets.sigma_bins = .125/(6/2047)
        assert len(saved['state_dict']) == 837
        assert saved['state_dict']['coeff_classifier.0.1.weight'].shape[0] == 2048
    config = load_config(base/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch); arch.vocab_size = config.dataset.vocab_size
    with torch.device('meta'):
        model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    model.eval().requires_grad_(False).to(device)
    del saved
    aux, _ = profile.tokenizer(device)
    aux.requires_grad_(False).eval()
    reference = json.loads((ROOT/'outputs/imagenet-laser-joint-predictability-20261016/original_rq-final.json').read_text())
    if cache_only:
        assert not is_rq
        dataset = TarImageNet(base/'data', 'val', create_transforms(config.dataset, split='val'))
        pairs = [dataset[index] for index in reference['splits']['heldout_val']['indices'][:2]]
        images = torch.stack([pair[0] for pair in pairs]).to(device)
        labels = torch.tensor([pair[1] for pair in pairs], device=device)
        _, codes = profile.targets(aux, images)
        assert torch.equal(model.scalar_tokens(model.pack(codes)), codes)
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            expected = model(codes, aux, labels, amp=True)
        prefix = torch.zeros_like(codes); prefix[..., 1::2] = aux.num_atoms+aux.coeff_vocab_size//2
        model.init_cache()
        maximum = {'atom':0., 'coefficient':0.}
        for h in range(8):
            for w in range(8):
                for event in range(8):
                    with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                        actual = model.cached_forward(prefix[:, :h+1], aux, labels, amp=True, sample_loc=(h,w,event))
                    key = 'atom_logits' if event%2 == 0 else 'coeff_logits'
                    kind = 'atom' if event%2 == 0 else 'coefficient'
                    actual = actual[:, :aux.num_atoms] if event%2 == 0 else actual[:, aux.num_atoms:]
                    p = expected[key][:,h,w,event//2].float().log_softmax(-1)
                    q = actual.float().log_softmax(-1)
                    kl = torch.where(p.isfinite(), p.exp()*(p-q), 0.).sum(-1)
                    assert torch.isfinite(kl).all()
                    maximum[kind] = max(maximum[kind], float(kl.max()))
                    prefix[:,h,w,event] = codes[:,h,w,event]
        assert max(maximum.values()) < 2.e-4
        proof = dict(passed=True, case=case, epoch=epoch, global_step=epoch*626,
            checkpoint=str(path), strict_reload=True, scalar_cache_events=512,
            maximum_fp16_full_vs_cache_kl=maximum, exact_integer_pack_roundtrip=True,
            training_updates_added=0, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
        write_json(OUT/f'{case}-epoch{epoch:03d}-cache.json', proof)
        print(json.dumps(proof), flush=True)
        return
    report = dict(case=case, epoch=epoch, global_step=epoch*626, checkpoint=str(path),
        parameter_count=parameter_count, splits={}, precision='FP16, uncached autocast',
        source_stage1_unchanged=True, training_updates_added=0,
        coefficients=2048 if case == 'compound2048' else 512 if not is_rq else None)
    hidden = []
    hook = None if is_rq else model.classifier.register_forward_pre_hook(lambda module, args: hidden.append(args[0]))
    for split, label in [('train', 'calibration_train'), ('val', 'heldout_val')]:
        indices = reference['splits'][label]['indices'][:count]
        dataset = TarImageNet(base/'data', split, create_transforms(config.dataset, split='val'))
        sums = {}
        image_rows = []
        def add(name, value):
            assert torch.isfinite(value).all(), name
            reduced = value.double().sum((0, 1, 2)).cpu()
            sums[name] = sums.get(name, torch.zeros_like(reduced))+reduced
        for first in range(0, count, 2):
            pairs = [dataset[index] for index in indices[first:first+2]]
            images = torch.stack([pair[0] for pair in pairs]).to(device)
            labels = torch.tensor([pair[1] for pair in pairs], device=device)
            torch.manual_seed(261016001+first)
            targets, codes = profile.targets(aux, images)
            hidden.clear()
            with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                output = model(codes, aux, labels, amp=True)
            states = None if is_rq else hidden.pop()
            probabilities = targets if is_rq else torch.nn.functional.one_hot(targets['atoms'], 16384).float()
            logits = output if is_rq else output['atom_logits']
            atom_ce = ce(logits, probabilities)
            atom_entropy = -(probabilities*probabilities.clamp_min(1.e-30).log()).sum(-1)
            add('atom_or_rq_ce', atom_ce)
            add('atom_or_rq_target_entropy', atom_entropy)
            logq = logits.float().log_softmax(-1)
            add('atom_or_rq_predicted_entropy', -torch.where(logq.isfinite(), logq.exp()*logq, 0.).sum(-1))
            target_ids = codes if is_rq else targets['atoms']
            add('atom_or_rq_top1', (logits.argmax(-1) == target_ids).float())
            order = logits.topk(1024, -1).indices
            for k in [10, 256, 512, 1024]:
                add(f'atom_or_rq_target_mass_top{k}', probabilities.gather(-1, order[..., :k]).sum(-1))
            for temperature in TEMPERATURES:
                add(f'atom_or_rq_ce_T{temperature}', ce(logits/temperature, probabilities))
            sampling_temperature = 1. if is_rq else .9
            filtered = logits.float().reshape(-1, logits.shape[-1])/sampling_temperature
            if is_rq: filtered = top_k_logits(filtered, 256)
            filtered = top_p_probs(filtered.softmax(-1), .95 if is_rq else .9).reshape_as(logits)
            add('target_mass_retained_by_active_sampler', (probabilities*(filtered > 0)).sum(-1))
            if not is_rq:
                p = targets['coefficient_probabilities']
                c = output['coeff_logits'].float()
                coeff_ce = ce(c, p)
                entropy = -(p*p.clamp_min(1.e-30).log()).sum(-1)
                add('coefficient_ce', coeff_ce); add('coefficient_target_entropy', entropy)
                add('coefficient_excess_ce', coeff_ce-entropy)
                add('pair_excess_ce', atom_ce+coeff_ce-entropy)
                cq = c.softmax(-1)
                correct_sign = (aux.coeff_bins > 0) == (targets['coefficients'][..., None] > 0)
                add('coefficient_correct_sign_probability', (cq*correct_sign).sum(-1))
                add('coefficient_binary_sign_accuracy', (((cq*(aux.coeff_bins > 0)).sum(-1) > .5)
                    == (targets['coefficients'] > 0)).float())
                mean = (cq*aux.coeff_bins).sum(-1)
                add('coefficient_normalized_mean_mse', (mean-targets['coefficients']).square())
                add('coefficient_normalized_sample_mse', (cq*(aux.coeff_bins-targets['coefficients'][..., None]).square()).sum(-1))
                for temperature in TEMPERATURES:
                    add(f'coefficient_ce_T{temperature}', ce(c/temperature, p))
                vectors = aux.dictionary.T[targets['atoms']]
                with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                    refined = model.coefficient_hidden(states, torch.zeros_like(vectors))
                    unconditional = torch.stack([head(refined[..., d, :]) for d, head in enumerate(model.coeff_classifier)], -2)
                add('coefficient_ce_without_current_atom', ce(unconditional, p))
                hard = codes.clone()
                from src.training.laser_hard_ar import nearest_coefficient_ids
                hard[..., 1::2] = nearest_coefficient_ids(targets['coefficients'], aux.coeff_bins)+aux.num_atoms
                hidden.clear()
                with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                    clean = model(hard, aux, labels, amp=True)
                hidden.clear()
                add('clean_prefix_atom_ce', ce(clean['atom_logits'], probabilities))
                add('clean_prefix_coefficient_ce', ce(clean['coeff_logits'], p))
                add('clean_prefix_correct_sign_probability', (clean['coeff_logits'].float().softmax(-1)*correct_sign).sum(-1))
                physical = aux.coeff_bins[codes[..., 1::2]-aux.num_atoms]*aux.coeff_scales
                clean_physical = aux.coeff_bins[hard[..., 1::2]-aux.num_atoms]*aux.coeff_scales
                add('prefix_physical_coefficient_noise_mse', (physical-clean_physical).square())
                del c, cq, clean, unconditional, refined, states, vectors, correct_sign, physical, clean_physical
            hidden.clear()
            with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                wrong = model(codes, aux, (labels+499)%1000, amp=True)
            hidden.clear()
            add('correct_class_ce_gain', ce(wrong if is_rq else wrong['atom_logits'], probabilities)-atom_ce)
            image_rows.extend(atom_ce.mean((1, 2)).cpu().tolist())
            del images, output, logits, targets, codes, probabilities, atom_ce, atom_entropy, logq, order, filtered, wrong
            if first % 16 == 0:
                write_json(OUT/f'progress-{case}.json', dict(epoch=epoch, split=label, images=first+len(pairs), time=time.time()))
        report['splits'][label] = dict(images=count, indices=indices,
            per_depth={name:(value/(count*64)).tolist() for name,value in sums.items()},
            per_image_atom_ce=image_rows)
        write_json(OUT/f'{case}-epoch{epoch:03d}.json', report)
        print(json.dumps(dict(case=case, epoch=epoch, split=label,
            metrics=report['splits'][label]['per_depth'])), flush=True)
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device)/1024**3
    report['limitations'] = ['Teacher-forced diagnostics are not generated-image FID.',
        'Different tokenizers define different prediction tasks; cross-tokenizer CE is descriptive.',
        'Archived 2048 model is strictly reloaded into the equivalent maintained native adapter; arithmetic is FP16 instead of historical BF16.',
        'Training and validation center crops match the earlier predictability audit.']
    write_json(OUT/f'{case}-epoch{epoch:03d}.json', report)
    if hook: hook.remove()
    del model, aux
    gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['compound512', 'compound2048', 'original_rq'], required=True)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--images', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(10.5*1024**3/total, device)
    def uncached_autocast(enabled=True):
        return torch.autocast('cuda', enabled=enabled, dtype=torch.float16, cache_enabled=False)
    for method in [rq_upstream.RQTransformer.forward, rq_upstream.RQTransformer.cached_forward]:
        method.__globals__['autocast'] = uncached_autocast
    audit(args.case, args.epoch, args.images, device, args.cache_only)


if __name__ == '__main__':
    main()
