#!/usr/bin/env python3
"""Calibrate compact ImageNet targets and reuse immutable encoder caches."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0] = [str(ROOT), str(UPSTREAM)]
os.environ.setdefault('TORCH_HOME', '/workspace/tmp/official-rqvae-eval-cache')
import numpy as np
import torch
from torch.utils.data import DataLoader
from rqvae.img_datasets.transforms import create_transforms
from src.compact_rq_training import FrozenCompactTokenizer
from src.imagenet_scaled_stage2 import CachedClassLatents, ManifestImages, load_imagenet_config
from src.original_rq_training import atomic_json, file_sha256, state_sha256
from calibrate_scaled_atom_temperature import measure


def link(source, dest):
    source = source.resolve()
    if dest.exists():
        assert dest.resolve() == source
    else:
        dest.symlink_to(source)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT/'outputs/imagenet-rfid421-rq8-refit-20260913')
    p.add_argument('--study', type=Path, default=ROOT/'outputs/imagenet-compact-rq-study-20260913')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--levels', type=int, choices=[2,4], default=2)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--finalize-cache', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out, source = args.output.resolve(), args.source.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'assets').mkdir(exist_ok=True)
    old = json.loads((source/'cache/complete.json').read_text())
    link(Path(old['checkpoint']), out/'assets/best_rfid_slot1_model.pt')
    for split in ('train','val'):
        link(source/f'{split}-manifest.json', out/f'{split}-manifest.json')
    book = out/'scaled-atom-codebook.pt'
    if not book.exists():
        shutil.copy2(args.study/f'compact{args.levels}-codebook.pt', book)
    assert file_sha256(book)==file_sha256(args.study/f'compact{args.levels}-codebook.pt')
    tokenizer = FrozenCompactTokenizer(out/'assets/best_rfid_slot1_model.pt', book).to(args.device).eval()
    frozen_hash = state_sha256(tokenizer)
    if not args.finalize_cache:
        prior_path = source/'temperature-calibration.json'
        prior = json.loads(prior_path.read_text())
        fit = json.loads((args.study/f'fit{args.levels}.json').read_text())
        assert not set(fit['fit_indices']).intersection(prior['calibration_indices'])
        values = np.load(source/'cache/train-view0-latents.npy', mmap_mode='r')
        ids = prior['calibration_indices']
        latents = torch.from_numpy(values[ids].copy()).to(args.device)
        # Confirm the reused view against fresh encoder output and class labels.
        manifest = json.loads((source/'train-manifest.json').read_text())
        dataset = ManifestImages(Path(old['data'])/'train', manifest,
            create_transforms(load_imagenet_config(UPSTREAM).dataset,split='train'), seed=old['seed'], indices=ids[:4])
        images, labels, indices = next(iter(DataLoader(dataset,batch_size=4)))
        direct = tokenizer.encode(images.to(args.device))
        torch.testing.assert_close(direct,latents[:4],atol=2e-5,rtol=2e-5)
        np.testing.assert_array_equal(labels.numpy(),np.load(source/'cache/train-labels.npy')[ids[:4]])
        limit = prior['original_rq_control']['sampled_to_hard_residual_mse_ratio']+.02
        sweep = []
        for temperature in [.5,.25,.125,.0625,.03125,.015625,.0078125]:
            row = measure(tokenizer.quantizer,latents,temperature,99200)
            sweep.append(row)
            print(json.dumps(dict(phase='temperature_calibration',**row)),flush=True)
            if row['sampled_to_hard_residual_mse_ratio']<=limit:
                break
        assert sweep[-1]['sampled_to_hard_residual_mse_ratio']<=limit
        atomic_json(out/'temperature-calibration.json',dict(selected_temperature=sweep[-1]['temperature'],
            original_rq_control=prior['original_rq_control'],compact_rq_sweep=sweep,allowed_mse_ratio=limit,
            calibration_indices=ids,fit_indices=fit['fit_indices'],level_count=args.levels,
            source_checkpoint_sha256=old['checkpoint_sha256'],codebook_sha256=file_sha256(book),
            source_rfid=prior['source_rfid'],control_checkpoint_sha256=prior['control_checkpoint_sha256'],
            reused_control_report=str(prior_path),reused_control_report_sha256=file_sha256(prior_path),
            selection_rule=prior['selection_rule'],fresh_encoder_cache_parity=True))
        smoke = out/'smoke-cache'
        smoke.mkdir(exist_ok=False)
        for split, selected in [('train',fit['fit_indices'][:32]),('val',list(range(32)))]:
            z = np.load(source/f'cache/{split}-view0-latents.npy',mmap_mode='r')[selected].copy()
            hard = tokenizer.quantizer.quantize(torch.from_numpy(z).to(args.device))['codes']
            np.save(smoke/f'{split}-view0-latents.npy',z)
            np.save(smoke/f'{split}-view0-codes.npy',hard.cpu().numpy().astype(np.uint32))
            np.save(smoke/f'{split}-labels.npy',np.load(source/f'cache/{split}-labels.npy')[selected])
        atomic_json(smoke/'complete.json',dict(train_views=1,train_images=32,val_images=32,
            checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
            codebook=str(book),codebook_sha256=file_sha256(book),frozen_state_sha256=frozen_hash,smoke_only=True))
    else:
        result = json.loads((args.study/'results-50000.json').read_text())
        chosen = result['results'][f'adaptive{args.levels}']
        assert chosen['codebook_sha256']==file_sha256(book)
        quality = json.loads((source/'fidelity-gate.json').read_text())
        assert result['reference_sha256']==quality['reference_statistics_sha256']
        gate = dict(checkpoint_sha256=old['checkpoint_sha256'],codebook_sha256=file_sha256(book),
            images=50000,same_validation_images=True,same_reference_statistics=True,
            original_rfid=quality['original_rfid'],converted_rfid=chosen['rfid'],
            source_reported_rfid=quality['source_reported_rfid'],reference_statistics=result['reference'],
            reference_statistics_sha256=result['reference_sha256'],evaluation=str(args.study.resolve()),
            variant=f'adaptive{args.levels}',previous_shared8_rfid=quality['converted_rfid'])
        atomic_json(out/'fidelity-gate.json',gate)
        cache = out/'cache'
        cache.mkdir(exist_ok=False)
        marker = json.loads((source/'cache/in-progress.json').read_text())
        marker.update(checkpoint=str((out/'assets/best_rfid_slot1_model.pt').resolve()),
            codebook=str(book),codebook_sha256=file_sha256(book),vocab_size=tokenizer.quantizer.vocab_size,
            coefficient_levels=args.levels,token_formula=f'0=zero; 1+atom_id*{args.levels}+coefficient_bin')
        atomic_json(cache/'in-progress.json',marker)
        for split in ('train','val'):
            link(source/f'cache/{split}-labels.npy',cache/f'{split}-labels.npy')
            for view in range(old['train_views'] if split=='train' else 1):
                link(source/f'cache/{split}-view{view}-latents.npy',cache/f'{split}-view{view}-latents.npy')
        codes_path = args.study/f'adaptive{args.levels}-50000-codes.npy'
        link(codes_path,cache/'val-view0-codes.npy')
        atomic_json(cache/'complete.json',dict(**marker,train_images=old['train_images'],val_images=old['val_images'],
            complete=True,frozen_state_sha256=frozen_hash,updated_unix=time.time(),
            hard_training_codes_cached=False,encoder_cache_source=str(source/'cache'),
            encoder_cache_source_manifest_sha256=file_sha256(source/'cache/complete.json'),
            validation_codes_sha256=file_sha256(codes_path)))
        atomic_json(cache/'tokenizer-rfid.json',gate)
        train = CachedClassLatents(cache,include_hard_codes=False)
        val = CachedClassLatents(cache,'val')
        for index in [0,499,12777,49999]:
            z, label, ids = val[index]
            actual = tokenizer.quantizer.quantize(z[None].to(args.device))['codes'][0].cpu()
            assert torch.equal(actual,ids)
        assert len(train)==1281167 and len(val)==50000
        atomic_json(out/'cache-smoke-verification.json',dict(passed=True,full_cache_reused=True,
            training_hard_ids_unused=True,validation_codes_recomputed=True,train_views=2,
            production_arrays_are_links=True,codebook_sha256=file_sha256(book)))
    assert state_sha256(tokenizer)==frozen_hash
    print(json.dumps(dict(phase='complete',finalize_cache=args.finalize_cache,output=str(out))),flush=True)


if __name__ == '__main__':
    main()
