#!/usr/bin/env python3
"""Matched original-image reconstruction FID for the selected Church codec."""
import argparse
import codecs
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import ChurchImages, sha256_file
from src.training.rqtransformer import LaserAux, atomic_torch_save
from src.church_support_pattern_training import pattern_targets
from src.complete_sparse_codec import sparse_latents
from src.rqvae_metrics import OriginalRQVAEInception, frechet_distance
from src.support_pattern_integer_codec import decode_support_pattern_integers


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, default=ROOT/'outputs/church-support-pattern-integer-20260911/results.json')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/church-support-pattern-integer-20260911/reconstruction-fid')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('Batch size must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.3)
    started = time.monotonic()

    def log(row):
        row = {'elapsed_seconds':time.monotonic()-started, **row}
        print(json.dumps(row,allow_nan=False),flush=True)
        with (args.output/'history.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')

    calibration = json.loads(args.calibration.read_text())
    book_path = Path(calibration['selected_codebook'])
    book = torch.load(book_path,map_location='cpu',weights_only=True)
    raw = torch.load(calibration['settings']['cache'],map_location='cpu',weights_only=False)
    assert sha256_file(book_path) == calibration['candidates'][str(calibration['selected_vocabulary'])]['codebook_sha256']
    assert sha256_file(Path(calibration['settings']['cache'])) == calibration['cache_sha256']
    stage1 = Path(calibration['settings']['stage1'])
    assert sha256_file(stage1) == raw['meta']['checkpoint_sha256'] == book['checkpoint_sha256']
    aux = LaserAux(stage1,16384,2048,3.,coeff_scales=raw['meta']['coeff_scales'],sparsity_level=4,
        soft_target_physical=True,clamp_coeffs=False,coefficient_patterns=book['coefficient_patterns']).cuda().eval().requires_grad_(False)
    inception = OriginalRQVAEInception().cuda().eval().requires_grad_(False)
    frozen = [(t,t._version) for m in (aux,inception) for t in (*m.parameters(),*m.buffers())]
    cached = raw['validation']
    dataset = ChurchImages('/tmp/laser-sign-data/church/church_outdoor_val_lmdb')
    assert set(dataset.keys) == {key.encode('ascii') for key in cached['keys']}
    dataset.keys = [key.encode('ascii') for key in cached['keys']]
    assert len(dataset) == len(cached['atoms']) == 300
    selected = torch.load(book_path.parent/'validation.pt',map_location='cpu',weights_only=True)
    assert torch.equal(selected['image_indices'],torch.arange(len(dataset)))
    recovered_atoms,recovered_bins = decode_support_pattern_integers(selected['complete_site_integers'],book['pattern_coefficient_ids'])
    assert torch.equal(recovered_atoms.reshape_as(cached['atoms']),cached['atoms'].long())
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    features = {name:[] for name in ('original','continuous_cached','nearest_scalar_cached','joint_pattern_cached','continuous_fp32_encoder')}
    dictionary,bins,scales = aux.dictionary.t(),aux.coeff_bins,aux.coeff_scales

    def decode(z):
        return ((aux.decoder(aux.post_quant_conv(z.permute(0,3,1,2).contiguous())).clamp(-1,1)+1)/2).clamp(0,1)

    seen = 0
    for images in loader:
        count = len(images)
        images = images.cuda(non_blocking=True)
        atoms = cached['atoms'][seen:seen+count].cuda().long()
        physical = cached['coefficients'][seen:seen+count].cuda()
        continuous = (dictionary[atoms]*physical[...,None]).sum(-2)
        scalar_ids = torch.bucketize((physical/scales).contiguous(),(bins[1:]+bins[:-1])/2)
        scalar = sparse_latents(atoms,scalar_ids,dictionary,bins,scales)
        ids = pattern_targets(aux,atoms,physical)
        assert torch.equal(ids.cpu().int(),selected['pattern_ids'][seen:seen+count])
        integer_bins = recovered_bins.reshape_as(cached['atoms'])[seen:seen+count].cuda()
        pattern_z = sparse_latents(atoms,integer_bins,dictionary,bins,scales)
        torch.testing.assert_close(pattern_z,aux.coefficient_pattern_latents(atoms,ids),atol=0,rtol=0)
        native_atoms,native_normalized = aux.encode_sparse_components(images)
        native_z = (dictionary[native_atoms.long()]*(native_normalized*scales)[...,None]).sum(-2)
        values = {'original':((images+1)/2).clamp(0,1), 'continuous_cached':decode(continuous),
            'nearest_scalar_cached':decode(scalar),'joint_pattern_cached':decode(pattern_z),
            'continuous_fp32_encoder':decode(native_z)}
        for name,value in values.items():
            feature,_ = inception(value)
            features[name].append(feature.cpu())
        seen += count
        if seen % 80 == 0 or seen == len(dataset):
            log({'phase':'features','images':seen,'total':len(dataset)})
    features = {name:torch.cat(rows) for name,rows in features.items()}
    atomic_torch_save({'features':features,'image_keys':cached['keys']},args.output/'features.pt')
    stats = {name:(v.double().numpy().mean(0),np.cov(v.double().numpy(),rowvar=False)) for name,v in features.items()}
    result = {'phase':'complete','images':len(dataset),'split':'all 300 official Church validation images',
        'reference':'original images with the same Resize(256)/CenterCrop(256) transform',
        'backend':'original-rqvae TensorFlow-FID-compatible Inception, 2048 features; FP64 mean/covariance',
        'precision':'FP32 decoder and Inception, TF32 disabled; cached source uses BF16 encoder/FP32 OMP; additional FP32-encoder baseline reported separately',
        'stage1_sha256':book['checkpoint_sha256'],'codebook_sha256':sha256_file(book_path),
        'vocabulary':len(book['coefficient_patterns']),'complete_site_bits':book['nominal_bits_per_site'],
        'reconstruction_fid':{},'not_unconditional_generation_fid':True,'neural_weights_updated':False}
    for name in features:
        if name != 'original':
            result['reconstruction_fid'][name] = frechet_distance(*stats['original'],*stats[name])
            log({'phase':'rfid','condition':name,'rfid':result['reconstruction_fid'][name]})
    result['pattern_minus_continuous_cached_rfid'] = result['reconstruction_fid']['joint_pattern_cached']-result['reconstruction_fid']['continuous_cached']
    assert all(t._version==version and t.grad is None for t,version in frozen)
    result['frozen_weights_verified'] = True
    result['elapsed_seconds'] = time.monotonic()-started
    result['source_sha256'] = {}
    source = args.output/'source'
    source.mkdir()
    for name in ('scripts/evaluate_church_support_pattern_rfid.py','scripts/tools/build_sign_probe_cache.py',
        'scripts/train_official_rqtransformer_laser_stage2.py','src/church_support_pattern_training.py',
        'src/complete_sparse_codec.py','src/support_pattern_integer_codec.py','src/rqvae_metrics.py'):
        result['source_sha256'][name] = sha256_file(ROOT/name)
        shutil.copy2(ROOT/name,source/Path(name).name)
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    log({'phase':'complete',**result})


if __name__ == '__main__':
    main()
