#!/usr/bin/env python3
"""Fresh paired LASER/RVQ training with an enforced 6000-bit/s packet limit."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
import lightning as pl
import torch
from torch.nn import functional as F

from archive.scripts.train_mdctcodec_matched import train,sha
from src.audio_hard6k_bitstream import FRAME_BITS
from src.mdctcodec_matched import PairedAudioDataset,tensor_hash
from src.mdctcodec_hard6k import HardRateLASER,HardRateRVQ,HardRateStatistics,reconstruct_hard6k
from src.scaled_atom_rq import fit_signed_levels
from src.training.mdctcodec_continuation import AudioContinuationMedia,GracefulBudget

ROOT=REPO/'outputs/mdctcodec_k4_a4096_hard6k_20260913'
SOURCE=REPO/'outputs/mdctcodec_matched_6kbps_rangefix'


def write(path,data):path.write_text(json.dumps(data,indent=2))


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'prepared.json').exists():
        assert sha(root/'protocol.json')==json.loads((root/'prepared.json').read_text())['protocol_sha256']
        return
    if list(root.glob('*_initial.pt')):raise RuntimeError('Inspect incomplete preparation before replacing initializations')
    source=json.loads((SOURCE/'protocol.json').read_text())
    assert sha(SOURCE/'manifest.json')==source['manifest_sha256']
    shutil.copy2(SOURCE/'manifest.json',root/'manifest.json')
    models={};initializations={}
    for arm,cls in [('laser',HardRateLASER),('rvq',HardRateRVQ)]:
        item=source['initializations'][arm];assert sha(item['path'])==item['sha256']
        original=torch.load(item['path'],map_location='cpu',weights_only=False)
        kwargs=dict(original['hyper_parameters'])
        kwargs.update(hard_rate_cap_bps=6000,log_images_every_n_steps=0,enable_val_latent_visuals=False)
        if arm=='laser':
            kwargs.update(num_embeddings=4096,sparsity_level=4,dictionary_update_max_atoms_per_step=4096,
                          coefficient_quantization_bits=4,coefficient_quantization_max=1.)
        else:kwargs['sparsity_level']=4
        pl.seed_everything(source['seed'],workers=True)
        model=cls(**kwargs)
        if arm=='rvq':model.load_state_dict(original['state_dict'],strict=True)
        for name,digest in source['shared_initialization_sha256'].items():
            state={k.removeprefix(name+'.'):v for k,v in original['state_dict'].items() if k.startswith(name+'.')}
            getattr(model,name).load_state_dict(state,strict=True)
            assert tensor_hash(getattr(model,name).state_dict())==digest
        models[arm]=model
    books=[q.codebook.weight for q in models['rvq'].bottleneck.quantizer.quantizers]
    assert len(books)==4 and all(b.shape==(1024,32) for b in books)
    assert models['laser'].bottleneck.dictionary.shape==(32,4096)
    assert sum(b.numel() for b in books)==models['laser'].bottleneck.dictionary.numel()==131072
    manifest=json.loads((root/'manifest.json').read_text())
    paths=json.loads((SOURCE/'calibration.json').read_text())['paths'][:256]
    assert set(paths)<=set(manifest['train']) and not set(paths)&set(manifest['validation']+manifest['test'])
    dataset=PairedAudioDataset(paths,seed=manifest['seed'])
    model=models['laser'].to('cuda:0').eval();values=[]
    with torch.inference_mode():
        dictionary=F.normalize(model.bottleneck.dictionary.float(),dim=0)
        for start in range(0,len(paths),32):
            x=torch.stack([dataset[(-1,i)][0] for i in range(start,min(start+32,len(paths)))]).to('cuda:0')
            pooled,_,_=model.budget_latents(x)
            signals=pooled.permute(0,2,3,1).reshape(-1,32).T.contiguous()
            _,coefficients=model.bottleneck.batch_omp_with_support(signals,dictionary)
            values.append(coefficients.flatten())
        levels=fit_signed_levels(torch.cat(values),8)
        model.bottleneck.coefficient_levels.copy_(torch.cat((levels[:4],levels.new_zeros(1),levels[4:])))
    model.cpu()
    for arm,model in models.items():
        path=root/f'{arm}_initial.pt'
        torch.save({'state_dict':model.state_dict(),'hyper_parameters':dict(model.hparams)},path)
        initializations[arm]={'path':str(path),'sha256':sha(path),'bottleneck':model.bottleneck_type,
            'parameter_count':sum(p.numel() for p in model.parameters()),'learned_dictionary_scalars':131072}
    write(root/'calibration.json',{'paths':paths,'crop_epoch':-1,'training_only':True,
        'uses_rate_limited_encoder_latents':True,'initial_coefficient_levels':models['laser'].bottleneck.coefficient_levels.tolist(),
        'policy':'Eight signed centers plus zero; training-only Lloyd EMA after each complete batch; no entropy model or evaluation fitting'})
    protocol=dict(source)
    protocol.pop('range_policy',None);protocol.pop('supersedes',None)
    protocol.update(name='mdctcodec-k4-a4096-hard6k',group='mdctcodec-k4-a4096-hard6k-20260913',
        created_utc=datetime.now(timezone.utc).isoformat(),initializations=initializations,
        rate_is_target_only=False,hard_rate_cap_bps=6000,
        packet_rule='For N original 48kHz samples: total packet bytes <= floor(N/64), INCLUDING the 16-byte header/CRC and all coefficients/padding.',
        frame_rule='floor((floor(N/64)-16)*8/bits_per_frame); applied before quantization during training and inference',
        frame_bits=FRAME_BITS,minimum_supported_samples=1536,
        coefficient_max=None,
        laser='4096 shared learned atoms, OMP K4; eight signed coefficient centers plus zero; 57-bit sorted-support combinatorial format',
        rvq='Four independent 1024-entry learned codebooks, depth4; unchanged authors quantizer; 40 bits/frame',
        dictionary_capacity={'laser_vectors':4096,'rvq_total_vectors':4096,'laser_dictionary_scalars':131072,'rvq_dictionary_scalars':131072},
        temporal_controller='Shared adaptive average pooling before quantization, linear interpolation back to the native decoder grid. No additional learned parameters.',
        frame_rate_note='Coded frames approach 6000/57=105.26Hz for LASER and 150Hz for RVQ before packet headers; header cost reduces both. Native backbone grid remains150Hz.',
        scope='Fresh paired codec methods with identical common initialization, data/crops, losses and 200k updates. Both learn through the hard rate controller. '
              'Total dictionary vectors match. Coded frame rates differ because LASER transmits coefficients; auxiliary learning and RVQ projections differ. '
              'This is a rate-constrained system comparison, not a pure quantizer-only ablation or comprehensive SOTA claim.',
        compute_ceiling_gpu_hours_per_arm=12,
        runtime_source_files=['src/audio_hard6k_bitstream.py','src/mdctcodec_hard6k.py','src/mdctcodec_k4.py',
            'scripts/tools/train_mdctcodec_hard6k.py','src/models/dictionary_learner.py',
            'src/training/mdctcodec_continuation.py','tests/test_audio_hard6k.py','docs/mdctcodec-hard6k-2026-09-13.md'])
    write(root/'protocol.json',protocol)
    write(root/'prepared.json',{'protocol_sha256':sha(root/'protocol.json'),'shared_initialization_verified':True,
        'dictionary_capacity':protocol['dictionary_capacity'],'hard_limit_bps':6000,'initializations':initializations})
    print('HARD6K_PREPARED',json.dumps(json.loads((root/'prepared.json').read_text())),flush=True)


def callbacks(output,manifest,arm):
    return [HardRateStatistics(output=output,reference_order=SOURCE/'rvq/data_order.jsonl'),GracefulBudget(output),
        AudioContinuationMedia(output,manifest,arm,reconstruct_fn=reconstruct_hard6k,
            rate_label='hard 6 kbps maximum including header and coefficients')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--arm',choices=['laser','rvq']);p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--mode',choices=['online','disabled'],default='online')
    args=p.parse_args();os.chdir(REPO);args.root=args.root.resolve()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    if args.prepare:prepare(args.root);return
    if not args.arm:p.error('--arm is required')
    args.workers=args.metric_workers=2 if args.smoke else 8
    args.updates=12 if args.smoke else 0;args.train_batches=2 if args.smoke else 0
    args.validation_limit=2 if args.smoke else 0
    args.output=args.output or args.root/(f'preflight_{args.arm}' if args.smoke else args.arm)
    train(args,model_class=HardRateLASER if args.arm=='laser' else HardRateRVQ,extra_callbacks_factory=callbacks)


if __name__=='__main__':main()
