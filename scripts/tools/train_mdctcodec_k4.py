#!/usr/bin/env python3
"""Train K4/A4096 LASER with the same total dictionary vectors as 4x1024 RVQ."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import lightning as pl
import torch
from torch.nn import functional as F

from archive.scripts.train_mdctcodec_matched import train, sha
from src.mdctcodec_matched import tensor_hash, PairedAudioDataset
from src.mdctcodec_k4 import K4AudioModel, K4TrainingStatistics, reconstruct_k4
from src.scaled_atom_rq import fit_signed_levels
from src.training.mdctcodec_continuation import AudioContinuationMedia, GracefulBudget

DEFAULT_ROOT=REPO/'outputs/mdctcodec_k4_a4096_20260913'
SOURCE=REPO/'outputs/mdctcodec_matched_6kbps_rangefix'


def write(path,data): path.write_text(json.dumps(data,indent=2))


def prepare(root, atoms=4096):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'prepared.json').exists():
        p=json.loads((root/'protocol.json').read_text())
        assert p['laser_architecture']['learned_atoms']==atoms and p['laser_architecture']['sparsity_level']==4
        assert sha(root/'protocol.json')==json.loads((root/'prepared.json').read_text())['protocol_sha256']
        print('ALREADY_PREPARED',root);return
    if (root/'laser_initial.pt').exists(): raise RuntimeError('Incomplete preparation; inspect output before replacement')
    original=json.loads((SOURCE/'protocol.json').read_text())
    assert sha(SOURCE/'manifest.json')==original['manifest_sha256']
    shutil.copy2(SOURCE/'manifest.json',root/'manifest.json')
    parent=torch.load(SOURCE/'laser_initial.pt',map_location='cpu',weights_only=False)
    kwargs=dict(parent['hyper_parameters'])
    kwargs.update(num_embeddings=atoms,sparsity_level=4,dictionary_update_max_atoms_per_step=atoms,
                  coefficient_quantization_bits=4,coefficient_quantization_max=1.,
                  log_images_every_n_steps=0,enable_val_latent_visuals=False)
    pl.seed_everything(original['seed'],workers=True)
    model=K4AudioModel(**kwargs)
    shared={}
    for name,digest in original['shared_initialization_sha256'].items():
        state={k.removeprefix(name+'.'):v for k,v in parent['state_dict'].items() if k.startswith(name+'.')}
        getattr(model,name).load_state_dict(state,strict=True)
        shared[name]=tensor_hash(getattr(model,name).state_dict())
        assert shared[name]==digest
    manifest=json.loads((root/'manifest.json').read_text())
    previous_calibration=json.loads((SOURCE/'calibration.json').read_text())
    paths=previous_calibration['paths'][:256]
    assert set(paths)<=set(manifest['train'])
    assert not set(paths)&set(manifest['validation']+manifest['test'])
    dataset=PairedAudioDataset(paths,seed=manifest['seed'])
    model.to('cuda:0').eval(); coefficients=[]
    with torch.inference_mode():
        dictionary=F.normalize(model.bottleneck.dictionary.float(),dim=0)
        for start in range(0,len(dataset),32):
            x=torch.stack([dataset[(-1,i)][0] for i in range(start,min(start+32,len(dataset)))]).to('cuda:0')
            z=model._to_bottleneck_input(model.pre_bottleneck(model.encoder(x)))
            signals=z.permute(0,2,3,1).reshape(-1,32).T.contiguous()
            _, values=model.bottleneck.batch_omp_with_support(signals,dictionary)
            coefficients.append(values.flatten())
        levels=fit_signed_levels(torch.cat(coefficients),8)
        model.bottleneck.coefficient_levels.copy_(torch.cat((levels[:4],levels.new_zeros(1),levels[4:])))
    model.cpu()
    assert model.bottleneck.dictionary.shape==(32,atoms)
    rvq_initial=original['initializations']['rvq']
    assert sha(rvq_initial['path'])==rvq_initial['sha256']
    rvq_state=torch.load(rvq_initial['path'],map_location='cpu',weights_only=False)['state_dict']
    books=[value for key,value in rvq_state.items() if key.startswith('bottleneck.') and key.endswith('.codebook.weight')]
    assert len(books)==4 and all(book.shape==(1024,32) for book in books)
    rvq_vectors=sum(book.shape[0] for book in books)
    rvq_scalars=sum(book.numel() for book in books)
    if atoms==4096:
        assert atoms==rvq_vectors and model.bottleneck.dictionary.numel()==rvq_scalars
    path=root/'laser_initial.pt'
    torch.save({'state_dict':model.state_dict(),'hyper_parameters':dict(model.hparams)},path)
    calibration={'paths':paths,'crop_epoch':-1,'source':f'fresh shared encoder, new {atoms}-atom dictionary; training only',
        'initial_coefficient_levels':model.bottleneck.coefficient_levels.tolist(),
        'policy':'Eight signed scalar centers + fixed zero; online Lloyd EMA .1 after complete training batches; frozen during validation/inference',
        'entropy_policy':'Per-depth joint-token and temporal-repeat counts from training only; EMA .999; deterministic Huffman with .25 pseudo-count; 12-byte frame/bit-count/CRC header counted'}
    write(root/'calibration.json',calibration)
    protocol=dict(original)
    protocol.pop('range_policy',None); protocol.pop('supersedes',None)
    joint_vocabulary=1+atoms*8
    protocol.update(name=f'mdctcodec-k4-a{atoms}-entropy-target6k',group=f'mdctcodec-k4-a{atoms}-20260913',
        created_utc=datetime.now(timezone.utc).isoformat(),
        laser=f'{atoms} learned shared atoms, K4, eight signed scalar levels plus zero, alternating residual updates',
        rate_is_target_only=True,rate_target_kbps=6.,raw_joint_nominal_kbps=(joint_vocabulary-1).bit_length()*4*150/1000,
        laser_architecture={'learned_atoms':atoms,'sparsity_level':4,'latent_dimension':32,'frames_per_second':150,
            'coefficient_nonzero_levels':8,'joint_vocabulary':joint_vocabulary,'shared_dictionary':True},
        rvq_architecture={'learned_vectors_per_codebook':1024,'residual_depth':4,'separate_codebooks':4,
            'frames_per_second':150,'raw_nominal_kbps':6.,'total_learned_vectors':rvq_vectors},
        dictionary_capacity={'laser_vectors':atoms,'rvq_total_vectors':rvq_vectors,
            'laser_dictionary_scalars':model.bottleneck.dictionary.numel(),'rvq_dictionary_scalars':rvq_scalars,
            'total_dictionary_vectors_match':atoms==rvq_vectors,
            'note':'This matches learned dictionary vectors/scalars, not the extra RVQ projection parameters or the transmitted token vocabulary.'},
        initializations={'laser':{'path':str(path),'sha256':sha(path),'bottleneck':'dictionary',
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'learned_dictionary_scalars':atoms*32}},
        shared_initialization_sha256=shared,
        rvq_control={'root':str(SOURCE/'rvq'),'run':json.loads((SOURCE/'rvq/run.json').read_text()),
            'initialization':original['initializations']['rvq'],'completed_generator_updates':200000,
            'reuse_reason':'Already completed identical common initialization, data/crops, seed, losses and 200k update budget. No need to spend on an identical rerun.'},
        scope=f'One matched seed; LASER has {atoms} learned vectors in a shared dictionary versus RVQ four separate tables of 1024 vectors, {rvq_vectors} total. Coefficient and dictionary learning differ. '
              'Same encoder/decoder/discriminator initialization, 150Hz frame rate, training data/crops, losses and 200k budget. '
              'Actual entropy bytes include coefficients, frame/bit-count header, checksum and padding. 6kbps is a target, not a achieved result. '
              'Do not claim a rate-matched win unless measured rates meet the declared budget; also entropy-code the RVQ control in final rate-distortion evaluation.',
        coefficient_max=None,
        checkpoint_upload='latest + top3 validation ViSQOL, every5 completed epochs and successful exit',
        compute_ceiling_gpu_hours=12,
        runtime_source_files=['src/mdctcodec_k4.py','src/audio_k4_entropy.py','src/models/dictionary_learner.py',
            'scripts/tools/train_mdctcodec_k4.py','src/training/mdctcodec_continuation.py','tests/test_mdctcodec_k4.py',
            'docs/mdctcodec-k4-a4096-2026-09-13.md'])
    write(root/'protocol.json',protocol)
    write(root/'prepared.json',{'protocol_sha256':sha(root/'protocol.json'),'shared_initialization_verified':True,
        'learned_atoms':atoms,'sparsity_level':4,'rvq_learned_vectors_per_codebook':1024,
        'dictionary_capacity':protocol['dictionary_capacity'],
        'initial_checkpoint_sha256':sha(path)})
    print('K4_PREPARED',json.dumps(json.loads((root/'prepared.json').read_text())),flush=True)


def callbacks(output,manifest,arm):
    atoms=json.loads((Path(output).parent/'protocol.json').read_text())['laser_architecture']['learned_atoms']
    return [K4TrainingStatistics(output=output,reference_order=SOURCE/'rvq/data_order.jsonl'),GracefulBudget(output),
        AudioContinuationMedia(output,manifest,arm,reconstruct_fn=reconstruct_k4,
            rate_label=f'K4 / {atoms} learned atoms | measured entropy rate; 6 kbps target')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--atoms',type=int,choices=[1024,4096],default=4096,
                   help='4096 matches total RVQ vectors; 1024 is only for explicit historical reproduction')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--mode',choices=['online','disabled'],default='online')
    args=p.parse_args();os.chdir(REPO)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    args.root=args.root.resolve()
    if args.prepare: prepare(args.root,args.atoms);return
    protocol=json.loads((args.root/'protocol.json').read_text())
    assert protocol['laser_architecture']['learned_atoms']==args.atoms, 'Requested atom count differs from prepared experiment'
    args.arm='laser';args.workers=args.metric_workers=2 if args.smoke else 8
    args.updates=12 if args.smoke else 0;args.train_batches=2 if args.smoke else 0
    args.validation_limit=2 if args.smoke else 0
    args.output=args.output or args.root/('preflight_laser' if args.smoke else 'laser')
    train(args,model_class=K4AudioModel,extra_callbacks_factory=callbacks)


if __name__=='__main__': main()
