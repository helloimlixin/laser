#!/usr/bin/env python3
"""Distortion-aware 20k pair compression, with a matched uniform-budget control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
SOURCE = Path('/tmp/laser-rq-native-joint32k-imagenet-20261016')
FIT = Path('/tmp/laser-compact-joint-vocabulary-20261016/fit-latents.npy')
CACHE = Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
BASE = Path('/tmp/laser-adaptive-joint20k-20261017')
OUT = ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017'
os.environ['RQ_UPSTREAM_ROOT'] = str(SOURCE/'upstream')
os.environ.setdefault('TORCH_HOME', str(SOURCE/'torch-cache'))
sys.path[:0] = [str(ROOT), str(SOURCE/'upstream'), str(ROOT/'runtime')]
import numpy as np
import torch
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ, FrozenJointLaserTokenizer
from src.training.laser_joint_vocabulary import fit_coefficient_candidates, allocate_shared_coefficients, pack_codebook
from src.training.laser_compact_joint_vocabulary import refit_compact_book


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, **values):
    path = OUT/name
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(dict(values, time=time.time()), indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def progress(phase, **values):
    value = dict(phase=phase, **values)
    write('progress.json', **value)
    print(json.dumps(value), flush=True)


def quantizer(path, device):
    spec = torch.load(path, map_location='cpu', weights_only=True)
    return JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)


@torch.inference_mode()
def fit(args, device):
    manifest = json.loads((ROOT/'outputs/imagenet-laser-compact-joint-vocabulary-20261016/calibration-data.json').read_text())
    held_meta = json.loads((ROOT/'outputs/imagenet-laser-joint-vocabulary-study-20261015/calibration-data.json').read_text())
    assert sha(FIT) == manifest['fit_latents_sha256']
    assert sha(CACHE/'temperature-latents.npy') == held_meta['temperature_latents_sha256']
    assert set(manifest['fit_indices']).isdisjoint(held_meta['temperature_indices'])
    prior_path = ROOT/'outputs/imagenet-laser-compact-joint-vocabulary-20261016/compact8192-refined-codebook.pt'
    prior = quantizer(prior_path, device)
    x = torch.from_numpy(np.load(FIT).reshape(-1, 256)).to(device)
    held = torch.from_numpy(np.load(CACHE/'temperature-latents.npy').reshape(-1, 256)).to(device)
    progress('collecting_pair_distortion', training_views=8192, unique_training_images=4096)
    encoded = prior.quantize(x, chunk_size=1024, return_projections=True)
    importance = torch.zeros(8192, device=device)
    books, errors = [], []
    for depth in range(4):
        ids = encoded['codes'][:, depth]
        observed = ids != 0
        atoms = prior.atom_ids[depth, ids[observed]]
        values = encoded['projections'][observed, depth]
        importance += torch.bincount(atoms, weights=values.square(), minlength=8192)
        depth_books, depth_errors = fit_coefficient_candidates(atoms, values, 8192, counts=(2,4,6,8))
        books.append(depth_books); errors.append(depth_errors)
        progress('fitted_coefficient_candidates', depth=depth+1, observed_pairs=int(observed.sum()))
    allocation = allocate_shared_coefficients(errors, 20480)
    assert len(allocation) == 8192 and sum(allocation) == 20480
    adaptive = pack_codebook(prior.dictionary, books, [allocation]*4)
    selected = importance.argsort(descending=True)[:5120]
    coefficients = torch.cat((prior.coefficients[:, :1],
        prior.coefficients[:, 1:].view(4,8192,4)[:, selected].flatten(1)), -1)
    atom_ids = torch.cat((torch.zeros(1, device=device, dtype=torch.long),
        torch.arange(5120, device=device).repeat_interleave(4))).expand(4,-1)
    uniform = dict(dictionary=prior.dictionary[:, selected], atom_ids=atom_ids, coefficients=coefficients)
    write('calibration-data.json', fitting_split='train', training_views=8192, unique_training_images=4096,
        fit_indices=manifest['fit_indices'], heldout_training_indices=held_meta['temperature_indices'],
        fit_latents_sha256=manifest['fit_latents_sha256'], heldout_latents_sha256=held_meta['temperature_latents_sha256'],
        source_stage1_sha256=manifest['source_stage1_sha256'], source_joint32k_sha256=sha(prior_path),
        coefficient_allocation_histogram={str(n):allocation.count(n) for n in [2,4,6,8]},
        retained_adaptive_directions=8192, retained_uniform_directions=5120,
        vocabulary=20481, code_zero='null vector', frozen_encoder_decoder=True,
        allocation='pooled training-only coefficient distortion reduction per added pair entry',
        constant_layout_across_depths=True, lost_information='lossy quantization; no claim of lossless pair relabeling')
    del encoded, prior, books, errors
    for name, spec in [('adaptive8192',adaptive),('uniform5120',uniform)]:
        q = JointAtomCoefficientRQ(spec['dictionary'], spec['atom_ids'], spec['coefficients']).to(device)
        best, best_error, trace = None, float('inf'), []
        for iteration in range(args.passes+1):
            result = q.quantize(x, chunk_size=1024)
            error = float((x-result['quantized']).square().mean())
            held_error = float((held-q.quantize(held,chunk_size=1024)['quantized']).square().mean())
            row = dict(iteration=iteration, fit_latent_mse=error, heldout_training_latent_mse=held_error)
            if held_error < best_error:
                best_error = held_error
                best = {k:getattr(q,k).cpu().clone() for k in ['dictionary','atom_ids','coefficients']}
            if iteration < args.passes:
                row.update(refit_compact_book(q,x,result['codes']))
            trace.append(row)
            progress('refitting_pair_prototypes', candidate=name, **row)
            write(name+'-fit.json', completed=iteration==args.passes, trace=trace, best_heldout_mse=best_error)
            del result
        for gain in [1.,1.1]:
            label = name+('-gain110' if gain > 1 else '')
            path = OUT/(label+'-codebook.pt')
            assert not path.exists(), 'Refusing to overwrite a fitted candidate'
            book = dict(best, coefficients=best['coefficients']*gain,
                kind='laser_joint_atom_coefficient_rq', format_version=2,
                source_stage1_checkpoint_sha256=manifest['source_stage1_sha256'],
                source_joint_codebook_sha256=sha(prior_path), learned_compact_dictionary=True,
                encoder_decoder_frozen=True, fitting_split='train', fitting_views=8192,
                coefficient_gain=gain, vocabulary=20481, adaptive_pair_allocation=name=='adaptive8192',
                allocation_training_only=True, shared_integer_atom_and_level_layout=True)
            torch.save(book,path)
        del q
        torch.cuda.empty_cache()
    write('fit-completed.json', completed=True, vocabulary=20481, candidates=['adaptive8192','uniform5120'],
        stage2_launched=False, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)


@torch.inference_mode()
def screen(device):
    from src import rqvae_metrics as metrics
    metrics.INCEPTION_PATH = SOURCE/'upstream/rqvae/metrics/inception.py'
    inception = metrics.OriginalRQVAEInception().to(device).eval().requires_grad_(False)
    aux = FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt', SOURCE/'inputs/joint-codebook.pt').to(device)
    real = np.load(CACHE/'screen-real-features.npy')
    latents = np.load(CACHE/'screen-latents.npy',mmap_mode='r')
    assert real.shape == (2048,2048) and len(latents) == 2048
    results = {}
    for name in ['adaptive8192','adaptive8192-gain110','uniform5120','uniform5120-gain110']:
        path = OUT/(name+'-codebook.pt')
        q = quantizer(path,device)
        fake,error = [],0.
        for first in range(0,len(latents),16):
            x = torch.from_numpy(latents[first:first+16].copy()).to(device)
            z = q.quantize(x,chunk_size=1024)['quantized']
            image = aux.decode(z)
            features = inception(image.mul(.5).add(.5).clamp(0,1))
            fake.append((features[0] if isinstance(features,tuple) else features).cpu().numpy())
            error += float((x-z).square().sum())
            if first%512 == 0:progress('screening_reconstruction',candidate=name,images=first+len(x))
        fake = np.concatenate(fake)
        fid = float(metrics.frechet_distance(fake.mean(0),np.cov(fake,rowvar=False),real.mean(0),np.cov(real,rowvar=False)))
        results[name] = dict(matched_subset_rfid=fid,images=2048,latent_mse=error/latents.size,
            vocabulary=q.vocab_size,codebook_path=str(path),codebook_sha256=sha(path))
        progress('screen_result',candidate=name,**results[name])
        write('screen-results.json',results=results,selection_only=True,full_50000_reconstruction_fid_required=True)
        del q
    selected = min(results,key=lambda name:results[name]['matched_subset_rfid'])
    write('selected.json',candidate=selected,**results[selected],selection_only=True)


@torch.inference_mode()
def full(args,device):
    from torch.utils.data import DataLoader,Subset
    import torch.distributed as dist
    from rqvae.utils.config import load_config
    from rqvae.img_datasets.transforms import create_transforms
    from src.training.rq_imagenet_runtime import TarImageNet
    from src import rqvae_metrics as metrics
    rank,world = int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    if world>1:dist.init_process_group('nccl')
    chosen = json.loads((OUT/'selected.json').read_text())
    path = Path(chosen['codebook_path'])
    assert sha(path)==chosen['codebook_sha256']
    metrics.INCEPTION_PATH = SOURCE/'upstream/rqvae/metrics/inception.py'
    aux = FrozenJointLaserTokenizer(SOURCE/'inputs/stage1-model.pt',path).to(device)
    config = load_config(SOURCE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(SOURCE/'data','val',create_transforms(config.dataset,split='val'))
    assert len(data)==50000
    loader = DataLoader(Subset(data,range(rank,len(data),world)),batch_size=16,num_workers=4,pin_memory=True)
    metric = metrics.DistributedOriginalRQVAEMetrics(device)
    count=0
    for images,_ in loader:
        images=images.to(device)
        reconstruction,_,_=aux(images)
        metric.update(images.mul(.5).add(.5).clamp(0,1),real=True)
        metric.update(reconstruction.mul(.5).add(.5).clamp(0,1),real=False)
        count+=len(images)
        if rank==0 and count%512==0:progress('full_reconstruction_fid',images_on_rank=count,target_on_rank=12500)
    fid,_,_=metric.compute()
    assert int(metric.real_count)==int(metric.fake_count)==50000
    if rank==0:
        write('full-rfid.json',completed=True,measured_rfid=fid,passed=fid<=5.25,maximum_rfid=5.25,
            real_images=50000,reconstructed_images=50000,reference='full validation, original RQ Inception',
            candidate=chosen['candidate'],codebook_path=str(path),codebook_sha256=sha(path),vocabulary=20481,
            stage2_launched=False,encoder_decoder_frozen=True,
            gate_scope='two-epoch generative pilot; reconstruction loss is measured explicitly')
    if world>1:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['fit','screen','full'])
    parser.add_argument('--passes',type=int,default=12)
    parser.add_argument('--memory-fraction',type=float,default=.15)
    args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)))
    torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='fit':fit(args,device)
    elif args.action=='screen':screen(device)
    else:full(args,device)


if __name__=='__main__':main()
