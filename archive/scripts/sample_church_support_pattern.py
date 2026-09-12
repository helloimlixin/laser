#!/usr/bin/env python3
"""Sample a frozen Church joint-pattern prior and save complete-site integers."""
import argparse
import codecs
import json
import os
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from archive.scripts.diagnose_church_support_pattern_generation import deterministic_top_p
from scripts.tools.build_sign_probe_cache import sha256_file
from src.training.rqtransformer import LaserAux,atomic_torch_save
from src.church_support_pattern_training import support_pattern_prior
from src.models.rqtransformer import transformers
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
from src.support_pattern_integer_codec import pack_support_pattern,unpack_support_pattern


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--atom-top-k',type=int,default=2048)
    p.add_argument('--pattern-top-p',type=float,default=.5)
    p.add_argument('--seed',type=int,default=18701)
    p.add_argument('--fid',action='store_true')
    args=p.parse_args()
    if args.samples<1 or not 0<args.pattern_top_p<=1 or not 0<=args.atom_top_k<=16384:
        p.error('Expected positive sample count, 0 < pattern-top-p <= 1, and atom-top-k in [0,16384]')
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.3)
    transformers._top_p_probs=deterministic_top_p
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config']
    epoch=saved['epoch']
    model=support_pattern_prior(cfg['coefficient_vocabulary'],cfg['dropout']).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'],strict=True)
    del saved
    book=torch.load(cfg['codebook_path'],map_location='cpu',weights_only=True)
    assert sha256_file(Path(cfg['codebook_path']))==cfg['codebook_sha256']
    aux=LaserAux(Path(cfg['stage1']),16384,2048,3.,coeff_scales=cfg['tokenizer']['coeff_scales'],
        sparsity_level=4,soft_target_physical=True,clamp_coeffs=False,
        coefficient_patterns=book['coefficient_patterns']).cuda().eval().requires_grad_(False)
    metric=DistributedOriginalRQVAEMetrics('cuda',reference_stats_path=Path(cfg['fid_stats'])) if args.fid else None
    frozen=[(v,v._version) for m in (model,aux) for v in (*m.parameters(),*m.buffers())]
    torch.manual_seed(args.seed)
    atoms_saved,ids_saved,integers=[],[],[]
    for first in range(0,args.samples,128):
        count=min(128,args.samples-first)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            atoms,ids=model.sample_compound(count,aux,atom_top_k=args.atom_top_k,atom_top_p=None,
                coeff_top_k=0,coeff_top_p=args.pattern_top_p if args.pattern_top_p<1 else None,
                atom_temperature=1.,coeff_temperature=1.,amp=True)
        packed=pack_support_pattern(atoms,ids,num_patterns=len(aux.coefficient_patterns))
        recovered_atoms,recovered_ids=unpack_support_pattern(packed,num_patterns=len(aux.coefficient_patterns))
        assert torch.equal(recovered_atoms.reshape_as(atoms),atoms.cpu())
        assert torch.equal(recovered_ids.reshape_as(ids),ids.cpu())
        ordered=atoms.sort(-1).values
        assert (ordered[...,1:]!=ordered[...,:-1]).all()
        atoms_saved.append(atoms.cpu().short())
        ids_saved.append(ids.cpu().short())
        integers.extend(packed)
        for offset in range(0,count,32):
            rgb=(aux.decode_coefficient_patterns(atoms[offset:offset+32],ids[offset:offset+32])+1)/2
            if metric is not None: metric.update(rgb,real=False)
            if first==0 and offset==0: save_image(rgb,args.output/'samples.png',nrow=8)
        print(json.dumps({'generated':first+count,'total':args.samples}),flush=True)
    atomic_torch_save({'atoms':torch.cat(atoms_saved),'pattern_ids':torch.cat(ids_saved),
        'complete_site_integers':integers,'integer_grid_shape':[args.samples,8,8]},args.output/'generated-codes.pt')
    assert all(v._version==version and v.grad is None for v,version in frozen)
    result={'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha256_file(args.checkpoint),'epoch':epoch,
        'codebook_sha256':cfg['codebook_sha256'],'samples':args.samples,'seed':args.seed,'batch':128,
        'atom_top_k':args.atom_top_k,'pattern_top_p':args.pattern_top_p,'temperature':1.,
        'weights_changed':False,'integer_roundtrip_verified':True,'source_sha256':sha256_file(Path(__file__))}
    if metric is not None: result['fid']=float(metric.compute()[0])
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
