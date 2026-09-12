#!/usr/bin/env python3
"""Verify full-grid FP32/BF16 cache consistency for a matched-order Church prior."""
import argparse
import codecs
import json
import os
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.tools.build_sign_probe_cache import sha256_file
from src.training.rqtransformer import LaserAux
from src.church_pattern_order import pattern_order_prior
from src.church_support_pattern_training import pattern_targets


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.2)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['config']
    model=pattern_order_prior(cfg['coefficient_vocabulary'],cfg['dropout'],cfg['ordering']).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'],strict=True)
    del saved
    raw=torch.load(cfg['cache'],map_location='cpu',weights_only=False)
    book=torch.load(cfg['codebook_path'],map_location='cpu',weights_only=True)
    aux=LaserAux(Path(cfg['stage1']),16384,2048,3.,coeff_scales=cfg['tokenizer']['coeff_scales'],
        sparsity_level=4,soft_target_physical=True,clamp_coeffs=False,
        coefficient_patterns=book['coefficient_patterns']).cuda().eval().requires_grad_(False)
    atoms=raw['holdout']['atoms'][:2].cuda().long()
    physical=raw['holdout']['coefficients'][:2].cuda()
    ids=pattern_targets(aux,atoms,physical)
    packed=model.pack(atoms,ids)
    results={}
    for amp in (False,True):
        errors={'atom_max_abs_logit':[],'atom_tv':[],'pattern_max_abs_logit':[],'pattern_tv':[]}
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            teacher=model(packed,model_aux=aux,amp=amp)
            model.init_cache()
            for h in range(8):
                for w in range(8):
                    for event in range(5):
                        hidden=model.cached_hidden(packed,aux,(h,w,event),amp=amp)
                        is_pattern=event==(0 if model.ordering=='pattern-first' else 4)
                        if is_pattern:
                            cached=model.pattern_classifier(hidden).float()
                            target=teacher['pattern_logits'][:,h,w].float()
                            kind='pattern'
                        else:
                            depth=event-1 if model.ordering=='pattern-first' else event
                            cached=model.classifier(hidden).float()
                            if depth: cached.scatter_(1,atoms[:,h,w,:depth],-float('inf'))
                            target=teacher['atom_logits'][:,h,w,depth].float()
                            kind='atom'
                        assert torch.equal(torch.isfinite(cached),torch.isfinite(target))
                        finite=torch.isfinite(target)
                        errors[kind+'_max_abs_logit'].append(float((cached[finite]-target[finite]).abs().max()))
                        errors[kind+'_tv'].extend((.5*(cached.softmax(-1)-target.softmax(-1)).abs().sum(-1)).cpu().tolist())
            model.init_cache()
        summary={key:{'mean':sum(v)/len(v),'max':max(v)} for key,v in errors.items()}
        if not amp:
            assert summary['atom_max_abs_logit']['max']<2e-4
            assert summary['pattern_max_abs_logit']['max']<2e-4
            assert summary['atom_tv']['max']<1e-5 and summary['pattern_tv']['max']<1e-5
        results['bf16' if amp else 'fp32']=summary
        print(json.dumps({'amp':amp,**summary}),flush=True)
    args.output.write_text(json.dumps({'checkpoint_sha256':sha256_file(args.checkpoint),
        'ordering':cfg['ordering'],'images':2,'sites_per_image':64,'real_history_at_every_site':True,'passed_fp32':True,
        'metrics':results,'source_sha256':sha256_file(Path(__file__))},indent=2)+'\n')


if __name__=='__main__':
    main()
