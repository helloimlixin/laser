#!/usr/bin/env python3
"""Measure cross-atom pair redundancy without changing the active pilot."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
OUT=ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_joint_redundancy import nearest_cross_atom_pairs


@torch.inference_mode()
def main():
    torch.set_num_threads(4);device=torch.device('cuda:3');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.08,device)
    torch.backends.cuda.matmul.allow_tf32=False
    selected=json.loads((OUT/'selected.json').read_text())
    path=Path(selected['codebook_path'])
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==selected['codebook_sha256']
    spec=torch.load(path,map_location='cpu',weights_only=True)
    q=JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients']).to(device)
    result=nearest_cross_atom_pairs(q.dictionary,q.atom_ids,q.coefficients)
    x=torch.from_numpy(np.load('/tmp/laser-compact-joint-vocabulary-20261016/fit-latents.npy').reshape(-1,256)).to(device)
    codes=q.quantize(x,chunk_size=1024)['codes']
    histogram=torch.stack([torch.bincount(codes[:,d],minlength=q.vocab_size) for d in range(4)]).cpu()
    frequency=histogram.sum(0).double();weight=frequency[1:]/frequency[1:].sum()
    relative=result['relative_vector_rmse'][1:];bound=result['omitted_direction_lower_bound']
    thresholds={}
    for t in [.05,.1,.2,.3]:
        close=relative<=t
        thresholds[str(t)]=dict(pair_fraction=float(close.double().mean()),
            training_token_fraction=float(weight[close].sum()),
            threshold_decision_certified_fraction=float(((bound>t)|close).double().mean()))
    usage=[]
    for d,h in enumerate(histogram):
        p=h.double()/h.sum();entropy=float(-(p*p.clamp_min(1.e-30).log()).sum())
        usage.append(dict(depth=d+1,active_codes=int((h>0).sum()),null_probability=float(p[0]),
            hard_code_entropy=entropy,hard_code_perplexity=float(np.exp(entropy))))
    report=dict(completed=True,codebook_sha256=selected['codebook_sha256'],
        vocabulary=q.vocab_size,training_views=8192,training_tokens=int(histogram.sum()),
        per_depth_usage=usage,relative_vector_rmse_thresholds=thresholds,
        candidate_neighbor_directions=result['candidate_directions'],
        global_nearest_certified_fraction=float((relative<=bound).double().mean()),
        geometry='sum of squared physical pair-vector distances over four depths / source pair energy',
        maximum_cross_atom_cosine_quantiles={str(t):float(result['maximum_cross_atom_cosine'].quantile(t))
            for t in [.5,.9,.99,1.]},
        read_only=True,active_training_recipe_changed=False,
        generation_quality_not_measured_by_this_audit=True,time=time.time())
    torch.save(dict(result,hard_training_histogram=histogram),OUT/'pair-redundancy-details.pt')
    (OUT/'pair-redundancy-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
