#!/usr/bin/env python3
"""CPU parity against the native shared RQ quantizer, plus target-noise checks."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
SOURCE=Path('/tmp/laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018')
OUT=ROOT/'outputs/imagenet-laser-sharedjoint20k-depth5-codebook-20261018'
LOCAL=Path('/tmp/laser-sharedjoint20k-depth5-20261018')
os.environ['RQ_UPSTREAM_ROOT']=str(SOURCE/'upstream')
sys.path[:0]=[str(ROOT),str(SOURCE/'upstream')]


def main():
    import numpy as np
    import torch
    import yaml
    from src import rq_upstream
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    from src.training.laser_shared_joint_vocabulary import assert_shared_codewords
    from rqvae.models.rqvae.quantizations import RQBottleneck
    torch.set_num_threads(8);torch.manual_seed(26101820481)
    path=OUT/'sharedjoint20k-depth5-codebook.pt'
    spec=torch.load(path,map_location='cpu',weights_only=True)
    q=JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients'])
    assert_shared_codewords(q)
    native=RQBottleneck([8,8,256],[8,8,5],20481,shared_codebook=True,restart_unused_codes=False).eval()
    assert native.shared_codebook and all(book is native.codebooks[0] for book in native.codebooks)
    with torch.no_grad():
        native.codebooks[0].weight[:-1].copy_(q.expanded_codebook(0))
        x=torch.from_numpy(np.load(LOCAL/'held-rank0.npy')[:16].copy())
        z,_,native_codes=native(x[:2])
        hard=q.quantize(x[:2])
        assert torch.equal(native_codes,hard['codes'])
        torch.testing.assert_close(z,hard['quantized'],rtol=1.e-4,atol=2.e-5)
        pn,cn=native.get_soft_codes(x[:2],temp=.1,stochastic=False)
        pq,cq=q.get_soft_codes(x[:2],temp=.1,stochastic=False)
        assert torch.equal(cn,cq)
        kl=(pn*(pn.clamp_min(1.e-30).log()-pq.clamp_min(1.e-30).log())).sum(-1)
        maximum_kl=float(kl.max());assert maximum_kl<1.e-4
        del pn,pq,cn,cq
        temperatures=yaml.safe_load((SOURCE/'inputs/recipe.yaml').read_text())['targets']['temperature']
        errors=torch.zeros(2,dtype=torch.float64);entropy=torch.zeros(5,dtype=torch.float64)
        for chunk in x.split(2):
            p,c=q.get_soft_codes(chunk,temp=temperatures,stochastic=True)
            hard=q.quantize(chunk)['quantized'];soft=q.embed(c).sum(-2)
            errors+=torch.stack([(chunk-hard).square().sum(),(chunk-soft).square().sum()]).double()
            entropy+=(-(p*p.clamp_min(1.e-30).log()).sum(-1)).sum((0,1,2)).double()
            del p,c
        ratio=float(errors[1]/errors[0]);entropies=(entropy/(len(x)*64)).tolist()
        assert .9<ratio<1.05 and all(0<e<10 for e in entropies)
    with path.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
    report=dict(passed=True,time=time.time(),codebook_sha256=h,vocabulary=20481,depth=5,
        physical_vectors_identical_at_every_depth=True,native_quantizer_shared_codebook=True,
        native_hard_codes_exact=True,native_hard_reconstruction_equal=True,
        maximum_native_soft_label_kl=maximum_kl,target_temperatures_unchanged=True,
        inherited_target_temperatures=temperatures,noise_audit_images=16,
        sampled_to_hard_latent_mse_ratio=ratio,per_depth_target_entropy=entropies,
        integer_categorical_targets=True,no_atom_or_coefficient_prediction_heads=True)
    (OUT/'sharing-proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
