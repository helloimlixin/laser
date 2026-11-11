"""Verify scores, tie selection, complete beam pools, and sampled soft labels."""
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE=Path('/tmp/laser-aligned-beam451-20261105/frozen')
OUT=Path('/tmp/laser-speed-aligned-20261105')
sys.path[:0]=[str(BASE/'source'),str(BASE)]
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
import torch
from src import rq_upstream
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from runtime import beam_teacher
spec=importlib.util.spec_from_file_location('exact_beam_tested',Path(__file__).with_name('exact_beam.py'))
fast=importlib.util.module_from_spec(spec);spec.loader.exec_module(fast)


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.04,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    spec=torch.load(BASE/'inputs/joint-codebook.pt',weights_only=True,map_location='cpu')
    book=JointAtomCoefficientRQ(spec['dictionary'],spec['atom_ids'],spec['coefficients'],zero_code_id=None).cuda()
    torch.manual_seed(26110577)
    for depth in range(4):
        r=torch.randn(64,16,256,device='cuda')*.2
        expected=r.square().sum(-1,keepdim=True)-book.scores(r.reshape(-1,256),depth).reshape(64,16,-1)
        actual=fast.candidate_cost(book,r,depth)
        assert torch.equal(expected,actual),dict(depth=depth,max_error=float((expected-actual).abs().max()))
        for mode in [0,1]:
            os.environ['LASER_FAST_BEAM_TOPK']=str(mode)
            assert torch.equal(fast.select(actual,16),expected.flatten(1).topk(16,dim=-1,largest=False).indices)
        del r,expected,actual
        torch.cuda.empty_cache()
    # Deliberate interior ties and cutoff ties exercise the native fallback.
    for size in [1,64,256]:
        tied=torch.zeros(size,16,32,device='cuda')
        assert torch.equal(fast.select(tied,16),tied.flatten(1).topk(16,dim=-1,largest=False).indices)
    tests=[]
    for n in [128,257]:
        ids=torch.randint(16384,(n,4),device='cuda')
        latent=book.embed(ids).sum(-2)+torch.randn(n,256,device='cuda')*.05
        reference=beam_teacher.beam_pool(book,latent)
        for mode in [0,1]:
            os.environ['LASER_FAST_BEAM_TOPK']=str(mode)
            result=fast.beam_pool(book,latent)
            assert all(torch.equal(reference[k],result[k]) for k in ['candidates','costs','canonical_index','valid'])
            p0=beam_teacher.posterior(reference,2.)
            p1=beam_teacher.posterior(result,2.);assert torch.equal(p0,p1)
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.manual_seed(26110577);chosen0=torch.multinomial(p0,1).squeeze(-1)
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.manual_seed(26110577);chosen1=torch.multinomial(p1,1).squeeze(-1)
            assert torch.equal(chosen0,chosen1)
            row=torch.arange(n,device='cuda')
            c0=reference['candidates'][row,chosen0];c1=result['candidates'][row,chosen1]
            t0=beam_teacher.conditional_targets(reference['candidates'],p0,c0,16384)
            t1=beam_teacher.conditional_targets(result['candidates'],p1,c1,16384)
            assert torch.equal(t0,t1) and torch.equal(c0,c1)
            tests.append(dict(latent_sites=n,hierarchical_topk=bool(mode),beam_pool_and_soft_targets_bitwise_equal=True))
            del result,p0,p1,t0,t1
        del reference,latent,ids
        torch.cuda.empty_cache()
    proof=dict(passed=True,fp32_candidate_costs_bitwise_equal=True,
        native_tie_behavior_preserved=True,sampled_codes_bitwise_equal=True,
        conditional_targets_bitwise_equal=True,tests=tests,stats=fast.STATS)
    (OUT/'exact-beam-parity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
