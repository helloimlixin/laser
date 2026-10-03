#!/usr/bin/env python3
"""Experimental depth-specific OMP noise; does not alter the active trainer."""
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/laser-church/dropout-experiment')
sys.path.insert(0, str(ROOT/'runtime'))
import torch
from src.training.rqtransformer import load_stage1_checkpoint
from src.stochastic_compound import stochastic_omp


@torch.inference_mode()
def sample(signals, dictionary, gram, temperatures):
    x = signals.reshape(-1, signals.shape[-1])
    correlation = x @ dictionary
    residual_correlation = correlation
    available = torch.ones_like(correlation, dtype=torch.bool)
    rows = torch.arange(len(x), device=x.device)
    support = torch.empty(len(x), 0, dtype=torch.long, device=x.device)
    chol = None
    entropy = []
    for step, temperature in enumerate(temperatures):
        scores = residual_correlation.square().masked_fill(~available, -torch.inf)
        probabilities = ((scores-scores.amax(-1, keepdim=True))/temperature).softmax(-1)
        entropy.append(-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1))
        atom = torch.multinomial(probabilities, 1).squeeze(-1)
        available[rows, atom] = False
        if step == 0:
            chol = gram[atom, atom].sqrt()[:, None, None]
        else:
            cross = gram[support, atom[:, None]].unsqueeze(-1)
            solved = torch.linalg.solve_triangular(chol, cross, upper=False).transpose(1,2)
            diagonal = gram[atom,atom][:,None,None]-solved.square().sum(-1,keepdim=True)
            assert (diagonal>1e-7).all()
            chol = torch.cat((torch.cat((chol,x.new_zeros(len(x),step,1)),dim=-1),
                              torch.cat((solved,diagonal.sqrt()),dim=-1)),dim=-2)
        support = torch.cat((support, atom[:,None]),dim=-1)
        coefficients = torch.cholesky_solve(correlation.gather(1,support).unsqueeze(-1),chol).squeeze(-1)
        residual_correlation = correlation-coefficients.unsqueeze(1).bmm(gram[support]).squeeze(1)
    quantized = (dictionary.T[support]*coefficients[...,None]).sum(-2)
    return dict(atoms=support,coefficients=coefficients,quantized=quantized,
                entropy=torch.stack(entropy,-1))


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.cuda.set_device(3)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:3')
    p=load_stage1_checkpoint(Path('/mnt/laser-church/assets/tokenizer.pt'))
    state=p['state_dict']
    key='quantizer.dictionary' if 'quantizer.dictionary' in state else 'bottleneck.dictionary'
    dictionary=torch.nn.functional.normalize(state[key].float(),dim=0).to(device)
    del p,state
    gram=dictionary.T@dictionary
    datasets=torch.load(ROOT/'noise-calibration/encoded-probe.pt',weights_only=True)
    output={}
    for split,data in datasets.items():
        x=data['latents'].reshape(-1,256).to(device)
        hard=stochastic_omp(x,dictionary,depth=4,temperature=0.,gram=gram)
        mse=(x-hard['quantized']).square().mean().item()
        torch.manual_seed(781)
        a=stochastic_omp(x[:128],dictionary,depth=4,temperature=.0625,gram=gram)
        torch.manual_seed(781)
        b=sample(x[:128],dictionary,gram,[.0625]*4)
        assert torch.equal(a['atoms'],b['atoms'])
        torch.testing.assert_close(a['coefficients'],b['coefficients'],atol=0,rtol=0)
        rows=[]
        for profile in [[.0625]*4,[.125,.125,.0625,.0625],[.25,.125,.0625,.0625],[.25,.25,.125,.0625]]:
            trials=[]
            for seed in range(3):
                torch.manual_seed(2026092440+seed)
                r=sample(x,dictionary,gram,profile)
                trials.append(dict(mse_ratio=(x-r['quantized']).square().mean().item()/mse,
                    entropy=r['entropy'].mean(0),changed=(r['atoms']!=hard['atoms']).any(-1).float().mean().item()))
            row=dict(temperatures=profile,latent_mse_ratio=sum(t['mse_ratio'] for t in trials)/3,
                     entropy_by_depth=torch.stack([t['entropy'] for t in trials]).mean(0).tolist(),
                     changed_ordered_support=sum(t['changed'] for t in trials)/3)
            rows.append(row);print(json.dumps(dict(split=split,**row)),flush=True)
        output[split]=rows
    # Bank enlargement changes finite-bank coverage, not the underlying noise temperature.
    x=datasets['val']['latents'].reshape(-1,256).to(device)[::16]
    draws=[]
    for seed in range(64):
        torch.manual_seed(2026092500+seed)
        a=sample(x,dictionary,gram,[.0625]*4)['atoms']
        draws.append(((a[:,0]*16384+a[:,1])*16384+a[:,2])*16384+a[:,3])
    codes=torch.stack(draws,dim=-1)
    bank=[]
    for size in [16,32,64]:
        ordered=codes[:,:size].sort(-1).values
        count=1+(ordered[:,1:]!=ordered[:,:-1]).sum(-1)
        same=(codes[:,:size,None]==codes[:,None,:size]).float().mean((1,2))
        bank.append(dict(variants=size,sites=len(x),mean_unique=float(count.float().mean()),
                         fraction_single=float((count==1).float().mean()),
                         probability_two_visits_same_ordered_support=float(same.mean())))
    result=dict(passed=True,uniform_profile_matches_original=True,training_policy_changed=False,
                profiles=output,bank_expansion=bank)
    (ROOT/'noise-calibration/depth-and-bank.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'bank_expansion':bank}),flush=True)


if __name__=='__main__':
    main()
