#!/usr/bin/env python3
"""Audit a frozen trained compound prior without altering either live run."""
import argparse
import codecs
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.church_ffhq_archived import make_prior, full_training_cache, targets
from src.church_relative_noise import RelativeChurchAux
from src.ffhq_v4_archived import compound_objective
from scripts.tools.build_sign_probe_cache import sha256_file


def comparison(actual, expected):
    a, b = actual.float(), expected.float()
    p, logp, logq = b.softmax(-1), b.log_softmax(-1), a.log_softmax(-1)
    return {'max_logit_error':float((a-b).abs().max()),
        'mean_absolute_logit_error':float((a-b).abs().mean()),
        'mean_kl_reference_to_test':float((p*(logp-logq)).sum(-1).mean()),
        'top1_disagreement':float((a.argmax(-1) != b.argmax(-1)).float().mean())}


@torch.no_grad()
def cached_logits(model, aux, packed, amp):
    atoms, _ = model.unpack(packed)
    buffer = torch.full_like(packed, 1024)
    result = {'atom_logits':[], 'coeff_logits':[]}
    model.init_cache()
    for h in range(8):
        for w in range(8):
            for d in range(4):
                hidden = model.cached_head_output(buffer, aux, None, (h,w,d), amp=amp)
                result['atom_logits'].append(model.classifier(hidden))
                result['coeff_logits'].append(model.coefficient_logits(hidden, aux.dictionary.t()[atoms[:,h,w,d]], d))
                buffer[:,h,w,d] = packed[:,h,w,d]
        print(json.dumps({'phase':'cache', 'amp':amp, 'rows':h+1}), flush=True)
    model.init_cache()
    return {k:torch.stack(v,1).reshape(len(packed),8,8,4,-1) for k,v in result.items()}


@torch.no_grad()
def geometry_audit(model, aux, atoms, physical):
    captured=[]
    hook=model.head_transformer.register_forward_hook(lambda m,args,out:captured.append(out))
    packed, probabilities=targets(aux, atoms, physical, stochastic=False)
    out=model(packed,model_aux=aux,amp=False)
    hook.remove()
    hidden=captured[0].reshape(len(atoms),8,8,4,-1)
    logits, candidates=out['atom_logits'].float().topk(4,-1)
    gt=out['atom_logits'].float().gather(-1,atoms[...,None])
    gt=gt.masked_fill((candidates==atoms[...,None]).any(-1,keepdim=True),-torch.inf)
    weights=torch.cat([logits,gt],-1).softmax(-1)
    candidates=torch.cat([candidates,atoms[...,None]],-1)
    vectors=aux.dictionary.t()[candidates]
    old_coeff=(out['coeff_logits'].float().softmax(-1)*aux.coeff_bins).sum(-1)*aux.coeff_scales
    old_prediction=(weights[...,None]*vectors).sum(-2)*old_coeff[...,None]
    means=[]
    for d in range(4):
        h=hidden[:,:,:,d,None,:].expand(-1,-1,-1,5,-1)
        coefficient_logits=model.coefficient_logits(h,vectors[:,:,:,d],d)
        means.append((coefficient_logits.float().softmax(-1)*aux.coeff_bins).sum(-1)*aux.coeff_scales[d])
    means=torch.stack(means,-2)
    joint_prediction=(weights[...,None]*vectors*means[...,None]).sum(-2)
    target=aux.dictionary.t()[atoms]*physical[...,None]
    def geometry(prediction):
        return float(.5*((prediction-target).square().mean()/target.square().mean()
            +(prediction.sum(-2)-target.sum(-2)).square().mean()/target.sum(-2).square().mean()))
    actual_prob=out['atom_logits'].float().softmax(-1)
    mass=actual_prob.gather(-1,candidates[...,:4]).sum(-1)
    wrong=candidates != atoms[...,None]
    return {'archived_geometry':geometry(old_prediction), 'candidate_conditional_geometry':geometry(joint_prediction),
        'prediction_difference_relative_energy':float((old_prediction-joint_prediction).square().mean()/target.square().mean()),
        'candidate_coefficient_mean_minus_gt_conditioned_mae':float((means-old_coeff[...,None]).abs()[wrong].mean()),
        'candidate_coefficient_mean_sign_disagreement':float(((means>=0)!=(old_coeff[...,None]>=0))[wrong].float().mean()),
        'top4_atom_probability_mass':float(mass.mean()),
        'gt_atom_in_top4_fraction':float((candidates[...,:4]==atoms[...,None]).any(-1).float().mean()),
        'candidate_approximation':'both comparisons retain archived top4 + ground-truth candidate renormalization'}


def logit_gradient_audit(model,aux,atoms,physical):
    with torch.no_grad():
        packed,probabilities=targets(aux,atoms,physical,stochastic=False)
        out=model(packed,model_aux=aux,amp=False)
    a=out['atom_logits'].detach().requires_grad_()
    c=out['coeff_logits'].detach().requires_grad_()
    _,values=compound_objective(a,c,None,atoms,probabilities,aux.dictionary.t()[atoms]*physical[...,None],
        atom_weight=1.5,geometry_weight=.05,accumulation=1,distribution_geometry=True,
        geometry_dictionary=aux.dictionary,geometry_coeff_bins=aux.coeff_bins,geometry_coeff_scales=aux.coeff_scales,geometry_top_k=4)
    ce_grad=torch.autograd.grad(values['classification'],(a,c),retain_graph=True)
    geo_grad=torch.autograd.grad(.05*values['geometry'],(a,c))
    return {name:{'weighted_geometry_over_ce_gradient_norm':float(g.norm()/e.norm()),
        'gradient_cosine':float(F.cosine_similarity(e.flatten(),g.flatten(),dim=0))}
        for name,e,g in zip(('atom','coefficient'),ce_grad,geo_grad)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.serialization.add_safe_globals([codecs.encode]);torch.manual_seed(64011)
    snapshot=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=snapshot['config']
    raw=torch.load(config['cache'],map_location='cpu',weights_only=False,mmap=True)
    data,scales=full_training_cache(raw)
    model=make_prior('control').cuda().eval().requires_grad_(False)
    model.load_state_dict(snapshot['state_dict'],strict=True)
    aux=RelativeChurchAux(Path(config['stage1']),16384,2048,3.,coeff_scales=scales,sparsity_level=4,
        sigma_cap=config['coefficient_noise_sigma_cap'],relative_sigma=config['relative_sigma'],truncate=config['truncate']).cuda().eval().requires_grad_(False)
    assert raw['meta']['checkpoint_sha256']==config['stage1_sha256']==sha256_file(Path(config['stage1']))
    assert torch.equal(raw['dictionary'],aux.dictionary.cpu())
    result={'checkpoint_sha256':sha256_file(args.checkpoint),'step':snapshot['step'],
        'stage1_sha256':config['stage1_sha256'],'source_hashes_match':all(sha256_file(ROOT/n)==h for n,h in config['source_hashes'].items()),
        'cache_dictionary_exact_match':True,'scope':'read-only trained-checkpoint audit; live training unchanged'}
    del snapshot
    atoms=data['validation']['atoms'][:2].cuda().long()
    physical=data['validation']['coefficients'][:2].cuda()
    packed,_=targets(aux,atoms,physical,stochastic=False)
    a,c=model.unpack(packed)
    assert torch.equal(a,atoms)
    assert int(packed.max())<16384*2048 and int(packed.min())>=0
    direct=aux.dictionary.t()[a]*(aux.coeff_bins[c]*aux.coeff_scales)[...,None]
    with torch.no_grad():
        embedded=aux.compound_embeddings(a,c)
        decoded=aux.decoder(aux.post_quant_conv(direct.sum(-2).permute(0,3,1,2).contiguous())).clamp(-1,1)
        aux_decoded=aux.decode_compound(a,c)
        continuous=(aux.dictionary.t()[a]*physical[...,None]).sum(-2)
        clean=aux.decoder(aux.post_quant_conv(continuous.permute(0,3,1,2).contiguous())).clamp(-1,1)
        save_image((torch.stack([clean,decoded],1).flatten(0,1)+1)/2,args.output/'token-roundtrip.png',nrow=2)
        result['token_roundtrip']={'atoms_equal':True,'contribution_max_error':float((direct-embedded).abs().max()),
            'decoder_max_error':float((decoded-aux_decoded).abs().max()),
            'nearest_bin_relative_latent_mse':float((direct.sum(-2)-continuous).square().mean()/continuous.square().mean())}
        print(json.dumps({'phase':'token_roundtrip',**result['token_roundtrip']}),flush=True)
        generated=torch.load(ROOT/'outputs/church-relative-noise-20260911/relative/evaluations/step-004930/full/generated-codes.pt',map_location='cpu',weights_only=False)
        generation_tokens=generated['atoms'][:1].long()*2048+generated['coefficient_ids'][:1].long()
        packed=torch.cat([packed[:1],generation_tokens.cuda()])
        del generated
        torch.backends.cuda.matmul.allow_tf32=False
        teacher=model(packed,model_aux=aux,amp=False)
        cached=cached_logits(model,aux,packed,amp=False)
        result['strict_fp32_cache']={k:comparison(cached[k],teacher[k]) for k in teacher}
        assert all(v['max_logit_error']<.002 for v in result['strict_fp32_cache'].values())
        # Poison current coefficient and all later tokens. Earlier outputs and
        # the current pair's prediction must be unchanged; current atom stays.
        changed=packed.clone().reshape(2,-1);query=81
        changed[:,query+1:]=torch.randint(16384*2048,changed[:,query+1:].shape,device='cuda')
        changed[:,query]=(changed[:,query]//2048)*2048+(changed[:,query]+713)%2048
        poisoned=model(changed.reshape_as(packed),model_aux=aux,amp=False)
        result['future_poison_max_error']={k:float((v.flatten(1,3)[:,:query+1]-poisoned[k].flatten(1,3)[:,:query+1]).abs().max()) for k,v in teacher.items()}
        assert all(v==0 for v in result['future_poison_max_error'].values())
        # Changing either part of a completed previous event must affect the next.
        result['previous_pair_effect']={}
        for name,increment in [('support',2048),('coefficient',137)]:
            changed=packed.clone().reshape(2,-1)
            if name=='support':changed[:,query-1]=((changed[:,query-1]//2048+1)%16384)*2048+changed[:,query-1]%2048
            else:changed[:,query-1]=(changed[:,query-1]//2048)*2048+(changed[:,query-1]+increment)%2048
            changed_out=model(changed.reshape_as(packed),model_aux=aux,amp=False)
            result['previous_pair_effect'][name]=float((changed_out['atom_logits'].flatten(1,3)[:,query]-teacher['atom_logits'].flatten(1,3)[:,query]).abs().max())
            assert result['previous_pair_effect'][name]>0
        torch.backends.cuda.matmul.allow_tf32=True
        teacher=model(packed,model_aux=aux,amp=False)
        cached=cached_logits(model,aux,packed,amp=True)
        result['production_amp_cache']={k:comparison(cached[k],teacher[k]) for k in teacher}
        print(json.dumps({'phase':'cache_and_causality', **{k:result[k] for k in ['strict_fp32_cache','production_amp_cache','future_poison_max_error','previous_pair_effect']}}),flush=True)
        result['geometry']={}
        for split in ('train_probe','validation'):
            a=data[split]['atoms'][:32].cuda().long();c=data[split]['coefficients'][:32].cuda()
            result['geometry'][split]=geometry_audit(model,aux,a,c)
            print(json.dumps({'phase':'geometry','split':split,**result['geometry'][split]}),flush=True)
    result['geometry_logit_gradients']=logit_gradient_audit(model,aux,atoms,physical)
    result['frozen_parameters_have_no_grad']=all(v.grad is None for v in (*model.parameters(),*aux.parameters()))
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'phase':'complete','geometry_logit_gradients':result['geometry_logit_gradients']}),flush=True)


if __name__=='__main__':main()
