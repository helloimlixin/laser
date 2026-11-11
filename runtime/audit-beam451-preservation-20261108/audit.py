"""Audit the running frozen codec without loading another prior onto the GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

BASE=Path('/tmp/laser-beam451-stage2-4h100-20261107')
OUT=Path('/workspace/Projects/laser/outputs/imagenet-beam451-aligned-joint16k-scratch100-4h100-20261107/audit-stage1-preservation-20261108')
sys.path[:0]=[str(BASE/'source'),str(BASE)]
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
os.environ['LASER_FAST_BEAM_TOPK']='0'
import numpy as np
import torch
from omegaconf import OmegaConf
from runtime.aligned_recipe import AlignedBeamRecipe
from runtime import beam_teacher
from runtime.speed.exact_beam import beam_pool as fast_pool
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.img_datasets.transforms import create_transforms


def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    tmp=OUT/(name+'.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(OUT/name)


@torch.no_grad()
def codec(images_per_split):
    # The training process retains its allocation; this audit has a hard cap.
    torch.cuda.set_per_process_memory_fraction(.035,0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(4)
    torch.manual_seed(26110845)
    profile=AlignedBeamRecipe(BASE)
    hashes={}
    for name,expected in json.loads((BASE/'prepared.json').read_text())['hashes'].items():
        with (BASE/'inputs'/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,name
        hashes[name]=actual
    aux,_=profile.tokenizer('cuda:0')
    assert not aux.training and all(not p.requires_grad for p in aux.parameters())
    before=profile.digest(aux)
    for depth,table in enumerate(aux.quantizer.codebooks):
        torch.testing.assert_close(table.weight[:-1],aux.physical_book.expanded_codebook(depth),rtol=0,atol=0)
        assert not table.training
    config=OmegaConf.load(BASE/'inputs/stage2-training-config.yaml')
    saved=[]; results={}; start=time.monotonic()
    for split in ['val','train']:
        data=TarImageNet(BASE/'data',split,create_transforms(config.dataset,split=split))
        indices=np.random.default_rng(26110845).choice(len(data),images_per_split,replace=False)
        totals={name:dict(pixel_mse=0.,latent_mse=0.) for name in ['canonical_beam','aligned_teacher','greedy','native_soft05']}
        expected_excess=[]; sampled_excess=[]; entropy=[]; canonical_mass=[]
        max_valid_excess=0.; code_differences=0
        for i,index in enumerate(indices):
            image,label=data[int(index)]; image=image[None].cuda()
            latent=aux.encode(image)
            pool=fast_pool(aux.physical_book,latent)
            if i<2:
                native=beam_teacher.beam_pool(aux.physical_book,latent)
                for key in ['candidates','costs','canonical_index','valid']:
                    torch.testing.assert_close(native[key],pool[key],rtol=0,atol=0)
                del native
            weights=beam_teacher.posterior(pool,profile.target_temperature())
            row=torch.arange(len(weights),device='cuda')
            canonical=pool['candidates'][row,pool['canonical_index']].reshape(1,8,8,4)
            chosen=torch.multinomial(weights,1).squeeze(-1)
            codes=pool['candidates'][row,chosen]
            targets=beam_teacher.conditional_targets(pool['candidates'],weights,codes,16384)
            torch.testing.assert_close(targets.sum(-1),torch.ones_like(targets[...,0]),rtol=2e-5,atol=2e-5)
            selected=targets.gather(-1,codes[...,None]).squeeze(-1).prod(-1)
            # Sum duplicate trajectories before comparing sequence probability.
            sequence_mass=(weights*(pool['candidates']==codes[:,None]).all(-1)).sum(-1)
            torch.testing.assert_close(selected,sequence_mass,rtol=2e-5,atol=2e-6)
            beam_cost=pool['costs'][row,pool['canonical_index']]
            delta=(pool['costs']-beam_cost[:,None]).clamp_min(0)
            assert ((delta<=.01*beam_cost[:,None].clamp_min(1e-8)+1e-6)|(weights==0)).all()
            relative=delta/beam_cost[:,None].clamp_min(1e-8)
            expected_excess.extend((relative*weights).sum(-1).cpu().tolist())
            sampled_excess.extend(relative[row,chosen].cpu().tolist())
            max_valid_excess=max(max_valid_excess,float(relative.masked_fill(weights==0,0).max()))
            entropy.extend((-(weights*weights.clamp_min(1e-30).log()).sum(-1)/4).cpu().tolist())
            canonical_mass.extend(weights[row,pool['canonical_index']].cpu().tolist())
            actual=aux.quantize_reconstruction(latent)['codes']
            assert torch.equal(canonical,actual)
            teacher=codes.reshape(1,8,8,4)
            code_differences+=int((teacher!=canonical).any(-1).sum())
            greedy=aux.physical_book._quantize_greedy(latent)['codes']
            _,soft=aux.quantizer.get_soft_codes(latent,temp=.5,stochastic=True)
            for name,tokens in [('canonical_beam',canonical),('aligned_teacher',teacher),('greedy',greedy),('native_soft05',soft)]:
                physical=aux.physical_book.embed(tokens).sum(-2)
                embedded=aux.quantizer.embed_code(tokens)
                torch.testing.assert_close(physical,embedded,rtol=0,atol=0)
                depth_values,_=aux.get_code_emb_with_depth(tokens)
                torch.testing.assert_close(depth_values,aux.physical_book.embed(tokens),rtol=0,atol=0)
                pixels=aux.decode_code(tokens).clamp(-1,1)
                assert pixels.isfinite().all()
                totals[name]['pixel_mse']+=float((pixels-image).square().mean())/4
                totals[name]['latent_mse']+=float((embedded-latent).square().mean())
            if split=='val' and len(saved)<4:saved.append(dict(codes=canonical.cpu(),label=label))
            if (i+1)%16==0:
                status=dict(phase='codec',split=split,images=i+1,elapsed_seconds=time.monotonic()-start,
                            peak_gib=torch.cuda.max_memory_allocated()/1024**3)
                write('progress.json',status);print(json.dumps(status),flush=True)
            del pool,weights,targets,image,latent
        for values in totals.values():
            for key in values:values[key]/=images_per_split
        results[split]=dict(images=images_per_split,reconstruction=totals,
            posterior_expected_relative_latent_error=float(np.mean(expected_excess)),
            sampled_relative_latent_error=float(np.mean(sampled_excess)),
            max_candidate_relative_latent_error=max_valid_excess,
            conditional_entropy_per_depth=float(np.mean(entropy)),
            mean_canonical_posterior_mass=float(np.mean(canonical_mass)),
            changed_site_fraction=code_differences/(images_per_split*64))
    assert profile.digest(aux)==before
    torch.save(saved,OUT/'probe-codes.pt')
    result=dict(passed=True,hashes=hashes,frozen_tokenizer_digest=before,frozen_tokenizer_unchanged=True,
        all_65536_physical_codewords_bitwise_equal=True,canonical_search_and_integer_roundtrips_passed=True,
        exact_prefix_conditional_probability_passed=True,teacher_candidate_error_bound_passed=True,
        precision='FP32 encoder, codebook, decoder; TF32 disabled for matrix multiplication',
        pixel_mse_units='RGB [0,1]',
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,results=results,
        limitation='Subset paired errors; this does not remeasure full-validation rFID or prove generation quality.')
    write('codec-audit.json',result);print(json.dumps(result),flush=True)


def prior():
    torch.set_num_threads(8)
    checkpoint=json.loads((BASE/'checkpoints/latest.json').read_text())
    path=Path(checkpoint['path'])
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    from src.rq_upstream import RQTransformer
    with torch.device('meta'):
        model=RQTransformer(OmegaConf.create(state['config']['arch']))
    model.load_state_dict(state['state_dict'],assign=True,strict=True)
    model.eval().requires_grad_(False)
    step=state['global_step'];del state
    profile=AlignedBeamRecipe(BASE);aux,_=profile.tokenizer('cpu')
    saved=torch.load(OUT/'probe-codes.pt',weights_only=True)
    codes=torch.cat([s['codes'] for s in saved]);labels=torch.tensor([s['label'] for s in saved])
    with torch.no_grad():
        full=model(codes,model_aux=aux,cond=labels,amp=False)
        wrong=model(codes,model_aux=aux,cond=(labels+1)%1000,amp=False)
        per_depth=torch.nn.functional.cross_entropy(full.reshape(-1,16384),codes.reshape(-1),reduction='none').reshape(4,8,8,4).mean((0,1,2))
        changed=(full-wrong).abs().mean().item()
        assert changed>0
        # Full teacher forcing against native cached generation at first/middle/last site.
        errors=[]
        for h,w in [(0,0),(3,5),(7,7)]:
            model.init_cache()
            for d in range(4):
                cached=model.cached_forward(codes[:1,:h+1],model_aux=aux,cond=labels[:1],amp=False,sample_loc=(h,w,d))
                expected=full[:1,h,w,d]
                error=float((cached-expected).abs().max());errors.append(error)
                torch.testing.assert_close(cached,expected,rtol=2e-4,atol=2e-4)
            print(json.dumps(dict(phase='cached_parity',site=[h,w],max_error=max(errors))),flush=True)
        perturbed=codes.clone().reshape(4,-1)
        perturbed[:,17:]=(perturbed[:,17:]+7919)%16384
        other=model(perturbed.reshape_as(codes),model_aux=aux,cond=labels,amp=False)
        torch.testing.assert_close(full.reshape(4,256,16384)[:,:18],other.reshape(4,256,16384)[:,:18],rtol=0,atol=0)
        from runtime.training_acceleration import install
        os.environ['LASER_TRAIN_SDPA']='1';os.environ['LASER_SDPA_MIN_SEQ']='16';install()
        with torch.enable_grad():accelerated=model(codes,model_aux=aux,cond=labels,amp=False)
        torch.testing.assert_close(accelerated,full,rtol=2e-4,atol=2e-4)
    result=dict(passed=True,checkpoint=str(path),checkpoint_step=step,actual_full_1387529728_parameter_prior=True,
        probe_images=4,cached_positions_tested=12,max_cached_logit_error=max(errors),
        causal_future_token_invariance=True,sdpa_vs_native_max_logit_error=float((accelerated-full).abs().max()),
        class_label_changes_logits=True,mean_class_label_logit_change=changed,canonical_ce_by_depth=per_depth.tolist(),
        limitation='FP32 CPU parity; four-image CE is diagnostic, not a held-out likelihood benchmark.')
    write('prior-audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['codec','prior']);parser.add_argument('--images-per-split',type=int,default=128)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='codec':codec(args.images_per_split)
    else:prior()
