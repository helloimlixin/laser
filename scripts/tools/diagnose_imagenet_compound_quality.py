"""Trained-checkpoint diagnostics and matched generation measurements."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.training import rqtransformer as training
from src.training.imagenet_compound_runtime import make_aux
from src.training.rq_imagenet_runtime import TarImageNet, write_json

BASE=Path('/tmp/laser-imagenet-compound512-audit-20261026')
POLICIES={
    'current':dict(atom_temperature=1.,atom_top_k=16384,atom_top_p=.95,
                   coeff_temperature=1.,coeff_top_k=0,coeff_top_p=.95),
    'prior_validated':dict(atom_temperature=.9,atom_top_k=16384,atom_top_p=.9,
                          coeff_temperature=1.,coeff_top_k=0,coeff_top_p=.85),
    'lower_atom_temperature':dict(atom_temperature=.8,atom_top_k=16384,atom_top_p=.9,
                                 coeff_temperature=1.,coeff_top_k=0,coeff_top_p=.85),
}


def load(checkpoint,device):
    p=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    c=p['config']
    with torch.device('meta'):
        model=training.build_model(16896,16384,coeff_vocab_size=512,sparsity_level=4,
            physical_pair_context=True,efficient_pair_backbone=True,
            compound_micro_transformer_layers=2,compound_depth_specific_coeff_heads=True,
            coefficient_history_layers=2,coefficient_history_width=512,coefficient_history_heads=8,
            coefficient_history_normalize_fields=True)
    model.load_state_dict(p['state_dict'],strict=True,assign=True)
    model=model.to(device).eval()
    metadata=dict(step=p['global_step'],fid=p.get('fid'),inception_score=p.get('inception_score'),config=c)
    del p
    return model,make_aux(BASE/'inputs/stage1-tokenizer.pt',device),metadata


def field_losses(output,atoms,ids):
    a=-output['atom_logits'].float().log_softmax(-1).gather(-1,atoms[...,None]).squeeze(-1)
    c=-output['coeff_logits'].float().log_softmax(-1).gather(-1,ids[...,None]).squeeze(-1)
    return a,c


@torch.inference_mode()
def diagnose(model,aux,metadata,device,out):
    from torchvision.utils import save_image
    dataset=TarImageNet(BASE/'data','val',training.val_image_transform())
    images,labels=zip(*[dataset[i] for i in range(0,1600,100)])
    sums={}; pictures=[]; first=None
    for start in range(0,len(images),4):
        x=torch.stack(images[start:start+4]).to(device)
        cond=torch.tensor(labels[start:start+4],device=device)
        atoms,values=aux.encode_sparse_components(x)
        tokens,_=aux.sparse_targets(atoms,values,hard=True,stochastic=False,compact=True)
        ids=tokens[...,1::2]-16384
        with torch.autocast('cuda',dtype=torch.bfloat16):
            correct=model(tokens,aux,cond)
            original=model.refine_coefficient_hidden
            def shuffled(hidden,vectors):
                changed=vectors.reshape(-1,vectors.shape[-1]).roll(53,0).reshape_as(vectors)
                return original(hidden,changed)
            model.refine_coefficient_hidden=shuffled
            try: wrong_atom=model(tokens,aux,cond)
            finally:model.refine_coefficient_hidden=original
            wrong_class=model(tokens,aux,(cond+499)%1000)
        ac,cc=field_losses(correct,atoms,ids)
        wa,wc=field_losses(wrong_atom,atoms,ids)
        ca,cc_wrong_class=field_losses(wrong_class,atoms,ids)
        probabilities=correct['coeff_logits'].float().softmax(-1)
        correct_sign=(aux.coeff_bins>0)==(values[...,None]>0)
        metrics=dict(atom_ce=ac.mean((0,1,2)),coefficient_ce=cc.mean((0,1,2)),
            coefficient_ce_current_atom_shuffle_increase=(wc-cc).mean((0,1,2)),
            atom_ce_current_atom_shuffle_increase=(wa-ac).mean((0,1,2)),
            atom_ce_wrong_class_increase=(ca-ac).mean((0,1,2)),
            coefficient_sign_probability=(probabilities*correct_sign).sum(-1).mean((0,1,2)))
        for key,value in metrics.items():sums[key]=sums.get(key,0)+value.cpu()/4
        argmax=correct['coeff_logits'].argmax(-1)
        sampled=torch.multinomial(probabilities.reshape(-1,512),1).reshape_as(ids)
        real=aux.decode_tokens(tokens)
        predicted=aux.decode_compound(atoms,argmax)
        random=aux.decode_compound(atoms,sampled)
        pictures.append(torch.stack((x,real,predicted,random),dim=1).flatten(0,1).cpu())
        if first is None:first=(tokens[:1].clone(),cond[:1].clone())
    save_image(torch.cat(pictures).mul(.5).add(.5),out/'real-reconstruction-oracle-atom-predicted-coefficients.png',nrow=4)
    target,cond=first
    with torch.autocast('cuda',dtype=torch.bfloat16):expected=model(target,aux,cond)
    generated=torch.zeros_like(target);generated[...,1::2]=16384+256
    model.init_cache();errors={'atom':[],'coefficient':[]}
    for h in range(8):
        for w in range(8):
            for event in range(8):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model.cached_forward(generated,aux,cond,amp=False,sample_loc=(h,w,event))
                field='coefficient' if event%2 else 'atom'
                actual=logits[:,16384:] if event%2 else logits[:,:16384]
                reference=expected['coeff_logits' if event%2 else 'atom_logits'][:,h,w,event//2].float()
                q=reference.softmax(-1);logp=actual.float().log_softmax(-1)
                kl=torch.where(q>0,q*(q.clamp_min(1.e-30).log()-logp),0.).sum(-1)
                errors[field].append(float(kl.max()))
                generated[:,h,w,event]=target[:,h,w,event]
    model.init_cache()
    parity={k:max(v) for k,v in errors.items()}
    report=dict(checkpoint_step=metadata['step'],images=16,
        metrics={k:v.tolist() for k,v in sums.items()},
        trained_cache_maximum_kl=parity,cache_parity_passed=max(parity.values())<.005,
        reconstruction_grid_columns=['real','quantized reconstruction','oracle atoms + teacher-forced coefficient argmax','oracle atoms + teacher-forced sampled coefficients'],
        limitations='Small fixed validation probe. Oracle atoms and true prefixes isolate coefficient prediction; these are not free generation or FID scores.')
    write_json(out/'trained-diagnostics.json',report)
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['diagnose','evaluate'])
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=5000);p.add_argument('--policy',choices=POLICIES,default='current')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);local=int(os.environ.get('LOCAL_RANK','0'))
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=True
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if 'WORLD_SIZE' in os.environ:dist.init_process_group('nccl',device_id=device)
    model,aux,metadata=load(args.checkpoint,device)
    if args.mode=='diagnose':diagnose(model,aux,metadata,device,args.output)
    else:
        started=time.monotonic()
        fid,is_mean,is_std=training.evaluate_generation_metrics(model,aux,None,args.samples,64,
            fid_seed=261027,metric_backend='original-rqvae',
            fid_reference_stats=BASE/'inputs/imagenet_256_train.npz',**POLICIES[args.policy])
        if not dist.is_initialized() or dist.get_rank()==0:
            report=dict(policy=args.policy,sampling=POLICIES[args.policy],checkpoint_step=metadata['step'],
                samples=args.samples,fid=fid,inception_score=is_mean,inception_std=is_std,
                stored_baseline_fid50k=metadata['fid'],stored_baseline_inception=metadata['inception_score'],
                same_checkpoint=True,fid_seed=261027,reference_images=1281167,seconds=time.monotonic()-started)
            write_json(args.output/'result.json',report);print(json.dumps(report),flush=True)
            if args.samples==50000:
                with torch.random.fork_rng(devices=[device.index]):
                    torch.manual_seed(261026)
                    training.sample_class_grid(model,aux,training.class_names_for_dataset('imagenet'),
                        args.output/'preview',metadata['step'],num_samples=64,sample_batch_size=8,
                        samples_per_class=8,setting_name=args.policy,**POLICIES[args.policy])
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
