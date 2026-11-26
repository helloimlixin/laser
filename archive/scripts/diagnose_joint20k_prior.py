#!/usr/bin/env python3
"""Read-only, matched-code probes of context use and prior calibration."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--images', type=int, default=64)
    parser.add_argument('--generated', type=int, default=512)
    args = parser.parse_args()
    if args.images < 16 or args.images % 4 or args.generated < 32:
        parser.error('Use a multiple of four >=16 probe images and >=32 generated images')
    base, out = args.base.resolve(), args.output
    os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')
    sys.path[:0] = [str(base/'source'), str(base/'upstream'), str(base/'runtime'), str(base)]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config

    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(12*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    torch.manual_seed(261018031)
    profile = JointLaserRecipe(base)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    profile.verify_resume(saved)
    step = int(saved['global_step'])
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model = model.requires_grad_(False).eval().to(device)
    profile.adapt(model, None)
    del saved
    aux, _ = profile.tokenizer(device)
    depth, vocab = aux.quantizer.depth, aux.quantizer.vocab_size
    config = load_config(base/'inputs/stage2-training-config.yaml')
    data = TarImageNet(base/'data','val',create_transforms(config.dataset,split='val'))
    indices = np.random.default_rng(261018031).permutation(len(data))[:args.images].tolist()
    temperatures = [.7,.85,1.,1.15,1.3]
    nll = torch.zeros(2,len(temperatures),depth,dtype=torch.float64,device=device)
    counts = torch.zeros(2,dtype=torch.float64,device=device)
    diagnostics = {name:torch.zeros(depth,dtype=torch.float64,device=device) for name in
        ['nll','wrong_class_nll','shuffled_prefix_nll','entropy','accuracy','confidence','input_rms','output_rms']}
    real_hist = torch.zeros(depth,vocab,dtype=torch.float64,device=device)
    null_mass = torch.zeros(depth,dtype=torch.float64,device=device)
    original_mass = torch.zeros(depth,4,dtype=torch.float64,device=device)
    tokens = 0

    def input_hook(module, inputs):
        x = inputs[0].float()
        diagnostics['input_rms'].add_(x.square().mean((0,2)).sqrt()*len(x))

    def output_hook(module, inputs, output):
        x = output.float()
        diagnostics['output_rms'].add_(x.square().mean((0,2)).sqrt()*len(x))

    def per_depth_nll(logits, codes):
        return -logits.float().log_softmax(-1).gather(-1,codes[...,None]).squeeze(-1).sum((0,1,2))

    state = dict(run_id=profile.run_id, checkpoint_global_step=step, images=args.images,
        val_indices=indices, training_modified=False, seed=261018031,
        limits='Small diagnostic on common deterministic hard codes; neither a generation score nor a causal explanation.')
    write(out/'status.json',dict(phase='probing',pid=os.getpid(),time=time.time(),**state))
    with torch.inference_mode():
        for first in range(0,args.images,2):
            records = [data[i] for i in indices[first:first+2]]
            images = torch.stack([r[0] for r in records]).to(device)
            labels = torch.tensor([r[1] for r in records],device=device)
            codes = aux.quantizer.quantize(aux.encode(images))['codes']
            ih = model.head_transformer.register_forward_pre_hook(input_hook)
            oh = model.head_transformer.register_forward_hook(output_hook)
            logits = model(codes,aux,labels,amp=True)
            ih.remove(); oh.remove()
            logp = logits.float().log_softmax(-1)
            p = logp.exp()
            diagnostics['nll'] += per_depth_nll(logits,codes)
            diagnostics['entropy'] += -(p*logp).sum(-1).sum((0,1,2))
            diagnostics['accuracy'] += (logits.argmax(-1)==codes).sum((0,1,2))
            diagnostics['confidence'] += p.max(-1).values.sum((0,1,2))
            null_mass += p[...,0].sum((0,1,2))
            for d in range(depth):
                real_hist[d] += torch.bincount(codes[...,d].flatten(),minlength=vocab)
            ordered = logits.float().sort(-1,descending=True).values.softmax(-1)
            for j,k in enumerate([256,1024,2048,vocab]):
                original_mass[:,j] += ordered[...,:k].sum(-1).sum((0,1,2))
            split = int(first >= args.images//2)
            for j,t in enumerate(temperatures):
                nll[split,j] += per_depth_nll(logits/t,codes)
            counts[split] += len(images)*64
            tokens += len(images)*64
            del logits,logp,p,ordered
            wrong = model(codes,aux,(labels+501)%1000,amp=True)
            diagnostics['wrong_class_nll'] += per_depth_nll(wrong,codes)
            del wrong
            shuffled = model(codes.flip(0),aux,labels,amp=True)
            diagnostics['shuffled_prefix_nll'] += per_depth_nll(shuffled,codes)
            del shuffled,images,codes
        nll = nll/counts[:,None,None]
        selected = nll[0].argmin(0)
        selected_temperatures = [temperatures[int(j)] for j in selected.cpu()]
        heldout_nll = nll[1].gather(0,selected[None]).squeeze(0)
        calibration = dict(temperatures=temperatures,fit_nll_by_temperature=nll[0].cpu().tolist(),
            validation_nll_by_temperature=nll[1].cpu().tolist(),
            selected_temperatures=selected_temperatures,
            selected_validation_nll=heldout_nll.cpu().tolist(),
            unit_temperature_validation_nll=nll[1,2].cpu().tolist(),
            heldout_nll_improvement_over_unit=(nll[1,2]-heldout_nll).cpu().tolist(),
            selection_uses_validation=False,generation_improvement_established=False)
        rows=[]
        for d in range(depth):
            row=dict(depth=d+1,**{name:float(value[d]/tokens) for name,value in diagnostics.items()},
                null_probability=float(null_mass[d]/tokens),
                top_k_original_mass={str(k):float(original_mass[d,j]/tokens)
                                    for j,k in enumerate([256,1024,2048,vocab])})
            row['class_context_gain']=row['wrong_class_nll']-row['nll']
            row['prefix_context_gain']=row['shuffled_prefix_nll']-row['nll']
            rows.append(row)
        write(out/'teacher-probes.json',dict(**state,per_depth=rows,calibration=calibration))
        write(out/'status.json',dict(phase='measuring_generated_codes',pid=os.getpid(),time=time.time(),**state))
        generated_hist=torch.zeros_like(real_hist)
        torch.manual_seed(261018032)
        for first in range(0,args.generated,16):
            labels=torch.arange(first,min(first+16,args.generated),device=device)%1000
            codes=profile.sample(model,aux,len(labels),labels)
            for d in range(depth):
                generated_hist[d] += torch.bincount(codes[...,d].flatten(),minlength=vocab)
        margins=[]
        for d in range(depth):
            a=real_hist[d]/real_hist[d].sum()
            b=generated_hist[d]/generated_hist[d].sum()
            m=(a+b)/2
            js=.5*(a*(a.clamp_min(1.e-30).log()-m.clamp_min(1.e-30).log())).sum()
            js+=.5*(b*(b.clamp_min(1.e-30).log()-m.clamp_min(1.e-30).log())).sum()
            margins.append(dict(depth=d+1,code_js_divergence=float(js),
                real_null_fraction=float(a[0]),generated_null_fraction=float(b[0]),
                real_marginal_entropy=float(-(a*a.clamp_min(1.e-30).log()).sum()),
                generated_marginal_entropy=float(-(b*b.clamp_min(1.e-30).log()).sum())))
        result=dict(**state,completed=True,per_depth=rows,calibration=calibration,
            generated_images=args.generated,generated_code_marginals=margins,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            generated_marginal_limit='Real sample is small and uses hard codes; JS includes sample-size bias.',time=time.time())
        write(out/'report.json',result)
        write(out/'status.json',dict(phase='completed',pid=os.getpid(),time=time.time()))
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
