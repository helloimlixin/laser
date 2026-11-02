"""Trained-model parity, search-cost measurements, and fixed-seed joint previews."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from src.joint_pair_topk import joint_pair_topk
from src.training import rqtransformer as training
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from experiment_coefficient_mixture import load_base
from src.training.imagenet_compound_runtime import make_aux

BASE = Path('/tmp/laser-imagenet-compound512-audit-20261026')


def history_state(model):
    return (model._offset_hidden, model._offset_location, model._offset_coefficients,
            model._coefficient_sample_context, model.coefficient_history_decoder._next_event,
            [block._kv for block in model.coefficient_history_decoder.blocks])


def restore_history(model, state):
    (model._offset_hidden, model._offset_location, model._offset_coefficients,
     model._coefficient_sample_context, model.coefficient_history_decoder._next_event, caches) = state
    for block, cache in zip(model.coefficient_history_decoder.blocks, caches, strict=True):
        block._kv = cache


@torch.no_grad()
def benchmark(model, aux, device, output, chunk):
    dataset = TarImageNet(BASE/'data', 'val', training.val_image_transform())
    image, label = dataset[1701]
    atoms, values = aux.encode_sparse_components(image[None].to(device))
    target, _ = aux.sparse_targets(atoms, values, hard=True, stochastic=False, compact=True)
    cond = torch.tensor([label], device=device)
    tokens = target.clone()
    rows = torch.zeros(1, dtype=torch.long, device=device)
    report = dict(searches=[], parity=[], parameter_count=sum(p.numel() for p in model.parameters()))
    model.init_cache()
    try:
        for h in range(8):
            for w in range(8):
                for depth in range(4):
                    event = (h*8+w)*4+depth
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model.cached_forward(tokens, aux, cond, amp=True,
                            sample_loc=(h,w,2*depth))[:, :16384]
                        if event in [0,1,2,3,255]:
                            scorer = model.joint_coefficient_scorer(aux)
                            candidates = logits.topk(4,-1).indices
                            actual = scorer(rows, candidates).float()
                            state = history_state(model)
                            maximum = 0.
                            for i in range(candidates.shape[1]):
                                branch = tokens.clone()
                                branch[:,h,w,2*depth] = candidates[:,i]
                                expected = model.cached_forward(branch, aux, cond, amp=True,
                                    sample_loc=(h,w,2*depth+1))[:,16384:].float()
                                q = expected.softmax(-1)
                                error = (q*(expected.log_softmax(-1)-actual[:,i].log_softmax(-1))).sum(-1)
                                maximum = max(maximum,float(error.max()))
                                restore_history(model,state)
                            report['parity'].append(dict(event=event, maximum_kl=maximum))
                            write_json(output/'benchmark.json',report)
                            if maximum >= .005:
                                raise RuntimeError(f'Candidate/native conditional mismatch: {maximum}')
                            for k in ([256,4096,16384] if event in [0,3] else [4096]):
                                torch.cuda.synchronize(device)
                                started = time.monotonic()
                                scores, ids, stats = joint_pair_topk(logits,scorer,top_k=k,candidate_chunk=chunk)
                                torch.cuda.synchronize(device)
                                result = dict(event=event,top_k=k,seconds=time.monotonic()-started,
                                    evaluated_atoms=int(stats['evaluated_atoms'].max()),
                                    retained_untempered_mass=float(stats['retained_mass'].mean()),
                                    certified=stats['certified'])
                                report['searches'].append(result)
                                write_json(output/'benchmark.json',report)
                                print(json.dumps(result),flush=True)
                        model.cached_forward(tokens,aux,cond,amp=True,sample_loc=(h,w,2*depth+1))
    finally:
        model.init_cache()
    report['passed'] = max(r['maximum_kl'] for r in report['parity']) < .005
    write_json(output/'benchmark.json',report)


@torch.no_grad()
def preview(model, aux, metadata, device, args):
    settings = (dict(joint_pair_top_k=args.top_k, joint_pair_temperature=args.temperature,
        joint_pair_top_p=args.top_p, joint_pair_candidate_chunk=args.chunk) if args.top_k else
        dict(atom_temperature=.8,atom_top_k=16384,atom_top_p=.9,
             coeff_temperature=1.,coeff_top_k=0,coeff_top_p=.85))
    name = f'joint_t{args.temperature:g}_k{args.top_k}_p{args.top_p:g}' if args.top_k else 'separate_fields'
    started = time.monotonic()
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(261026)
        path = training.sample_class_grid(model,aux,training.class_names_for_dataset('imagenet'),
            args.output,metadata['source_step'],num_samples=args.samples,
            sample_batch_size=args.batch_size,samples_per_class=args.samples_per_class,
            class_ids=[933,849,258,22,277,926,106,269],
            setting_name=name,**settings)
    write_json(args.output/'preview-result.json',dict(path=str(path),source_step=metadata['source_step'],
        seconds=time.monotonic()-started,samples=args.samples,seed=261026,sampling=settings,
        certified_global_top_k=bool(args.top_k),weights_changed=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['benchmark','preview'])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--top-k',type=int,default=4096)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--samples-per-class',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--compile-candidates',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device('cuda',0);torch.cuda.set_device(device)
    # The read-only probe shares one GPU with training. Bound only this
    # process's allocator; it must not exhaust the production worker's memory.
    torch.cuda.set_per_process_memory_fraction(.18,device)
    torch.set_float32_matmul_precision('high')
    model,metadata=load_base(args.checkpoint,device)
    aux=make_aux(BASE/'inputs/stage1-tokenizer.pt',device)
    if args.compile_candidates:
        from src.models.rqtransformer.attentions import AttentionBlock
        for block in model.coeff_micro_transformer.modules():
            if isinstance(block,AttentionBlock):
                block.attn.short_attention_backend='compiled'
                block.forward=torch.compile(block.forward,fullgraph=True,dynamic=True)
        for block in model.coefficient_history_decoder.blocks:
            block.forward_candidates=torch.compile(block.forward_candidates,fullgraph=True,dynamic=True)
    write_json(args.output/'started.json',dict(pid=os.getpid(),mode=args.mode,source=metadata,
        gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),compile_candidates=args.compile_candidates,
        memory_fraction=.18,training_weights_changed=False))
    if args.mode=='benchmark':benchmark(model,aux,device,args.output,args.chunk)
    else:preview(model,aux,metadata,device,args)


if __name__=='__main__':main()
