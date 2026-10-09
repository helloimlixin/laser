"""Fixed training-cache diagnostics; this is not held-out validation.

Restricting trained cross reads is an inference ablation, not a retrained control.
The latest encoded record still contains history through the causal encoder.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/tmp/laser-imagenet-pair-memory-20261007')
OUT = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-investigation-20261007')
PAIR_OUT = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-trial-20261007')
ASSETS = Path('/tmp/laser-imagenet-classcond-20261007')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--branch', choices=('cross', 'mlp', 'baseline'), default='cross')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--memory-only', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', TORCH_HOME=str(ASSETS / 'torch-cache'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE / 'checkpoint-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    sys.path[:0] = [str(BASE / 'source'), str(BASE / 'source/runtime')]
    import numpy as np
    import torch
    from src.training import rqtransformer as training
    from src.training import k4_checkpoint_io as io
    from src.training.pair_memory_cross_attention import attach_pair_memory_queries

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.18, 0)
    cache = torch.load(ASSETS / 'inputs/compound-cache.pt', map_location='cpu', mmap=True, weights_only=False)
    labels = cache['labels'].numpy()
    generator = np.random.default_rng(7319)
    classes = np.linspace(0, 999, args.samples, dtype=int) if args.samples <= 1000 else np.arange(args.samples) % 1000
    indices = np.array([generator.choice(np.flatnonzero(labels == label)) for label in classes])
    atoms = cache['atoms'][indices].long().clone()
    coefficients = cache['coeffs'][indices].clone()
    conditions = cache['labels'][indices].long().clone()
    del cache
    source = ((OUT / 'baseline/train/checkpoints/last.pt') if args.branch == 'baseline'
              else PAIR_OUT / args.branch / 'train/checkpoints/last.pt').resolve()
    local = io._checkpoint_upload_source(source)
    assert local != source
    payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    assert payload['global_step'] == 3798
    with torch.device('meta'):
        model = training.build_model(16384+2048, 16384, compound=True,
            coeff_vocab_size=2048, sparsity_level=4, compound_pair_autoregressive=True,
            model_preset='imagenet-1400m')
        if args.branch != 'baseline':
            attach_pair_memory_queries(model, width=512, heads=8, memory_layers=1,
                                       query_layers=2, mode=args.branch)
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    model = model.requires_grad_(False).eval().to('cuda')
    del payload
    gc.collect()
    aux = training.LaserAux(ASSETS / 'inputs/stage1-tokenizer.pt', 16384, 2048,
        coeff_max=3., coeff_scale=6.4, coeff_scales=[8.203365325927734,4.265638828277588,
        3.0662174224853516,1.8273425102233887], attn_resolutions=(8,),
        soft_target_physical=False, clamp_coeffs=False, sparsity_level=4).requires_grad_(False).eval().to('cuda')

    modes = ('full', 'atom_latest', 'coefficient_latest', 'both_latest') if args.branch == 'cross' else ('full',)
    if args.branch == 'cross' and args.extended:
        modes += ('atom_uniform','coefficient_uniform','both_uniform',
                  'atom_local','coefficient_local','both_local')
    if args.memory_only:
        assert args.branch == 'cross'
        modes = ('full','memory_bos','memory_zero')
    original = {}
    original_encode = model.pair_memory_queries.encode_memory if args.branch == 'cross' else None
    if args.branch == 'cross':
        for head in ('atom', 'coefficient'):
            for index, block in enumerate(getattr(model.pair_memory_queries, head + '_blocks')):
                original[(head, index)] = block.forward

    def restricted_read(block, query, memory, read, *, cached=False):
        assert not cached and query.shape[:2] == memory.shape[:2]
        batch, length, width = query.shape
        keys, values = block.key_value(block.memory_norm(memory)).reshape(
            batch, length, 2, block.heads, width // block.heads).unbind(2)
        if read == 'latest':
            # A diagonal mask has one legal value per query, so softmax=1.
            attended = values.reshape(batch, length, width)
        elif read == 'uniform':
            counts = torch.arange(1,length+1,device=query.device).view(1,length,1,1)
            attended = (values.float().cumsum(1)/counts).to(values.dtype).reshape(batch,length,width)
        else:
            assert read == 'local'
            q = block.query(block.query_norm(query)).reshape(batch,length,block.heads,-1).transpose(1,2)
            t = torch.arange(length,device=query.device)[:,None]
            j = torch.arange(length,device=query.device)[None,:]
            mask = (j==0) | ((j>0) & ((j-1)//4==t//4) & (j<=t))
            attended = torch.nn.functional.scaled_dot_product_attention(
                q, keys.transpose(1,2), values.transpose(1,2),attn_mask=mask).transpose(1,2).reshape(batch,length,width)
        query = query + block.projection(attended)
        return query + block.ffn(block.ffn_norm(query))

    attention = {}
    residuals = []
    scales_by_block = {}
    diagnostic_images = 0

    def residual_summary(module, inputs, outputs):
        hidden = inputs[0].float()
        atom = outputs[0].float()-hidden
        coefficient = outputs[1].float()
        fields = (hidden,module.hidden_projection(inputs[0]).float(),
                  module.prefix_projection(inputs[2]).float(),
                  module.coefficient_atom(module.atom_norm(inputs[1])).float(),
                  atom,coefficient)
        residuals.append(torch.stack([x.square().mean(-1).sqrt().mean(0)
                                     for x in fields],-1).cpu().numpy())
    def attention_summary(block, query, memory, name):
        batch, length, width = query.shape
        q = block.query(block.query_norm(query)).reshape(batch,length,block.heads,-1).transpose(1,2)
        k, v = block.key_value(block.memory_norm(memory)).reshape(
            batch,length,2,block.heads,-1).permute(2,0,3,1,4).unbind(0)
        attended = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        write = block.projection(attended.transpose(1,2).reshape(batch,length,width))
        ffn = block.ffn(block.ffn_norm(query+write))
        uniform = (v.float().cumsum(-2)/torch.arange(1,length+1,device=query.device).view(1,1,length,1)).to(v.dtype)
        uniform_write = block.projection(uniform.transpose(1,2).reshape(batch,length,width))
        centered_values = v.float()-v.float().mean(-2,keepdim=True)
        cosine_bos = torch.nn.functional.cosine_similarity(v[:,:,1:].float(),v[:,:,:1].float(),dim=-1).mean()
        scales_by_block.setdefault(name, []).append(torch.stack([
            x.float().square().mean(-1).sqrt().mean() for x in (query,memory,write,ffn)
        ]+[v.float().square().mean().sqrt(),centered_values.square().mean().sqrt(),
            (write.float()-uniform_write.float()).square().mean().sqrt(),cosine_bos]).cpu().numpy())
        with torch.autocast('cuda', enabled=False):
            scores = (q.float() @ k.float().transpose(-1,-2)) / (width // block.heads) ** .5
            t = torch.arange(length,device=query.device)[:,None]
            j = torch.arange(length,device=query.device)[None,:]
            legal = j <= t
            probs = scores.masked_fill(~legal, -torch.inf).softmax(-1).mean((0,1))
            same = (j>0) & ((j-1)//4 == t//4) & legal
            prior = (j>0) & ((j-1)//4 < t//4) & legal
            distant = (t-j>=64) & (j>0) & legal
            # Entropy is averaged over samples/heads before normalizing.
            head_probs = scores.masked_fill(~legal, -torch.inf).softmax(-1)
            entropy = -(head_probs * head_probs.clamp_min(1e-30).log()).sum(-1).mean((0,1))
            values = torch.stack((probs[:,0],probs.diagonal(),(probs*same).sum(-1),
                (probs*prior).sum(-1),(probs*distant).sum(-1),
                entropy/torch.arange(1,length+1,device=query.device).float().log().clamp_min(1e-8),
                same.sum(-1)/(t[:,0]+1),prior.sum(-1)/(t[:,0]+1)),dim=-1)
            values[0,5] = 0
        attention.setdefault(name, []).append(values.cpu().numpy())

    results = {}
    for mode in modes:
        if args.branch == 'cross':
            def memory_ablation(*inputs, kind=mode, **kwargs):
                memory = original_encode(*inputs, **kwargs)
                return memory[:,:1].expand_as(memory) if kind == 'memory_bos' else torch.zeros_like(memory)
            model.pair_memory_queries.encode_memory = memory_ablation if mode.startswith('memory_') else original_encode
            for head in ('atom', 'coefficient'):
                for index, block in enumerate(getattr(model.pair_memory_queries, head + '_blocks')):
                    selected,read = mode.split('_') if mode != 'full' else ('none','none')
                    restrict = selected in ('both',head)
                    block.forward = (lambda q,m,cached=False,b=block,r=read: restricted_read(b,q,m,r,cached=cached)) if restrict else original[(head,index)]
        torch.manual_seed(7319)
        torch.cuda.manual_seed(7319)
        values = []
        started = time.time()
        for start in range(0,args.samples,args.batch_size):
            stop = min(args.samples,start+args.batch_size)
            target = atoms[start:stop].cuda()
            coeff = coefficients[start:stop].cuda()
            cond = conditions[start:stop].cuda()
            hooks = []
            if args.branch == 'cross' and mode == 'full' and (args.samples <= 64 or start % 64 == 0):
                diagnostic_images += stop-start
                hooks.append(model.pair_memory_queries.register_forward_hook(residual_summary))
                for head in ('atom','coefficient'):
                    for index,block in enumerate(getattr(model.pair_memory_queries,head+'_blocks')):
                        name=f'{head}_{index}'
                        hooks.append(block.register_forward_pre_hook(
                            lambda b,inputs,name=name: attention_summary(b,*inputs,name)))
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                ids,probs = aux.compound_coeff_ids(coeff,temp=.5,stochastic=True,hard=False)
                tokens = target*2048+ids
                output = model(tokens,model_aux=aux,cond=cond)
            for hook in hooks:
                hook.remove()
            atom_log = output['atom_logits'].float().log_softmax(-1)
            coeff_log = output['coeff_logits'].float().log_softmax(-1)
            nll = -atom_log.gather(-1,target[...,None]).squeeze(-1)
            kl = (probs*(probs.clamp_min(1e-30).log()-coeff_log)).sum(-1)
            correct = (atom_log.argmax(-1)==target).float()
            scales = aux.coeff_scales.view(1,1,1,4)
            mae = ((coeff_log.exp()*aux.coeff_bins).sum(-1)-(probs*aux.coeff_bins).sum(-1)).abs()*scales
            # Per image and depth, for paired analysis on identical inputs.
            values.append(torch.stack([x.mean((1,2)) for x in (nll,kl,correct,mae)],-1).cpu().numpy())
            if start % 100 == 0:
                print(json.dumps(dict(branch=args.branch,mode=mode,completed=stop,samples=args.samples,time=time.time())),flush=True)
            del output,atom_log,coeff_log,probs,nll,kl,correct,mae
        array = np.concatenate(values)
        names = ('atom_nll','coefficient_kl','atom_top1','coefficient_physical_mae')
        results[mode] = dict(images=args.samples,seconds=time.time()-started,
            overall={name:float(array[:,:,i].mean()) for i,name in enumerate(names)},
            by_depth={name:array[:,:,i].mean(0).tolist() for i,name in enumerate(names)},
            per_image_and_depth=array.tolist())
    summaries = {}
    fields = ('bos_mass','latest_mass','same_site_mass','earlier_sites_mass','at_least_64_events_back_mass',
              'normalized_entropy','uniform_same_site_mass','uniform_earlier_sites_mass')
    for name,arrays in attention.items():
        mean = np.stack(arrays).mean(0)
        summaries[name] = dict(overall={field:float(mean[1:,i].mean()) for i,field in enumerate(fields)},
            by_depth={field:[float(mean[np.arange(256)%4==d,i].mean()) for d in range(4)] for i,field in enumerate(fields)})
    report = dict(branch=args.branch,checkpoint=str(source),checkpoint_step=3798,
        dataset='fixed training-cache probe; not held-out validation',samples=args.samples,
        covered_classes=int(len(np.unique(classes))),indices=indices.tolist(),coefficient_sampling_seed=7319,
        inference_precision='BF16 teacher-forced forward, FP32 scoring',
        restriction='latest: one contextualized record; uniform: all legal records with uniform weights; local: BOS and completed same-site pairs. The causal memory encoder retains all history in every case.',
        interpretation='inference ablation of trained weights, not a retrained control',
        results=results,attention=summaries,diagnostic_attention_images=diagnostic_images,
        residual_rms=(dict(zip(('backbone_hidden','hidden_query_projection','local_prefix_projection',
                              'current_atom_projection','atom_added_residual','coefficient_added_residual'),
                              np.stack(residuals).mean((0,1)).tolist())) if residuals else None),
        block_rms={name:dict(zip(('query','memory','attention_write','ffn_write'),
                                np.stack(values).mean(0).tolist())) for name,values in scales_by_block.items()},
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    for name,values in scales_by_block.items():
        report['block_rms'][name].update(dict(zip(('value_rms','value_centered_across_records_rms',
            'attention_write_minus_uniform_write_rms','value_cosine_to_BOS'),np.stack(values).mean(0)[4:].tolist())))
    path=OUT/f'{args.branch}-teacher-probe-{args.samples}{"-memory" if args.memory_only else ""}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(completed=True,report=str(path),peak_allocated_gib=report['peak_allocated_gib'])),flush=True)


if __name__ == '__main__':
    main()
