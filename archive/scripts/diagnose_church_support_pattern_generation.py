#!/usr/bin/env python3
"""Frozen-prior atom baselines, oracle-support rollout, and sampling screen."""
import argparse
import codecs
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import sha256_file
from src.training.rqtransformer import LaserAux, atomic_torch_save
from src.church_support_pattern_training import support_pattern_prior, pattern_targets
from src.models.lpips import LPIPS
from src.models.rqtransformer import transformers
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


def deterministic_top_p(probs, p):
    # The original CUDA cumsum is unavailable with strict determinism. This
    # diagnostic alone moves the probability scan to CPU; live sources are intact.
    ordered, indices = torch.sort(probs, descending=True)
    remove = ordered.cpu().cumsum(-1).to(probs.device) >= p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = remove.scatter(-1, indices, remove)
    kept = probs.masked_fill(remove, 0.)
    return kept / kept.sum(-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def fixed_support_rollout(model, aux, atoms):
    ids = torch.zeros(atoms.shape[:-1], dtype=torch.long, device=atoms.device)
    packed = model.pack(atoms, ids)
    model.init_cache()
    initial_logits = None
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for h in range(8):
                for w in range(8):
                    for d in range(4):
                        hidden = model.cached_head_output(packed, aux, None, (h, w, d), amp=True)
                    logits = model.coefficient_pattern_logits(hidden, aux.dictionary.t()[atoms[:, h, w]])
                    if h == 0 and w == 0:
                        initial_logits = logits.float()
                    ids[:, h, w] = logits.argmax(-1)
                    packed[:, h, w, -1] = atoms[:, h, w, -1] * len(aux.coefficient_patterns) + ids[:, h, w]
    finally:
        model.init_cache()
    replay_atoms, replay_ids = model.unpack(packed)
    assert torch.equal(atoms, replay_atoms) and torch.equal(ids, replay_ids)
    return ids, initial_logits


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe-images', type=int, default=256)
    p.add_argument('--screen-samples', type=int, default=1024)
    args = p.parse_args()
    assert 0 < args.probe_images <= 1024 and args.screen_samples > 1
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'results.json').exists()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.manual_seed(29703)
    for cutoff in (.1, .5, .85, 1.):
        example = torch.softmax(torch.randn(7, 53), -1)
        torch.testing.assert_close(deterministic_top_p(example, cutoff), transformers._top_p_probs(example, cutoff), atol=0, rtol=0)
    transformers._top_p_probs = deterministic_top_p
    started = time.monotonic()

    def log(row):
        row = {'seconds':time.monotonic()-started, **row}
        print(json.dumps(row, allow_nan=False), flush=True)
        with (args.output/'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']
    epoch = saved['epoch']
    model = support_pattern_prior(cfg['coefficient_vocabulary'], cfg['dropout']).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    raw = torch.load(cfg['cache'], map_location='cpu', weights_only=False)
    book = torch.load(cfg['codebook_path'], map_location='cpu', weights_only=True)
    assert sha256_file(Path(cfg['codebook_path'])) == cfg['codebook_sha256']
    aux = LaserAux(Path(cfg['stage1']),16384,2048,3.,coeff_scales=cfg['tokenizer']['coeff_scales'],
        sparsity_level=4,soft_target_physical=True,clamp_coeffs=False,
        coefficient_patterns=book['coefficient_patterns']).cuda().eval().requires_grad_(False)
    perceptual = LPIPS().cuda().eval().requires_grad_(False)
    frozen = [(v,v._version) for m in (model,aux) for v in (*m.parameters(),*m.buffers())]
    q = torch.stack([torch.bincount(raw['train']['atoms'][...,d].long().flatten(), minlength=16384).double()+.5 for d in range(4)])
    q /= q.sum(-1, keepdim=True)
    nll, accuracy, marginal_nll, masked_marginal_nll = [], [], [], []
    rows = {name:[] for name in ('teacher_argmax','rollout_argmax')}
    mean_metrics, cache_tv = [], []
    for first in range(0,args.probe_images,16):
        atoms = raw['holdout']['atoms'][first:first+min(16,args.probe_images-first)].cuda().long()
        physical = raw['holdout']['coefficients'][first:first+len(atoms)].cuda()
        targets = pattern_targets(aux,atoms,physical)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predicted = model(model.pack(atoms,targets), model_aux=aux, amp=True)
        logits = predicted['atom_logits'].float()
        nll.append(F.cross_entropy(logits.reshape(-1,16384),atoms.flatten(),reduction='none').reshape_as(atoms).mean((1,2)).cpu())
        accuracy.append((logits.argmax(-1)==atoms).float().mean((1,2)).cpu())
        baseline, masked_baseline = [], []
        for d in range(4):
            targets_cpu = atoms[...,d].cpu()
            base = -q[d,targets_cpu].log()
            remaining = 1-q[d,atoms[...,:d].cpu()].sum(-1)
            baseline.append(base.mean((1,2)))
            masked_baseline.append((base+remaining.log()).mean((1,2)))
        marginal_nll.append(torch.stack(baseline,-1))
        masked_marginal_nll.append(torch.stack(masked_baseline,-1))
        teacher_ids = predicted['pattern_logits'].argmax(-1)
        mean_coefficients = predicted['pattern_logits'].float().softmax(-1) @ aux.coefficient_patterns
        mean_metrics.append(torch.stack(((mean_coefficients-physical).abs().mean((1,2,3)),
            ((mean_coefficients>=0)==(physical>=0)).float().mean((1,2,3))),-1).cpu())
        rollout_ids, initial_logits = fixed_support_rollout(model,aux,atoms)
        # BF16 SDPA teacher forcing and explicit cached attention round
        # differently. Measure distribution error, not equality of argmax IDs.
        cache_tv.append((.5*(initial_logits.softmax(-1)-predicted['pattern_logits'][:,0,0].float().softmax(-1)).abs().sum(-1)).cpu())
        if first == 0:
            one = model.pack(atoms[:1],targets[:1])
            fp32_teacher = model(one,model_aux=aux,amp=False)['pattern_logits'][:,0,0]
            model.init_cache()
            for d in range(4):
                hidden = model.cached_head_output(one,aux,None,(0,0,d),amp=False)
            fp32_cached = model.coefficient_pattern_logits(hidden,aux.dictionary.t()[atoms[:1,0,0]])
            fp32_error = float((fp32_teacher-fp32_cached).abs().max())
            torch.testing.assert_close(fp32_cached,fp32_teacher,atol=2e-4,rtol=2e-4)
            model.init_cache()
            log({'phase':'cache_numerics','fp32_first_site_max_logit_error':fp32_error,
                'bf16_first_site_tv':float(torch.cat(cache_tv).mean())})
        reference = aux.decode_coefficient_patterns(atoms,targets)
        images = [reference]
        for name,ids in [('teacher_argmax',teacher_ids),('rollout_argmax',rollout_ids)]:
            decoded = aux.decode_coefficient_patterns(atoms,ids)
            images.append(decoded)
            coeff = aux.coefficient_patterns[ids]
            rows[name].append(torch.stack((perceptual(decoded,reference).flatten(),
                -10*((decoded-reference).square().mean((1,2,3))/4).clamp_min(1e-12).log10(),
                ((coeff>=0)==(physical>=0)).float().mean((1,2,3)),
                (coeff-physical).abs().mean((1,2,3))),-1).cpu())
        if first == 0:
            save_image((torch.stack([x[:8] for x in images],1).flatten(0,1)+1)/2,args.output/'fixed-support.png',nrow=3)
        if (first+len(atoms))%64==0 or first+len(atoms)==args.probe_images:
            log({'phase':'fixed_support','images':first+len(atoms)})
    diag = {'images':args.probe_images,'split':'first holdout images in saved cache order',
        'bf16_first_site_teacher_cache_tv_mean':torch.cat(cache_tv).mean().item(),
        'bf16_first_site_teacher_cache_tv_max':torch.cat(cache_tv).max().item(),
        'fp32_first_site_max_logit_error':fp32_error,
        'atom_nll_by_depth':torch.cat(nll).mean(0).tolist(),
        'atom_accuracy_by_depth':torch.cat(accuracy).mean(0).tolist(),
        'train_frequency_nll_by_depth':torch.cat(marginal_nll).mean(0).tolist(),
        'train_frequency_distinct_support_nll_by_depth':torch.cat(masked_marginal_nll).mean(0).tolist(),
        'uniform_nll_by_depth':[(torch.tensor(16384-d,dtype=torch.double).log().item()) for d in range(4)],
        'teacher_pattern_mean_mae':torch.cat(mean_metrics).mean(0)[0].item(),
        'teacher_pattern_mean_sign_accuracy':torch.cat(mean_metrics).mean(0)[1].item(),
        'fixed_support':{name:dict(zip(('lpips','psnr','sign_accuracy','coefficient_mae'),torch.cat(v).mean(0).tolist())) for name,v in rows.items()},
        'interpretation':'Oracle atom identities supplied throughout. Reconstruction stress test against true-pattern reconstructions, not unconditional generation. Rollout uses its own past patterns; teacher forcing uses real past patterns.'}
    atomic_torch_save({'atom_nll':torch.cat(nll),'marginal_nll':torch.cat(marginal_nll),
        'masked_marginal_nll':torch.cat(masked_marginal_nll),**{k:torch.cat(v) for k,v in rows.items()}},args.output/'per-image.pt')
    (args.output/'diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    log({'phase':'diagnostics_complete',**diag})
    del perceptual
    original = Path(cfg['output'])/'evaluations'/f'epoch-{epoch:03d}'/'screen'
    baseline_codes = torch.load(original/'generated-codes.pt',map_location='cpu',weights_only=False)
    assert cfg['generation_batch'] == 128
    torch.manual_seed(18701)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        check_atoms,check_ids = model.sample_compound(128,aux,atom_top_k=2048,atom_top_p=None,
            coeff_top_k=0,coeff_top_p=None,atom_temperature=1.,coeff_temperature=1.,amp=True)
    assert torch.equal(check_atoms.cpu(),baseline_codes['atoms'][:128].long())
    assert torch.equal(check_ids.cpu(),baseline_codes['pattern_ids'][:128].long())
    log({'phase':'baseline_replay','exact_first_128':True,'epoch':epoch})

    def sampling(name,atom_k,pattern_p,count,use_saved=False):
        directory = args.output/name
        directory.mkdir(exist_ok=False)
        metric = DistributedOriginalRQVAEMetrics('cuda',reference_stats_path=Path(cfg['fid_stats']))
        torch.manual_seed(18701)
        atoms_saved,ids_saved = [],[]
        for first in range(0,count,128):
            batch=min(128,count-first)
            if use_saved:
                atoms=baseline_codes['atoms'][first:first+batch].cuda().long()
                ids=baseline_codes['pattern_ids'][first:first+batch].cuda().long()
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    atoms,ids=model.sample_compound(batch,aux,atom_top_k=atom_k,atom_top_p=None,
                        coeff_top_k=0,coeff_top_p=pattern_p,atom_temperature=1.,coeff_temperature=1.,amp=True)
            atoms_saved.append(atoms.cpu().short());ids_saved.append(ids.cpu().short())
            for offset in range(0,batch,32):
                rgb=(aux.decode_coefficient_patterns(atoms[offset:offset+32],ids[offset:offset+32])+1)/2
                metric.update(rgb,real=False)
                if first==0 and offset==0: save_image(rgb,directory/'samples.png',nrow=8)
            if (first+batch)%512==0 or first+batch==count:
                log({'phase':'sampling','condition':name,'images':first+batch,'total':count})
        fid,_,_=metric.compute()
        result={'fid':float(fid),'samples':count,'seed':18701,'batch':128,'atom_top_k':atom_k,
            'pattern_top_p':pattern_p,'temperature':1.,'checkpoint_epoch':epoch,'uses_saved_codes':use_saved}
        atomic_torch_save({'atoms':torch.cat(atoms_saved),'pattern_ids':torch.cat(ids_saved)},directory/'codes.pt')
        (directory/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        log({'phase':'sampling_complete','condition':name,**result})
        return result

    conditions={'baseline':(2048,None),'atom250':(250,None),'pattern05':(2048,.5),'both':(250,.5)}
    scores={name:sampling(name,k,p,args.screen_samples,name=='baseline') for name,(k,p) in conditions.items()}
    winner=min(scores,key=lambda name:scores[name]['fid'])
    confirmation=None
    if winner!='baseline':
        confirmation=sampling(winner+'-4096',*conditions[winner],4096)
    assert all(v._version==version and v.grad is None for v,version in frozen)
    result={'epoch':epoch,'checkpoint_sha256':sha256_file(args.checkpoint),'diagnostics':diag,
        'screen':scores,'selected_sampling':winner,'selected_sampling_4096':confirmation,
        'original_4096':json.loads((original/'metrics.json').read_text()),'weights_changed':False,
        'baseline_128_exact_replay':True,'source_sha256':sha256_file(Path(__file__)),
        'limitations':'Single frozen checkpoint and seed. Sampling selected with 1024 images; 4096 comparison shares the same seed, not independent confirmation or FID-50000. Cross-run differences do not isolate architecture from optimization.'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    log({'phase':'complete','selected_sampling':winner,'selected_sampling_4096':confirmation})


if __name__=='__main__':
    main()
