#!/usr/bin/env python3
"""Read-only prediction and sampler checks on a committed LASER checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3'))
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=3)
    args = parser.parse_args()
    base = args.base.resolve()
    os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')
    os.environ['TORCH_HOME'] = str(base/'torch-cache')
    sys.path[:0] = [str(base/'source'),str(base/'upstream'),str(base/'runtime')]
    import torch
    from omegaconf import OmegaConf
    from torch.nn import functional as F
    from src import rq_upstream
    from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from src.training.rq_sample_panels import CLASSES, preview_labels, save_panels
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config
    from rqvae.utils.utils import top_k_logits, top_p_probs

    root = Path('/workspace/Projects/laser')
    profile = CompoundReplayRecipe(base)
    run_out = root/'outputs'/profile.run_id
    out = run_out/'collapse-diagnosis'/f'epoch{args.epoch:03d}'
    out.mkdir(parents=True, exist_ok=True)
    status = out/'status.json'
    torch.set_num_threads(4)
    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(10.5*2**30/torch.cuda.get_device_properties(device).total_memory, device)
    # The patched context is confined to this diagnostic process.
    for method in (rq_upstream.RQTransformer.forward, rq_upstream.RQTransformer.cached_forward):
        method.__globals__['autocast'] = lambda enabled=True: torch.autocast(
            'cuda', dtype=torch.float16, enabled=enabled, cache_enabled=False)
    receipts = [json.loads(p.read_text()) for p in
        (run_out/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    candidates = [r for r in receipts if r['global_step'] == args.epoch*626]
    assert len(candidates) == 1
    receipt = candidates[0]
    checkpoint = Path(receipt['local'])
    assert checkpoint.stat().st_size == receipt['bytes']
    saved = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=False)
    assert saved['global_step'] == args.epoch*626
    profile.verify_resume(saved)
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).eval().to(device)
    del saved
    aux, _ = profile.tokenizer(device)
    frozen = profile.digest(aux)
    record(status, dict(phase='loaded', checkpoint_step=args.epoch*626, pid=os.getpid(), time=time.time()))
    policies = {
        'current_rq':dict(atom_temperature=1.,atom_top_k=256,atom_top_p=.95,
            coefficient_temperature=1.,coefficient_top_k=256,coefficient_top_p=.95),
        'no_atom_topk':dict(atom_temperature=1.,atom_top_k=0,atom_top_p=.95,
            coefficient_temperature=1.,coefficient_top_k=256,coefficient_top_p=.95),
        'previous_laser':dict(atom_temperature=.9,atom_top_k=0,atom_top_p=.9,
            coefficient_temperature=1.,coefficient_top_k=0,coefficient_top_p=.85),
    }
    assert OmegaConf.to_container(profile.recipe.sampling, resolve=True) == policies['current_rq']
    config = load_config(base/'inputs/stage2-training-config.yaml')
    reference = json.loads((root/'outputs/imagenet-compound512-gap-20261016'/
        f'compound512-epoch{args.epoch:03d}.json').read_text())
    indices = reference['splits']['heldout_val']['indices'][:128]
    dataset = TarImageNet(base/'data','val',create_transforms(config.dataset,split='val'))
    sums = {}
    rows = []

    def add(name, value):
        assert torch.isfinite(value).all(), name
        sums[name] = sums.get(name, torch.zeros(4,dtype=torch.float64)) + value.double().sum((0,1,2)).cpu()

    def filtered(logits, temperature, k, p):
        shape = logits.shape
        value = logits.float().reshape(-1,shape[-1])/temperature
        if k:
            value = top_k_logits(value,k)
        return top_p_probs(value.softmax(-1),p).reshape(shape)

    with torch.inference_mode():
        for first in range(0,len(indices),2):
            pairs = [dataset[i] for i in indices[first:first+2]]
            images = torch.stack([p[0] for p in pairs]).to(device)
            labels = torch.tensor([p[1] for p in pairs],device=device)
            torch.manual_seed(261016001+first)
            targets,codes = profile.targets(aux,images)
            logits = model(codes,aux,labels,amp=True)
            atoms,coeff = logits['atom_logits'].float(),logits['coeff_logits'].float()
            atom_ce = F.cross_entropy(atoms.reshape(-1,16384),targets['atoms'].reshape(-1),
                reduction='none').reshape_as(targets['atoms'])
            p = targets['coefficient_probabilities']
            coefficient_ce = -(p*coeff.log_softmax(-1)).sum(-1)
            add('atom_ce',atom_ce)
            add('coefficient_ce',coefficient_ce)
            add('coefficient_target_entropy',-(p*p.clamp_min(1.e-30).log()).sum(-1))
            add('atom_top1',(atoms.argmax(-1)==targets['atoms']).float())
            q = atoms.log_softmax(-1)
            add('atom_predicted_entropy',-torch.where(q.isfinite(),q.exp()*q,0.).sum(-1))
            sign = (aux.coeff_bins>0)==(targets['coefficients'][...,None]>0)
            add('correct_coefficient_sign_probability',(coeff.softmax(-1)*sign).sum(-1))
            add('coefficient_binary_sign_accuracy',((coeff.softmax(-1)[...,aux.coeff_bins>0].sum(-1)>.5)
                ==(targets['coefficients']>0)).float())
            for name,settings in policies.items():
                selected = filtered(atoms,settings['atom_temperature'],settings['atom_top_k'],settings['atom_top_p'])
                retained = selected.gather(-1,targets['atoms'][...,None]).squeeze(-1)>0
                add(name+'_true_atom_retained',retained.float())
                selected_c = filtered(coeff,settings['coefficient_temperature'],
                    settings['coefficient_top_k'],settings['coefficient_top_p'])
                add(name+'_coefficient_target_mass_retained',(p*(selected_c>0)).sum(-1))
            wrong = model(codes,aux,(labels+499)%1000,amp=True)['atom_logits'].float()
            wrong_ce = F.cross_entropy(wrong.reshape(-1,16384),targets['atoms'].reshape(-1),
                reduction='none').reshape_as(atom_ce)
            add('correct_class_atom_ce_gain',wrong_ce-atom_ce)
            rows.extend(atom_ce.mean((1,2)).cpu().tolist())
            if first == 0:
                # Verify trained weights, not merely initialization, use the
                # same distributions under full and cached autoregression.
                prefix = torch.zeros_like(codes)
                prefix[...,1::2] = 16384+256
                maximum = 0.
                model.init_cache()
                for h in range(8):
                    for w in range(8):
                        for event in range(8):
                            cached = model.cached_forward(prefix[:,:h+1],aux,labels,amp=True,sample_loc=(h,w,event))
                            full = logits['atom_logits' if event%2==0 else 'coeff_logits'][:,h,w,event//2].float()
                            cached = cached[:,:16384] if event%2==0 else cached[:,16384:]
                            lp,lq = full.log_softmax(-1),cached.float().log_softmax(-1)
                            maximum = max(maximum,float(torch.where(lp.isfinite(),lp.exp()*(lp-lq),0.).sum(-1).max()))
                            prefix[:,h,w,event] = codes[:,h,w,event]
                model.init_cache()
                assert maximum < 2.e-4
                record(out/'cache-verification.json',dict(passed=True,checkpoint_step=args.epoch*626,
                    cache_events=512,maximum_kl=maximum))
            del logits,atoms,coeff,wrong,targets,codes,images,selected,selected_c,p,q
            record(status,dict(phase='teacher_forced',images=first+len(pairs),target=128,pid=os.getpid(),time=time.time()))
        metrics = {k:(v/(len(indices)*64)).tolist() for k,v in sums.items()}
        report = dict(checkpoint=receipt,epoch=args.epoch,images=len(indices),indices=indices,
            per_depth=metrics,per_image_atom_ce=rows,training_updates_added=0,
            current_loss_weight=float(profile.recipe.atom_ce_weight),
            coefficient_temperature=float(profile.recipe.targets.temperature),
            previous_epoch2_metrics=reference['splits']['heldout_val']['per_depth'],
            limits=['Teacher-forced metrics use real prefixes and are not generation quality.',
                'Different coefficient scales/targets prevent direct raw coefficient CE comparisons.'])
        record(out/'teacher-forced.json',report)
        summaries = {}
        all_labels = preview_labels(device)
        for name,settings in policies.items():
            profile.recipe.sampling = OmegaConf.create(settings)
            torch.manual_seed(261017001)
            generated,images = [],[]
            started = time.monotonic()
            for first in range(0,384,32):
                codes = profile.sample(model,aux,32,all_labels[first:first+32])
                generated.append(codes.cpu())
                images.extend([profile.decode(aux,part).float().mul(.5).add(.5).clamp(0,1)
                    .mul(255).add(.5).to(torch.uint8).cpu() for part in codes.split(8)])
                record(status,dict(phase='sample_comparison',policy=name,generated=first+32,
                    target=384,pid=os.getpid(),time=time.time()))
            tokens = torch.cat(generated)
            pictures = torch.cat(images)
            paths = save_panels(pictures,out/name,args.epoch*626)
            ids = tokens[...,0::2]
            physical = aux.coeff_bins.cpu()[tokens[...,1::2]-16384]*aux.coeff_scales.cpu()
            vectors = aux.dictionary.cpu().T[ids]
            latent = (vectors*physical[...,None]).sum(-2)
            counts = torch.stack([torch.bincount(ids[...,d].flatten(),minlength=16384) for d in range(4)])
            marginal = counts.double()/counts.sum(-1,keepdim=True)
            pic = pictures.float()/255
            spatial_std = pic.std((-1,-2)).mean(-1)
            gradients = ((pic[:,:,:,1:]-pic[:,:,:,:-1]).abs().mean((1,2,3))+
                (pic[:,:,1:]-pic[:,:,:-1]).abs().mean((1,2,3)))/2
            summary = dict(sampling=settings,generated_images=384,classes=list(CLASSES),seed=261017001,
                atom_unique_per_depth=(counts>0).sum(-1).tolist(),
                atom_marginal_entropy_per_depth=(-(marginal*marginal.clamp_min(1.e-30).log()).sum(-1)).tolist(),
                atom_adjacent_agreement_per_depth=(.5*((ids[:,1:]==ids[:,:-1]).float().mean((0,1,2))+
                    (ids[:,:,1:]==ids[:,:,:-1]).float().mean((0,1,2)))).tolist(),
                physical_coefficient_rms_per_depth=physical.square().mean((0,1,2)).sqrt().tolist(),
                latent_rms=float(latent.square().mean().sqrt()),
                latent_spatial_std=float(latent.std((1,2)).mean()),
                image_spatial_std=float(spatial_std.mean()),
                fraction_image_std_below_0_05=float((spatial_std<.05).float().mean()),
                image_neighbor_difference=float(gradients.mean()),
                elapsed_seconds=time.monotonic()-started,
                overview=str(paths['samples/fixed']),
                not_fid=True,training_updates_added=0)
            torch.save(tokens,out/name/'tokens.pt')
            record(out/name/'summary.json',summary)
            summaries[name] = summary
            print(json.dumps(dict(policy=name,summary=summary)),flush=True)
            del pic,pictures,physical,latent,vectors,tokens,ids,generated,images,codes
        assert profile.digest(aux)==frozen
        record(out/'sampling-comparison.json',dict(passed=True,weights_unchanged=True,
            tokenizer_unchanged=True,checkpoint_step=args.epoch*626,policies=summaries,
            limits=['Designated-class previews are qualitative diagnostics, not FID50k.',
                'Spatial variance and atom diversity do not establish semantic quality.']))
        record(status,dict(phase='completed',pid=os.getpid(),training_updates_added=0,time=time.time()))


if __name__ == '__main__':
    main()
