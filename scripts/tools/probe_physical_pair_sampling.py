"""Compare frozen epoch77 sampling policies beside the continuing training run."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration-images',type=int,default=64)
    parser.add_argument('--samples-per-class',type=int,default=2)
    parser.add_argument('--classes',type=int,nargs='+',default=[269,612,265,628])
    parser.add_argument('--memory-limit-gib',type=float,default=8.)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(args.base/'source/runtime'),str(args.base/'support'),str(repo)]
    import torch
    from torchvision.datasets import ImageFolder
    from src.training.rqtransformer import LaserAux,build_model,val_image_transform,save_class_labeled_grid
    from src.data.imagenet_labels import class_names_for_dataset
    # Import the experimental helper without replacing the active frozen runtime.
    import importlib.util
    spec=importlib.util.spec_from_file_location('sampling_probe',repo/'src/physical_pair_sampling.py')
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    torch.set_num_threads(4)
    device=torch.device('cuda',0)
    torch.cuda.set_device(device)
    total=torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*2**30/total,device)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    def record(name,value):
        path=args.output/name
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)
    def status(phase,**extra):
        value=dict(phase=phase,elapsed_seconds=time.time()-started,time=time.time(),**extra)
        record('status.json',value);print(json.dumps(value),flush=True)
    source=args.base/'inputs/source-epoch077-full.pt'
    payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    assert (payload['epoch'],payload['global_step'])==(77,48202)
    config=payload['config']
    names=class_names_for_dataset('imagenet')
    record('specification.json',dict(source_checkpoint=str(source),source_epoch=77,
        source_global_step=48202,source_fid=15.24941539209243,training_unchanged=True,
        classes=[dict(id=x,name=names[x]) for x in args.classes],
        seed=261006,samples_per_class=args.samples_per_class,
        calibration_split='train',calibration_images=args.calibration_images,
        memory_limit_gib=args.memory_limit_gib,evaluation_metrics_logged=False,
        purpose='Qualitative sampler pilot, not a FID/IS measurement.'))
    status('loading_frozen_tokenizer')
    aux=LaserAux(args.base/'inputs/resume-stage1-tokenizer.pt',config['num_atoms'],
        config['coeff_vocab_size'],config['coeff_max'],config['coeff_scale'],
        coeff_scales=config['coeff_scales'],soft_target_physical=False,
        clamp_coeffs=False,sparsity_level=config['sparsity_level']).to(device).eval()
    status('calibrating_on_local_training_images')
    dataset=ImageFolder(Path(config['data'])/'train',transform=val_image_transform())
    assert len(dataset)==1281167
    generator=torch.Generator().manual_seed(261006)
    selected=torch.randperm(len(dataset),generator=generator)[:args.calibration_images].tolist()
    record('training-calibration-indices.json',selected)
    atoms_all,coefficients_all=[],[]
    originals,reconstructions=[],[]
    with torch.inference_mode():
        for start in range(0,len(selected),4):
            images=torch.stack([dataset[index][0] for index in selected[start:start+4]]).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                atoms,coefficients=aux.encode_sparse_components(images)
                # Match the existing stochastic coefficient context, not validation labels.
                torch.manual_seed(261006+start)
                tokens,_=aux.sparse_targets(atoms,coefficients,temp=.01125,stochastic=True,compact=True)
                physical=aux.coeff_bins[tokens[...,1::2]-aux.num_atoms]*aux.coeff_scales
                if start==0:
                    recon=aux.decode_tokens(tokens).float().cpu()
                    originals.append(images.float().cpu());reconstructions.append(recon)
            atoms_all.append(atoms.cpu());coefficients_all.append(physical.cpu())
        calibration=helper.calibrate_geometry(aux.dictionary.cpu(),torch.cat(atoms_all),torch.cat(coefficients_all))
        record('training-geometry-calibration.json',calibration)
        original=torch.cat(originals).add(1).mul(.5).clamp(0,1)
        reconstruction=torch.cat(reconstructions).add(1).mul(.5).clamp(0,1)
        from src.training.rqtransformer import save_unlabeled_grid
        save_unlabeled_grid(torch.cat((original,reconstruction)),args.output/'real-vs-tokenized.png',nrow=len(original))
        del images,tokens,atoms,coefficients,physical,atoms_all,coefficients_all
        torch.cuda.empty_cache()
        status('loading_frozen_prior')
        with torch.device('meta'):
            model=build_model(config['num_atoms']+config['coeff_vocab_size'],config['num_atoms'],
                physical_pair_context=True,sparsity_level=config['sparsity_level'],
                coeff_vocab_size=config['coeff_vocab_size'],model_preset=config['model_preset'])
        model.load_state_dict(payload['state_dict'],strict=True,assign=True)
        model.requires_grad_(False).to(device).eval()
        del payload
        policies={
            'native':helper.PairSamplingPolicy(mode='ancestral'),
            'depth-sharp':helper.PairSamplingPolicy(mode='ancestral',coefficient_temperatures=(.55,.7,.85,1.)),
            'joint-prior':helper.PairSamplingPolicy(mode='joint',atom_proposal='prior',candidate_atoms=32),
            'joint-prior-geometry':helper.PairSamplingPolicy(mode='joint',atom_proposal='prior',candidate_atoms=32,geometry_weight=2.),
            'joint-prior-sharp':helper.PairSamplingPolicy(mode='joint',atom_proposal='prior',candidate_atoms=32,coefficient_temperatures=(.55,.7,.85,1.)),
        }
        record('policies.json',{name:asdict(policy) for name,policy in policies.items()})
        labels=torch.tensor(args.classes,device=device).repeat_interleave(args.samples_per_class)
        results={}
        for name,policy in policies.items():
            status('sampling',policy=name,completed=0,total=len(labels))
            images_all,tokens_all,diagnostics_all=[],[],[]
            for start in range(0,len(labels),2):
                torch.manual_seed(261006+start)
                diagnostics={}
                tokens=helper.sample_physical_pairs(model,len(labels[start:start+2]),aux,labels[start:start+2],
                    policy=policy,calibration=calibration,amp=True,diagnostics=diagnostics)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    images=aux.decode_tokens(tokens).float().add(1).mul(.5).clamp(0,1)
                assert torch.isfinite(images).all()
                support=tokens[...,0::2].sort(-1).values
                assert (support.diff(dim=-1)>0).all()
                images_all.append(images.cpu());tokens_all.append(tokens.cpu());diagnostics_all.append(diagnostics)
                status('sampling',policy=name,completed=start+len(images),total=len(labels),
                    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
            images=torch.cat(images_all)
            save_class_labeled_grid(images,torch.tensor(args.classes),names,args.output/f'{name}.png',
                samples_per_class=args.samples_per_class)
            torch.save(dict(tokens=torch.cat(tokens_all),labels=labels.cpu(),policy=asdict(policy)),args.output/f'{name}-tokens.pt')
            results[name]=dict(images=len(images),all_images_finite=True,unique_support=True,
                diagnostics=diagnostics_all,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
            record('pilot-results.json',results)
        status('complete',policies=list(results),official_fid_is_measured=False)


if __name__=='__main__':main()
