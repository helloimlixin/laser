#!/usr/bin/env python3
"""Learn a synthesis correction from integer lookups without transmitting extra bits."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from src.models.entropy_constrained_dictionary import EntropyConstrainedBook
from src.models.joint_rate_distortion import NeuralJointDecoder


def study_module(args):
    import importlib.util
    path=args.base/'source/scripts/tools/study_dictionary_rate_distortion.py'
    spec=importlib.util.spec_from_file_location('dictionary_rd_study',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def book(args):
    spec=torch.load(args.book,weights_only=True,map_location='cpu')
    q=EntropyConstrainedBook(spec['vectors'],spec['probabilities']).to(args.device)
    return spec,q


def train(args):
    study=study_module(args);spec,q=book(args)
    fit=study.load_cache(args,'fit')['latents'].to(args.device)
    held=study.load_cache(args,'held')['latents'].to(args.device)
    with torch.no_grad():
        fit_codes=q.quantize(fit,rate_weight=spec['beta'],sweeps=spec['sweeps'])['codes']
        held_codes=q.quantize(held,rate_weight=spec['beta'],sweeps=spec['sweeps'])['codes']
    torch.manual_seed(261024)
    model=NeuralJointDecoder().to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3.e-4,weight_decay=1.e-4)
    generator=torch.Generator().manual_seed(261025)
    best,best_error,trace=None,math.inf,[]
    @torch.no_grad()
    def assess():
        total=0.
        for first in range(0,len(held),64):
            target=held[first:first+64]
            prediction=model(q.embed(held_codes[first:first+64]))
            total+=float((prediction-target).double().square().sum())
        return total/held.numel()
    for step in range(args.steps+1):
        if step in [0,100,300,600,args.steps]:
            error=assess();trace.append(dict(step=step,heldout_training_mse=error))
            if error<best_error:
                best_error=error;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(json.dumps(dict(candidate=spec['name'],step=step,heldout_mse=error)),flush=True)
        if step==args.steps:break
        selected=torch.randint(len(fit),(64,),generator=generator).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        prediction=model(q.embed(fit_codes[selected]))
        loss=(prediction-fit[selected]).square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    target=args.base/f"neural/{spec['name']}.pt";target.parent.mkdir(exist_ok=True)
    torch.save(dict(state_dict=best,name=spec['name']+'-neural',source_book_sha256=study.sha256(args.book),
        vocabulary=spec['vocabulary'],
        source_book=str(args.book),trace=trace,heldout_training_mse=best_error,
        decoder_parameters=sum(p.numel() for p in model.parameters()),objective='training latent MSE',
        decoder_inputs='integer codeword lookups only',continuous_side_information=False,
        tokens_per_image=256,validation_used_for_fitting=False,image_cnns_frozen=True),target)


@torch.inference_mode()
def evaluate(args):
    import torch.distributed as dist
    from torch.utils.data import DataLoader,Subset
    from src import rqvae_metrics as metrics
    study=study_module(args)
    aux,dataset=study.setup_runtime(args);aux=aux.to(args.device).eval()
    inception=metrics.OriginalRQVAEInception().to(args.device).eval().requires_grad_(False)
    local=torch.load(args.base/f'cache/validation-rank{args.rank}.pt',weights_only=True,map_location='cpu')
    loader=DataLoader(Subset(dataset('val'),local['indices'].tolist()),batch_size=16,num_workers=4,
        pin_memory=True,persistent_workers=True)
    report=json.loads((args.base/'results.json').read_text())
    by_name={r['name']:r for r in report['results']}
    measured_names=set(by_name)
    candidates={}
    for path in sorted((args.base/'neural').glob('*.pt')):
        candidate=torch.load(path,weights_only=True,map_location='cpu')
        vocabulary=candidate.get('vocabulary',int(path.stem.split('-')[0].removeprefix('ecvq')))
        group=(vocabulary,candidate['objective'])
        if group not in candidates or candidate['heldout_training_mse']<candidates[group][0]:
            candidates[group]=(candidate['heldout_training_mse'],path)
    if args.rank==0:
        study.write(args.base/'neural/selection.json',dict(selection_split='heldout ImageNet training',
            validation_used_for_selection=False,selected={str(v):dict(path=str(path),heldout_mse=error)
                for v,(error,path) in candidates.items()}))
    for _,path in sorted(candidates.values(),key=lambda item:str(item[1])):
        neural=torch.load(path,weights_only=True,map_location='cpu')
        if neural['name'] in measured_names:continue
        args.book=Path(neural['source_book']);spec,q=book(args)
        if study.sha256(args.book)!=neural['source_book_sha256']:raise ValueError('Source book changed')
        decoder=NeuralJointDecoder().to(args.device).eval();decoder.load_state_dict(neural['state_dict'])
        state=metrics.DistributedOriginalRQVAEMetrics(args.device,inception=inception,compute_inception_score=False)
        real=local['features'].to(args.device,dtype=torch.float64)
        state.real_sum.copy_(real.sum(0));state.real_cross.addmm_(real.T,real);state.real_count.fill_(len(real))
        histogram=torch.zeros(4,q.vocab_size,device=args.device,dtype=torch.float64)
        errors=torch.zeros(4,device=args.device,dtype=torch.float64);started=time.monotonic()
        for batch,(images,_) in enumerate(loader):
            first=batch*16;x=local['latents'][first:first+len(images)].to(args.device)
            codes=q.quantize(x,chunk_size=512,rate_weight=spec['beta'],sweeps=spec['sweeps'])['codes']
            z=decoder(q.embed(codes))*neural.get('decode_gain',1.)
            reconstructed=aux.decode(z).mul(.5).add(.5).clamp(0,1)
            reference=images.to(args.device).mul(.5).add(.5).clamp(0,1)
            state.update(reconstructed,real=False);histogram.add_(study.counts(codes,q.vocab_size))
            errors.add_(torch.stack([(x-z).double().square().sum(),x.new_tensor(x.numel(),dtype=torch.float64),
                (reconstructed-reference).double().square().sum(),x.new_tensor(reference.numel(),dtype=torch.float64)]))
            if args.rank==0 and batch%50==0:
                print(json.dumps(dict(candidate=neural['name'],local_images=first+len(images))),flush=True)
        dist.all_reduce(errors);dist.all_reduce(histogram)
        rfid,_,_=state.compute()
        if int(state.real_count)!=50000 or int(state.fake_count)!=50000:raise RuntimeError('Incomplete evaluation')
        original_counts=torch.load(args.base/f"{spec['name']}-counts.pt",weights_only=True,map_location='cpu')
        torch.testing.assert_close(histogram.cpu(),original_counts['validation_counts'],rtol=0,atol=0)
        row=dict(by_name[spec['name']],name=neural['name'],rfid=rfid,latent_mse=float(errors[0]/errors[1]),
            pixel_mse=float(errors[2]/errors[3]),psnr_db=float(-10*torch.log10(errors[2]/errors[3])),
            neural_decoder=True,identical_integer_codes_to=spec['name'],
            neural_training_objective=neural['objective'],
            neural_decoder_parameters=neural['decoder_parameters'],neural_decoder_sha256=study.sha256(path),
            decode_gain=neural.get('decode_gain',1.),
            seconds=time.monotonic()-started)
        report['results'].append(row)
        if args.rank==0:
            study.write(args.base/'neural-results.json',report)
            print(json.dumps(row),flush=True)
        del q,decoder,state
    if args.rank==0:
        study.write(args.base/'results.json',report)
    dist.barrier()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['train','evaluate'])
    p.add_argument('--base',type=Path,default=Path('/dev/shm/laser-entropy-rd-20261024'))
    p.add_argument('--book',type=Path)
    p.add_argument('--steps',type=int,default=1000)
    args=p.parse_args();torch.set_num_threads(4)
    args.rank=int(os.environ.get('RANK',0));args.world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)));args.device=torch.device('cuda',torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ.setdefault('TORCH_HOME',str(args.base/'torch-cache'))
    if args.phase=='evaluate':
        import torch.distributed as dist
        dist.init_process_group('nccl',device_id=args.device)
    globals()[args.phase](args)
    if args.phase=='evaluate':dist.destroy_process_group()


if __name__=='__main__':main()
