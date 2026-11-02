"""Fit a finite-code neural decompressor through the frozen image CNN and LPIPS."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset, DataLoader

from src.models.joint_rate_distortion import JointVectorBook, NeuralJointDecoder
from src.models.lpips import LPIPS
from src.training.joint_rate_distortion import load_cache, tokenizer, image_transform, write
from src.training.rq_imagenet_runtime import TarImageNet


class CodecViews(Dataset):
    def __init__(self, base, cache, codes):
        self.images = TarImageNet(base/'data', 'train', image_transform(True), seed=261001)
        self.cache, self.codes = cache, codes
    def __len__(self):return len(self.codes)
    def __getitem__(self, index):
        return (self.images[int(self.cache['indices'][index])][0],
                self.cache['latents'][index], self.codes[index])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--vocabulary',type=int,default=16384)
    p.add_argument('--steps',type=int,default=200)
    args=p.parse_args();args.device=torch.device('cuda',0)
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(261030)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    fit,held=load_cache(args,'fit'),load_cache(args,'held')
    v=args.vocabulary
    vectors=torch.load(args.base/f'books/vq{v}.pt',map_location='cpu',weights_only=True)['vectors']
    q=JointVectorBook(vectors).to(args.device)
    codes=torch.load(args.base/f'books/vq{v}-training-codes.pt',weights_only=True)
    data=CodecViews(args.base,fit,codes['fit_codes'])
    held={k:value[:128] for k,value in held.items()}
    validation=CodecViews(args.base,held,codes['held_codes'][:128])
    loader=DataLoader(data,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(261031),
        num_workers=4,persistent_workers=True,pin_memory=True,multiprocessing_context='spawn')
    valid_loader=DataLoader(validation,batch_size=8,num_workers=0,pin_memory=True)
    aux=tokenizer(args)
    perceptual=LPIPS().to(args.device).eval().requires_grad_(False)
    cnn_versions=[t._version for t in aux.state_dict().values()]
    learner=NeuralJointDecoder().to(args.device)
    source=torch.load(args.base/f'books/neural{v}.pt',map_location='cpu',weights_only=True)
    learner.load_state_dict(source['state_dict'])
    optimizer=torch.optim.AdamW(learner.parameters(),lr=1.e-4,weight_decay=1.e-4)
    trace=[];best_state=None;best_loss=float('inf');started=time.monotonic();checked=False

    def objective(images,latent,code):
        images,latent,code=[value.to(args.device,non_blocking=True) for value in (images,latent,code)]
        restored=learner(q.embed(code.long()))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            reconstructed=aux.decode(restored)
            perceptual_loss=perceptual(reconstructed,images).mean()
            pixel=(reconstructed.float()-images).abs().mean()
        latent_loss=(restored-latent).square().mean()
        total=latent_loss+.1*perceptual_loss+.1*pixel
        return total,torch.stack([total.detach(),latent_loss.detach(),perceptual_loss.detach(),pixel.detach()])

    @torch.no_grad()
    def assess(step):
        nonlocal best_state,best_loss
        total=torch.zeros(4,device=args.device)
        for batch in valid_loader:total.add_(objective(*batch)[1])
        total.div_(len(valid_loader));score=float(total[0])
        row=dict(step=step,heldout_objective=score,latent_mse=float(total[1]),lpips=float(total[2]),pixel_l1=float(total[3]),seconds=time.monotonic()-started)
        trace.append(row)
        if score<best_loss:
            best_loss=score;best_state={k:t.detach().cpu().clone() for k,t in learner.state_dict().items()}
        print(json.dumps(row),flush=True)
        write(args.base,f'perceptual{v}-trace.json',dict(trace=trace,completed=step==args.steps))

    assess(0);step=0
    while step<args.steps:
        for batch in loader:
            if not checked:
                images,target,_=batch
                with torch.no_grad():error=(aux.encode(images.to(args.device))-target.to(args.device)).abs().max()
                if float(error)>1.e-3:raise RuntimeError('Cached training latent does not match its image view')
                checked=True
            optimizer.zero_grad(set_to_none=True);loss,values=objective(*batch)
            loss.backward();torch.nn.utils.clip_grad_norm_(learner.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            if step in [1,20,50,100,args.steps]:print(json.dumps(dict(train_step=step,loss=float(loss),seconds=time.monotonic()-started)),flush=True)
            if step in [100,args.steps]:assess(step)
            if step>=args.steps:break
    assert cnn_versions==[t._version for t in aux.state_dict().values()]
    torch.save(dict(state_dict=best_state,vocabulary=v,trace=trace,
        objective='latent MSE + 0.1 LPIPS + 0.1 pixel L1',
        selected_by='128 heldout training images; no validation fitting',
        trained_through_frozen_image_decoder=True,image_cnns_frozen=True,
        continuous_side_information=False,decoder_parameters=sum(t.numel() for t in learner.parameters()),
        updates=args.steps),args.base/f'books/perceptual{v}.pt')
    write(args.base,'perceptual-refinement-verified.json',dict(completed=True,
        image_cnns_unchanged=True,integer_codes_unchanged=True,updates=step,
        fit_images=len(data),heldout_training_images=len(validation)))


if __name__=='__main__':main()
