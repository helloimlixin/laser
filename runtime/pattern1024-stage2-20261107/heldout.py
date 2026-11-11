"""Held-out validation NLL: empirical predictive cross-entropy, not exact entropy."""
import math

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from rqvae.utils.config import load_config
from rqvae.img_datasets.transforms import create_transforms
from src.training.rq_imagenet_runtime import TarImageNet, write_json


@torch.no_grad()
def evaluate(profile, model, aux, device, rank, world):
    was_training=model.training
    model.eval()
    sums=torch.zeros(5,device=device,dtype=torch.float64)
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(261107710+rank)
            torch.cuda.manual_seed(261107710+rank)
            config=load_config(profile.base/'inputs/stage2-training-config.yaml')
            data=TarImageNet(profile.base/'data','val',create_transforms(config.dataset,split='val'))
            indices=[]
            for label in range(1000):
                indices.extend(np.flatnonzero(data.labels==label)[:2].tolist())
            assert len(indices)==2000
            if rank==0:
                write_json(profile.base/'inputs/validation-nll-indices.json',dict(indices=indices,split='val',
                    images=2000,images_per_class=2,used_for_training_or_codebook_fit=False))
            loader=DataLoader(Subset(data,indices[rank::world]),batch_size=32,num_workers=2,
                pin_memory=True,prefetch_factor=1,multiprocessing_context='spawn')
            for images,labels in loader:
                images=images.to(device,non_blocking=True)
                labels=labels.to(device,non_blocking=True)
                targets,codes=profile.targets(aux,images)
                logits=model(codes,model_aux=aux,cond=labels,amp=True)
                atom=logits['atom_logits'].float().reshape(-1,16384)
                pattern=logits['pattern_logits'].float().reshape(-1,1024)
                a=F.cross_entropy(atom,targets['atoms'].reshape(-1),reduction='sum')
                p=F.cross_entropy(pattern,targets['patterns'].reshape(-1),reduction='sum')
                logp=pattern.log_softmax(-1)
                entropy=-(logp.exp()*logp).sum()
                sums+=torch.stack((a.double(),p.double(),entropy.double(),
                    a.new_tensor(len(images)*64,dtype=torch.float64),a.new_tensor(len(images),dtype=torch.float64)))
        dist.all_reduce(sums)
        assert int(sums[4])==2000 and torch.isfinite(sums).all()
        sites=sums[3]
        return {'eval/heldout_images':2000,'eval/atom_ce_nats':float(sums[0]/(4*sites)),
            'eval/pattern_ce_nats':float(sums[1]/sites),
            'eval/pattern_cross_entropy_bits':float(sums[1]/sites/math.log(2.)),
            'eval/pattern_predicted_entropy_bits':float(sums[2]/sites/math.log(2.)),
            'eval/joint_cross_entropy_bits_per_site':float((sums[0]+sums[1])/sites/math.log(2.)),
            'eval/nll_bits_per_image':float((sums[0]+sums[1])/sums[4]/math.log(2.)),
            'eval/teacher_entropy_nats':0.}
    finally:
        model.train(was_training)
        torch.cuda.empty_cache()
