#!/usr/bin/env python3
"""Read-only checkpoint probes of context usage, separate from training loss."""
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-rq-joint20k-imagenet-20261017')
RUN='imagenet-laser-joint20k-scratch2-4h100-20261017'
OUT=ROOT/'outputs'/RUN
STUDY=ROOT/'outputs/imagenet-laser-adaptive-joint20k-vocabulary-20261017'
os.environ['RQ_UPSTREAM_ROOT']=str(BASE/'upstream')
sys.path[:0]=[str(BASE/'source'),str(BASE/'upstream'),str(BASE/'runtime')]
import numpy as np
import torch
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_runtime import TarImageNet
from rqvae.utils.config import load_config,augment_arch_defaults
from rqvae.img_datasets.transforms import create_transforms


def write(name,report):
    path=OUT/'context-audit'/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)


@torch.inference_mode()
def audit(receipt):
    device=torch.device('cuda:3');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.18,device)
    saved=torch.load(receipt['local'],map_location='cpu',mmap=True,weights_only=False)
    assert saved['run_id']==RUN and saved['global_step']==receipt['global_step']
    profile=JointLaserRecipe(BASE);profile.verify_resume(saved)
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    with torch.device('meta'):model=profile.model(augment_arch_defaults(config.arch))
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    model=model.eval().to(device);aux,_=profile.tokenizer(device)
    fitted=json.loads((STUDY/'calibration-data.json').read_text())
    excluded=set(fitted['fit_indices']+fitted['heldout_training_indices'])
    indices=[int(i) for i in np.random.default_rng(261017935).permutation(1281167) if int(i) not in excluded][:16]
    data=TarImageNet(BASE/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
    data.set_epoch(0)
    details=torch.load(STUDY/'pair-redundancy-details.pt',map_location='cpu',weights_only=True)
    counts=details['hard_training_histogram'].double()+1.
    unigram_logp=(counts/counts.sum(-1,keepdim=True)).log().float().to(device)
    sums={k:torch.zeros(4,device=device,dtype=torch.float64) for k in
          ['correct_context','shuffled_image_prefix','shuffled_class_label','unigram_baseline','target_entropy']}
    tokens=0
    for first in range(0,len(indices),2):
        torch.manual_seed(261017935+first)
        records=[data[i] for i in indices[first:first+2]]
        images=torch.stack([r[0] for r in records]).to(device)
        labels=torch.tensor([r[1] for r in records],device=device)
        targets,codes=profile.targets(aux,images)
        assert labels[0]!=labels[1], 'Class-shuffle probe requires distinct classes'
        for name,input_codes,input_labels in [
                ('correct_context',codes,labels),
                ('shuffled_image_prefix',codes.flip(0),labels),
                ('shuffled_class_label',codes,labels.flip(0))]:
            logits=model(input_codes,aux,input_labels,amp=True)
            ce=-(targets*logits.float().log_softmax(-1)).sum(-1)
            sums[name]+=ce.sum((0,1,2)).double()
            del logits,ce
        sums['unigram_baseline']+=(-(targets*unigram_logp).sum(-1)).sum((0,1,2)).double()
        sums['target_entropy']+=(-(targets*targets.clamp_min(1.e-30).log()).sum(-1)).sum((0,1,2)).double()
        tokens+=codes.numel()//4
        del targets,codes,images
    means={k:(v/tokens).cpu().tolist() for k,v in sums.items()}
    per_depth=[]
    for d in range(4):
        row=dict(depth=d+1,**{k:v[d] for k,v in means.items()})
        row.update(prefix_context_gain=row['shuffled_image_prefix']-row['correct_context'],
            class_context_gain=row['shuffled_class_label']-row['correct_context'],
            gain_over_unigram=row['unigram_baseline']-row['correct_context'])
        per_depth.append(row)
    report=dict(completed=True,checkpoint_global_step=receipt['global_step'],
        checkpoint_sha256=receipt['sha256'],images=len(indices),training_indices=indices,
        disjoint_from_codebook_fitting=True,augmentation='original training crop and flip',
        per_depth=per_depth,mean={key:float(np.mean([r[key] for r in per_depth])) for key in per_depth[0] if key!='depth'},
        loss='joint soft cross entropy in nats; all variants use the same original-image targets',
        unigram_estimator='Laplace-smoothed hard-code frequencies from8192 training views',
        read_only=True,training_objective_changed=False,additional_training_losses=0,
        interpretation='Small diagnostic of context use; neither a generation score nor a causal proof of the FID gap',
        time=time.time())
    write(f'step{receipt["global_step"]:06d}.json',report);write('latest.json',report)
    del model,aux,saved,details
    gc.collect();torch.cuda.empty_cache()
    return report


def publish(report):
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    import wandb
    run=wandb.Api(timeout=45).run('helloimlixin-rutgers/laser/'+RUN)
    directory=BASE/'audit-uploads/context-audit';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'step{report["checkpoint_global_step"]:06d}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    run.upload_file(str(path),root=str(BASE/'audit-uploads'))
    run.summary['diagnostics/context_audit_step']=report['checkpoint_global_step']
    run.summary['diagnostics/prefix_context_gain']=report['mean']['prefix_context_gain']
    run.summary['diagnostics/class_context_gain']=report['mean']['class_context_gain']
    run.summary.update()


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    last=0
    write('watcher.json',dict(phase='waiting_for_first_production_checkpoint',pid=os.getpid(),
        training_recipe_unchanged=True,time=time.time()))
    while True:
        receipts=[json.loads(p.read_text()) for p in (OUT/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
        available=sorted((r for r in receipts if r['global_step']>last),key=lambda r:r['global_step'])
        for receipt in available:
            try:
                report=audit(receipt);publish(report)
                last=receipt['global_step']
                write('watcher.json',dict(phase='checkpoint_audited',last_step=last,pid=os.getpid(),time=time.time()))
            except Exception as error:
                write('watcher.json',dict(phase='audit_failed',error_type=type(error).__name__,
                    error=str(error),checkpoint_step=receipt['global_step'],time=time.time()))
                raise
        if (OUT/'completed.json').exists():return
        time.sleep(30)


if __name__=='__main__':main()
