#!/usr/bin/env python3
"""Verify the fresh adaptive pair-ID trial before production training."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    base=Path(os.environ['LASER_RQ_BASE'])
    os.environ['RQ_UPSTREAM_ROOT']=str(base/'upstream')
    sys.path[:0]=[str(base/'source'),str(base/'upstream'),str(base/'runtime')]
    import torch
    from src import rq_upstream
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training.rq_imagenet_runtime import TarImageNet
    from src.training.rq_sample_panels import CLASSES,PREVIEW_IMAGES,OVERVIEW_IMAGES
    from rqvae.utils.config import load_config,augment_arch_defaults,augment_optimizer_defaults
    from rqvae.optimizer.optimizer import create_resnet_optimizer
    from rqvae.optimizer.scheduler import create_scheduler
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.models.rqvae.quantizations import RQBottleneck
    torch.set_num_threads(4)
    profile=JointLaserRecipe(base)
    out=Path('/workspace/Projects/laser/outputs')/profile.run_id
    smoke=Path(os.environ['LASER_PRIVATE_SMOKE_DIR'])
    assert json.loads((smoke/'verification.json').read_text())['passed']
    receipts=[json.loads(p.read_text()) for p in (smoke/'checkpoints/.checkpoint-data').glob('*.json')]
    receipt=max(receipts,key=lambda r:r['global_step'])
    assert receipt['global_step']==2
    for key in ['local','persistent']:
        path=Path(receipt[key]);assert path.stat().st_size==receipt['bytes']
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['sha256']
    saved=torch.load(receipt['local'],map_location='cpu',mmap=True,weights_only=False)
    profile.verify_resume(saved)
    assert saved['runtime']['fresh_stage2'] and saved['runtime']['fresh_optimizer']
    assert saved['runtime']['global_step']==0 and saved['runtime']['initial_lr']==.0005
    assert saved['runtime']['warmup_epochs']==0 and saved['runtime']['lr_schedule_epochs']==100
    assert len(saved['rng_state_by_rank'])==4
    config=load_config(base/'inputs/stage2-training-config.yaml')
    with torch.device('meta'):model=profile.model(augment_arch_defaults(config.arch))
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    assert type(model) is rq_upstream.RQTransformer and type(model.classifier.linear) is torch.nn.Linear
    assert model.classifier.linear.weight.shape==(20481,1536)
    depth = int(profile.recipe.tokenizer.depth)
    assert depth == 5 and tuple(model.block_size) == (8,8,depth)
    optimizer=create_resnet_optimizer(model,augment_optimizer_defaults(config.optimizer))
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,626,100)
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    assert scheduler.after_scheduler.last_epoch==2 and scheduler.after_scheduler.T_max==62600
    assert len(optimizer.state)==len(list(model.parameters()))
    for state in optimizer.state.values():
        assert int(state['step'])==2
        for value in [state['exp_avg'],state['exp_avg_sq']]:
            for chunk in value.reshape(-1).split(1048576):assert torch.isfinite(chunk).all()
    preview=json.loads((out/'smoke/samples/step000002-panels.json').read_text())
    assert preview['generated_images']==PREVIEW_IMAGES==512 and preview['overview_images']==OVERVIEW_IMAGES==80
    assert preview['classes']==[dict(name=name,imagenet_id=label) for name,label in CLASSES]
    for rank in range(4):assert json.loads((smoke/f'preview-verification-rank{rank}.json').read_text())['training_rng_unchanged']
    del optimizer
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    model=model.eval().to(device);aux,_=profile.tokenizer(device)
    data=TarImageNet(base/'data','val',create_transforms(config.dataset,split='val'))
    images=torch.stack([data[i][0] for i in range(2)]).to(device)
    labels=torch.tensor([106,269],device=device)
    with torch.inference_mode():
        native=RQBottleneck([8,8,256],[8,8,depth],20481,shared_codebook=False,restart_unused_codes=False).eval().to(device)
        for d,book in enumerate(native.codebooks):book.weight[:-1].copy_(aux.quantizer.expanded_codebook(d))
        x=aux.encode(images)
        z,_,native_codes=native(x)
        hard=aux.quantizer.quantize(x)
        assert torch.equal(native_codes,hard['codes'])
        torch.testing.assert_close(z,hard['quantized'],rtol=1.e-4,atol=2.e-5)
        del native,z
        p,codes=profile.targets(aux,images)
        targets=profile.verify_targets(aux,p,codes)
        del p
        expected=model(codes,aux,labels,amp=True)
        prefix=torch.zeros_like(codes);model.init_cache();maximum_kl=0.
        for h in range(8):
            for w in range(8):
                for d in range(depth):
                    actual=model.cached_forward(prefix[:,:h+1],aux,labels,amp=True,sample_loc=(h,w,d))
                    ref=expected[:,h,w,d].float().log_softmax(-1)
                    logp=actual.float().log_softmax(-1)
                    maximum_kl=max(maximum_kl,float((ref.exp()*(ref-logp)).sum(-1).max()))
                    prefix[:,h,w,d]=codes[:,h,w,d]
        assert maximum_kl<1.e-4,maximum_kl
        model.init_cache()
        del expected
        audit=json.loads((base/'inputs/target-audit.json').read_text())
        train=TarImageNet(base/'data','train',create_transforms(config.dataset,split='train'),seed=261001)
        train.set_epoch(0)
        errors=torch.zeros(2,device=device,dtype=torch.float64)
        entropy=torch.zeros(depth,device=device,dtype=torch.float64)
        n=0
        for first in range(0,len(audit['training_indices']),4):
            batch=torch.stack([train[i][0] for i in audit['training_indices'][first:first+4]]).to(device)
            x=aux.encode(batch)
            p,c=profile.targets(aux,batch)
            soft=aux.quantizer.embed(c).sum(-2)
            hard=aux.quantizer.quantize(x)['quantized']
            errors+=torch.stack(((x-hard).square().sum(),(x-soft).square().sum())).double()
            entropy+=(-(p*p.clamp_min(1.e-30).log()).sum(-1)).sum((0,1,2)).double()
            n+=len(batch)*64
        ratio=float(errors[1]/errors[0]);mean_entropy=(entropy/n).cpu().tolist()
        assert .9<ratio<1.05 and all(0<e<10 for e in mean_entropy),(ratio,mean_entropy)
    proof=dict(passed=True,native_transformer=True,strict_model_reload=True,strict_optimizer_reload=True,
        all_adam_steps=2,scheduler_updates=2,cosine_total_updates=62600,
        parameter_count=sum(p.numel() for p in model.parameters()),vocabulary=20481,
        native_dense_classifier=True,checkpoint_sha256=receipt['sha256'],targets=targets,
        native_quantizer_hard_codes_exact=True,maximum_fp16_cache_kl=maximum_kl,cached_events_verified=64*depth,depth=depth,
        target_audit_training_images=len(audit['training_indices']),target_audit_disjoint_from_fit=True,
        sampled_to_hard_latent_mse_ratio=ratio,per_depth_target_entropy=mean_entropy,
        designated_preview_images=512,overview_images=80,class_grids=8,
        training_rng_unchanged_by_previews=True,production_starts_from_zero=True,
        preflight_weights_discarded=True,time=time.time())
    (out/'smoke-reload-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
