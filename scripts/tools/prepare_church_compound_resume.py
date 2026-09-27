#!/usr/bin/env python3
"""Recover and verify the requested Church compound run's immutable local inputs."""
import argparse, base64, hashlib, json, math, shutil, sys
from pathlib import Path
import torch


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    args=p.parse_args();base=args.base.resolve();inputs=args.inputs.resolve()
    sys.path.insert(0,str(base/'source'))
    from src.training import rqtransformer as recipe
    torch.set_num_threads(4)
    for kind in ('stage1','stage2'):
        metadata=json.loads((base/'recovery'/f'{kind}-files.json').read_text())
        for row in metadata:
            path=inputs/kind/row['name']
            if not path.is_file(): continue
            with path.open('rb') as f: digest=base64.b64encode(hashlib.file_digest(f,'md5').digest()).decode()
            assert digest==row['md5'],(path,digest,row['md5'])
    raw=torch.load(inputs/'stage1/last.ckpt',map_location='cpu',weights_only=False,mmap=True)
    assert raw['epoch']==3 and int(raw['state_dict']['_manual_train_step'])==2961
    allowed=('encoder.','decoder.','pre_bottleneck.','post_bottleneck.','bottleneck.dictionary')
    state={k:v.clone() for k,v in raw['state_dict'].items() if k.startswith(allowed)}
    assert state['bottleneck.dictionary'].shape==(256,16384)
    prepared=base/'prepared';prepared.mkdir(exist_ok=True)
    exported=prepared/'tokenizer.pt'
    provenance=dict(source_run='helloimlixin-rutgers/laser/church-laser-rfid421-full-ft3-20260918',
        source_file='last.ckpt',source_sha256=sha(inputs/'stage1/last.ckpt'),epoch=3,generator_updates=2961)
    torch.save(dict(state_dict=state,**provenance),exported)
    provenance['export_sha256']=sha(exported)
    (prepared/'tokenizer.json').write_text(json.dumps(provenance,indent=2)+'\n')
    del state,raw
    payload=torch.load(inputs/'stage2/last.pt',map_location='cpu',weights_only=False,mmap=True)
    c=payload['config'];assert c['wandb_id']=='church-laser-rfid421-ft3-compound-scratch90-20260918'
    assert c['compound_geometry_version']=='atom_conditional_v2'
    assert c['epochs']==c['lr_schedule_epochs']==90 and c['total_batch_size']==128
    assert c['world_size']==8 and c['batch_size']==16 and c['accumulation_steps']==1
    assert len(payload['rng_state_by_rank'])==8
    assert payload['epoch']*986+payload.get('batch_idx',0)==payload['global_step']
    assert payload['scheduler']['T_max']==88740 and payload['scheduler']['last_epoch']==payload['global_step']
    with torch.device('meta'):
        model=recipe.build_model(18432,16384,compound=True,coeff_vocab_size=2048,
            compound_micro_transformer_layers=2,compound_depth_specific_coeff_heads=True,
            compound_pair_autoregressive=True,sparsity_level=4,model_preset='lsun-church-350m')
    model.load_state_dict(payload['state_dict'],strict=True,assign=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=1e-4,betas=(.9,.95),fused=False)
    optimizer.load_state_dict(payload['optimizer'])
    assert len(optimizer.state)==517
    for parameter,s in optimizer.state.items():
        assert parameter.shape==s['exp_avg'].shape==s['exp_avg_sq'].shape
        assert int(s['step'])==payload['global_step']
    expected=c['lr']*.5*(1+math.cos(math.pi*payload['global_step']/88740))
    assert math.isclose(optimizer.param_groups[0]['lr'],expected,rel_tol=1e-9)
    schedule=recipe.create_cosine_lr_scheduler(optimizer,initial_lr=c['lr'],min_lr=0,total_steps=88740,
        completed_steps=payload['global_step'],state_dict=payload['scheduler'])
    assert math.isclose(optimizer.param_groups[0]['lr'],expected,rel_tol=1e-9)
    optimizer.step();schedule.step()
    next_lr=c['lr']*.5*(1+math.cos(math.pi*(payload['global_step']+1)/88740))
    assert math.isclose(optimizer.param_groups[0]['lr'],next_lr,rel_tol=1e-9)
    aux=recipe.LaserAux(exported,16384,2048,3.,6.4,attn_resolutions=(8,),coeff_scales=c['coeff_scales'],
        soft_target_physical=True,sparsity_level=4)
    assert not any(p.requires_grad for p in aux.parameters())
    (prepared/'source-config.json').write_text(json.dumps(c,indent=2,default=str)+'\n')
    old=Path('/scratch/xl598/runs/laser/church-laser-ft3ep-scratch-adaptive-lr-20260916-amarel')
    (base/'reference').mkdir(exist_ok=True)
    for name in ('real-statistics.npz','data-protocol.json','complete.json'):
        shutil.copy2(old/'reference'/name,base/'reference'/name)
    reference_hash=sha(base/'reference/real-statistics.npz')
    assert reference_hash=='ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364'
    (base/'data/church').mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        dest=base/'data/church'/f'church_outdoor_{split}_lmdb'
        if not dest.exists():dest.symlink_to(old/'assets/data'/f'church_outdoor_{split}_lmdb',target_is_directory=True)
    report=dict(passed=True,epoch=payload['epoch'],batch_idx=payload['batch_idx'],step=payload['global_step'],
        parameters=sum(p.numel() for p in model.parameters()),optimizer_entries=len(optimizer.state),
        checkpoint_lr=expected,next_lr=next_lr,world_size=8,global_batch=128,steps_per_epoch=986,
        schedule_steps=88740,coeff_scales=c['coeff_scales'],geometry_version=c['compound_geometry_version'],
        tokenizer=provenance,reference_sha256=reference_hash,stage2_sha256=sha(inputs/'stage2/last.pt'),
        saved_best_fid=payload['best_fid'],cache_rebuild_required=True,gpu_training_verified=False)
    (base/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
