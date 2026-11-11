"""Create an isolated continuation from a selected original late checkpoint."""
from pathlib import Path
import hashlib
import json
import math
import py_compile
import shutil
from omegaconf import OmegaConf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=Path('/tmp/laser-rq24576-native-8h100-20261108')


def install(plan):
    base=Path(plan['base']);out=Path(plan['output'])
    base.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    shutil.copytree(PARENT/'source',base/'source',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    shutil.copytree(PARENT/'runtime',base/'runtime',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    for name in ['upstream','data','torch-cache','compile-cache']:
        target=base/name
        if not target.exists():target.symlink_to(PARENT/name,target_is_directory=True)
    for name in ['inputs','checkpoints','smoke','wandb','wandb-cache']:(base/name).mkdir(exist_ok=True)
    for path in (PARENT/'inputs').iterdir():
        target=base/'inputs'/path.name
        if path.suffix in ['.pt','.npz']:
            if not target.exists():target.symlink_to(path.resolve())
        elif path.is_file():shutil.copyfile(path,target)
    shutil.copyfile(PARENT/'train.py',base/'train.py')
    shutil.copyfile(PARENT/'inputs/recipe.yaml',base/'inputs/parent-recipe.yaml')
    shutil.copyfile(PARENT/'runtime/native_recipe.py',base/'runtime/native_parent_recipe.py')
    for name in ['native_recipe.py','refinement.py']:shutil.copyfile(HERE/name,base/'runtime'/name)
    shutil.copyfile(ROOT/'outputs/imagenet-rq24576-late-teacher-calibration-20261110/report.json',base/'inputs/teacher-calibration.json')
    (base/'inputs/late-refinement.json').write_text(json.dumps(plan,indent=2)+'\n')
    recipe=OmegaConf.load(PARENT/'inputs/recipe.yaml')
    recipe.version='native-rq24576-late-noise-ema-v1'
    recipe.run_id=plan['run_id'];recipe.name=plan['name'];recipe.epochs=plan['target_epoch']
    recipe.targets.temperature=plan['teacher']['final']
    recipe.targets.temperature_schedule=plan['teacher']
    recipe.sampling=OmegaConf.create(plan['sampler'])
    recipe.evaluation.include_epoch_one=False
    recipe.initialization.stage2_checkpoint_loaded=True
    recipe.initialization.policy='restore original late model and all Adam moments; initialize EMA from that model'
    recipe.lineage.fresh_stage2=False;recipe.lineage.fresh_optimizer=False
    recipe.lineage.parent_run_id=plan['parent_run_id'];recipe.lineage.source_epoch=plan['source_epoch']
    recipe.refinement=OmegaConf.create(dict(lr=plan['lr'],ema_decay=plan['ema_decay']))
    OmegaConf.save(recipe,base/'inputs/recipe.yaml')
    config=OmegaConf.load(base/'inputs/stage2-training-config.yaml')
    config.experiment.epochs=plan['target_epoch'];OmegaConf.save(config,base/'inputs/stage2-training-config.yaml')
    execution=json.loads((base/'inputs/execution.json').read_text())
    execution.update(epochs=plan['target_epoch'],fresh_stage2=False,fresh_optimizer=False,
        model_and_optimizer_restored=True,source_epoch=plan['source_epoch'],
        temperature_schedule=plan['teacher'],ema_decay=plan['ema_decay'])
    execution.pop('measured_images_per_second',None)
    (base/'inputs/execution.json').write_text(json.dumps(execution,indent=2)+'\n')

    path=base/'source/src/training/rq_imagenet_shared.py';s=path.read_text()
    def replace(old,new):
        nonlocal s
        assert s.count(old)==1,(s.count(old),old[:120])
        s=s.replace(old,new,1)
    replace("    if args.new_wandb_run and not args.resume:p.error('--new-wandb-run requires --resume')",
        "    if not args.resume:p.error('Late continuation requires a full retained checkpoint')\n    assert args.epochs==PROFILE.epochs")
    replace("    restored_evaluation=None\n","    restored_evaluation=None\n    saved_lr_policy=None\n    from runtime.refinement import ModelEMA,LearningRateRefinement\n")
    replace("payload['run_id'] in (RUN,PROFILE.run_id)","payload['run_id'] in (RUN,PROFILE.run_id,PROFILE.parent_run_id)")
    replace("        model.load_state_dict(payload['state_dict'],strict=True);optimizer.load_state_dict(payload['optimizer'])",
        "        raw_key=payload.get('optimizer_weights_key','state_dict')\n        model.load_state_dict(payload[raw_key],strict=True);optimizer.load_state_dict(payload['optimizer'])\n        ema=ModelEMA(model,PROFILE.refinement['ema_decay'])\n        if payload['run_id']==PROFILE.run_id:\n            ema.model.load_state_dict(payload['state_dict'],strict=True);ema.updates=int(payload['ema_updates'])")
    replace("        scheduler.load_state_dict(payload['scheduler']);scaler.load_state_dict(payload['scaler'])",
        "        scheduler.load_state_dict(payload['scheduler']);scaler.load_state_dict(payload['scaler'])\n        saved_lr_policy=payload['scheduler'].get('late_refinement')")
    replace("        restored_evaluation=payload.get('evaluation')",
        "        restored_evaluation=payload.get('evaluation')\n        if payload['run_id']==PROFILE.parent_run_id:\n            restored_evaluation=dict(PROFILE.refinement['baseline'])\n            restored_best_fid=restored_evaluation['fid']")
    replace("    ddp=DDP(model,",
        "    PROFILE.set_training_step(global_step)\n    scheduler=LearningRateRefinement(scheduler,optimizer,PROFILE.refinement['lr'],global_step,saved_lr_policy)\n    assert ema.updates==global_step-PROFILE.refinement['source_step']\n    ddp=DDP(model,")
    replace("lr_schedule_epochs=PROFILE.schedule_epochs,initial_lr=.0005,minimum_lr=0.,warmup_epochs=0,",
        "lr_schedule_epochs=PROFILE.schedule_epochs,initial_lr=optimizer.param_groups[0]['lr'],minimum_lr=PROFILE.refinement['lr']['min_lr'],warmup_epochs=1,\n        ema_updates_at_start=ema.updates,checkpoint_training_weights='state_dict_raw',inference_weights='state_dict',")
    replace("def generation(model,aux,device,rank,world,epoch,run,batch=None,seed=261001):",
        "def generation(model,aux,device,rank,world,epoch,run,batch=None,seed=261001):")
    # Keep IS split assignment identical to the matched frozen sampler studies.
    replace("metric.compute(shuffle_for_inception=True)","metric.compute(shuffle_for_inception=True,inception_shuffle_seed=261002)")
    replace("        run.define_metric('eval/inception_score_std',step_metric='train/epoch')",
        "        run.define_metric('eval/inception_score_std',step_metric='train/epoch')\n        run.define_metric('eval/raw_fid',step_metric='train/epoch',summary='min,last')\n        run.define_metric('eval/raw_inception_score',step_metric='train/epoch',summary='max,last')\n        run.define_metric('eval/raw_inception_score_std',step_metric='train/epoch')")
    replace("            'training/auxiliary_loss_terms':0,'training/objective':PROFILE.metadata()['objective']})",
        "            'training/auxiliary_loss_terms':0,'training/objective':PROFILE.metadata()['objective'],\n            'evaluation/published_target_fid':11.56,'evaluation/primary_weights':'ema'})")
    replace("                'eval/inception_score_std':restored_evaluation['inception_score_std']})",
        "                'eval/inception_score_std':restored_evaluation['inception_score_std'],\n                'eval/raw_fid':restored_evaluation['fid'],'eval/raw_inception_score':restored_evaluation['inception_score'],\n                'eval/raw_inception_score_std':restored_evaluation['inception_score_std']})")
    replace("state=cpu_snapshot(dict(run_id=RUN,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),",
        "state=cpu_snapshot(dict(run_id=RUN,state_dict=ema.model.state_dict(),state_dict_raw=model.state_dict(),\n                ema_updates=ema.updates,optimizer=optimizer.state_dict(),")
    replace("        images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)\n        sync=",
        "        PROFILE.set_training_step(global_step)\n        images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)\n        sync=")
    replace("                scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1",
        "                ema.update(model)\n                scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1")
    replace("'train/grad_norm':float(norm),'train/amp_scale':scaler.get_scale(),",
        "'train/grad_norm':float(norm),'train/amp_scale':scaler.get_scale(),\n                    'train/teacher_temperature':PROFILE.target_temperature(),'train/ema_updates':ema.updates,")
    replace("                if not args.smoke_updates and (global_step==10 or global_step%250==0):",
        "                if not args.smoke_updates and (global_step==PROFILE.refinement['source_step']+2 or global_step%500==0):")
    replace("                            frozen_tokenizer_unchanged=True,initialization=initial))",
        "                            frozen_tokenizer_unchanged=True,ema_updates=ema.updates,\n                            checkpoint_schema='EMA inference + raw optimizer weights',initialization=initial))")
    replace("                    checkpoint(epoch,epoch,absolute+1)\n                    if rank==0:\n                        save_future.result()",
        "                    write_json(folder/f'ema-generation-interface-rank{rank}.json',\n                               verify_generation(PROFILE,ema.model,aux,device))\n                    checkpoint(epoch,epoch,absolute+1)\n                    if rank==0:\n                        save_future.result()")
    old="""        if epoch==0 or (epoch+1)%int(config.experiment.test_freq)==0:
            evaluation=generation(model,aux,device,rank,world,epoch+1,run)
            assert PROFILE.digest(aux)==before
            if rank==0:
                row={'train/epoch':epoch+1,'train/global_step':global_step,
                    'eval/fid':evaluation['fid'],'eval/inception_score':evaluation['inception_score'],
                    'eval/inception_score_std':evaluation['inception_score_std']}
                write_json(OUT/f'evaluations/epoch{epoch+1:03d}.json',dict(row,generated_images=50000,
                    real_images=1281167,real_split='train',rejection_sampling=False,sampler=evaluation['sampler']))
                print(json.dumps(row),flush=True)
                if run:run.log(row);run.save(str(OUT/f'evaluations/epoch{epoch+1:03d}.json'),base_path=str(OUT),policy='now')"""
    new="""        if not args.smoke_updates and ((epoch+1-PROFILE.refinement['source_epoch'])%2==0 or epoch+1==args.epochs):
            evaluation=generation(ema.model,aux,device,rank,world,epoch+1,run)
            evaluation.update(weights='ema',ema_decay=ema.decay,ema_updates=ema.updates)
            class EMAMedia:
                def log(self,values):
                    run.log({('ema_samples/'+key[len('samples/'):] if key.startswith('samples/') else key):value
                             for key,value in values.items()})
            runtime_support.sample_panels(PROFILE,ema.model,aux,device,rank,world,global_step,EMAMedia() if run else None,OUT/'ema')
            raw=None
            offset=epoch+1-PROFILE.refinement['source_epoch']
            if offset==2 or offset%4==0 or epoch+1==args.epochs:
                raw=generation(model,aux,device,rank,world,epoch+1,run)
                raw.update(weights='raw')
            assert PROFILE.digest(aux)==before
            if rank==0:
                row={'train/epoch':epoch+1,'train/global_step':global_step,
                    'eval/fid':evaluation['fid'],'eval/inception_score':evaluation['inception_score'],
                    'eval/inception_score_std':evaluation['inception_score_std']}
                if raw:
                    row.update({'eval/raw_fid':raw['fid'],'eval/raw_inception_score':raw['inception_score'],
                                'eval/raw_inception_score_std':raw['inception_score_std']})
                record=dict(row,generated_images=50000,real_images=1281167,real_split='train',
                    rejection_sampling=False,sampler=evaluation['sampler'],ema=evaluation,raw=raw,
                    seed=261001,is_shuffle_seed=261002,target_fid=11.56)
                write_json(OUT/f'evaluations/epoch{epoch+1:03d}.json',record)
                previous=json.loads((OUT/'best-evaluation.json').read_text()) if (OUT/'best-evaluation.json').exists() else None
                winner=dict(raw if raw is not None and raw['fid']<evaluation['fid'] else evaluation,
                    epoch=epoch+1,global_step=global_step,generated_images=50000,reference_images=1281167,
                    reference_split='train',seed=261001,is_shuffle_seed=261002)
                if previous is None or winner['fid']<previous['fid']:write_json(OUT/'best-evaluation.json',winner)
                print(json.dumps(row),flush=True)
                if run:
                    run.log(row);run.save(str(OUT/f'evaluations/epoch{epoch+1:03d}.json'),base_path=str(OUT),policy='now')
                    run.summary.update({'evaluation/best_raw_or_ema_fid':min(winner['fid'],previous['fid'] if previous else math.inf),
                        'evaluation/target_reached_single_seed':min(winner['fid'],previous['fid'] if previous else math.inf)<11.56})"""
    replace(old,new)
    path.write_text(s)

    metrics=base/'source/src/rqvae_metrics.py';s=metrics.read_text()
    s=s.replace('def compute(self, *, shuffle_for_inception: bool = True):',
        'def compute(self, *, shuffle_for_inception: bool = True, inception_shuffle_seed=None):',1)
    needle='''                if shuffle_for_inception:
                    # Upstream uses DataLoader(..., shuffle=True) before its
                    # ten contiguous splits.
                    probabilities = probabilities[
                        torch.randperm(probabilities.shape[0], device=self.device)
                    ]'''
    assert needle in s
    s=s.replace(needle,'''                if inception_shuffle_seed is not None:
                    probabilities=probabilities.cpu()
                    order=torch.randperm(len(probabilities),generator=torch.Generator().manual_seed(inception_shuffle_seed))
                    probabilities=probabilities[order]
                elif shuffle_for_inception:
                    probabilities=probabilities[torch.randperm(probabilities.shape[0],device=self.device)]''',1)
    metrics.write_text(s)
    support=base/'source/src/training/joint_runtime_support.py';s=support.read_text()
    s=s.replace("metadata={'native_rq':True,'fresh':True,'source_epoch':0,'warmup_epochs':0}",
        "metadata={'native_rq':True,'fresh':False,'source_epoch':"+str(plan['source_epoch'])+",'late_refinement':True}")
    s=s.replace("    run.use_artifact('helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9')\n",'')
    s=s.replace("top-k={sampler['top_k'][0]}, top-p={sampler['top_p'][0]}","top-k={sampler['top_k']}, top-p={sampler['top_p']}")
    s=s.replace("weights='training model'","weights='ema' if Path(out).name=='ema' else 'raw'")
    support.write_text(s)
    store=(HERE.parent/'rq24576-lr-rewind68-8h100-20261110/checkpoint_store.py').read_text()
    store=store.replace("        keep.add('step-0042568.pt')\n",'')
    (base/'runtime/checkpoint_store.py').write_text(store)
    files=[path,metrics,support,base/'runtime/checkpoint_store.py',base/'runtime/native_recipe.py',base/'runtime/refinement.py']
    for file in files:py_compile.compile(str(file),doraise=True)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(installed=True,base=str(base),run_id=plan['run_id'])))


if __name__=='__main__':install(json.loads((HERE/'plan.json').read_text()))
