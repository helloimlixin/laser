"""Install the isolated LR experiment without modifying the completed run."""
from pathlib import Path
import json,shutil,py_compile

HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'plan.json').read_text());base=Path(plan['base'])
for name in ['lr_policy.py']:
 shutil.copyfile(HERE/name,base/'runtime'/name)
profile=base/'runtime/native_recipe.py'
s=profile.read_text()
s=s.replace('import torch\n','import torch\nimport json\nfrom omegaconf import OmegaConf\n',1)
s+='''
    def __init__(self, base):
        super().__init__(base)
        self.branch_spec=json.loads((self.base/'inputs/lr-branch.json').read_text())
        self.parent_run_id=self.branch_spec['parent_run_id']
        self._trial_sampling_override=None

    def verify_resume(self, payload):
        expected=OmegaConf.to_container(self.recipe,resolve=True)
        if payload['run_id']==self.parent_run_id:
            parent=OmegaConf.to_container(OmegaConf.load(self.base/'inputs/parent-recipe.yaml'),resolve=True)
            assert payload['recipe']==parent
            assert payload['global_step']==self.branch_spec['source_step']
            assert payload['next_epoch']==self.branch_spec['source_epoch'] and payload['next_microbatch']==0
            assert payload['scheduler'].get('learning_rate_branch') is None
        else:
            assert payload['run_id']==self.run_id and payload['recipe']==expected
            assert payload['scheduler']['learning_rate_branch']==self.branch_spec['policy']
        assert payload['joint_codebook_sha256']==self.codebook_sha256

    def checkpoint_extra(self):
        result=super().checkpoint_extra()
        result.update(learning_rate_branch=self.branch_spec['policy'],
            rewind_source=dict(parent_run_id=self.parent_run_id,source_epoch=self.branch_spec['source_epoch'],
                source_checkpoint_sha256=self.branch_spec['source_checkpoint_sha256'],
                lr_change_epoch=self.branch_spec['requested_rewind_epoch']))
        return result

    def metadata(self):
        result=super().metadata()
        result.update(learning_rate_branch=self.branch_spec['policy'],
            parent_run_id=self.parent_run_id,source_checkpoint_epoch=self.branch_spec['source_epoch'],
            source_checkpoint_sha256=self.branch_spec['source_checkpoint_sha256'],
            requested_rewind_epoch=self.branch_spec['requested_rewind_epoch'],
            exact_epoch68_checkpoint_recovered=False,replaying_to_epoch68=True,
            original_optimizer_moments_retained=True,branch_peak_lr=self.branch_spec['policy']['peak_lr'])
        result['initialization']=dict(result['initialization'],stage2_checkpoint_loaded=True,
            source_epoch=self.branch_spec['source_epoch'])
        return result

    def sampling_settings(self):
        if getattr(self,'_trial_sampling_override',None) is not None:
            return dict(self._trial_sampling_override)
        return super().sampling_settings()
'''
profile.write_text(s)
training=base/'source/src/training/rq_imagenet_shared.py'
s=training.read_text()
def replace(old,new,count=1):
 global s
 assert s.count(old)>=count,old[:100]
 s=s.replace(old,new,count)
replace("    restored_evaluation=None\n", "    restored_evaluation=None\n    saved_lr_policy=None\n")
replace("    if args.new_wandb_run and not args.resume:p.error('--new-wandb-run requires --resume')", "    if args.new_wandb_run and not args.resume:p.error('--new-wandb-run requires --resume')\n    if not args.resume:p.error('The LR rewind trial requires a retained parent checkpoint')")
replace("payload['run_id'] in (RUN,PROFILE.run_id)","payload['run_id'] in (RUN,PROFILE.run_id,PROFILE.parent_run_id)")
replace("        scheduler.load_state_dict(payload['scheduler']);scaler.load_state_dict(payload['scaler'])", "        scheduler.load_state_dict(payload['scheduler']);scaler.load_state_dict(payload['scaler'])\n        saved_lr_policy=payload['scheduler'].get('learning_rate_branch')")
replace("    ddp=DDP(model,", "    from runtime.lr_policy import LearningRateBranch\n    scheduler=LearningRateBranch(scheduler,optimizer,PROFILE.branch_spec['policy'],global_step,saved_lr_policy)\n    ddp=DDP(model,")
replace("lr_schedule_epochs=PROFILE.schedule_epochs,initial_lr=.0005,minimum_lr=0.,warmup_epochs=0,", "lr_schedule_epochs=PROFILE.schedule_epochs,initial_lr=.0005,minimum_lr=PROFILE.branch_spec['policy']['min_lr'],warmup_epochs=0,")
replace("def generation(model,aux,device,rank,world,epoch,run,batch=None,seed=261001):", "def generation(model,aux,device,rank,world,epoch,run,batch=None,seed=261001,inception_shuffle_seed=None):")
replace("metric.compute(shuffle_for_inception=True)","metric.compute(shuffle_for_inception=True,inception_shuffle_seed=inception_shuffle_seed)")
replace("        run.define_metric('eval/inception_score_std',step_metric='train/epoch')", "        run.define_metric('eval/inception_score_std',step_metric='train/epoch')\n        run.define_metric('eval/tuned_fid',step_metric='train/epoch',summary='min,last')\n        run.define_metric('eval/tuned_inception_score',step_metric='train/epoch',summary='max,last')\n        run.define_metric('eval/tuned_inception_score_std',step_metric='train/epoch')")
needle="        if evaluation is not None or (epoch+1)%int(config.experiment.save_ckpt_freq)==0:\n"
insert='''        if not args.smoke_updates and epoch+1 in PROFILE.branch_spec['evaluation']['tuned_epochs']:
            assert PROFILE._trial_sampling_override is None
            PROFILE._trial_sampling_override=PROFILE.branch_spec['evaluation']['tuned_sampler']
            try:
                tuned=generation(model,aux,device,rank,world,epoch+1,run,inception_shuffle_seed=261002)
                class TunedMedia:
                    def log(self,values):
                        run.log({('tuned_samples/'+key[len('samples/'):] if key.startswith('samples/') else key):value
                                 for key,value in values.items()})
                runtime_support.sample_panels(PROFILE,model,aux,device,rank,world,global_step,TunedMedia() if run else None,OUT/'tuned')
            finally:
                PROFILE._trial_sampling_override=None
            assert PROFILE.digest(aux)==before
            if rank==0:
                tuned_row=dict(epoch=epoch+1,global_step=global_step,fid=tuned['fid'],inception_score=tuned['inception_score'],
                    inception_score_std=tuned['inception_score_std'],sampler=tuned['sampler'],generated_images=50000,
                    reference_images=1281167,reference_split='train',seed=261001,is_shuffle_seed=261002)
                write_json(OUT/f'tuned-evaluations/epoch{epoch+1:03d}.json',tuned_row)
                prior_best=json.loads((OUT/'tuned-best.json').read_text()) if (OUT/'tuned-best.json').exists() else None
                if prior_best is None or tuned['fid']<prior_best['fid']:write_json(OUT/'tuned-best.json',tuned_row)
                if run:
                    run.log({'train/global_step':global_step,'train/epoch':epoch+1,'eval/tuned_fid':tuned['fid'],
                        'eval/tuned_inception_score':tuned['inception_score'],'eval/tuned_inception_score_std':tuned['inception_score_std']})
                    run.save(str(OUT/f'tuned-evaluations/epoch{epoch+1:03d}.json'),base_path=str(OUT),policy='now')
'''
replace(needle,insert+needle)
training.write_text(s)
metrics=base/'source/src/rqvae_metrics.py'
s=metrics.read_text().replace('def compute(self, *, shuffle_for_inception: bool = True):','def compute(self, *, shuffle_for_inception: bool = True, inception_shuffle_seed=None):',1)
old='''                if shuffle_for_inception:
                    # Upstream uses DataLoader(..., shuffle=True) before its
                    # ten contiguous splits.
                    probabilities = probabilities[
                        torch.randperm(probabilities.shape[0], device=self.device)
                    ]'''
new='''                if inception_shuffle_seed is not None:
                    probabilities=probabilities.cpu()
                    order=torch.randperm(len(probabilities),generator=torch.Generator().manual_seed(inception_shuffle_seed))
                    probabilities=probabilities[order]
                elif shuffle_for_inception:
                    # Preserve the parent's routine evaluation exactly.
                    probabilities = probabilities[
                        torch.randperm(probabilities.shape[0], device=self.device)
                    ]'''
assert old in s;s=s.replace(old,new,1);metrics.write_text(s)
support=base/'source/src/training/joint_runtime_support.py'
s=support.read_text()
s=s.replace("metadata={'native_rq':True,'fresh':True,'source_epoch':0,'warmup_epochs':0}","metadata={'native_rq':True,'fresh':False,'source_epoch':62,'lr_change_epoch':68,'warmup_epochs':0}")
s=s.replace("    run.use_artifact('helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9')\n",'')
s=s.replace("top-k={sampler['top_k'][0]}, top-p={sampler['top_p'][0]}","top-k={sampler['top_k']}, top-p={sampler['top_p']}")
support.write_text(s)
for f in [profile,training,metrics,support,base/'runtime/lr_policy.py']:
 py_compile.compile(str(f),doraise=True)
print(json.dumps(dict(installed=True,base=str(base),modified_files=[str(f.relative_to(base)) for f in [profile,training,metrics,support,base/'runtime/lr_policy.py']])) )
