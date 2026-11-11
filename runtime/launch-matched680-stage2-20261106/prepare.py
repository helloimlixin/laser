"""Freeze the selected stage1 checkpoint and prepare class-conditional VAR stage2."""
import json,hashlib,shutil,sys,tarfile,difflib,time
from pathlib import Path
import numpy as np
import torch
BASE=Path('/tmp/laser-var-stage2-matched680-20261106')
STAGE1=Path('/tmp/laser-var-full-stage1-20261105')
HERE=Path(__file__).resolve().parent
OUT=Path('/workspace/Projects/laser/outputs/imnet-s2-laser-matched680-var-d16-classcond-4gpu-20261106')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 BASE.mkdir(exist_ok=True);(BASE/'inputs').mkdir(exist_ok=True)
 best=json.loads((STAGE1/'run/best-reconstruction.json').read_text());source=STAGE1/'run/tokenizer-best-rfid.pt'
 assert best['step']==100090 and best['epoch']==10 and best['rfid']==2.069105617509024
 assert sha(source)==best['checkpoint']['checkpoint_sha256']=='cdea5ae33dff9943653f0fa8919db277547e6a4be39588472f26c45f0ce80599'
 target=BASE/'inputs/tokenizer.pt';assert not target.exists(),'Do not overwrite a staged tokenizer';target.hardlink_to(source)
 runtime=BASE/'runtime';shutil.copytree(STAGE1/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copyfile(HERE/'frozen_tokenizer.py',runtime/'frozen_tokenizer.py')
 sys.path.insert(0,str(runtime))
 from frozen_tokenizer import load_source,digest_state
 model=load_source(target,'laser','cuda:0')
 book=model.quantize.export_codebook();book.update(source_checkpoint_sha256=sha(target),source_artifact=best['checkpoint']['artifact'],source_step=100090,source_rfid=best['rfid'],fitted_after_training=False)
 torch.save(book,BASE/'inputs/joint-codebook.pt')
 source_digest=digest_state(model)
 del model;torch.cuda.empty_cache()
 old=json.loads((STAGE1/'run/config.json').read_text())
 reference=BASE/'inputs/validation-fid-reference.npz'
 with np.load(old['fid_reference']) as ref:np.savez(reference,mu=ref['mu'],sigma=ref['sigma'])
 with np.load(reference) as r,np.load(old['fid_reference']) as q:assert np.array_equal(r['mu'],q['mu']) and np.array_equal(r['sigma'],q['sigma'])
 original=(runtime/'train_joint4096_var.py').read_text();code=original
 def replace(a,b):
  nonlocal code
  assert code.count(a)==1,(a,code.count(a));code=code.replace(a,b)
 replace('from evaluate_local import load_source, LASER_HASH','from frozen_tokenizer import load_source, LASER_HASH, digest_state')
 replace('from src.rqvae_metrics import OriginalRQVAEInception, frechet_distance','from src.rqvae_metrics import OriginalRQVAEInception, frechet_distance, DistributedOriginalRQVAEMetrics')
 replace("        self.q=JointMultiScaleQuantizer(self.source.quantize,spec['dictionary'],spec['coefficients']).to(self.device)","        self.q=self.source.quantize\n        torch.testing.assert_close(self.q.dictionary.cpu(),spec['dictionary'],rtol=0,atol=0)\n        torch.testing.assert_close(self.q.coefficients.cpu(),spec['coefficients'],rtol=0,atol=0)\n        self.frozen_tokenizer_digest=digest_state(self.source)")
 replace("raise ValueError('Requires the full-evaluation rFID1.8393 book')", "raise ValueError('Requires the trained matched680/4096-symbol tokenizer')")
 replace("        if smoke:\n            self.train=SmokeImages(cfg['smoke_cache']);self.val=None", "        if smoke:\n            self.train=ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed']);self.val=None")
 replace("group='imagenet-var-joint4096-20261102'", "group='imagenet-var-matched680-stage2-20261106'")
 replace("'stage1/full_validation_rfid':1.8392875492706935", "'stage1/full_validation_rfid':cfg['stage1_rfid']")
 replace("        self.log('setup',parameters=", "        if self.run:\n            self.run.use_artifact(cfg['tokenizer_artifact'])\n            self.run.summary.update({'training/stage':'stage2','tokenizer/tokens_per_image':680,'tokenizer/vocabulary':4096,'tokenizer/nominal_bits_per_image':8160,'training/world_size':self.world,'training/microbatch_per_gpu':self.micro,'training/accumulation':self.accum,'evaluation/cadence':'epoch_end_only'})\n        self.log('setup',parameters=")
 replace("        if p['contract_sha256']!=self.contract_sha or p['codebook_sha256']!=self.cfg['codebook_sha256'] or p['world_size']!=self.world:", "        if p['contract_sha256']!=self.contract_sha or p['codebook_sha256']!=self.cfg['codebook_sha256'] or p['world_size']!=self.world or p['tokenizer_sha256']!=LASER_HASH:")
 replace("torch.nn.utils.clip_grad_norm_(self.model.parameters(),.1,error_if_nonfinite=True)","torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg['gradient_clip'],error_if_nonfinite=True)")
 replace("                if epoch==0 or (epoch+1)%5==0:self.fid()", "                self.fid()")
 replace("    torch.set_num_threads(4);torch.cuda.set_device", "    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.cuda.set_device")
 replace("            if any(p.requires_grad or p.grad is not None for p in experiment.source.parameters()):raise ValueError('Frozen source received gradients')", "            if any(p.requires_grad or p.grad is not None for p in experiment.source.parameters()):raise ValueError('Frozen source received gradients')\n            if digest_state(experiment.source)!=experiment.frozen_tokenizer_digest:raise ValueError('Tokenizer changed during prior preflight')\n            assert experiment.model.num_classes==1000 and experiment.model.cond_drop_rate==.1\n            assert experiment.model.L*experiment.model.sparsity==680\n            assert experiment.updates==1668")
 replace("parameters=sum(p.numel() for p in experiment.model.parameters()),global_batch=experiment.global_batch))", "parameters=sum(p.numel() for p in experiment.model.parameters()),global_batch=experiment.global_batch,microbatch=experiment.micro,accumulation=experiment.accum,updates_per_epoch=experiment.updates,tokenizer_digest_unchanged=True,class_conditional=True,num_classes=1000,tokens_per_image=680,vocabulary=4096,gradient_clip=cfg['gradient_clip'],peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))")
 start=code.index('    @torch.no_grad()\n    def fid(self):')
 end=code.index('\n    def train_loop(',start)
 new_fid='''    @torch.no_grad()
    def fid(self):
        assert self.progress['batch']==0 and self.progress['step']==self.progress['epoch']*self.updates
        with evaluation_rng(self.device):
            self.model.eval()
            if self.inception is None:self.inception=OriginalRQVAEInception().eval().requires_grad_(False).to(self.device)
            metric=DistributedOriginalRQVAEMetrics(self.device,compute_inception_score=True,reference_stats_path=self.cfg['fid_reference'],inception=self.inception)
            class_counts=torch.zeros(1000,device=self.device,dtype=torch.int64)
            positions=list(range(self.rank,50000,self.world));started=time.monotonic()
            for first in range(0,len(positions),self.cfg['evaluation_batch']):
                selected=positions[first:first+self.cfg['evaluation_batch']];labels=torch.tensor([i%1000 for i in selected],device=self.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    latent=self.model.sample(labels,cfg=self.cfg['sampling']['cfg'],top_k=self.cfg['sampling']['top_k'],top_p=self.cfg['sampling']['top_p'],seed=self.cfg['evaluation_seed']+1000003*self.progress['epoch']+selected[0])
                    for part in latent.split(self.cfg['decode_batch']):
                        images=self.source.decoder(self.source.post_quant_conv(part)).float().clamp(-1,1)
                        with torch.autocast('cuda',enabled=False):metric.update((images+1)*.5,real=False)
                class_counts+=torch.bincount(labels,minlength=1000)
                if self.rank==0:atomic(self.out/'generation-progress.json',dict(epoch=self.progress['epoch'],generated=min(50000,(first+len(selected))*self.world),target=50000,time=time.time()))
            assert int(metric.fake_count)==12500
            torch.manual_seed(self.cfg['evaluation_seed']+self.progress['epoch'])
            score,inception,inception_std=metric.compute(shuffle_for_inception=True);dist.all_reduce(class_counts)
            if int(metric.fake_count)!=50000 or not (class_counts==50).all():raise ValueError('Generation evaluation is not complete/class balanced')
            if self.rank==0:
                record=dict(step=self.progress['step'],epoch=self.progress['epoch'],validation_fid50k=score,inception_score=inception,inception_score_std=inception_std,count=50000,reference_count=50000,reference_split='validation',cfg=self.cfg['sampling']['cfg'],top_k=self.cfg['sampling']['top_k'],top_p=self.cfg['sampling']['top_p'],elapsed_seconds=time.monotonic()-started,cadence='epoch_end_only',tokenizer_sha256=LASER_HASH)
                atomic(self.out/f"fid-epoch{self.progress['epoch']:03d}.json",record);self.log('generation',**record)
            assert digest_state(self.source)==self.frozen_tokenizer_digest
            dist.barrier();self.model.train()
'''
 code=code[:start]+new_fid+code[end:]
 (runtime/'train_stage2.py').write_text(code)
 (BASE/'stage2-runtime.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='previous_prior/train_joint4096_var.py',tofile='matched680/train_stage2.py')))
 cfg=dict(output=str(BASE/'run'),checkpoint=str(target),source_checkpoint_sha256=sha(target),codebook=str(BASE/'inputs/joint-codebook.pt'),codebook_sha256=sha(BASE/'inputs/joint-codebook.pt'),tokenizer_artifact=best['checkpoint']['artifact'],stage1_rfid=best['rfid'],source_bundle=str(BASE/'source.tar.gz'),fresh_prior=True,seed=0,prior_seed=1001,epochs=200,lr=3e-4,warmup_epochs=4,lr_schedule='lin0',final_lr_ratio=.1,betas=[.9,.95],weight_decay=.05,gradient_clip=1.,microbatch=192,accumulation=1,global_batch=768,world_size=4,workers=4,checkpoint_every=1000,preview_every=500,class_dropout=.1,image_size=256,class_conditional=True,num_classes=1000,vocabulary=4096,directions=1024,coefficient_levels=4,sparsity=2,spatial_positions=340,joint_tokens_per_image=680,cnn_frozen=True,codebook_frozen=True,augmentation='Resize288 Lanczos; fresh deterministic RandomCrop256 each epoch; no horizontal flip',train_archive=old['train_archive'],training_archive_receipt=old['training_archive_receipt'],train_index=old['train_index'],val_archive=old['val_archive'],val_index=old['val_index'],val_labels=old['val_labels'],fid_reference=str(reference),fid_reference_split='validation',fid_samples=50000,evaluation_cadence='epoch_end_only',evaluation_batch=64,decode_batch=32,evaluation_seed=26110601,smoke_cache=old['smoke_cache'],sampling=dict(cfg=1.5,top_k=900,top_p=.96,temperature=1.),wandb=dict(id=OUT.name,name='ImageNet LASER VAR-d16 | class conditional | 680 tokens | frozen rFID2.0691 | 4 GPUs'))
 rows=json.loads(Path('/tmp/laser-var-stage1-20261105/recent-runs.json').read_text());vq=next(x for x in rows if x['id']=='imagenet256-var-vq-d16-stage2-20261026')
 v=vq['config']['prior'];assert v['lr']==cfg['lr'] and v['epochs']==cfg['epochs'] and v['weight_decay']==cfg['weight_decay'] and v['warmup_epochs']==cfg['warmup_epochs'] and v['final_lr_ratio']==cfg['final_lr_ratio']
 cfg['comparison']=dict(vq_baseline_run=vq['id'],shared_body='FoundationVision VAR-d16 depth16 width1024 heads16',global_batch=768,prior_seed=1001,optimizer='AdamW .9/.95, LR3e-4, weight decay .05, clip1.0, 4epoch warmup, lin0 floor0.1, 200epochs',token_budget='680 categorical IDs with vocabulary4096; 8160 nominal bits/image',representation_difference='LASER 340 spatial sites times two joint IDs; VQ 680 sites times one VQ ID',sampler='CFG1.5 top-k900 top-p.96; no rejection',metrics='FID50k and IS on 50 per ImageNet class, same float pixels and validation reference as existing local VQ control',stage1_history='LASER tokenizer selected after warm-start/tuning; stage1 comparison is not equal total tuning compute')
 keys=['train_stage2.py','frozen_tokenizer.py','src/models/joint_multiscale_var.py','src/models/trainable_joint_var_tokenizer.py','src/models/multiscale_laser_var.py','src/rqvae_metrics.py','src/original_rq_training.py','third_party/FoundationVision_VAR/models/var.py','third_party/FoundationVision_VAR/models/basic_var.py','third_party/FoundationVision_VAR/utils/lr_control.py']
 cfg['runtime_sha256']={n:sha(runtime/n) for n in keys}
 (BASE/'run').mkdir(exist_ok=True);write(BASE/'config.json',cfg);write(OUT/'config.json',cfg)
 for micro,accum in [(192,1),(96,2),(64,3)]:
  c=dict(cfg,output=str(BASE/f'preflight-mb{micro}'),microbatch=micro,accumulation=accum);Path(c['output']).mkdir(exist_ok=True);write(BASE/f'preflight-mb{micro}.json',c)
 report=dict(passed=True,stage1_checkpoint_sha256=sha(target),stage1_artifact=best['checkpoint']['artifact'],stage1_epoch=10,stage1_step=100090,stage1_rfid=best['rfid'],frozen_tokenizer_state_digest=source_digest,codebook_sha256=cfg['codebook_sha256'],token_budget=680,vocabulary=4096,nominal_bits_per_image=8160,spatial_positions=340,sparsity=2,prior_initialization='fresh VAR-d16',world_size=4,global_batch=768,target_epochs=200,updates_per_epoch=1668,target_updates=333600,matched_vq_stage2_optimizer_recipe=True,full_evaluation_cadence='once at each epoch end',reference_mu_sigma_unchanged=True,time=time.time())
 write(BASE/'source-verification.json',report);write(OUT/'source-verification.json',report);write(OUT/'comparison.json',cfg['comparison'])
 with tarfile.open(BASE/'source.tar.gz','w:gz') as a:
  for p in runtime.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:a.add(p,arcname='runtime/'+str(p.relative_to(runtime)))
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
