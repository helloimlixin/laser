"""Prepare a reproducible continuation of the matched680 experiment with vector IDs."""
import hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-matched680-vector6144-20261111')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/imnet-s2-matched680-vector6144-var-d16-4h100-20261111'

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')

def main():
 runtime=BASE/'runtime/runtime';OUT.mkdir(parents=True,exist_ok=True)
 assert sha(BASE/'checkpoint/checkpoint.pt')=='cdea5ae33dff9943653f0fa8919db277547e6a4be39588472f26c45f0ce80599'
 for name in ['src/models/residual_vector_var.py','src/models/trainable_joint_var_tokenizer.py','src/rqvae_metrics.py']:
  shutil.copyfile(ROOT/name,runtime/name)
 shutil.copyfile(ROOT/'runtime/launch-matched680-stage2-20261106/frozen_tokenizer.py',runtime/'frozen_tokenizer.py')
 shutil.copyfile(HERE/'quality.py',runtime/'vector_quality.py')
 original=(BASE/'source/train_joint4096_var.py').read_text();code=original
 def replace(a,b):
  nonlocal code
  assert code.count(a)==1,(a,code.count(a));code=code.replace(a,b)
 replace('from evaluate_local import load_source, LASER_HASH','from frozen_tokenizer import load_source, LASER_HASH, digest_state\nfrom src.models.residual_vector_var import ResidualVectorQuantizer, ResidualVectorVAR')
 replace("        self.q=JointMultiScaleQuantizer(self.source.quantize,spec['dictionary'],spec['coefficients']).to(self.device)\n        self.q.quantize_scale=functools.partial(self.q.quantize_scale,chunk_size=2048)\n        if (self.q.vocab_size,self.q.num_directions,self.q.levels)!=(4096,1024,4):raise ValueError('Requires the full-evaluation rFID1.8393 book')", "        self.q=ResidualVectorQuantizer(self.source.quantize.source,spec['vectors'],temperature=cfg['teacher_temperature']).to(self.device)\n        assert self.q.vocab_size==6144 and self.q.sparsity==2\n        self.frozen_tokenizer_digest=digest_state(self.source)")
 replace('self.model=JointLaserVAR(self.q,depth=16,num_classes=1000)', 'self.model=ResidualVectorVAR(self.q,depth=16,num_classes=1000)')
 replace("self.train=SmokeImages(cfg['smoke_cache']);self.val=None", "self.train=ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed']);self.val=None")
 replace("group='imagenet-var-joint4096-20261102'", "group='matched680-vector6144-20261111'")
 replace("'stage1/full_validation_rfid':1.8392875492706935", "'stage1/full_validation_rfid':cfg['stage1_rfid'],'tokenizer/vocabulary':6144,'tokenizer/tokens_per_image':680,'tokenizer/teacher_temperature':cfg['teacher_temperature'],'tokenizer/representation':'independent_scale_depth_vector_ids','training/world_size':self.world")
 replace("            a=wandb.Artifact(cfg['wandb']['id']+'-source'", "            self.run.use_artifact(cfg['tokenizer_artifact'])\n            a=wandb.Artifact(cfg['wandb']['id']+'-source'")
 replace("return self.q.decompose(latent)", "return self.q.decompose(latent,stochastic=True)")
 code=code.replace("codes['inputs'],codes['codes'])", "codes['inputs'],codes['codes'],codes['residuals'])")
 code=code.replace("code['inputs'],code['codes'])", "code['inputs'],code['codes'],code['residuals'])")
 replace("torch.nn.utils.clip_grad_norm_(self.model.parameters(),.1,error_if_nonfinite=True)","torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)")
 code=code.replace('(math.log(4096)+math.log(4092))*.5','math.log(self.q.vocab_size)')
 code=code.replace("images=self.source.decoder(self.source.post_quant_conv(latent)).float().clamp(-1,1)", "images=torch.cat([self.source.decoder(self.source.post_quant_conv(part)).float().clamp(-1,1) for part in latent.split(8)])")
 replace("num_workers=self.cfg['workers'] if not self.smoke else 0,pin_memory=True,drop_last=True,multiprocessing_context='spawn' if self.cfg['workers'] and not self.smoke else None", "num_workers=self.cfg['workers'],pin_memory=True,drop_last=True,prefetch_factor=2,multiprocessing_context='spawn'")
 # The checkpoint state verifies codebook identity; the source digest additionally checks the frozen CNN.
 replace("            if any(p.requires_grad or p.grad is not None for p in experiment.source.parameters()):raise ValueError('Frozen source received gradients')", "            if any(p.requires_grad or p.grad is not None for p in experiment.source.parameters()):raise ValueError('Frozen source received gradients')\n            assert digest_state(experiment.source)==experiment.frozen_tokenizer_digest")
 replace("                if epoch==0 or (epoch+1)%5==0:self.fid()", "                self.fid()")
 replace("    torch.set_num_threads(4);torch.cuda.set_device", "    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_device")
 code=code.replace('from utils.lr_control import lr_wd_annealing','from utils.lr_control import lr_wd_annealing\nfrom async_upload import submit as submit_upload, finish as finish_uploads')
 start=code.index("                a=wandb.Artifact(self.cfg['wandb']['id']+'-prior'")
 end=code.index("        dist.barrier()",start)
 code=code[:start]+"                aliases=['latest',f\"step-{self.progress['step']:09d}\"]\n                if self.progress['batch']==0:aliases.append(f\"epoch-{self.progress['epoch']:03d}\")\n                submit_upload(self.run,p,Path(self.cfg['codebook']),self.out/'config.json',self.out,receipt,aliases)\n"+code[end:]
 code=code.replace('self.run.finish()', 'finish_uploads();self.run.finish()')
 code=code.replace('experiment.run.finish()', 'finish_uploads();experiment.run.finish()')
 shutil.copyfile(HERE/'async_upload.py',runtime/'async_upload.py')
 code=code.replace("        self.wrapped=DDP(self.model,device_ids=[self.device.index],broadcast_buffers=False)","        self.wrapped=DDP(self.model,device_ids=[self.device.index],broadcast_buffers=False)\n        if cfg.get('compile',False):self.wrapped=torch.compile(self.wrapped,mode='default',dynamic=False)")
 (runtime/'train_vector_stage2.py').write_text(code)
 old=json.loads((ROOT/'outputs/imnet-s2-laser-beam16-joint4096-var-d16-4gpu-20261106/config.json').read_text())
 for key in ['runtime_sha256','beam_width','beam_chunk_size','beam_teacher','teacher','comparison','coefficient_representation','pair_token_format','directions','coefficient_levels']:old.pop(key,None)
 old.update(output=str(BASE/'train'),checkpoint=str(BASE/'checkpoint/checkpoint.pt'),source_bundle=str(BASE/'source-vector6144.tar.gz'),codebook=str(BASE/'quality/selected-codebook.pt'),codebook_sha256='',vocabulary=6144,microbatch=96,accumulation=2,workers=4,teacher_temperature=.5,stage1_rfid=2.069105617509024,
  train_archive=str(BASE/'data/ILSVRC2012_img_train.tar'),val_archive=str(BASE/'data/ILSVRC2012_img_val.tar'),
  train_index=str(BASE/'full-source/data/train-index.npz'),val_index=str(BASE/'data/val-index.npz'),val_labels=str(BASE/'full-source/data/val-labels.npy'),fid_reference=str(BASE/'full-source/data/real-feature-moments.npz'),training_archive_receipt=str(BASE/'data/ILSVRC2012_img_train.tar.verified.json'),
  wandb=dict(id=OUT.name,name='ImageNet | matched680 stage1 | vector6144 | fresh classcond VAR-d16 | 4H100'),
  tokenization=dict(kind='native_residual_vector_ids',tables=[4,2,6144,32],zero_id=0,coefficient_tokens=False,beam_search=False,fit_images=50000,heldout_training_images=10000,fit_passes=24),
  comparison=dict(vq_run='imagenet256-var-vq-d16-stage2-20261026',vq_tokens=680,vq_vocab=4096,laser_tokens=680,laser_vocab=6144,equal_token_count=True,equal_bitrate=False,stage1_history='warm-started and tuned; not equal stage1 compute'),checkpoint_every=1000,preview_every=500)
 shutil.copyfile(ROOT.parent/'data/imagenet/validation-verified-20261107/val-index.npz',BASE/'data/val-index.npz')
 write(BASE/'config.json',old);write(OUT/'config.json',old)
 write(OUT/'source-verification.json',dict(passed=True,checkpoint_sha256=sha(BASE/'checkpoint/checkpoint.pt'),stage1_artifact=old['tokenizer_artifact'],stage1_rfid=old['stage1_rfid'],stage1_run='imnet-s1-laser-matched680-full-continue-best-4gpu-20261105',token_budget=680,source_vocab=4096,new_vocab=6144,encoder_decoder_dictionary_frozen=True,prior_initialization='fresh',world_size=4))
 print('Prepared',runtime,flush=True)
if __name__=='__main__':main()
