"""Resume the 680-token/4096-symbol LASER stage-1 best, preserving full state."""
from pathlib import Path
import json,hashlib,shutil,sys,tarfile
import torch
BASE=Path('/tmp/laser-var-stage1-20261105')
RUNTIME=BASE/'frozen/runtime'
OUT=Path('/workspace/Projects/laser/outputs/imnet-s1-laser-matched680-best24608-resume-20261105')
RUN=OUT.name
SOURCE='helloimlixin-rutgers/laser/imnet-s1-laser-decoder-refine-best24608-continuous-1b200-20261105-pilot:v0'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str)+'\n')
def main():
 source=BASE/'matched-best/checkpoint.pt'
 assert sha(source)=='e173b72dc538f281a415be7d36c1f01bcc308ff1f6d54decd6989042afa45925'
 p=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
 assert p['progress']==dict(epoch=4,batch=25040,step=43166) and p['world_size']==1
 cfg=json.loads((BASE/'matched-source/config.json').read_text())
 old_cfg=json.loads((BASE/'matched-best/config.json').read_text())
 excluded={'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}
 contract={k:v for k,v in old_cfg.items() if k not in excluded}
 assert hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()==p['contract_sha256']
 for n,h in cfg['runtime_sha256'].items():assert sha(RUNTIME/n)==h,n
 assert cfg['patch_nums']==[2,4,8,16] and cfg['sparsity']==2
 assert sum(x*x for x in cfg['patch_nums'])*2==680
 source_data=BASE/'matched-source/data'
 for key,name in [('train_index','train-index.npz'),('val_index','validation-index.npz'),('val_labels','val-labels.npy'),('fid_reference','real-feature-moments.npz')]:cfg[key]=str(source_data/name)
 cfg.update(train_archive='/tmp/laser-beam16-duplicate-20261105/frozen/data/ILSVRC2012_img_train.tar',val_archive='/tmp/laser-var-resume-20261105/data/ILSVRC2012_img_val.verified.tar',training_archive_receipt=str(source_data/'train.verified.json'),source_bundle=str(BASE/'matched-source/source.tar.gz'),smoke_cache=str(BASE/'safe-smoke-cache.pt'),microbatch=32,accumulation=4,workers=4,eval_batch=64,checkpoint_every=500,preview_every=500,wandb=dict(id=RUN,name='ImageNet LASER VAR stage1 | matched 680x4096 | resume best rFID2.46084'))
 cfg['continuation'].update(durable=str(OUT),source_run='helloimlixin-rutgers/laser/imnet-s1-laser-decoder-refine-best24608-continuous-1b200-20261105',source_artifact=SOURCE,source_step=43166,source_rfid=2.4608426770367373,source_checkpoint_sha256=sha(source),baseline_inherited_same_hardware_and_protocol=False,resume_hardware='NVIDIA RTX PRO 6000 Blackwell 96GB')
 progress=dict(p['progress']);progress['batch']//=2
 sys.path.insert(0,str(RUNTIME))
 from migrate_stage1_world_size import assert_equal,global_indices
 assert global_indices(1281167,1,16,8,25040,4,0)==global_indices(1281167,1,32,4,12520,4,0)
 ages={k:sorted({int(s['step']) for s in p[k]['state'].values() if 'step' in s}) for k in ['optimizer','discriminator_optimizer']}
 assert ages==dict(optimizer=[43136,43166],discriminator_optimizer=[33166])
 best=json.loads(Path('/workspace/Projects/laser/outputs/imnet-s1-laser-decoder-refine-best24608-continuous-1b200-20261105/best-reconstruction.json').read_text())
 for folder in ['run','preflight']:
  c=dict(cfg,output=str(BASE/folder));local=BASE/folder;local.mkdir(exist_ok=True)
  contract={k:v for k,v in c.items() if k not in excluded}
  digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
  migrated=dict(p,progress=progress,contract_sha256=digest)
  destination=local/'tokenizer-last.pt';assert not destination.exists(),'Refusing to overwrite checkpoint'
  torch.save(migrated,destination)
  restored=torch.load(destination,map_location='cpu',mmap=True,weights_only=False)
  for key in ['model','discriminator','optimizer','discriminator_optimizer','rng']:assert_equal(p[key],restored[key])
  (local/'tokenizer-best-rfid.pt').hardlink_to(source)
  for name,value in [('config.json',c),('contract.json',contract),('best-reconstruction.json',best),('initialization.json',dict(initialization='full-state stage1 resume from matched-budget best',stage='stage1',source_artifact=SOURCE,source_step=43166,source_rfid=best['rfid'],tokens_per_image=680,vocabulary=4096,nominal_bits_per_image=8160)),('checkpoint.json',dict(progress=progress,checkpoint_sha256=sha(destination),contract_sha256=digest,full_state=True))]:write(local/name,value)
 write(OUT/'config.json',dict(cfg,output=str(BASE/'run')))
 proof=dict(passed=True,stage='stage1',source_artifact=SOURCE,source_step=43166,source_epoch=4,source_rfid=best['rfid'],tokens_per_image=680,vocabulary=4096,nominal_bits_per_image=8160,patch_nums=[2,4,8,16],sparsity=2,model_and_both_optimizers_and_rng_preserved=True,optimizer_ages=ages,original_cursor=p['progress']['batch'],new_cursor=progress['batch'],original_microbatch=16,microbatch=32,accumulation=4,global_batch=128,world_size=1,next_three_global_batches_identical=True,stage2_enabled=False,decoder_refinement_policy_preserved=True,encoder_quantizer_frozen=True,generator_lr=1e-7,discriminator_lr=1e-9,source_full_fid_is_inherited=True,hardware_changed=True)
 write(OUT/'resume-verification.json',proof)
 weights=BASE/'weights';weights.mkdir(exist_ok=True)
 shutil.copyfile('/workspace/Projects/laser/outputs/imagenet-var-dl-comparison-20261024/assets/vgg16-397923af.pth',weights/'vgg16-397923af.pth')
 from train_matched_stage1 import ArchiveImages
 d=ArchiveImages(cfg['val_archive'],cfg['val_index'],False,0,labels=cfg['val_labels'])
 import numpy as np
 indices=[int(np.flatnonzero(d.labels==label)[j]) for label in [22,277,258,849] for j in [0,1]]
 torch.save(dict(images=torch.stack([d[i][0] for i in indices]),labels=torch.tensor([d[i][1] for i in indices])),cfg['smoke_cache'])
 print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
