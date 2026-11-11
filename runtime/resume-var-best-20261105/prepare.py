"""Restore the immutable joint4096 VAR contract and select the best evaluated prior."""
import json,hashlib,shutil,os
from pathlib import Path
import torch
BASE=Path('/tmp/laser-var-resume-20261105')
OUT=Path('/workspace/Projects/laser/outputs/imnet-var-j4096-bestep1-continue200-4rtx6k-20261105')
RUN='imnet-var-j4096-bestep1-continue200-4rtx6k-20261105'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,default=str)+'\n')
def link(destination,source):
 destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.is_symlink():assert destination.resolve()==source.resolve(),destination
 elif destination.exists():assert destination.resolve()==source.resolve(),destination
 else:destination.symlink_to(source,target_is_directory=source.is_dir())

def main():
 c=json.loads((BASE/'source-artifact/config.json').read_text())
 contract={k:v for k,v in c.items() if k not in ['output','wandb','checkpoint_every','preview_every','workers','source_bundle']}
 digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
 assert digest=='c89e8f24fafe6b0852a65d8be67f63e2c49dc04810573be41ed5d3be3e2a3fee'
 prior=BASE/'prior-artifact/checkpoint.pt';tokenizer=BASE/'tokenizer-artifact/checkpoint.pt'
 assert sha(prior)=='9cff13c1532c46b36a4dccd3d378270befb9cff4c5d19fc2bab5647d6e9e858c'
 assert sha(tokenizer)==c['source_checkpoint_sha256']
 for name,expected in c['runtime_sha256'].items():assert sha(BASE/'recovered/runtime'/name)==expected,name
 p=torch.load(prior,map_location='cpu',mmap=True,weights_only=False)
 assert p['progress']==dict(epoch=1,batch=0,step=1668)
 assert p['world_size']==4 and p['contract_sha256']==digest and len(p['rng'])==4
 ages=sorted({int(s['step']) for s in p['optimizer']['state'].values() if 'step' in s})
 assert ages==[1668],ages
 assert p['tokenizer_sha256']==c['source_checkpoint_sha256'] and p['codebook_sha256']==c['codebook_sha256']
 link(BASE/'runtime',BASE/'recovered/runtime')
 link(Path('/tmp/laser-imagenet-stage2-joint4096-20261102'),BASE)
 link(Path(c['checkpoint']),tokenizer)
 link(BASE/'source.tar.gz',BASE/'source-artifact/source.tar.gz')
 for name in ['train-index.npz','val-labels.npy']:
  shutil.copyfile(BASE/'source-artifact/data'/name,BASE/'data'/name)
 for key,name in [('train_index','train-index.npz'),('val_labels','val-labels.npy')]:assert sha(BASE/'data'/name)==c[key+'_sha256']
 full=Path('/tmp/laser-var-joint-full-eval-20261102');(full/'full').mkdir(parents=True,exist_ok=True)
 for name in ['validation-index.npz','real-feature-moments.npz']:shutil.copyfile(BASE/'source-artifact/data'/name,full/'full'/name)
 assert sha(c['val_index'])==c['val_index_sha256'] and sha(c['fid_reference'])==c['fid_reference_sha256']
 validation=json.loads((BASE/'data/validation-archive.json').read_text());assert validation['verified'] and validation['md5']=='29b22e2961454d5413ddabcf34fc5622'
 link(Path(c['val_archive']),BASE/'data/ILSVRC2012_img_val.verified.tar')
 train=Path('/tmp/laser-beam16-duplicate-20261105/frozen/data/ILSVRC2012_img_train.tar')
 verified=json.loads(train.with_suffix('.verified.json').read_text());assert verified['verified'] and verified['md5']=='1d675b47d978889d74fa0da5fadfb00e'
 link(Path(c['train_archive']),train)
 receipt=json.loads((BASE/'source-artifact/data/training-archive.json').read_text());write(Path(c['training_archive_receipt']),receipt)
 c.update(output=str(BASE/'run'),wandb=dict(id=RUN,name='ImageNet VAR joint4096 | best evaluated ep1 continuation | 4 RTX6000'))
 write(BASE/'config.json',c);write(OUT/'config.json',c)
 for folder in ['run','preflight']:
  target=BASE/folder;target.mkdir(exist_ok=True)
  checkpoint=target/'prior-last.pt'
  if not checkpoint.exists():checkpoint.hardlink_to(prior)
  write(target/'initialization.json',dict(initialization='resume_best_evaluated_prior',source_step=1668,source_epoch=1,prior_checkpoint_sha256=sha(prior),source_checkpoint_sha256=c['source_checkpoint_sha256'],codebook_sha256=c['codebook_sha256']))
 smoke=dict(c,output=str(BASE/'preflight'))
 write(BASE/'preflight-config.json',smoke)
 report=dict(passed=True,source_run='imnet-var-j4096-a1024-l4-d16-scratch200-4h100-20261102-r2',source_artifact='helloimlixin-rutgers/laser/imnet-var-j4096-a1024-l4-d16-scratch200-4h100-20261102-r2-prior:v2',selection='Best available completed 50k evaluation; epoch 1 is the only evaluated checkpoint',source_validation_fid50k=196.2973291315157,source_step=1668,source_epoch=1,optimizer_ages=ages,optimizer_parameter_states=len(p['optimizer']['state']),all_rank_rng_states=len(p['rng']),contract_sha256=digest,original_runtime_hashes_match=True,world_size=4,microbatch=64,accumulation=3,global_batch=768,schedule_epochs=200,schedule_updates=333600,warmup_updates=6672,schedule_reset=False,tokenizer_frozen=True,codebook_frozen=True,hardware='4 NVIDIA RTX PRO 6000 Blackwell 96GB',training_archive_md5=verified['md5'],validation_archive_md5=validation['md5'])
 write(OUT/'source-verification.json',report)
 print(json.dumps(report),flush=True)

if __name__=='__main__':main()
