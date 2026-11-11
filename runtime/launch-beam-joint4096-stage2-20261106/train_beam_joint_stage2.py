"""Resumable class-conditional ImageNet VAR-d16 over frozen joint integers."""
import argparse, contextlib, fcntl, functools, hashlib, io, json, math, os, random
import signal, tarfile, time
from datetime import timedelta
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from frozen_beam_tokenizer import load_source, LASER_HASH, digest_state
from src.models.beam_joint_var import BeamAlignedJointVAR
from src.models.scratch_var import state_digest
from src.original_rq_training import FeatureMoments
from src.rqvae_metrics import OriginalRQVAEInception, frechet_distance, DistributedOriginalRQVAEMetrics
from utils.lr_control import lr_wd_annealing


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def optimizer_groups(model):
    exempt=('pos_1LC','pos_start','lvl_embed','gamma','beta','ada_gss','scale_mul')
    groups=[dict(params=[],wd_sc=0.,lr_sc=1.),dict(params=[],wd_sc=1.,lr_sc=1.)]
    for name,p in model.named_parameters():
        if p.requires_grad:groups[0 if p.ndim==1 or name.endswith('bias') or any(x in name for x in exempt) else 1]['params'].append(p)
    return groups

class ArchiveImages(Dataset):
    def __init__(self,archive,index,train,seed,labels=None):
        self.archive=str(archive);self.train=train;self.seed=seed;self.epoch=0;self.fd=None;self.pid=None
        with np.load(index,allow_pickle=False) as data:
            self.offsets=data['offsets'];self.sizes=data['sizes']
            self.labels=data['labels'] if train else np.load(labels,allow_pickle=False)
        if len(self.offsets)!=(1281167 if train else 50000) or len(self.labels)!=len(self.offsets):raise ValueError('Incomplete dataset index')
        self.transform=transforms.Compose([transforms.Resize(288,interpolation=InterpolationMode.LANCZOS),transforms.RandomCrop(256) if train else transforms.CenterCrop(256),transforms.ToTensor(),transforms.Normalize(.5,.5)])
    def __len__(self):return len(self.labels)
    def __getitem__(self,index):
        index=int(index)
        if self.pid!=os.getpid():
            if self.fd is not None:os.close(self.fd)
            self.fd=os.open(self.archive,os.O_RDONLY);self.pid=os.getpid()
        raw=os.pread(self.fd,int(self.sizes[index]),int(self.offsets[index]))
        if len(raw)!=int(self.sizes[index]):raise OSError('Truncated original ImageNet JPEG')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed+self.epoch*10000019+index)
            with Image.open(io.BytesIO(raw)) as image:x=self.transform(image.convert('RGB'))
        return x,int(self.labels[index])
    def __getstate__(self):return dict(self.__dict__,fd=None,pid=None)
    def __del__(self):
        if getattr(self,'fd',None) is not None:os.close(self.fd)

class SmokeImages(Dataset):
    def __init__(self,path):
        d=torch.load(path,map_location='cpu',weights_only=True);self.images=d['images'];self.labels=d['labels'];self.epoch=0
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):return self.images[i],self.labels[i]

class OffsetSampler(Sampler):
    def __init__(self,sampler,offset):self.sampler,self.offset=sampler,offset
    def __iter__(self):return iter(list(self.sampler)[self.offset:])
    def __len__(self):return max(0,len(self.sampler)-self.offset)

@contextlib.contextmanager
def evaluation_rng(device):
    python=random.getstate();numpy=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index]):yield
    finally:random.setstate(python);np.random.set_state(numpy)

def capture_rng():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())

def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda']);random.setstate(state['python']);np.random.set_state(state['numpy'])

def should_stop(local_stop,stop_file,max_steps,step):
    return bool(local_stop or stop_file or (max_steps is not None and step>=max_steps))

class Experiment:
    def __init__(self,cfg,smoke=False):
        self.cfg=cfg;self.smoke=smoke;self.rank=dist.get_rank();self.world=dist.get_world_size()
        self.device=torch.device('cuda',int(os.environ['LOCAL_RANK']));self.out=Path(cfg['output']);self.out.mkdir(parents=True,exist_ok=True)
        self.stop=False;self.run=None;self.inception=None
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True));signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
        self.contract={k:v for k,v in cfg.items() if k not in ['output','wandb','checkpoint_every','preview_every','workers','source_bundle']}
        self.contract_sha=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
        for name,expected in cfg.get('runtime_sha256',{}).items():
            if sha(Path(__file__).parent/name)!=expected:raise ValueError('Frozen runtime changed: '+name)
        if self.rank==0:
            lock=(self.out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.lock=lock
            if (self.out/'contract.json').exists() and json.loads((self.out/'contract.json').read_text())!=self.contract:raise ValueError('Training contract changed')
            atomic(self.out/'contract.json',self.contract);atomic(self.out/'config.json',cfg)
        if cfg['source_checkpoint_sha256']!=LASER_HASH or sha(cfg['checkpoint'])!=LASER_HASH:raise ValueError('Wrong frozen source checkpoint')
        if sha(cfg['codebook'])!=cfg['codebook_sha256']:raise ValueError('Wrong joint codebook')
        self.source=load_source(Path(cfg['checkpoint']),'laser',self.device)
        spec=torch.load(cfg['codebook'],map_location='cpu',weights_only=True)
        if spec['source_checkpoint_sha256']!=LASER_HASH:raise ValueError('Codebook/source mismatch')
        self.q=self.source.quantize
        torch.testing.assert_close(self.q.dictionary.cpu(),spec['dictionary'],rtol=0,atol=0)
        torch.testing.assert_close(self.q.coefficients.cpu(),spec['coefficients'],rtol=0,atol=0)
        self.frozen_tokenizer_digest=digest_state(self.source)
        if (self.q.vocab_size,self.q.num_directions,self.q.levels)!=(4096,1024,4):raise ValueError('Requires the trained matched680/4096-symbol tokenizer')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['prior_seed']);self.model=BeamAlignedJointVAR(self.q,depth=16,num_classes=1000)
        if self.rank==0 and not (self.out/'initialization.json').exists():
            atomic(self.out/'initialization.json',dict(initialization='fresh_random_prior',seed=cfg['prior_seed'],prior_sha256=state_digest(self.model),parameters=sum(p.numel() for p in self.model.parameters()),codebook_sha256=cfg['codebook_sha256'],source_checkpoint_sha256=LASER_HASH))
        self.model.to(self.device);self.optimizer=torch.optim.AdamW(optimizer_groups(self.model),lr=cfg['lr'],betas=(.9,.95),weight_decay=.05,fused=True)
        self.progress=dict(epoch=0,batch=0,step=0)
        self.wrapped=DDP(self.model,device_ids=[self.device.index],broadcast_buffers=False)
        torch.manual_seed(cfg['seed']+self.rank);np.random.seed(cfg['seed']+self.rank);random.seed(cfg['seed']+self.rank)
        if smoke:
            self.train=ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed']);self.val=None
        else:
            ready=json.loads(Path(cfg['training_archive_receipt']).read_text())
            if not ready.get('verified') or ready['md5']!='1d675b47d978889d74fa0da5fadfb00e':raise ValueError('Training archive not verified')
            self.train=ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed'])
            self.val=ArchiveImages(cfg['val_archive'],cfg['val_index'],False,cfg['seed'],labels=cfg['val_labels'])
        self.micro=cfg['microbatch'];self.accum=cfg['accumulation'];self.global_batch=self.micro*self.accum*self.world
        if self.global_batch!=cfg['global_batch']:raise ValueError('Global batch mismatch')
        self.updates=((len(self.train)+self.world-1)//self.world//self.micro)//self.accum
        if self.updates<1:raise ValueError('Dataset too small for effective batch')
        if (self.out/'prior-last.pt').exists():self.restore()
        elif not cfg['fresh_prior']:raise ValueError('No prior checkpoint available')
        if self.rank==0 and not smoke:
            import wandb
            self.run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=cfg['wandb']['id'],name=cfg['wandb']['name'],group='imagenet-var-beam16-joint4096-stage2-20261106',job_type='class-conditional-stage2',resume='allow',dir=str(self.out),config=cfg)
            self.run.define_metric('train/step');self.run.define_metric('train/*',step_metric='train/step')
            self.run.summary.update({'training/status':'running','tokenizer/source_greedy_stage1_rfid':cfg['source_greedy_stage1_rfid'],'tokenizer/frozen':True,'prior/fresh_initialization':True,'prior/parameters':sum(p.numel() for p in self.model.parameters()),'training/global_batch':self.global_batch,'training/updates_per_epoch':self.updates,'training/target_epochs':cfg['epochs'],'training/class_conditional':True})
            a=wandb.Artifact(cfg['wandb']['id']+'-source',type='source',metadata={'contract_sha256':self.contract_sha,'codebook_sha256':cfg['codebook_sha256']})
            a.add_file(cfg['source_bundle'],name='source.tar.gz');a.add_file(str(self.out/'config.json'));a.add_file(cfg['codebook'],name='joint-codebook.pt')
            for key in ['train_index','val_index','val_labels','fid_reference','training_archive_receipt']:a.add_file(cfg[key],name='data/'+Path(cfg[key]).name)
            self.run.log_artifact(a).wait()
            atomic(self.out/'wandb.json',{'url':self.run.url,'id':self.run.id})
        if self.run:
            self.run.use_artifact(cfg['tokenizer_artifact'])
            self.run.summary.update({'training/stage':'stage2','tokenizer/tokens_per_image':680,'tokenizer/vocabulary':4096,'tokenizer/nominal_bits_per_image':8160,'training/world_size':self.world,'training/microbatch_per_gpu':self.micro,'training/accumulation':self.accum,'evaluation/cadence':'epoch_end_only','tokenizer/beam_width':16,'tokenizer/pair_format':'atom_id*4+level_id','tokenizer/separate_coefficient_stream':False,'teacher/beam_aligned':True})
        self.log('setup',parameters=sum(p.numel() for p in self.model.parameters()),global_batch=self.global_batch,updates_per_epoch=self.updates)

    def log(self,phase,**values):
        if self.rank==0:
            record=dict(phase=phase,time=time.time(),**values);atomic(self.out/'status.json',record);print(json.dumps(record),flush=True)
            if self.run:self.run.log({phase+'/'+k:v for k,v in values.items()})

    @torch.no_grad()
    def tokenize(self,images,*,teacher=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):latent=self.source.quant_conv(self.source.encoder(images))
        with torch.autocast('cuda',enabled=False):
            result=self.q.decompose(latent,teacher=teacher)
            if self.smoke:
                assert result['codes'].dtype==torch.long and result['codes'].shape[1:]==(340,2)
                assert int(result['codes'].min())>=0 and int(result['codes'].max())<4096
                decoded,inputs=self.q.from_codes(result['codes'])
                torch.testing.assert_close(decoded,result['latent'],atol=0,rtol=0)
                torch.testing.assert_close(inputs,result['inputs'],atol=0,rtol=0)
            return result

    def save(self,publish=False):
        rng=[None]*self.world;dist.all_gather_object(rng,capture_rng())
        if self.rank==0:
            p=self.out/'prior-last.pt';temporary=p.with_suffix('.tmp')
            torch.save(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),progress=self.progress,rng=rng,contract_sha256=self.contract_sha,tokenizer_sha256=LASER_HASH,codebook_sha256=self.cfg['codebook_sha256'],world_size=self.world),temporary);temporary.replace(p)
            receipt=dict(progress=self.progress,bytes=p.stat().st_size,checkpoint_sha256=sha(p),contract_sha256=self.contract_sha,full_state=True)
            atomic(self.out/'checkpoint.json',receipt)
            if publish and self.run:
                import wandb
                a=wandb.Artifact(self.cfg['wandb']['id']+'-prior',type='model',metadata={**receipt,'codebook_sha256':self.cfg['codebook_sha256'],'full_state':True})
                a.add_file(str(p),name='checkpoint.pt',policy='immutable');a.add_file(str(self.out/'config.json'));a.add_file(self.cfg['codebook'],name='joint-codebook.pt')
                aliases=['latest',f"step-{self.progress['step']:09d}"]
                if self.progress['batch']==0:aliases.append(f"epoch-{self.progress['epoch']:03d}")
                logged=self.run.log_artifact(a,aliases=aliases);logged.wait();receipt['artifact']=logged.qualified_name;atomic(self.out/'checkpoint.json',receipt)
                self.run.summary['checkpoint/latest_step']=self.progress['step']
        dist.barrier()

    def restore(self):
        p=torch.load(self.out/'prior-last.pt',map_location='cpu',weights_only=False,mmap=True)
        if p['contract_sha256']!=self.contract_sha or p['codebook_sha256']!=self.cfg['codebook_sha256'] or p['world_size']!=self.world or p['tokenizer_sha256']!=LASER_HASH:raise ValueError('Incompatible prior checkpoint')
        self.model.load_state_dict(p['model']);self.optimizer.load_state_dict(p['optimizer']);self.progress=dict(p['progress']);restore_rng(p['rng'][self.rank])
        self.log('resume',step=self.progress['step'],epoch=self.progress['epoch'],all_rank_rng_restored=True)

    def loader(self,dataset,indices=None,batch=None):
        return DataLoader(Subset(dataset,indices) if indices is not None else dataset,batch_size=batch or self.micro,num_workers=self.cfg['workers'] if not self.smoke else 0,pin_memory=True,multiprocessing_context='spawn' if self.cfg['workers'] and not self.smoke else None,generator=torch.Generator().manual_seed(self.cfg['seed']+self.rank))

    @torch.no_grad()
    def preview(self):
        # All ranks enter the RNG guard; only rank zero needs the preview batch.
        with evaluation_rng(self.device):
            self.model.eval()
            if self.rank==0:
                labels=torch.tensor([22,277,258,849],device=self.device).repeat_interleave(8)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    latent=self.model.sample(labels,cfg=1.5,top_k=900,top_p=.96,seed=7919)
                    images=self.source.decoder(self.source.post_quant_conv(latent)).float().clamp(-1,1)
                if not images.isfinite().all():raise FloatingPointError('Nonfinite generated images')
                path=self.out/f"samples-step{self.progress['step']:09d}.png";save_image((images+1)*.5,path,nrow=8)
                if self.run:
                    import wandb
                    self.run.log({'samples/class_conditional':wandb.Image(str(path),caption='Rows: bald eagle (22), red fox (277), Samoyed (258), teapot (849); CFG1.5, top-k900, top-p.96; 8 independent draws per class')})
                    self.run.summary['samples/latest_step']=self.progress['step']
                atomic(self.out/'last-preview.json',dict(step=self.progress['step'],path=str(path),class_ids=[22,277,258,849],cfg=1.5,top_k=900,top_p=.96,finite=True))
            dist.barrier();self.model.train()

    @torch.no_grad()
    def validate(self):
        with evaluation_rng(self.device):
            self.model.eval();drop=self.model.cond_drop_rate;self.model.cond_drop_rate=0.
            indices=np.random.default_rng(123).permutation(len(self.val))[:1024][self.rank::self.world]
            total=torch.zeros(3,device=self.device,dtype=torch.float64)
            try:
                for images,labels in self.loader(self.val,indices):
                    labels=labels.to(self.device);code=self.tokenize(images.to(self.device,non_blocking=True))
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=self.model(labels,code['inputs'],code['codes']);wrong=self.model((labels+1)%1000,code['inputs'],code['codes'])
                    total+=torch.stack((loss,wrong,loss.new_tensor(1.))).double()*len(images)
                dist.all_reduce(total);self.log('validation',step=self.progress['step'],epoch=self.progress['epoch'],joint_nll=float(total[0]/total[2]),wrong_label_nll=float(total[1]/total[2]),class_conditioning_gain=float((total[1]-total[0])/total[2]),images=int(total[2]))
            finally:self.model.cond_drop_rate=drop;self.model.train()

    @torch.no_grad()
    def fid(self):
        assert self.progress['batch']==0 and self.progress['step']==self.progress['epoch']*self.updates
        receipt=self.out/f"fid-epoch{self.progress['epoch']:03d}.json"
        done=torch.tensor(int(receipt.exists()) if self.rank==0 else 0,device=self.device);dist.broadcast(done,src=0)
        if done.item():return
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

    def train_loop(self,max_steps=None):
        if max_steps and self.progress['step']>=max_steps:return
        if not self.smoke and self.progress['epoch']>0 and self.progress['batch']==0:self.fid()
        for epoch in range(self.progress['epoch'],self.cfg['epochs']):
            self.train.epoch=epoch;sampler=DistributedSampler(self.train,self.world,self.rank,shuffle=True,seed=self.cfg['seed']);sampler.set_epoch(epoch)
            begin=self.progress['batch'];loader=DataLoader(self.train,batch_size=self.micro,sampler=OffsetSampler(sampler,begin*self.micro),num_workers=self.cfg['workers'] if not self.smoke else 0,pin_memory=True,drop_last=True,multiprocessing_context='spawn' if self.cfg['workers'] and not self.smoke else None,generator=torch.Generator().manual_seed(self.cfg['seed']+epoch+self.rank))
            self.model.train();self.optimizer.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device=self.device);tick=time.monotonic()
            for batch,(images,labels) in enumerate(loader,start=begin):
                if batch>=self.updates*self.accum:break
                if batch%self.accum==0:lr_wd_annealing('lin0',self.optimizer,self.cfg['lr'],.05,.05,self.progress['step'],self.cfg['warmup_epochs']*self.updates,self.cfg['epochs']*self.updates,wp0=.005,wpe=.1)
                labels=labels.to(self.device,non_blocking=True);codes=self.tokenize(images.to(self.device,non_blocking=True),teacher=True);sync=(batch+1)%self.accum==0
                with (contextlib.nullcontext() if sync else self.wrapped.no_sync()),torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=self.wrapped(labels,codes['inputs'],codes['codes'],codes['targets']);(loss/self.accum).backward()
                loss_sum+=loss.detach()/self.accum
                if not sync:continue
                class_gradient=float(self.model.class_emb.weight.grad.norm())
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg['gradient_clip'],error_if_nonfinite=True);self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
                self.progress=dict(epoch=epoch,batch=batch+1,step=self.progress['step']+1)
                dist.all_reduce(loss_sum);loss_sum/=self.world
                if self.progress['step']<=10 or self.progress['step']%10==0:self.log('train',step=self.progress['step'],epoch=epoch,epoch_progress=epoch+(batch+1)/(self.updates*self.accum),joint_nll=float(loss_sum),uniform_joint_nll=(math.log(4096)+math.log(4092))*.5,gain_vs_uniform=(math.log(4096)+math.log(4092))*.5-float(loss_sum),gradient_norm=float(norm),class_embedding_gradient_norm=class_gradient,lr=self.optimizer.param_groups[0]['lr'],images_per_second=self.global_batch/(time.monotonic()-tick),images_seen=self.progress['step']*self.global_batch,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                loss_sum.zero_();tick=time.monotonic()
                flag=torch.tensor(int(should_stop(self.stop,(self.out/'STOP').exists(),max_steps,self.progress['step'])),device=self.device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
                checkpoint=self.progress['step'] in [1,10] or self.progress['step']%self.cfg['checkpoint_every']==0 or flag.item()
                if checkpoint:self.save(publish=not self.smoke and (self.progress['step']==10 or self.progress['step']%self.cfg['checkpoint_every']==0 or bool(flag.item())))
                if self.progress['step']==1 or self.progress['step']%self.cfg['preview_every']==0:self.preview()
                if flag.item():return
            self.progress.update(epoch=epoch+1,batch=0);self.save(publish=not self.smoke)
            if not self.smoke:
                self.validate()
                self.fid()
        if self.run:self.run.summary['training/status']='completed';self.run.finish()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--max-steps',type=int);a=p.parse_args();cfg=json.loads(a.config.read_text())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl',timeout=timedelta(hours=2))
    experiment=None
    try:
        experiment=Experiment(cfg,a.smoke);experiment.train_loop(a.max_steps)
        if a.smoke:
            previous=dict(experiment.progress);experiment.restore()
            if experiment.progress!=previous:raise ValueError('Checkpoint progress failed to restore')
            saved=torch.load(experiment.out/'prior-last.pt',map_location='cpu',weights_only=False,mmap=True)
            current=capture_rng();expected=saved['rng'][experiment.rank]
            if not torch.equal(current['torch'],expected['torch']) or not torch.equal(current['cuda'],expected['cuda']):raise ValueError('RNG roundtrip differs')
            if not torch.equal(experiment.model.head.weight.detach().cpu(),saved['model']['head.weight']):raise ValueError('Model roundtrip differs')
            ages={int(s['step']) for s in experiment.optimizer.state.values() if 'step' in s}
            if ages!={previous['step']}:raise ValueError('Optimizer age roundtrip differs')
            if any(p.requires_grad or p.grad is not None for p in experiment.source.parameters()):raise ValueError('Frozen source received gradients')
            if digest_state(experiment.source)!=experiment.frozen_tokenizer_digest:raise ValueError('Tokenizer changed during prior preflight')
            assert experiment.model.num_classes==1000 and experiment.model.cond_drop_rate==.1
            assert experiment.model.L*experiment.model.sparsity==680
            assert experiment.updates==1668
            if experiment.rank==0:atomic(experiment.out/'smoke-complete.json',dict(completed=True,progress=previous,full_state_roundtrip=True,model_head_bit_exact=True,all_rank_rng_bit_exact=True,optimizer_age=list(ages),source_frozen=True,sampling_finite=True,parameters=sum(p.numel() for p in experiment.model.parameters()),global_batch=experiment.global_batch,microbatch=experiment.micro,accumulation=experiment.accum,updates_per_epoch=experiment.updates,tokenizer_digest_unchanged=True,class_conditional=True,num_classes=1000,tokens_per_image=680,vocabulary=4096,pair_id='atom_id*4+level_id',beam_width=16,beam_aligned_teacher=True,separate_coefficient_stream=False,actual_integer_roundtrip_bit_exact=True,gradient_clip=cfg['gradient_clip'],peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))
        elif experiment.run and experiment.progress['epoch']<cfg['epochs']:
            experiment.run.summary['training/status']='stopped_after_checkpoint';experiment.run.finish()
    except Exception as error:
        if experiment is not None:
            atomic(experiment.out/f'failure-rank{experiment.rank}.json',dict(type=type(error).__name__,error=str(error),progress=experiment.progress))
            if experiment.run:experiment.run.summary['training/status']='failed';experiment.run.finish(exit_code=1)
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
